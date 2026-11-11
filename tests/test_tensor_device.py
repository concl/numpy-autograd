import numpy as np
import pytest
from numba import cuda

from autograd_engine.engine import Tensor, cross_entropy_loss, matmul, relu, softmax
from autograd_engine.optimizers import SGD
import autograd_engine.engine as engine


def test_cpu_storage_and_invalid_device():
    x = Tensor([1, 2], requires_grad=True)
    assert x.device == "cpu"
    assert isinstance(x.data, np.ndarray)
    assert x.to("cpu") is x
    x.backward()
    np.testing.assert_array_equal(x.grad, [1, 1])
    with pytest.raises(ValueError, match="device must"):
        x.to("cuda:1")


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_storage_operations_and_gradients():
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True, device="cuda")
    b = Tensor(np.array([[2, 1], [1, 2]], dtype=np.float32), requires_grad=True, device="cuda")
    out = relu(matmul(a, b) + b)
    assert out.device == "cuda" and cuda.is_cuda_array(out.data)
    np.testing.assert_allclose(out.cpu().data, [[6, 6], [11, 13]])
    out.backward()
    assert cuda.is_cuda_array(a.grad) and cuda.is_cuda_array(b.grad)
    np.testing.assert_allclose(a.grad.copy_to_host(), np.full((2, 2), 3))
    np.testing.assert_allclose(b.grad.copy_to_host(), [[5, 5], [7, 7]])
    with pytest.raises(ValueError, match="same device"):
        _ = a + Tensor([1, 2])


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_transfers_loss_and_optimizer():
    original = Tensor(np.array([[1., 2.]], dtype=np.float32), requires_grad=True)
    x = original.cuda()
    assert x.device == "cuda" and x.cpu().device == "cpu"
    labels = Tensor(np.array([1]), device="cuda")
    probs = softmax(x)
    np.testing.assert_allclose(probs.cpu().data.sum(axis=-1), 1)
    loss = cross_entropy_loss(x, labels)
    assert loss.shape == () and loss.device == "cuda"
    loss.backward()
    assert isinstance(original.grad, np.ndarray)
    np.testing.assert_allclose(original.grad, [[0.2689414, -0.2689414]], rtol=1e-5)

    param = Tensor(np.array([1., 2.], np.float32), requires_grad=True, device="cuda")
    (param * param).backward()
    SGD([param], lr=0.1).step()
    assert param.device == "cuda" and cuda.is_cuda_array(param.data)
    np.testing.assert_allclose(param.cpu().data, [0.8, 1.6])


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_cuda_matmul_uses_triton_without_host_copy(monkeypatch, dtype):
    a_np = np.arange(15, dtype=dtype).reshape(3, 5)
    b_np = np.arange(35, dtype=dtype).reshape(5, 7)
    a = Tensor(a_np, requires_grad=True, device="cuda")
    b = Tensor(b_np, requires_grad=True, device="cuda")
    original = engine.matmul_triton
    calls = []

    def launch(left, right):
        calls.append((left, right))
        return original(left, right)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "matmul_triton", launch)
        patch.setattr(engine, "_to_numpy", lambda _: pytest.fail("unexpected host copy"))
        result = a @ b
    assert calls == [(a.data, b.data)]
    assert cuda.is_cuda_array(result.data)
    np.testing.assert_allclose(result.data.copy_to_host(), a_np @ b_np, rtol=0.01, atol=0.01)
    result.backward()
    np.testing.assert_allclose(a.grad.copy_to_host(), np.ones((3, 7)) @ b_np.T, rtol=0.01)
    np.testing.assert_allclose(b.grad.copy_to_host(), a_np.T @ np.ones((3, 7)), rtol=0.01)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matmul_empty_and_dtype_validation():
    a = Tensor(np.empty((2, 0), np.float32), device="cuda")
    b = Tensor(np.empty((0, 3), np.float32), device="cuda")
    np.testing.assert_array_equal((a @ b).data.copy_to_host(), np.zeros((2, 3)))
    a64 = Tensor(np.ones((2, 2), np.float64), device="cuda")
    with pytest.raises(TypeError, match="float16 or float32"):
        _ = a64 @ a64
