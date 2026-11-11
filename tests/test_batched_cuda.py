import numpy as np
import pytest
from numba import cuda

from autograd_engine.engine import Tensor, relu, softmax, _swap_last_two_cuda
from autograd_engine import triton_kernels as kernels

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("a_shape,b_shape", [
    ((2, 3, 4), (2, 4, 5)), ((2, 1, 3, 4), (1, 3, 4, 5)),
    ((3, 4), (2, 4, 5)), ((4,), (2, 4, 5)),
    ((2, 3, 4), (4,)), ((4,), (4,)), ((2, 3, 0), (1, 0, 5)),
    ((0, 3, 4), (1, 4, 5)),
])
def test_batched_matmul(a_shape, b_shape):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    out = kernels.matmul(cuda.to_device(a), cuda.to_device(b))
    np.testing.assert_allclose(out.copy_to_host(), a @ b, atol=1e-2, rtol=1e-2)


def test_broadcasted_batched_matmul_gradients():
    rng = np.random.default_rng(7)
    a_np = rng.standard_normal((2, 1, 3, 4)).astype(np.float32)
    b_np = rng.standard_normal((1, 3, 4, 5)).astype(np.float32)
    grad = rng.standard_normal((2, 3, 3, 5)).astype(np.float32)
    a = Tensor(a_np, requires_grad=True, device="cuda")
    b = Tensor(b_np, requires_grad=True, device="cuda")
    (a @ b).backward(grad)
    expected_a = (grad @ np.swapaxes(b_np, -1, -2)).sum(axis=1, keepdims=True)
    expected_b = (np.swapaxes(a_np, -1, -2) @ grad).sum(axis=0, keepdims=True)
    np.testing.assert_allclose(a.grad.copy_to_host(), expected_a, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(b.grad.copy_to_host(), expected_b, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("a_shape,b_shape", [((4,), (4,)), ((4,), (2, 4, 3)), ((2, 3, 4), (4,))])
def test_vector_matmul_gradients(a_shape, b_shape):
    rng = np.random.default_rng(6)
    a_np = rng.standard_normal(a_shape).astype(np.float32)
    b_np = rng.standard_normal(b_shape).astype(np.float32)
    a = Tensor(a_np, requires_grad=True, device="cuda")
    b = Tensor(b_np, requires_grad=True, device="cuda")
    result = a @ b
    result.backward()
    eps = 1e-2
    for tensor, original, other, left in ((a, a_np, b_np, True), (b, b_np, a_np, False)):
        numerical = np.zeros_like(original)
        for index in np.ndindex(original.shape):
            plus, minus = original.copy(), original.copy()
            plus[index] += eps
            minus[index] -= eps
            numerical[index] = (((plus @ other).sum() - (minus @ other).sum()) if left else
                                ((other @ plus).sum() - (other @ minus).sum())) / (2 * eps)
        np.testing.assert_allclose(tensor.grad.copy_to_host(), numerical, atol=2e-2, rtol=2e-2)


def test_batched_elementwise_and_softmax_gradients():
    rng = np.random.default_rng(8)
    x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b_np = rng.standard_normal((1, 3, 1)).astype(np.float32)
    x = Tensor(x_np, requires_grad=True, device="cuda")
    b = Tensor(b_np, requires_grad=True, device="cuda")
    out = softmax(relu(x + b) * b)
    grad = rng.standard_normal(out.shape).astype(np.float32)
    out.backward(grad)
    y = np.maximum(x_np + b_np, 0) * b_np
    p = np.exp(y - y.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    gy = p * (grad - (p * grad).sum(axis=-1, keepdims=True))
    gx = gy * b_np * ((x_np + b_np) > 0)
    gb = (gy * np.maximum(x_np + b_np, 0) + gx).sum(axis=(0, 2), keepdims=True)
    np.testing.assert_allclose(out.data.copy_to_host(), p, atol=1e-6)
    np.testing.assert_allclose(x.grad.copy_to_host(), gx, atol=1e-5)
    np.testing.assert_allclose(b.grad.copy_to_host(), gb, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_batched_strided_inputs(dtype):
    rng = np.random.default_rng(9)
    a = rng.standard_normal((2, 3, 4)).astype(dtype)
    b = rng.standard_normal((2, 3, 5)).astype(dtype)
    view = _swap_last_two_cuda(cuda.to_device(a))
    out = kernels.matmul(view, cuda.to_device(b))
    expected = np.swapaxes(a.astype(np.float32), -1, -2) @ b.astype(np.float32)
    np.testing.assert_allclose(out.copy_to_host(), expected, atol=2e-2, rtol=2e-2)


def test_cpu_batched_matmul_broadcast_gradients():
    a_np = np.arange(24, dtype=np.float32).reshape(2, 1, 3, 4)
    b_np = np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)
    a, b = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
    (a @ b).backward()
    grad = np.ones((2, 3, 3, 5), np.float32)
    np.testing.assert_allclose(a.grad, (grad @ np.swapaxes(b_np, -1, -2)).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(b.grad, (np.swapaxes(a_np, -1, -2) @ grad).sum(axis=0, keepdims=True))
