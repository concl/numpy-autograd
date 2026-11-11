import numpy as np
import pytest
from numba import cuda

from autograd_engine import engine
from autograd_engine.engine import Tensor, cross_entropy_loss, softmax
from autograd_engine.modules import FFN
from autograd_engine.optimizers import SGD


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_ffn_training_step_matches_cpu_without_host_copies(monkeypatch):
    np.random.seed(12)
    cpu_model = FFN(8, 6, 3)
    np.random.seed(12)
    gpu_model = FFN(8, 6, 3).to("cuda")
    rng = np.random.default_rng(3)
    inputs = rng.standard_normal((4, 8)).astype(np.float32)
    labels = np.array([0, 2, 1, 2], dtype=np.int64)
    cpu_x, cpu_y = Tensor(inputs), Tensor(labels)
    gpu_x, gpu_y = Tensor(inputs, device="cuda"), Tensor(labels, device="cuda")
    cpu_optim = SGD(cpu_model.parameters, lr=0.01, momentum=0.9)
    gpu_optim = SGD(gpu_model.parameters, lr=0.01, momentum=0.9)

    for _ in range(2):
        cpu_optim.zero_grad()
        cpu_loss = cross_entropy_loss(cpu_model(cpu_x), cpu_y)
        cpu_loss.backward()
        cpu_optim.step()
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_to_numpy", lambda _: pytest.fail("unexpected host transfer"))
            gpu_optim.zero_grad()
            gpu_loss = cross_entropy_loss(gpu_model(gpu_x), gpu_y)
            gpu_loss.backward()
            gpu_optim.step()
        np.testing.assert_allclose(gpu_loss.data.copy_to_host(), cpu_loss.data, rtol=2e-3, atol=2e-3)
        for cpu_p, gpu_p in zip(cpu_model.parameters, gpu_model.parameters):
            assert cuda.is_cuda_array(gpu_p.data) and cuda.is_cuda_array(gpu_p.grad)
            np.testing.assert_allclose(gpu_p.grad.copy_to_host(), cpu_p.grad, rtol=2e-2, atol=2e-3)
            np.testing.assert_allclose(gpu_p.data.copy_to_host(), cpu_p.data, rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_multiply_broadcast_and_softmax_backward():
    x_np = np.array([[1., 2., 3.], [2., 3., 4.]], np.float32)
    b_np = np.array([2., 3., 4.], np.float32)
    results = []
    for device in ("cpu", "cuda"):
        x = Tensor(x_np, requires_grad=True, device=device)
        b = Tensor(b_np, requires_grad=True, device=device)
        output = softmax(x * b)
        output.backward(np.array([[1., 2., -1.], [0., 1., 3.]], np.float32))
        results.append((engine._to_numpy(output.data), engine._to_numpy(x.grad), engine._to_numpy(b.grad)))
    for cpu, gpu in zip(*results):
        np.testing.assert_allclose(gpu, cpu, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_gradient_across_cpu_cuda_boundary():
    x = Tensor(np.array([[1., 2.]], dtype=np.float32), requires_grad=True)
    w = Tensor(np.array([[2.], [3.]], dtype=np.float32), requires_grad=True, device="cuda")
    result = (x.cuda() @ w).cpu()
    result.backward()
    np.testing.assert_allclose(x.grad, [[2., 3.]])
    np.testing.assert_allclose(w.grad.copy_to_host(), [[1.], [2.]])
