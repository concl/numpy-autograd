import numpy as np
import pytest
from numba import cuda

from autograd_engine.triton_kernels import matmul


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shape", [(3, 5, 7), (129, 17, 133), (5, 137, 3)])
def test_matmul_matches_numpy(dtype, shape):
    m, k, n = shape
    rng = np.random.default_rng(4)
    # Transposed views exercise the wrapper's input normalization.
    a = rng.standard_normal((k, m)).astype(dtype).T
    b = rng.standard_normal((n, k)).astype(dtype).T
    actual = matmul(cuda.to_device(a), cuda.to_device(b)).copy_to_host()
    expected = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
    assert actual.dtype == dtype


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_matmul_empty_and_validation():
    a = cuda.to_device(np.empty((2, 0), np.float32))
    b = cuda.to_device(np.empty((0, 3), np.float32))
    assert matmul(a, b).shape == (2, 3)
    np.testing.assert_array_equal(
        matmul(a, b).copy_to_host(),
        np.zeros((2, 3), np.float32),
    )
    with pytest.raises(ValueError, match="inner dimensions"):
        matmul(cuda.to_device(np.ones((2, 3), np.float32)), cuda.to_device(np.ones((4, 2), np.float32)))
    with pytest.raises(TypeError, match="float16 or float32"):
        matmul(cuda.to_device(np.ones((2, 2), np.float64)), cuda.to_device(np.ones((2, 2), np.float64)))


def test_matmul_rejects_numpy_arrays():
    with pytest.raises(TypeError, match="CUDA arrays"):
        matmul(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32))
