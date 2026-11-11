import numpy as np
import pytest
from numba import cuda

from autograd_engine.triton_kernels import relu


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shape", [(), (257,), (19, 23), (0, 3)])
def test_relu_matches_numpy(dtype, shape):
    rng = np.random.default_rng(23)
    x = np.asarray(rng.standard_normal(shape), dtype=dtype)
    result = relu(cuda.to_device(x))
    assert cuda.is_cuda_array(result)
    assert result.shape == shape and result.dtype == dtype
    np.testing.assert_array_equal(result.copy_to_host(), np.maximum(x, 0))


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_relu_strided_and_nan():
    x = np.array([[np.nan, -1, 2], [3, -4, 0]], dtype=np.float32)
    device = cuda.to_device(x).transpose()
    np.testing.assert_array_equal(relu(device).copy_to_host(), np.maximum(x.T, 0))


def test_relu_rejects_host_array():
    with pytest.raises(TypeError, match="CUDA array"):
        relu(np.ones(3, dtype=np.float32))


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_relu_rejects_unsupported_input():
    with pytest.raises(TypeError, match="float16 or float32"):
        relu(cuda.to_device(np.ones(3, dtype=np.int32)))
    actual = relu(cuda.to_device(np.ones((1, 1, 1), dtype=np.float32)))
    np.testing.assert_array_equal(actual.copy_to_host(), np.ones((1, 1, 1)))
