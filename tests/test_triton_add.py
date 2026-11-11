import numpy as np
import pytest
from numba import cuda

from autograd_engine.triton_kernels import add


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shapes", [((35, 37), (35, 37)), ((35, 37), (37,)),
                                     ((35, 1), (1, 37)), ((), ()), ((5,), ())])
def test_add_broadcast(dtype, shapes):
    rng = np.random.default_rng(5)
    a, b = (np.asarray(rng.standard_normal(shape), dtype=dtype) for shape in shapes)
    result = add(cuda.to_device(a), cuda.to_device(b)).copy_to_host()
    np.testing.assert_allclose(result, a + b, rtol=1e-3, atol=1e-3)
    assert result.shape == np.broadcast_shapes(*shapes)
    assert result.dtype == dtype


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_add_device_strides():
    a = np.arange(35 * 37, dtype=np.float32).reshape(37, 35)
    b = np.arange(35, dtype=np.float32)[:, None]
    a_device = cuda.to_device(a).transpose()
    b_device = cuda.to_device(b)
    result = add(a_device, b_device)
    assert cuda.is_cuda_array(result)
    np.testing.assert_array_equal(result.copy_to_host(), a.T + b)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is unavailable")
def test_add_validation_and_empty():
    result = add(cuda.to_device(np.empty((0, 3), np.float32)), cuda.to_device(np.ones((1, 3), np.float32)))
    assert result.shape == (0, 3)
    with pytest.raises(ValueError, match="broadcast"):
        add(cuda.to_device(np.ones((2, 3), np.float32)), cuda.to_device(np.ones((4,), np.float32)))
    with pytest.raises(TypeError, match="float16 or float32"):
        add(cuda.to_device(np.ones(2, np.int32)), cuda.to_device(np.ones(2, np.int32)))
    actual = add(cuda.to_device(np.ones((1, 1, 1), np.float32)), cuda.to_device(np.ones(1, np.float32)))
    np.testing.assert_array_equal(actual.copy_to_host(), np.full((1, 1, 1), 2))


def test_add_rejects_numpy_arrays():
    with pytest.raises(TypeError, match="CUDA arrays"):
        add(np.ones(2, np.float32), np.ones(2, np.float32))
