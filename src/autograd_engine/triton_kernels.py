import numpy as np
from numba import cuda
import triton
import triton.language as tl

# ----- Triton Kernels -----


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ai,
    stride_ak,
    stride_bk,
    stride_bj,
    stride_ci,
    stride_cj,
    BATCH_SHAPE: tl.constexpr,
    A_BATCH_STRIDES: tl.constexpr,
    B_BATCH_STRIDES: tl.constexpr,
):
    """
    a_ptr: lhs matrix
    b_ptr: rhs matrix
    c_ptr: output matrix

    M: rows of lhs matrix
    N: cols of rhs matrix
    K: inner dimension (cols of lhs matrix, rows of rhs matrix)
    """

    # Define block size
    BLOCK_SIZE_M: tl.constexpr = 128
    BLOCK_SIZE_N: tl.constexpr = 128
    BLOCK_SIZE_K: tl.constexpr = 32

    # Compute the block indices
    program = tl.program_id(0)
    tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    batch = program // (tiles_m * tiles_n)
    tile = program % (tiles_m * tiles_n)
    pid_i = tile // tiles_n
    pid_j = tile % tiles_n
    remaining = batch
    a_batch_offset = 0
    b_batch_offset = 0
    for axis in tl.static_range(len(BATCH_SHAPE) - 1, -1, -1):
        coordinate = remaining % BATCH_SHAPE[axis]
        remaining = remaining // BATCH_SHAPE[axis]
        a_batch_offset += coordinate * A_BATCH_STRIDES[axis]
        b_batch_offset += coordinate * B_BATCH_STRIDES[axis]
    a_ptr += a_batch_offset
    b_ptr += b_batch_offset
    c_ptr += batch * M * N

    start_i = pid_i * BLOCK_SIZE_M
    start_j = pid_j * BLOCK_SIZE_N

    block_result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_SIZE_K):

        block_ai = tl.arange(0, BLOCK_SIZE_M) + start_i
        block_ak = tl.arange(0, BLOCK_SIZE_K) + start_k

        block_bk = tl.arange(0, BLOCK_SIZE_K) + start_k
        block_bj = tl.arange(0, BLOCK_SIZE_N) + start_j

        block_a_idx = block_ai[:, None] * stride_ai + block_ak[None, :] * stride_ak
        block_b_idx = block_bk[:, None] * stride_bk + block_bj[None, :] * stride_bj

        block_a = tl.load(
            a_ptr + block_a_idx,
            (block_ai[:, None] < M) & (block_ak[None, :] < K),
            other=0,
        )
        block_b = tl.load(
            b_ptr + block_b_idx,
            (block_bk[:, None] < K) & (block_bj[None, :] < N),
            other=0,
        )

        block_result += tl.dot(block_a, block_b)

    block_ci = tl.arange(0, BLOCK_SIZE_M) + start_i
    block_cj = tl.arange(0, BLOCK_SIZE_N) + start_j
    block_c_idx = block_ci[:, None] * stride_ci + block_cj[None, :] * stride_cj

    tl.store(
        c_ptr + block_c_idx,
        block_result,
        (block_ci[:, None] < M) & (block_cj[None, :] < N),
    )


@triton.jit
def _zero_kernel(out_ptr, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, 0, offsets < size)


@triton.jit
def _elementwise_kernel(a_ptr, b_ptr, out_ptr, size, SHAPE: tl.constexpr,
                        A_STRIDES: tl.constexpr, B_STRIDES: tl.constexpr,
                        OP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remaining = offsets
    a_offsets = tl.full((BLOCK,), 0, tl.int64)
    b_offsets = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remaining % SHAPE[axis]
        remaining = remaining // SHAPE[axis]
        a_offsets += coord * A_STRIDES[axis]
        b_offsets += coord * B_STRIDES[axis]
    a = tl.load(a_ptr + a_offsets, offsets < size, other=0)
    if OP == 2:
        value = tl.where(a < 0, 0, a)
    else:
        b = tl.load(b_ptr + b_offsets, offsets < size, other=0)
        if OP == 0:
            value = a + b
        else:
            value = a * b
    tl.store(out_ptr + offsets, value, offsets < size)


@triton.jit
def _unbroadcast_kernel(grad_ptr, out_ptr, TARGET: tl.constexpr, GRAD: tl.constexpr,
                        GSTRIDES: tl.constexpr, REDUCE_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    out_index = tl.program_id(0)
    remaining = out_index
    base = 0
    for axis in tl.static_range(len(TARGET) - 1, -1, -1):
        coordinate = remaining % TARGET[axis]
        remaining = remaining // TARGET[axis]
        if TARGET[axis] != 1:
            base += coordinate * GSTRIDES[axis]
    accumulator = tl.full((BLOCK,), 0, tl.float32)
    for start in range(0, REDUCE_SIZE, BLOCK):
        indices = start + tl.arange(0, BLOCK)
        remainder = indices
        offset = tl.full((BLOCK,), 0, tl.int64) + base
        for axis in tl.static_range(len(TARGET) - 1, -1, -1):
            if TARGET[axis] == 1:
                reduction_coord = remainder % GRAD[axis]
                remainder = remainder // GRAD[axis]
                offset += reduction_coord * GSTRIDES[axis]
        accumulator += tl.load(grad_ptr + offset, indices < REDUCE_SIZE, other=0).to(tl.float32)
    tl.store(out_ptr + out_index, tl.sum(accumulator, 0))


# ----- API -----


class _TritonBuffer:
    """Keep a Numba allocation alive and expose its pointer to Triton."""

    def __init__(self, array):
        self.array = array
        self.dtype = array.dtype

    def data_ptr(self):
        return self.array.device_ctypes_pointer.value


def _element_strides(array):
    return tuple(s // array.dtype.itemsize for s in array.strides)


def _broadcast_strides(shape, strides, output_shape):
    pad = len(output_shape) - len(shape)
    shape = (1,) * pad + tuple(shape)
    strides = (0,) * pad + tuple(strides)
    return tuple(0 if dim == 1 else stride for dim, stride in zip(shape, strides))


def matmul(a_device, b_device):
    """NumPy-style batched matmul on CUDA arrays with matching floating dtypes."""
    if not cuda.is_cuda_array(a_device) or not cuda.is_cuda_array(b_device):
        raise TypeError("matmul expects CUDA arrays")
    a = cuda.as_cuda_array(a_device)
    b = cuda.as_cuda_array(b_device)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("matmul expects arrays with at least one dimension")
    if a.dtype != b.dtype or a.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("matmul expects matching float16 or float32 dtypes")
    a_vector, b_vector = a.ndim == 1, b.ndim == 1
    if a_vector:
        a = a.reshape((1, a.shape[0]))
    if b_vector:
        b = b.reshape((b.shape[0], 1))
    m, k = a.shape[-2:]
    if b.shape[-2] != k:
        raise ValueError("matmul inner dimensions must match")
    n = b.shape[-1]
    batch_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    batch_count = int(np.prod(batch_shape, dtype=np.int64))
    matrix_shape = batch_shape + (m, n)
    result_shape = batch_shape + (() if a_vector else (m,)) + (() if b_vector else (n,))
    c = cuda.device_array(matrix_shape, dtype=a.dtype)
    if c.size == 0:
        return c.reshape(result_shape)
    if k == 0:
        cuda.synchronize()
        _zero_kernel[(triton.cdiv(c.size, 256),)](_TritonBuffer(c), c.size, 256)
        cuda.synchronize()
        return c.reshape(result_shape)
    a_strides = _element_strides(a)
    b_strides = _element_strides(b)
    a_batch = _broadcast_strides(a.shape[:-2], a_strides[:-2], batch_shape)
    b_batch = _broadcast_strides(b.shape[:-2], b_strides[:-2], batch_shape)
    cuda.synchronize()
    _matmul_kernel[(triton.cdiv(m, 128) * triton.cdiv(n, 128) * batch_count,)](
        _TritonBuffer(a), _TritonBuffer(b), _TritonBuffer(c), m, n, k,
        *a_strides[-2:], *b_strides[-2:], n, 1, batch_shape, a_batch, b_batch,
    )
    cuda.synchronize()
    return c.reshape(result_shape)


def _elementwise(a_device, b_device, op):
    if not cuda.is_cuda_array(a_device) or not cuda.is_cuda_array(b_device):
        raise TypeError("operation expects CUDA arrays")
    a, b = cuda.as_cuda_array(a_device), cuda.as_cuda_array(b_device)
    if a.dtype != b.dtype or a.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("operation expects matching float16 or float32 dtypes")
    try:
        shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError as exc:
        raise ValueError("inputs cannot be broadcast together") from exc
    out = cuda.device_array(shape, dtype=a.dtype)
    if not out.size:
        return out
    sa = _broadcast_strides(a.shape, _element_strides(a), shape)
    sb = _broadcast_strides(b.shape, _element_strides(b), shape)
    cuda.synchronize()
    _elementwise_kernel[(triton.cdiv(out.size, 256),)](
        _TritonBuffer(a), _TritonBuffer(b), _TritonBuffer(out), out.size,
        shape, sa, sb, op, 256,
    )
    cuda.synchronize()
    return out


def add(a_device, b_device):
    """Add CUDA arrays with NumPy-style broadcasting."""
    return _elementwise(a_device, b_device, 0)


def multiply(a_device, b_device):
    """Multiply CUDA arrays with NumPy-style broadcasting."""
    return _elementwise(a_device, b_device, 1)


def relu(x_device):
    """Apply ReLU to a float16 or float32 CUDA array."""
    if not cuda.is_cuda_array(x_device):
        raise TypeError("relu expects a CUDA array")
    return _elementwise(x_device, x_device, 2)


def unbroadcast(grad_device, target_shape):
    """Sum broadcast dimensions back into a CUDA array of target_shape."""
    grad = cuda.as_cuda_array(grad_device)
    shape = tuple(target_shape)
    if len(shape) > grad.ndim or np.broadcast_shapes(shape, grad.shape) != grad.shape:
        raise ValueError("gradient cannot be reduced to target shape")
    if shape == grad.shape:
        return grad
    padded = (1,) * (grad.ndim - len(shape)) + shape
    out = cuda.device_array(shape, dtype=grad.dtype)
    if not out.size:
        return out
    reduce_size = int(np.prod([g for g, t in zip(grad.shape, padded) if t == 1], dtype=np.int64))
    cuda.synchronize()
    if reduce_size == 0:
        _zero_kernel[(triton.cdiv(out.size, 256),)](_TritonBuffer(out), out.size, 256)
    else:
        _unbroadcast_kernel[(out.size,)](_TritonBuffer(grad), _TritonBuffer(out), padded,
                                           grad.shape, _element_strides(grad), reduce_size, 256)
    cuda.synchronize()
    return out
