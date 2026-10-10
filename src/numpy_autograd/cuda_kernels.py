import numpy as np
import numba
from numba import cuda

TPB = 32


@cuda.jit
def _matmul_kernel(a, b, out):

    size_a_rows, size_a_cols = a.shape
    size_b_rows, size_b_cols = b.shape

    a_shared = cuda.shared.array((TPB, TPB), numba.float32)
    b_shared = cuda.shared.array((TPB, TPB), numba.float32)

    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    local_i = cuda.threadIdx.y
    local_j = cuda.threadIdx.x

    tmp = numba.float32(0.0)
    for curr_block in range((size_a_cols + TPB - 1) // TPB):
        if i < size_a_rows and (local_j + curr_block * TPB) < size_a_cols:
            a_shared[local_i, local_j] = a[i, local_j + curr_block * TPB]
        else:
            a_shared[local_i, local_j] = 0.0

        if j < size_b_cols and (local_i + curr_block * TPB) < size_b_rows:
            b_shared[local_i, local_j] = b[local_i + curr_block * TPB, j]
        else:
            b_shared[local_i, local_j] = 0.0
        cuda.syncthreads()

        if i < size_a_rows and j < size_b_cols:
            for k in range(TPB):
                tmp += a_shared[local_i, k] * b_shared[k, local_j]

        cuda.syncthreads()

    if i < size_a_rows and j < size_b_cols:
        out[i, j] = tmp


def matmul(a, b, out):

    size_a_rows, size_a_cols = a.shape
    size_b_rows, size_b_cols = b.shape

    assert (
        size_a_cols == size_b_rows
    ), "Inner dimensions must match for matrix multiplication"
    
    out_shape = (a.shape[0], b.shape[1])
    out_device = cuda.device_array(out_shape, dtype=a.dtype)
    
    threads_per_block = (TPB, TPB)
    blocks_per_grid_x = (size_b_cols + TPB - 1) // TPB
    blocks_per_grid_y = (size_a_rows + TPB - 1) // TPB

    _matmul_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
        a, b, out_device
    )
    
    return out_device
