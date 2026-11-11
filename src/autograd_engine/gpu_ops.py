"""CUDA operations used by the autograd engine's training path.

These kernels favor clarity over performance. Inputs to this module are
contiguous device arrays unless an operation explicitly accepts strides.
"""
import math

import numpy as np
from numba import cuda


@cuda.jit
def _fill(out, value):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = value


@cuda.jit
def _relu_backward(x, grad, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = grad[i] if x[i] > 0 else 0


@cuda.jit
def _softmax(x, out, rows, cols):
    row = cuda.grid(1)
    if row < rows:
        largest = -math.inf
        for j in range(cols):
            largest = max(largest, x[row * cols + j])
        total = 0.0
        for j in range(cols):
            total += math.exp(x[row * cols + j] - largest)
        for j in range(cols):
            out[row * cols + j] = math.exp(x[row * cols + j] - largest) / total


@cuda.jit
def _softmax_backward(p, grad, out, rows, cols):
    row = cuda.grid(1)
    if row < rows:
        dot = 0.0
        for j in range(cols):
            dot += p[row * cols + j] * grad[row * cols + j]
        for j in range(cols):
            i = row * cols + j
            out[i] = p[i] * (grad[i] - dot)


@cuda.jit
def _loss(p, labels, out, rows, cols):
    if cuda.grid(1) == 0:
        total = 0.0
        for row in range(rows):
            label = labels[row]
            if label < 0 or label >= cols:
                out[0] = math.nan
                return
            total -= math.log(p[row * cols + label] + 1e-9)
        out[0] = total / rows


@cuda.jit
def _loss_backward(p, labels, grad, out, rows, cols):
    i = cuda.grid(1)
    if i < out.size:
        row, col = i // cols, i % cols
        out[i] = (p[i] - (1.0 if labels[row] == col else 0.0)) * grad[0] / rows


@cuda.jit
def _sgd(param, grad, velocity, lr, momentum):
    i = cuda.grid(1)
    if i < param.size:
        velocity[i] = momentum * velocity[i] + lr * grad[i]
        param[i] -= velocity[i]


def _flat(x):
    return x.reshape(x.size)


def _launch(kernel, count, *args):
    if count:
        kernel[(count + 127) // 128, 128](*args)


def full(shape, dtype, value):
    out = cuda.device_array(shape, dtype=dtype)
    _launch(_fill, out.size, _flat(out), value)
    return out


def multiply(a, b):
    from autograd_engine.triton_kernels import multiply as triton_multiply
    return triton_multiply(a, b)


def unbroadcast(grad, shape):
    from autograd_engine.triton_kernels import unbroadcast as triton_unbroadcast
    return triton_unbroadcast(grad, shape)


def relu_backward(x, grad):
    out = cuda.device_array(x.shape, dtype=x.dtype)
    _launch(_relu_backward, out.size, _flat(x), _flat(grad), _flat(out))
    return out


def softmax(x):
    if x.ndim == 0:
        raise ValueError("softmax expects at least one dimension")
    n = x.shape[-1]
    m = x.size // n if n else 0
    if not n:
        raise ValueError("softmax requires a non-empty last dimension")
    out = cuda.device_array(x.shape, dtype=x.dtype)
    _launch(_softmax, m, _flat(x), _flat(out), m, n)
    return out


def softmax_backward(p, grad):
    n = p.shape[-1]
    m = p.size // n if n else 0
    out = cuda.device_array(p.shape, dtype=p.dtype)
    _launch(_softmax_backward, m, _flat(p), _flat(grad), _flat(out), m, n)
    return out


def loss(logits, labels):
    if len(logits.shape) != 2 or not logits.shape[0] or labels.size != logits.shape[0]:
        raise ValueError("cross entropy expects non-empty 2D logits and one label per row")
    if labels.dtype.kind not in "iu":
        raise TypeError("cross entropy labels must be integers")
    p = softmax(logits)
    out = cuda.device_array((), dtype=logits.dtype)
    _launch(_loss, 1, _flat(p), _flat(labels), _flat(out), *logits.shape)
    return out, p


def loss_backward(p, labels, grad):
    out = cuda.device_array(p.shape, dtype=p.dtype)
    _launch(_loss_backward, out.size, _flat(p), _flat(labels), _flat(grad), _flat(out), *p.shape)
    return out


def sgd(param, grad, velocity, lr, momentum):
    _launch(_sgd, param.size, _flat(param), _flat(grad), _flat(velocity), lr, momentum)
