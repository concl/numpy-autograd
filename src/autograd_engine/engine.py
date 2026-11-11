import numpy as np
from numba import cuda
from autograd_engine import gpu_ops

try:
    from autograd_engine.triton_kernels import matmul as matmul_triton, add as add_triton, relu as relu_triton

    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# ----- Helper functions -----


def _check_device(device):
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")


def _to_numpy(data):
    """Copy device storage to the host; leave NumPy values on the host."""
    if cuda.is_cuda_array(data):
        return data.copy_to_host()
    return np.asarray(data)


def _to_storage(data, device):
    if device == "cuda" and cuda.is_cuda_array(data):
        return data
    host = _to_numpy(data)
    if device == "cpu":
        return np.array(host, copy=True)
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return cuda.to_device(np.array(host, copy=True, order="C"))


def _same_device(*tensors):
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError("all tensors must be on the same device")
    return device


def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum grad over axes that were broadcasted to match target_shape."""
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _swap_last_two_cuda(array):
    """Create a strided view with the last two axes exchanged."""
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    shape = array.shape[:-2] + (array.shape[-1], array.shape[-2])
    strides = array.strides[:-2] + (array.strides[-1], array.strides[-2])
    return DeviceNDArray(shape, strides, array.dtype, gpu_data=array.gpu_data)


# ----- Autograd functions for basic operations -----


class AutogradFunction:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AddFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        if cuda.is_cuda_array(grad_output):
            return gpu_ops.unbroadcast(grad_output, a_shape), gpu_ops.unbroadcast(grad_output, b_shape)
        grad_output_a = unbroadcast(grad_output, a_shape)
        grad_output_b = unbroadcast(grad_output, b_shape)
        return grad_output_a, grad_output_b


class MulFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):

        # The derivative of a * b with respect to a is b,
        # and with respect to b is a
        a, b = ctx.saved_tensors
        if cuda.is_cuda_array(grad_output):
            return (gpu_ops.unbroadcast(gpu_ops.multiply(grad_output, b), a.shape),
                    gpu_ops.unbroadcast(gpu_ops.multiply(grad_output, a), b.shape))
        grad_output_a = unbroadcast(grad_output * b, a.shape)
        grad_output_b = unbroadcast(grad_output * a, b.shape)

        return grad_output_a, grad_output_b


class MatMulFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):

        # The derivative of a_ij with respect to an element c_ik in the output is is b_jk,
        # and the derivative of b_jk with respect to an element c_ik in the output  is a_ij
        # Using the chain rule, we get the partial for a_ij as:
        # grad_output_ik * b_jk (summed over k) and for b_jk as a_ij * grad_output_ik (summed over i)
        a, b = ctx.saved_tensors
        if cuda.is_cuda_array(grad_output):
            a_vector, b_vector = a.ndim == 1, b.ndim == 1
            a_matrix = a.reshape((1, a.shape[0])) if a_vector else a
            b_matrix = b.reshape((b.shape[0], 1)) if b_vector else b
            batch = np.broadcast_shapes(a_matrix.shape[:-2], b_matrix.shape[:-2])
            grad_matrix = grad_output.reshape(batch + (a_matrix.shape[-2], b_matrix.shape[-1]))
            grad_a = matmul_triton(grad_matrix, _swap_last_two_cuda(b_matrix))
            grad_b = matmul_triton(_swap_last_two_cuda(a_matrix), grad_matrix)
            grad_a = gpu_ops.unbroadcast(grad_a, a_matrix.shape)
            grad_b = gpu_ops.unbroadcast(grad_b, b_matrix.shape)
            return grad_a.reshape(a.shape), grad_b.reshape(b.shape)
        a, b = (_to_numpy(value) for value in (a, b))
        a_matrix = a.reshape((1, a.shape[0])) if a.ndim == 1 else a
        b_matrix = b.reshape((b.shape[0], 1)) if b.ndim == 1 else b
        batch = np.broadcast_shapes(a_matrix.shape[:-2], b_matrix.shape[:-2])
        grad_matrix = np.asarray(grad_output).reshape(batch + (a_matrix.shape[-2], b_matrix.shape[-1]))
        grad_a = np.matmul(grad_matrix, np.swapaxes(b_matrix, -1, -2))
        grad_b = np.matmul(np.swapaxes(a_matrix, -1, -2), grad_matrix)
        return (unbroadcast(grad_a, a_matrix.shape).reshape(a.shape),
                unbroadcast(grad_b, b_matrix.shape).reshape(b.shape))


class ReLUFunction(AutogradFunction):
    """
    ReLU activation function with autograd support.
    Should be used as a module for abstraction.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        if cuda.is_cuda_array(grad_output):
            return gpu_ops.relu_backward(x, grad_output)
        grad = grad_output * (x > 0).astype(x.dtype)
        return grad


class SoftmaxFunction(AutogradFunction):
    @staticmethod
    def forward(ctx, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        ctx.save_for_backward(softmax)
        return softmax

    @staticmethod
    def backward(ctx, grad_output):
        (softmax,) = ctx.saved_tensors
        if cuda.is_cuda_array(grad_output):
            return gpu_ops.softmax_backward(softmax, grad_output)
        grad_input = softmax * (
            grad_output - np.sum(grad_output * softmax, axis=-1, keepdims=True)
        )
        return grad_input


class CrossEntropyLossFunction(AutogradFunction):
    """
    Cross-entropy loss function with autograd support.
    Should be used as a module for abstraction.
    """

    @staticmethod
    def forward(ctx, logits, labels, eps=1e-9):
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Compute cross-entropy loss for integer labels
        batch_size = logits.shape[0]
        loss = (
            -np.sum(np.log(softmax_probs + eps)[np.arange(batch_size), labels])
            / batch_size
        )

        ctx.save_for_backward(softmax_probs, labels)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax_probs, labels = ctx.saved_tensors
        if cuda.is_cuda_array(grad_output):
            return gpu_ops.loss_backward(softmax_probs, labels, grad_output), None
        batch_size = softmax_probs.shape[0]

        # Gradient: softmax_probs - one_hot(labels)
        grad_logits = softmax_probs.copy()
        grad_logits[np.arange(batch_size), labels] -= 1
        grad_logits /= batch_size

        return grad_logits * grad_output, None


# ----- Autograd graph classes -----


class AutogradContext:
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class AutogradNode:
    """
    A node in the autograd computation graph.
    Each node wraps an AutoGradFunction and holds the context
    and references to input tensors for backward traversal.
    """

    def __init__(self, function: AutogradFunction, ctx: AutogradContext, inputs: list):
        self.function = function  # The AutogradFunction class (e.g. AddFunction)
        self.ctx = ctx  # Context holding saved tensors
        self.inputs = inputs  # List of input Tensors (for graph traversal)

    def backward(self, grad_output):
        grads = self.function.backward(self.ctx, grad_output)
        for i, input_tensor in enumerate(self.inputs):
            grad = grads[i] if isinstance(grads, tuple) else grads
            if grad is None:
                continue
            grad = _to_storage(grad, input_tensor.device)
            if input_tensor.requires_grad:
                if input_tensor.grad is None:
                    input_tensor.grad = grad
                elif input_tensor.device == "cuda":
                    input_tensor.grad = add_triton(input_tensor.grad, grad)
                else:
                    input_tensor.grad = input_tensor.grad + grad
            if input_tensor.grad_fn is not None:
                input_tensor.grad_fn.backward(grad)


# ----- Tensor class -----


class Tensor:
    """
    A wrapper around numpy arrays that tracks the computation graph
    for automatic differentiation.
    """

    def __init__(
        self, data, requires_grad=False, grad_fn: AutogradNode = None, dtype=np.float32,
        device="cpu",
    ):
        _check_device(device)
        host = _to_numpy(data)
        if not isinstance(data, np.ndarray) and not cuda.is_cuda_array(data):
            host = np.asarray(data, dtype=dtype)
        self.data = _to_storage(host, device)
        self._device = device
        self.requires_grad = requires_grad
        self.grad = None  # Accumulated gradient
        self.grad_fn = grad_fn  # Node that produced this tensor (None for leaves)

    @classmethod
    def _from_cuda_storage(cls, data, requires_grad, grad_fn):
        """Wrap an owned CUDA result without copying it through host memory."""
        tensor = cls.__new__(cls)
        tensor.data = data
        tensor._device = "cuda"
        tensor.requires_grad = requires_grad
        tensor.grad = None
        tensor.grad_fn = grad_fn
        return tensor

    def backward(self, grad_output=None):
        if isinstance(grad_output, Tensor):
            if grad_output.device != self.device:
                raise ValueError("gradient must be on the same device as the tensor")
            grad_output = grad_output.data
        elif grad_output is not None:
            grad_output = _to_storage(grad_output, self.device)
        if grad_output is None:
            grad_output = (gpu_ops.full(self.shape, self.data.dtype, 1) if self.device == "cuda"
                           else np.ones_like(self.data))
        if grad_output.shape != self.shape:
            raise ValueError("gradient shape must match the tensor shape")
        if self.grad_fn is not None:
            self.grad_fn.backward(grad_output)
        elif self.requires_grad:
            accumulated = (grad_output if self.grad is None else
                           add_triton(self.grad, grad_output) if self.device == "cuda" else self.grad + grad_output)
            self.grad = _to_storage(accumulated, self.device)

    @property
    def device(self):
        return self._device

    def to(self, device):
        """Return a tensor on device, preserving the gradient connection."""
        _check_device(device)
        if device == self.device:
            return self
        ctx = AutogradContext()
        grad_fn = AutogradNode(_TransferFunction, ctx, [self]) if self.requires_grad else None
        return Tensor(self.data, requires_grad=self.requires_grad, grad_fn=grad_fn, device=device)

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        if isinstance(other, Tensor):
            device = _same_device(self, other)
            ctx = AutogradContext()
            if device == "cuda":
                ctx.save_for_backward(self.shape, other.shape)
                result_data = add_triton(self.data, other.data)
            else:
                result_data = AddFunction.forward(ctx, self.data, other.data)
            grad_fn = AutogradNode(AddFunction, ctx, [self, other])
            if device == "cuda":
                return Tensor._from_cuda_storage(result_data, self.requires_grad or other.requires_grad, grad_fn)
            return Tensor(
                result_data,
                requires_grad=self.requires_grad or other.requires_grad,
                grad_fn=grad_fn,
                device=device,
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            device = _same_device(self, other)
            ctx = AutogradContext()
            if device == "cuda":
                ctx.save_for_backward(self.data, other.data)
                result_data = gpu_ops.multiply(self.data, other.data)
            else:
                result_data = MulFunction.forward(ctx, self.data, other.data)
            grad_fn = AutogradNode(MulFunction, ctx, [self, other])
            if device == "cuda":
                return Tensor._from_cuda_storage(result_data, self.requires_grad or other.requires_grad, grad_fn)
            return Tensor(
                result_data,
                requires_grad=self.requires_grad or other.requires_grad,
                grad_fn=grad_fn,
                device=device,
            )
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return matmul(self, other)
        else:
            return NotImplemented

    def __getitem__(self, key):
        return Tensor(_to_numpy(self.data)[key], requires_grad=self.requires_grad, device=self.device)

    def __repr__(self):
        return f"Tensor(data={_to_numpy(self.data)}, requires_grad={self.requires_grad}, device='{self.device}')"


# ----- Autograd function wrappers for user-friendly API -----


class _TransferFunction(AutogradFunction):
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def matmul(a: Tensor, b: Tensor) -> Tensor:
    device = _same_device(a, b)
    ctx = AutogradContext()
    if device == "cuda":
        if not HAS_CUDA:
            raise RuntimeError("Triton is required for CUDA matmul")
        ctx.save_for_backward(a.data, b.data)
        result_data = matmul_triton(a.data, b.data)
    else:
        result_data = MatMulFunction.forward(ctx, a.data, b.data)
    grad_fn = AutogradNode(MatMulFunction, ctx, [a, b])
    if device == "cuda":
        return Tensor._from_cuda_storage(result_data, a.requires_grad or b.requires_grad, grad_fn)
    return Tensor(
        result_data, requires_grad=a.requires_grad or b.requires_grad, grad_fn=grad_fn, device=device
    )


def relu(x: Tensor) -> Tensor:
    ctx = AutogradContext()
    if x.device == "cuda":
        ctx.save_for_backward(x.data)
        result_data = relu_triton(x.data)
        return Tensor._from_cuda_storage(result_data, x.requires_grad, AutogradNode(ReLUFunction, ctx, [x]))
    result_data = ReLUFunction.forward(ctx, _to_numpy(x.data))
    grad_fn = AutogradNode(ReLUFunction, ctx, [x])
    return Tensor(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn, device=x.device)


def softmax(x: Tensor) -> Tensor:
    ctx = AutogradContext()
    if x.device == "cuda":
        result_data = gpu_ops.softmax(x.data)
        ctx.save_for_backward(result_data)
        return Tensor._from_cuda_storage(result_data, x.requires_grad, AutogradNode(SoftmaxFunction, ctx, [x]))
    result_data = SoftmaxFunction.forward(ctx, _to_numpy(x.data))
    grad_fn = AutogradNode(SoftmaxFunction, ctx, [x])
    return Tensor(result_data, requires_grad=x.requires_grad, grad_fn=grad_fn, device=x.device)


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    _same_device(logits, labels)
    ctx = AutogradContext()
    if logits.device == "cuda":
        loss_data, probabilities = gpu_ops.loss(logits.data, labels.data)
        ctx.save_for_backward(probabilities, labels.data)
        return Tensor._from_cuda_storage(loss_data, logits.requires_grad, AutogradNode(CrossEntropyLossFunction, ctx, [logits, labels]))
    loss_data = CrossEntropyLossFunction.forward(
        ctx, _to_numpy(logits.data), _to_numpy(labels.data).flatten().astype(np.intp)
    )
    grad_fn = AutogradNode(CrossEntropyLossFunction, ctx, [logits, labels])
    return Tensor(loss_data, requires_grad=logits.requires_grad, grad_fn=grad_fn, device=logits.device)
