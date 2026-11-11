import numpy as np
import autograd_engine.engine as engine


class Module:
    """Base class for all modules. Defines the basic structure for forward and backward passes."""

    def __init__(self):
        self.parameters = []  # List of parameters (Tensors) in the module

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)

    def to(self, device):
        """Move module parameters in place, preserving their identities."""
        engine._check_device(device)
        for param in self.parameters:
            if param.grad_fn is not None:
                raise ValueError("module parameters must be leaf tensors")
        for param in self.parameters:
            if param.device != device:
                param.data = engine._to_storage(param.data, device)
                if param.grad is not None:
                    param.grad = engine._to_storage(param.grad, device)
                param._device = device
        return self


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weights and bias
        self.W = engine.Tensor(
            (np.random.randn(in_features, out_features) * 0.1).astype(np.float32), requires_grad=True
        )
        self.b = engine.Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        self.parameters.extend([self.W, self.b])

    def forward(self, x):
        return engine.matmul(x, self.W) + self.b


class FFN(Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = Linear(in_features, hidden_features)
        self.linear2 = Linear(hidden_features, hidden_features)
        self.linear3 = Linear(hidden_features, out_features)

        self.parameters.extend(self.linear1.parameters)
        self.parameters.extend(self.linear2.parameters)
        self.parameters.extend(self.linear3.parameters)

    def forward(self, x):

        x = self.linear1(x)
        x = engine.relu(x)
        x = self.linear2(x)
        x = engine.relu(x)
        x = self.linear3(x)
        return x
