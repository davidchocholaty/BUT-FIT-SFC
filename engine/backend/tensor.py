import numpy as np

from engine.backend.activation_function import ActivationFunction
from engine.nn.util import accumulative_add_by_shape, get_repeat_axis


class Tensor:
    def __init__(self, data, data_type=np.float64, _children=()):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            self.data = np.array(data, dtype=np.float32)

        if data_type:
            self.data = self.data.astype(data_type)

        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self.prev = set(_children)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def numpy(self):
        return self.data.copy()

    def set_backward_function(self, function):
        self._backward = function

    def flatten(self):
        return self.reshape(-1)

    def reshape(self, new_shape):
        out = Tensor(self.data.copy().reshape(new_shape), _children=(self,))
        out.grad = self.grad.copy().reshape(new_shape)

        def _backward():
            self.grad += out.grad.copy().reshape(self.grad.shape)

        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _children=(self,))

        def _backward():
            out_grad_expand = np.expand_dims(out.grad, axis=axis if axis else ())
            self.grad += out_grad_expand

        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), _children=(self,))

        def _backward():
            out_grad_expand = np.expand_dims(out.grad, axis=axis if axis else ())
            self.grad += (1 / (np.prod(self.data.shape) / np.prod(out.data.shape))) * out_grad_expand

        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward():
            self_not_repeat, self_repeat_axis = get_repeat_axis(self.grad.shape, out.grad.shape)
            other_not_repeat, other_repeat_axis = get_repeat_axis(other.grad.shape, out.grad.shape)

            accumulative_add_by_shape(self_not_repeat, self_repeat_axis, self.grad, out.grad)
            accumulative_add_by_shape(other_not_repeat, other_repeat_axis, other.grad, out.grad)

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward():
            self_delta = other.data * out.grad
            other_delta = self.data * out.grad

            self_not_repeat, self_repeat_axis = get_repeat_axis(self.grad.shape, self_delta.shape)
            other_not_repeat, other_repeat_axis = get_repeat_axis(other.grad.shape, other_delta.shape)

            accumulative_add_by_shape(self_not_repeat, self_repeat_axis, self.grad, self_delta)
            accumulative_add_by_shape(other_not_repeat, other_repeat_axis, other.grad, other_delta)

        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert self.data.shape[-1] == other.data.shape[0], 'self data last dim should equals other data first dim'
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "int or float supported only"
        out = Tensor(self.data ** other, _children=(self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        if isinstance(other, int):
            return self * (float(other) ** -1)

        if isinstance(other, (np.int32, np.int64)):
            return self * (other.astype(np.float32) ** -1)

        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        assert isinstance(other, (int, float)), "int or float supported only"
        return self * other

    def backward(self, grad=None):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        if grad is not None:
            self.grad = grad
        else:
            self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def relu(self):
        out = Tensor(ActivationFunction.ReLu.forward(self.data), _children=(self,))

        def _backward():
            self.grad += ActivationFunction.ReLu.backward(self.data, out.data, out.grad)

        out._backward = _backward

        return out

    def softmax(self, axis=None, batch_axis=0):
        out = Tensor(ActivationFunction.Softmax.forward(self.data, axis=axis))

        def _backward():
            self.grad += ActivationFunction.Softmax.backward(self.data, out.data, out.grad, batch_axis=batch_axis)

        out._backward = _backward

        return out
