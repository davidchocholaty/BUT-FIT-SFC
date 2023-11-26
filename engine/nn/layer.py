import numpy as np

from engine.backend.activation_function import ActivationFunction
from engine.backend.tensor import Tensor
from engine.nn.module import Module


class Layer(Module):
    def __init__(self, inputs_count, outputs_count, activation_function_str):
        # The seed is provided for the purpose of starting with the exact same defined model for all optimizers.
        np.random.seed(0)
        self.w = Tensor(np.random.uniform(low=-1, high=1, size=(inputs_count, outputs_count)))
        # self.w = Tensor(np.random.uniform(low=-1, high=1, size=(inputs_count, outputs_count)))
        self.b = Tensor(np.zeros(outputs_count))

        assert activation_function_str in ["ReLu", "Softmax", "Linear"], "invalid activation function definition"

        if activation_function_str == "ReLu":
            self.activation_function = ActivationFunction.ReLu
        elif activation_function_str == "Softmax":
            self.activation_function = ActivationFunction.Softmax
        else:
            # Linear
            self.activation_function = None

    def __call__(self, x):
        xw = x @ self.w

        if self.b:
            xw_b = xw + self.b
        else:
            xw_b = xw

        if self.activation_function:
            out = self.activation_function.forward(xw_b.data)
            xw_b_act = Tensor(out, _children=(xw_b,))

            def _backward():
                xw_b.grad += self.activation_function.backward(xw_b.data, out, xw_b_act.grad)

            xw_b_act.set_backward_function(_backward)
        else:
            xw_b_act = xw_b

        return xw_b_act

    def parameters(self):
        return [self.w, self.b] if self.b else [self.w]
