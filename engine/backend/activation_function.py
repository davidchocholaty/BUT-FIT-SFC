import numpy as np


class ActivationFunction:
    class Function:
        @staticmethod
        def forward(x_data, **kwargs):
            pass

        @staticmethod
        def backward(x_data, y_data, y_grad, **kwargs):
            pass

    class ReLu(Function):
        @staticmethod
        def forward(x_data, **kwargs):
            out_data = x_data.copy()
            out_data[out_data < 0] = 0

            return out_data

        @staticmethod
        def backward(x_data, y_data, y_grad, **kwargs):
            return (x_data > 0) * y_grad

    class Softmax(Function):
        @staticmethod
        def forward(x_data, axis=1, **kwargs):
            x_data_max = x_data.max(axis=axis, keepdims=True)
            exp_x = np.exp(x_data - x_data_max)

            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        @staticmethod
        def backward(x_data, y_data, y_grad, batch_axis=0, **kwargs):
            batch_size = y_data.shape[batch_axis]

            batch_grads = []

            for i in range(batch_size):
                yi = np.diag(y_data[i])
                yi_x_yj = np.outer(y_data[i], y_data[i])
                softmax_grad = yi - yi_x_yj
                x_grad = softmax_grad @ y_grad[i].T
                batch_grads.append(x_grad)

            return np.array(batch_grads, dtype=x_data.dtype)
