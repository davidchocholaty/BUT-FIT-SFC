from engine.nn.module import Module
from engine.nn.layer import Layer
from engine.opt.optimizer import Optimizer


class MultilayerPerceptron(Module):
    # MultilayerPerceptron(2, [(16, "ReLu"), (16, "ReLu"), (1, "Linear")])
    def __init__(self, inputs_count, layers, opt=Optimizer.NoOptimizer()):
        outputs_count_each_layer = []
        activation_function_each_layer = []

        for outputs_count, activation_function in layers:
            outputs_count_each_layer.append(outputs_count)
            activation_function_each_layer.append(activation_function)

        size_list = [inputs_count] + outputs_count_each_layer
        self.layers = [Layer(size_list[i], size_list[i+1], activation_function_str=activation_function_each_layer[i])
                       for i in range(len(outputs_count_each_layer))]
        self.opt = opt

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
