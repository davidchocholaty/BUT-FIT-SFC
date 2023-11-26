import numpy as np

from abc import ABC, abstractmethod


class Optimizer:
    class Function(ABC):
        @abstractmethod
        def update_params(self, model, learning_rate):
            pass

    class NoOptimizer(Function):
        def update_params(self, model, learning_rate):
            for p in model.parameters():
                p.data -= learning_rate * p.grad

    class RMSpropOptimizer(Function):
        # https://www.fit.vutbr.cz/study/courses/SFC/private/zboril2020/20sfc_2.pdf
        def __init__(self, eps=10 ** (-6), beta=0.9, params_size_list=None):
            if params_size_list is None:
                params_size_list = []
            self._eps = eps
            self._beta = beta
            self._v = [np.zeros(params) for params in params_size_list]

        @property
        def eps(self):
            return self._eps

        @property
        def beta(self):
            return self._beta

        def update_params(self, model, learning_rate):
            parameters = model.parameters()

            for i in range(len(parameters)):
                self._v[i] = self._beta * self._v[i] + (1 - self._beta) * (parameters[i].grad ** 2)
                parameters[i].data -= (learning_rate / np.sqrt(self._v[i] + self._eps)) * parameters[i].grad

    class AdamOptimizer(Function):
        # https://keras.io/api/optimizers/
        # https://www.fit.vutbr.cz/study/courses/SFC/private/zboril2020/20sfc_2.pdf
        def __init__(self,
                     beta_1=0.9,
                     beta_2=0.999,
                     eps=10 ** (-7),
                     params_size_list=None):
            if params_size_list is None:
                params_size_list = []
            self._beta_1 = beta_1
            self._beta_2 = beta_2
            self._eps = eps
            self._m = [np.zeros(params) for params in params_size_list]
            self._v = [np.zeros(params) for params in params_size_list]

        @property
        def beta_1(self):
            return self._beta_1

        @property
        def beta_2(self):
            return self._beta_2

        @property
        def eps(self):
            return self._eps

        @property
        def m(self):
            return self._m

        @property
        def v(self):
            return self._v

        def update_params(self, model, learning_rate):
            parameters = model.parameters()

            for i in range(len(parameters)):
                self._v[i] = self._beta_2 * self._v[i] + (1 - self._beta_2) * (parameters[i].grad ** 2)
                v = self._v[i] / (1 - self._beta_2)
                self._m[i] = self._beta_1 * self._m[i] + (1 - self._beta_1) * parameters[i].grad
                parameters[i].data -= (learning_rate / (np.sqrt(v) + self._eps)) * self._m[i]

    class AmsGradOptimizer(AdamOptimizer):
        # https://www.fit.vutbr.cz/study/courses/SFC/private/zboril2020/20sfc_2.pdf
        def __init__(self,
                     beta_1=0.9,
                     beta_2=0.999,
                     eps=10 ** (-7),
                     params_size_list=None):
            super().__init__(beta_1, beta_2, eps, params_size_list)
            self._v_2 = [np.zeros(params) for params in params_size_list]

        @property
        def v_2(self):
            return self._v_2

        @v_2.setter
        def v_2(self, new_v_2):
            self._v_2 = new_v_2

        def update_params(self, model, learning_rate):
            parameters = model.parameters()

            for i in range(len(parameters)):
                self._v[i] = self._beta_2 * self._v[i] + (1 - self._beta_2) * (parameters[i].grad ** 2)
                self._v_2[i] = np.maximum(self._v_2[i], self._v[i])
                self._m[i] = self._beta_1 * self._m[i] + (1 - self._beta_1) * parameters[i].grad
                parameters[i].data -= (learning_rate / (np.sqrt(self._v_2[i]) + self._eps)) * self._m[i]
