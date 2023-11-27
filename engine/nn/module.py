# Project: Demonstration of backpropagation learning - basic algorithm and selected optimizer
# Author: David Chocholaty <xchoch09@stud.fit.vutbr.cz>
# File: module.py

# The source code of the following file is based on the implementation of the Tensorgrad library.
# Source: https://github.com/hkxIron/tensorgrad/blob/6098d54eeeeeebf69ee89a2dcb0a7d8b60b95c16/tensorgrad/network.py

class Module:
    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def parameters(self):
        return []
