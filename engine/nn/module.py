class Module:
    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def parameters(self):
        return []
