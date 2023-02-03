from ..nn._container import Module

class Optim(Module):
    def zero_grad(self):
        for par in self.parameters:
            par.zero_grad()