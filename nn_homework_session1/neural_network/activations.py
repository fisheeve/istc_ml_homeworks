from .base_modules import Module
import numpy as np


class Sigmoid(Module):
    """
    Sigmoid
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def updateOutput(self, inpt):
        self.output = 1 / (1 + np.exp(-inpt))
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * (self.output * (1 - self.output))
        return self.gradInput

    def __repr__(self):
        return "Sigmoid"


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def updateOutput(self, inpt):
        self.output = 1 - 2 / (np.exp(2 * inpt) + 1)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        exp = np.exp(2 * inpt)
        self.gradInput = gradOutput * 4 * exp / (exp + 1)**2
        return self.gradInput

    def __repr__(self):
        return "Tanh"


class ReLU(Module):
    """
    Rectified Linear Unit non-linearity
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, inpt):
        self.output = np.maximum(inpt, np.zeros_like(inpt))
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * (inpt>0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        assert 0 < slope < 1
        super(LeakyReLU, self).__init__()
        self.slope = slope

    def updateOutput(self, inpt):
        self.output = np.maximum(inpt, self.slope * inpt)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * np.maximum(
            np.sign(inpt), self.slope * np.ones_like(inpt))
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, inpt):
        self.output = inpt * (inpt > 0) + self.alpha * (
                inpt <= 0) * (np.exp(inpt) -1)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * (
                (inpt > 0) + (self.alpha + self.output) * (inpt <= 0))
        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
     def __init__(self):
        super(SoftPlus, self).__init__()

     def updateOutput(self, inpt):
        self.output = np.log(1 + np.exp(inpt))
        return self.output

     def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput / (1 + np.exp(-inpt))
        return self.gradInput

     def __repr__(self):
        return "SoftPlus"