from .base_modules import Module


class Sigmoid(Module):
    """
    # Fixme: delete instructions after implementing
    Implement well-known Sigmoid non-linearity
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "Sigmoid"


class Tanh(Module):
    """
    # Fixme: delete instructions after implementing
    Implement hyperbolic tangent non-linearity (aka Tanh): Note that Tanh is
     scaled version of the sigmoid function.
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "Tanh"


class ReLU(Module):
    """
    # Fixme: delete instructions after implementing
    Implement Rectified Linear Unit non-linearity (aka ReLU)
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    """
    # Fixme: delete instructions after implementing
    Implement Leaky Rectified Linear Unit. Experiment with slope.
    http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs
    """
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()
        self.slope = slope

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    """
    # Fixme: delete instructions after implementing
    Implement Exponential Linear Units activations.
    http://arxiv.org/abs/1511.07289
    """
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    """
    # Fixme: delete instructions after implementing
    Implement SoftPlus activations. Look, how they look a lot like ReLU.
    https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29
    """
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"