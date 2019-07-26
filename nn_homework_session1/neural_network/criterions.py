import numpy as np
from .base_modules import Criterion


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        delta = input - target
        self.output = np.sum(delta * delta) / target.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = 2*(input - target)
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


class CrossEntropyCriterion(Criterion):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def updateOutput(self, inpt, target):
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        self.output = np.mean(-np.log(input_clamp) * target)
        return self.output

    def updateGradInput(self, inpt, target):
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        self.gradInput  =  - target / input_clamp
        return self.gradInput

    def __repr__(self):
        return "CrossEntropyCriterion"


class MultiLabelCriterion(Criterion):
    def __init__(self):
        super(MultiLabelCriterion, self).__init__()

    def updateOutput(self, inpt, target):
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        sigm = 1 / (1 + np.exp(-input_clamp))
        self.output = np.mean(np.sum(-target * np.log(sigm) -
                             (1 - target) * log(1-sigm), axis=1))
        return self.output

    def updateGradInput(self, inpt, target):
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        sigm = 1 / (1 + np.exp(-input_clamp))
        self.gradInput = -target * (1 - sigm) + (1 - target) * sigm
        return self.gradInput

    def __repr__(self):
        return "MultiLabelCriterion"