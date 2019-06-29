import numpy as np
from .base_modules import Criterion


class MSECriterion(Criterion):
    """
    # Fixme: delete instructions after implementing
    The MSECriterion, which is basic L2 norm usually used for regression.
    """
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
    """
    # Fixme: delete instructions after implementing
    You task is to implement the CrossEntropyCriterion. It should implement
    multiclass log loss (http://ml-cheatsheet.readthedocs.io/en/latest/
     loss_functions.html#cross-entropy).
    Nevertheless there is a sum over y (target) in that formula,
    remember that targets are one-hot encoded. This fact simplifies
    the computations a lot. Note, that criterions are the only places,
    where you divide by batch size.
    """
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def updateOutput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "CrossEntropyCriterion"


class MultiLabelCriterion(Criterion):
    def __init__(self):
        """
        # Fixme: delete instructions after implementing
        MultiLabelCriterion for attribute classification, i.e. target is
         multiple-hot encoded, could be multiple ones.
        """
        super(MultiLabelCriterion, self).__init__()

    def updateOutput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))

        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.maximum(1e-15, np.minimum(inpt, 1 - 1e-15))

        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "MultiLabelCriterion"