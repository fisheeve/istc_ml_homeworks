import numpy as np
from scipy.stats import bernoulli
from .base_modules import Module


class Linear(Module):
    """
    ### input:  batch_size x n_features1 ###
    ### output: batch_size x n_features2 ###

    A module which applies a linear transformation
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, inpt):
        self.output = inpt.dot(self.W.T) + self.b

        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput.dot(self.W)

        return self.gradInput

    def accGradParameters(self, inpt, gradOutput):
        self.gradW = gradOutput.T.dot(inpt)
        self.gradb = gradOutput.sum(axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q


class SoftMax(Module):
    """
    ### input:  batch_size x n_feats ###
    ### output: batch_size x n_feats ###
    """
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        exp = np.exp(input)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        diff = np.sum(gradOutput * self.output, axis=1, keepdims=True)
        self.gradInput = self.output * (gradOutput - diff)
        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class BatchMeanSubtraction(Module):
    """
    ### input:  batch_size x n_feats ###
    ### output: batch_size x n_feats ###
    """
    def __init__(self, alpha=0.95):
        super(BatchMeanSubtraction, self).__init__()

        if self.training:
            self.alpha = alpha
        else:
            self.alpha = 1

        self.old_mean = None

    def updateOutput(self, inpt):
        batch_mean = np.mean(inpt, axis=0)
        if self.old_mean is None:
            self.old_mean = batch_mean
        mean_to_subtract = (
                self.alpha * self.old_mean + (1 - self.alpha) * batch_mean)
        self.output = inpt - mean_to_subtract
        self.old_mean = mean_to_subtract
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self):
        return "BatchMeanNormalization"


class Dropout(Module):
    """
    ### input:  batch_size x n_feats ###
    ### output: batch_size x n_feats ###
    """
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.mask = None

    def updateOutput(self, inpt):
        if self.training:
            self.mask = bernoulli.rvs(1-self.p, size=inpt.shape[-1])
            self.output = self.mask * inpt
        else :
            self.output = inpt * (1 - self.p)
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        self.gradInput = gradOutput * self.mask
        return self.gradInput

    def __repr__(self):
        return "Dropout"