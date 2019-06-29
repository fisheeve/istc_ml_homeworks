import numpy as np
from scipy.stats import bernoulli
from .base_modules import Module


class Linear(Module):
    """
    # Fixme: delete instructions after implementing
    ### input:  batch_size x n_features1 ###
    ### output: batch_size x n_features2 ###

    A module which applies a linear transformation
    A common name is fully-connected layer, dense layer, InnerProductLayer
     in caffe(package).

    The module should work with 2D input of shape (n_samples, n_feature).
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
    # Fixme: delete instructions after implementing
    This one is probably the hardest but as others only takes 5 lines of code
     in total.

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
    # Fixme: delete instructions after implementing
    One of the most significant recent ideas that impacted NNs a lot is Batch
    normalization (https://arxiv.org/abs/1502.03167). The idea is simple,
    yet effective: the features should be whitened ( 𝑚𝑒𝑎𝑛=0 , 𝑠𝑡𝑑=1 ) all the
    way through NN. This improves the convergence for deep models letting it
    train them for days but not weeks. You are to implement a part of the
    layer: mean subtraction. That is, the module should calculate mean value for
    every feature (every column) and subtract it.

    Note, that you need to estimate the mean over the dataset to be able to
    predict on test examples. The right way is to create a variable which will
    hold smoothed mean over batches (exponential smoothing works good) and use
     it when forwarding test examples.

    When training calculate mean as following:
        mean_to_subtract = self.old_mean * alpha + batch_mean * (1 - alpha)
    when evaluating (self.training == False) set  𝑎𝑙𝑝ℎ𝑎=1 .

    ### input:  batch_size x n_feats ###
    ### output: batch_size x n_feats ###
    """
    def __init__(self, alpha=0.95):
        super(BatchMeanSubtraction, self).__init__()

        self.alpha = alpha
        self.old_mean = None

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "BatchMeanNormalization"


class Dropout(Module):
    """
    # Fixme: delete instructions after implementing
    Implement dropout (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    The idea and implementation is really simple: just multiply the input by
      𝐵𝑒𝑟𝑛𝑜𝑢𝑙𝑙𝑖(𝑝) mask.

    This is a very cool regularizer. In fact, when you see your net is
    overfitting try to add more dropout. It is hard to test, since every forward
    requires sampling a new mask, that is the only reason we need fix_mask
    parameter in there.

    While training (self.training == True) it should sample a mask on each
     iteration (for every batch).
    When testing this module should implement identity transform
     i.e. self.output = input * p.

    ### input:  batch_size x n_feats ###
    ### output: batch_size x n_feats ###
    """
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.mask = None

    def updateOutput(self, inpt):
        raise NotImplemented()
        return self.output

    def updateGradInput(self, inpt, gradOutput):
        raise NotImplemented()
        return self.gradInput

    def __repr__(self):
        return "Dropout"