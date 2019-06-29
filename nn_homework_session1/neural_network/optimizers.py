import numpy as np
from copy import copy


class SGD:
    """
    Simple stochastic gradient descent.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, weights, grad_weights):
        for layer_weights, layer_grad_weights in zip(copy(weights),
                                                     copy(grad_weights)):
            for w, grad_w in zip(layer_weights, layer_grad_weights):
                if grad_w.shape[0] == 1:
                    w = w.reshape(grad_w)
                np.add(w, -self.learning_rate * grad_w, out=w)


class SGDMomentum:
    """
    Simple stochastic gradient descent.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        # todo, add state

    def step(self, weights, grad_weights):
        raise NotImplemented()


class Adam:
    # https: // arxiv.org / abs / 1412.6980
    raise NotImplemented()
