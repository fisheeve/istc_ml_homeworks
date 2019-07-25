import numpy as np
from copy import copy


class SGD:
    """
    Simple stochastic gradient descent.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, weights, grad_weights):
        for layer_weights, layer_grad_weights in zip(weights, grad_weights):
            for w, grad_w in zip(layer_weights, layer_grad_weights):
                if grad_w.shape[0] == 1:
                    w = w.reshape(grad_w)
                np.add(w, -self.learning_rate * grad_w, out=w)


class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.state = np.array([])

    def step(self, weights, grad_weights):
        if self.state == np.array([]):
            self.state = np.zeros_like(weights)
        for i, layer_weights, layer_grad_weights in zip(self.state,
                                                        weights, grad_weights):
            for j, w, grad_w in zip(self.state, layer_weights,
                                    layer_grad_weights):
                if grad_w.shape[0] == 1:
                    w = w.reshape(grad_w)
                np.add(self.momentum * self.state[i][j],
                       -self.learning_rate * grad_w, out=self.state[i][j])
                np.add(w, self.state[i][j], out=w)




class Adam:
    # https: // arxiv.org / abs / 1412.6980
    pass
