import numpy as np


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
        self.state = None

    def step(self, weights, grad_weights):
        if self.state is None:
            self.state = []
            for w in weights:
                self.state.append([])
                for w_ik in w:
                    self.state[-1].append(np.zeros_like(w_ik))

        for layer_state, layer_weights, layer_grad_weights in zip(self.state,
                                                                  weights,
                                                                  grad_weights):
            for state, w, grad_w in zip(layer_state, layer_weights,
                                        layer_grad_weights):
                if grad_w.shape[0] == 1:
                    w = w.reshape(grad_w)
                    state = state.reshape(grad_w)

                np.add(-self.learning_rate * grad_w, self.momentum * state,
                       out=state)
                np.add(w, state, out=w)


class Adam:
    # https: // arxiv.org / abs / 1412.6980
    pass
