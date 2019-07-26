import numpy as np


def uniform(x):
    return int(-1 < x < 1) * 0.5


def gaussian(x):
    return np.exp(-x**2/2) / (2*np.pi)**0.5


def epaneshnikov(x):
    return 3 / 4 * max([1 - x**2, 0])


class KDE:
    def __init__(self, h, kernel=uniform):
        self.h = h
        self.kernel = kernel
        self.data = None

    def train(self, data):
        self.data = data

    def predict(self, point):
        return np.sum([self.kernel((point - x)/self.h) for x in self.data])/(
                self.h*len(self.data))
