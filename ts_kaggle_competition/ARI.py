import numpy as np
from numpy.linalg import inv


class ARI:
    def __init__(self, window, s, d=1, n=1):
        self.d = d
        self.s = s
        self.n = n
        self.window = window
        self.weights = np.array([])
        self.diff_matrix = []
        self.point = np.array([])

    def difference(self, data):
        old_diff_data = None
        diff_data = data
        self.diff_matrix.append(data)

        for k in range(len(self.s)):
            old_diff_data = diff_data
            diff_data = []
            assert self.s[k] < len(old_diff_data)
            for i in range(self.s[k], len(old_diff_data)):
                diff_data.append(
                    old_diff_data[i] - old_diff_data[i - self.s[k]])
            self.diff_matrix.append(diff_data)
        for _ in range(self.d):
            old_diff_data = diff_data
            diff_data = []
            for j in range(1, len(old_diff_data)):
                diff_data.append(old_diff_data[j] - old_diff_data[j - 1])
            self.diff_matrix.append(diff_data)

        return diff_data

    def regresion(self, features, targets):
        weights = ((inv((features.T).dot(features))).dot(
            features.T)).dot(targets)
        return weights

    def train(self, raw_data):
        data = self.difference(raw_data)

        t = len(data) - 1
        feature_matrix = []
        targets = []
        for i in range(self.window, t + 1):
            datum = np.array([data[k] for k in range(i - self.window, i)])
            feature_matrix.append(datum)
            targets.append(data[i])
        targets = np.array(targets)
        targets = targets.reshape((-1, 1))

        self.weights = self.regresion(np.array(feature_matrix), targets)
        self.point = data[len(data) - self.window:]

    def predict(self):
        prediction = []
        self.diff_matrix = np.array(self.diff_matrix)
        for _ in range(self.n):
            pred = ((self.weights).flatten()).dot(self.point)
            self.point.append(pred)
            del (self.point[0])
            for k in range(len(self.s), len(self.diff_matrix) - 1):
                pred = pred + self.diff_matrix[len(self.s) - 2 - k][-1]
                self.diff_matrix[len(self.s) - 2 - k].append(pred)
            for m in range(len(self.s)):
                pred = self.diff_matrix[len(self.s) - m - 1][-self.s[m]] + pred
                self.diff_matrix[len(self.s) - m - 1].append(pred)
            prediction.append(pred)
        return prediction
