import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from numpy.linalg import det, inv

pi =  3.14159265358979
class KMeans:

    def __init__(self, class_num, step_num, data):
        self.class_num = class_num
        self.step_num = step_num
        self.indicators = np.zeros((len(data), self.class_num))
        self.means = np.array(
            [data[np.random.randint(0, len(data), 1)].flatten()
             for _ in range(self.class_num)])

    def train(self, data):
        step = 0
        while step < self.step_num:
            for n in range(len(data)):
                k = np.argmin([np.sum((data[n] - self.means[i])**2)
                              for i in range(self.class_num)])
                self.indicators[n] = np.zeros(self.class_num)
                self.indicators[n, k] = 1

            for k in range(self.class_num):
                k_data = np.sum([self.indicators[n, k]*data[n]
                                 for n in range(len(data))], axis=0)
                k_sum = np.sum([self.indicators[n, k]
                               for n in range(len(data))])
                if k_sum != 0:
                    self.means[k] = k_data / k_sum
            step = step + 1

        return self.means

    def predict(self, data):
        prediction = np.array([np.argmax(self.indicators[n])
                               for n in range(len(data))])
        return prediction


class GaussMix:
    def __init__(self, class_num, step_num, convergence, means, cov, data) :
        self.class_num = class_num
        self.convergence = convergence
        self.step_num = step_num
        #step1: seting initial values
        self.means = means
        self.prior = 1 / self.class_num*np.ones(class_num)
        self.cov = cov
        self.post = np.zeros((len(data), self.class_num))
        self.efpoints = np.zeros(class_num)
        self.dim = len(data[0])
        self.loss = []

    def prob(self, vector, mean, cov):
        prob = 1/(2*pi)**(self.dim/2)*(1/(det(cov)**0.5))*np.exp(
            (-0.5)*(((vector - mean).T).dot(inv(cov))).dot(vector-mean))
        return prob

    def loss_func(self, data, c_num, priors, means, covs):
        loss = np.sum([np.log(np.sum([priors[k] * self.prob(
            data[n], means[k], covs[k]) for k in range(c_num)]))
                       for n in range(len(data))])
        return loss

    def train(self, data):
        loss = np.inf
        step = 0
        while loss > self.convergence and step < self.step_num:
        #step 2: E step
            for k in range(self.class_num):
                for n in range(len(data)):
                    self.post[n,k] = self.prior[k] * self.prob(
                        data[n], self.means[k], self.cov[k]) / (
                        self.prior.dot(np.array([self.prob(data[n],
                        self.means[j], self.cov[j]) for j in range(
                            self.class_num)])))
                self.efpoints[k] = np.sum(self.post[:, k])
            #step 3: M step
            for k in range(self.class_num):
                self.means[k] = 1 / self.efpoints[k] * np.sum(np.array(
                        [self.post[n, k] * data[n] for n in range(len(data))]),
                         axis=0)
                self.cov[k] = 1 / self.efpoints[k] * np.sum(np.array(
                         [self.post[n, k] * (data[n] - self.means[k]).dot((
                          data[n] - self.means[k]).T) for n in range(len(
                             data))]))
                self.prior[k] = self.efpoints[k] / len(data)
            #step 4: evaluate loss (-log(likelyhood_prob))
            loss = self.loss_func(data, self.class_num, self.prior,
                                  self.means, self.cov)
            self.loss.append(loss)
            step = step + 1

    def predict_proba(self):
        proba = []
        for n in range(len(self.post)):
            proba.append(np.max(self.post[n]))
        return proba

    def predict(self):
        prediction = []
        for n in range(len(self.post)):
            prediction.append(np.argmax(self.post[n]))
        return prediction
