import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid(x) * (1-sigmoid(x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class TheNetwork:
    def __init__(self):
        weights = np.array([1,1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, inputs):
        h1_out = self.h1.feedforward(inputs)
        h2_out = self.h2.feedforward(inputs)
        o1_out = self.o1.feedforward(np.array([h1_out, h2_out]))
        return o1_out




def mean_squared_error(yt, yp):
    return np.sum((yt-yp)**2)/len(yt)



