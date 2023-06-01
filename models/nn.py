"""
Defines a neural network.
"""
import math

import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim=1, hidden_layers=1, output_dim=1, layer_sizes=None, weights=None, biases=None,
                 nonlinearity=None):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim

        if layer_sizes is None:
            self.layer_sizes = [50 for i in range(hidden_layers)]
        elif len(layer_sizes) != hidden_layers:
            raise ValueError(f"length of layer_sizes ({len(layer_sizes)}) does not match number of hidden_layers "
                             f"({hidden_layers}).")
        else:
            self.layer_sizes = layer_sizes

        if weights is None:
            self.weights = []
            self.weights.append(np.random.normal(0, 2 / self.input_dim,
                                                 (self.layer_sizes[0], self.input_dim)))
            for i in range(1, self.hidden_layers):
                self.weights.append(np.random.normal(0, 2 / self.layer_sizes[i - 1],
                                                     (self.layer_sizes[i - 1], self.layer_sizes[i])))
            self.weights.append(np.random.normal(0, 2 / self.layer_sizes[-1],
                                                 (self.layer_sizes[-1], self.output_dim)))
        else:
            self.weights = weights

        if biases is None:
            self.biases = []
            for size in self.layer_sizes:
                self.biases.append(np.ones(size) * 0.1)
            self.biases.append(np.ones(output_dim) * 0.1)
        else:
            self.biases = biases

        if nonlinearity is None:
            self.nonlinearity = lambda x: np.maximum(x, 0)

    def predict_single(self, x):
        assert len(x) == self.input_dim
        z = self.weights[0] @ x + self.biases[0]
        layer_output = self.nonlinearity(z)

        for i in range(1, self.hidden_layers):
            z = self.weights[i] @ layer_output + self.biases[i]
            layer_output = self.nonlinearity(z)

        return self.weights[self.hidden_layers] @ layer_output + self.biases[self.hidden_layers]

    def predict(self, dataset):
        outputs = []
        for x in dataset:
            outputs.append(self.predict_single(x))
        return np.array(outputs)
