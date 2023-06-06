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
                                                     (self.layer_sizes[i], self.layer_sizes[i - 1])))
            self.weights.append(np.random.normal(0, 2 / self.layer_sizes[-1],
                                                 (self.output_dim, self.layer_sizes[-1])))
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

    def forward_prop(self, x):
        """Only works for ReLU"""
        assert len(x) == self.input_dim
        activations = []
        outputs = []

        z = self.weights[0] @ x + self.biases[0]
        layer_output = self.nonlinearity(z)
        n = len(layer_output)
        outputs.append(layer_output)
        activations.append(np.array([[1 if layer_output[i] > 0 and i == j else 0 for i in range(n)] for j in range(n)]))
        assert np.shape(activations[0]) == (n, n)

        for i in range(1, self.hidden_layers):
            z = self.weights[i] @ layer_output + self.biases[i]
            layer_output = self.nonlinearity(z)
            n = len(layer_output)
            outputs.append(layer_output)
            activations.append(np.array([[1 if layer_output[i] > 0 and i == j else 0
                                          for i in range(n)] for j in range(n)]))
            assert np.shape(activations[-1]) == (n, n)

        assert len(activations) == self.hidden_layers
        assert len(outputs) == self.hidden_layers

        y = self.weights[self.hidden_layers] @ layer_output + self.biases[self.hidden_layers]
        return y, activations, outputs

    def gradient(self, x, y):
        """Only works for ReLU, output_dim = 1"""
        assert len(x) == self.input_dim
        layers = self.hidden_layers
        prediction, activations, outputs = self.forward_prop(x)

        dl_dz = prediction - y
        d_weights = [outputs[-1]]
        d_biases = [np.array([1])]
        diff_layers = [self.weights[i + 1] @ activations[i] for i in range(layers)]

        for i in range(1, layers):
            diff_layers[layers - i - 1] = diff_layers[layers - i - 1] @ diff_layers[layers - i]

        for i in range(1, layers):
            d_weights.append(diff_layers[-i].T @ outputs[-i].T)
            assert np.shape(d_weights[-1]) == np.shape(self.weights[-i - 1])
            d_biases.append(diff_layers[-i])
            d_biases[-1] = d_biases[-1].reshape(self.layer_sizes[-i])

        d_weights.append(diff_layers[0].T @ x.T)
        d_weights[-1] = d_weights[-1].reshape((self.layer_sizes[0], self.input_dim))
        assert np.shape(d_weights[-1]) == np.shape(self.weights[0])
        d_biases.append(diff_layers[0])
        d_biases[-1] = d_biases[-1].reshape(self.layer_sizes[0])

        d_weights.reverse()
        d_biases.reverse()

        return [dl_dz[0] * d_w for d_w in d_weights], [dl_dz[0] * d_b for d_b in d_biases]

    def train_single(self, x, y, learning_rate_weights, learning_rate_biases):
        assert len(learning_rate_weights) == self.hidden_layers + 1 \
               and len(learning_rate_biases) == self.hidden_layers + 1

        d_weights, d_biases = self.gradient(x, y)

        for i in range(self.hidden_layers + 1):
            self.weights[i] -= learning_rate_weights[i] * d_weights[i]
            self.biases[i] -= learning_rate_biases[i] * d_biases[i]

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
