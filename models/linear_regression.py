"""
Defines a linear regression model used to find an approximation that minimizes mean squared error among a dataset.
"""
import numpy as np
from numpy.linalg import LinAlgError


class LinearRegression:
    def __init__(self, input_dim=1, order=1, basis_functions=None):
        self.input_dim = input_dim
        if basis_functions is None:
            self.basis_functions = []
            self.string = []
            if input_dim == 1:
                for i in range(order + 1):
                    self.basis_functions.append(lambda x, exp=i: x ** exp)
                    self.string.append("" if i == 0 else "x" if i == 1 else f"x^{i}")
            else:
                if order == 1:
                    self.basis_functions.append(lambda x: 1)
                    for i in range(self.input_dim):
                        self.basis_functions.append(lambda x, idx=i: x[idx])
                        self.string.append(f"x_{i + 1}")
                elif order == 2:
                    self.basis_functions.append(lambda x: 1)
                    for i in range(self.input_dim):
                        self.basis_functions.append(lambda x, idx=i: x[idx])
                        self.string.append(f"x_{i + 1}")
                    for i in range(self.input_dim):
                        for j in range(i, self.input_dim):
                            self.basis_functions.append(lambda x, idx1=i, idx2=j: x[idx1] * x[idx2])
                            self.string.append(f"x_{i + 1}^2" if i == j else f"x_{i + 1} x_{j + 1}")
                else:
                    raise Exception("invalid input_dim/order combination (only order 1 or 2 approximations currently"
                                    "supported for input dimensions higher than 1).")

    def _phi(self, x):
        phi = np.empty((len(x), len(self.basis_functions)))
        for i in range(len(x)):
            for j in range(len(self.basis_functions)):
                phi[i][j] = self.basis_functions[j](x[i])
        return phi

    def train(self, x, y):
        assert len(x) == len(y)
        if self.input_dim > 1:
            assert x.shape[1] == self.input_dim
        else:
            assert x.ndim == 1
        phi = self._phi(x)

        if phi.shape[1] > phi.shape[0]:
            raise Exception("Overparametrized! (not enough datapoints)")

        theta = np.linalg.solve(phi.T @ phi, phi.T @ y)
        loss = 0.5 * np.linalg.norm(phi @ theta - y) ** 2

        return theta, loss

    def predict(self, x, theta, y=None):
        if y is not None:
            assert len(x) == len(y)
        assert len(theta) == len(self.basis_functions)

        phi = self._phi(x)
        predictions = phi @ theta

        if y is None:
            return predictions

        loss = 0.5 * np.linalg.norm(predictions - y) ** 2
        return predictions, loss

    def to_string(self, theta, digits=4):
        assert len(theta) == len(self.string)
        string = ""
        for i in range(len(theta)):
            if i != 0:
                string += f" {'' if theta[i] < 0 else '+'}"
            string += f"{round(theta[i], digits)}{self.string[i]}"
        return string



