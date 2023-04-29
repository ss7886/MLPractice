"""
Generates data based on a polynomial function with Gaussian noise.
"""
import numpy as np
import random


class Polynomial:
    def __init__(self, order=2, dim=1, coefficients=None):
        self.order = order
        self.dim = dim

        self.coefficients = coefficients
        if self.coefficients is None:
            self.coefficients = []
            if self.dim == 1:
                for i in range(self.order + 1):
                    mean = 0
                    dev = 10 * 6 ** (-i)
                    self.coefficients.append(random.gauss(mean, dev))
            elif self.dim > 1 and self.order == 1:
                for i in range(self.dim + 1):
                    mean = 0
                    dev = random.gauss(0, 1)
                    self.coefficients.append(random.gauss(mean, dev))
            else:
                raise Exception("Illegal order/dimension combination (either order or dim must be 1).")

            self.coefficients = np.array(self.coefficients)

    def generate(self, num_points, x_dev=3, y_dev=1, bias=0):
        x = []
        y = []

        if not isinstance(x_dev, (list, tuple, np.ndarray)):
            x_dev = [x_dev] * self.dim

        for i in range(num_points):
            y_val = random.gauss(bias, y_dev)
            if self.dim == 1:
                x_val = random.gauss(0, x_dev[0])
                for j in range(self.order + 1):
                    y_val += self.coefficients[j] * x_val ** j
                x.append(x_val)
                y.append(y_val)
            else:
                x_vals = []
                for j in range(self.dim):
                    x_vals.append(random.gauss(0, x_dev[j]))
                x_vals = np.array(x_vals)
                y_val += self.coefficients[0]
                y_val += x_vals @ self.coefficients[1:]
                x.append(x_vals)
                y.append(y_val)

        x = np.array(x)
        y = np.array(y)
        return x, y

    def __str__(self):
        if self.dim == 1:
            string = f"{round(self.coefficients[0], 4)}"
            for i in range(1, len(self.coefficients)):
                string += f" {'-' if self.coefficients[i] < 0 else '+'} {round(abs(self.coefficients[i]), 4)}x" \
                          f"{'' if i == 1 else f'^{i}'}"
            return string

        return "Unable to display function"

