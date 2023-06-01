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
                mean = 0
                self.coefficients.append(random.gauss(mean, 16))  # constant
                for i in range(self.dim):
                    self.coefficients.append(random.gauss(mean, 4))  # first order terms
            elif self.dim > 1 and self.order == 2:
                self.coefficients.append(random.gauss(0, 16))  # constant
                for i in range(self.dim):  # first order terms
                    self.coefficients.append(random.gauss(0, 4))
                for i in range(self.dim):  # second order terms
                    for j in range(i, self.dim):
                        self.coefficients.append(random.gauss(0, 1))
            else:
                raise Exception("Illegal order/dimension combination (only first and second order polynomials allowed "
                                "when input dimension is larger than 1).")

            self.coefficients = np.array(self.coefficients)

    def generate(self, num_points, x_dev=3, y_dev=1, bias=0):
        x = np.empty((num_points, self.dim)) if self.dim > 1 else np.empty(num_points)
        y = np.empty(num_points)

        for i in range(num_points):
            x[i] = random.gauss(0, x_dev) if self.dim == 1 \
                else np.random.normal(loc=0, scale=x_dev, size=self.dim)

            noise = random.gauss(bias, y_dev)
            if self.dim == 1:
                y[i] = noise + sum([self.coefficients[j] * x[i] ** j for j in range(len(self.coefficients))])
            elif self.order == 1:
                y[i] = noise + self.coefficients[0] + x[i] @ self.coefficients[1:]
            elif self.order == 2:
                y[i] = noise + self.coefficients[0] + x[i] @ self.coefficients[1:self.dim + 1]
                index = self.dim + 1
                for a in range(self.dim):
                    for b in range(a, self.dim):
                        y[i] += x[i][a] * x[i][b] * self.coefficients[index]
                        index += 1

        return x, y

    def __str__(self):
        if self.dim == 1:
            string = f"{round(self.coefficients[0], 4)}"
            for i in range(1, len(self.coefficients)):
                string += f" {'-' if self.coefficients[i] < 0 else '+'} {round(abs(self.coefficients[i]), 4)}x" \
                          f"{'' if i == 1 else f'^{i}'}"
            return string

        return "Unable to display function"

