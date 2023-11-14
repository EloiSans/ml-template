from torch import nn
import numpy as np


class LstSqConvergence(nn.Module):  # TODO: structure as a decorator
    def __init__(self, window_size, epsilon):
        super().__init__()
        self.window_size = window_size
        self.epsilon = epsilon
        self.window = list()

    def forward(self, value):
        self.window.append(value)
        if len(self.window) >= self.window_size:
            self.window = self.window[-self.window_size:]
            return abs(self._get_slope()) < self.epsilon
        else:
            return False

    def _get_slope(self):
        x = np.array(list(range(self.window_size)))
        x = np.vstack([x, np.ones(len(x))]).T
        y = np.array(self.window)
        m, b = np.linalg.lstsq(x, y, rcond=None)[0]
        return m


class MaxEpoch(object):
    def __init__(self, func):
        super().__init__()
        self._func = func

    def __call__(self, start_epoch, max_epochs):
        def wrapper(**kwargs):
            for epoch in range(start_epoch, max_epochs):
                res = self._func(**kwargs)
            return res
        return wrapper
