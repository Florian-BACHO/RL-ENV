import numpy as np

class ArgmaxActionSelector:
    def __call__(self, values):
        return np.argmax(values, axis=1)
