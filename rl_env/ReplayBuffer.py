from collections import deque
import numpy as np

class ReplayBuffer(deque):
    # maxsize = 0 is infinite replay buffer
    def __init__(self, maxsize=10000, start_size=10000):
        self.start_size = start_size
        super(ReplayBuffer, self).__init__(maxlen=maxsize)

    def __call__(self, sample_size):
        size = len(self)
        if size < self.start_size:
            return None
        indices = np.random.choice(size, size if size < sample_size else sample_size, replace=False)

        return [self[i] for i in indices]
