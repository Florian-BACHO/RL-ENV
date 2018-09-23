import numpy as np
import random

class GreedyArgmaxActionSelector:
    def __init__(self, epsilon_start=1.0, epsilon_final=0.02, epsilon_decay=1e5):
        self.epsilon = self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.step = 0

    def __call__(self, values):
        if random.random() <= self.epsilon:
            return np.random.randint(len(values[0]), size=len(values))
        else:
            return np.argmax(values, axis=1)

    def update_epsilon(self):
        self.step += 1
        self.epsilon = max(self.epsilon_final, self.epsilon_start - self.step / self.epsilon_decay)
