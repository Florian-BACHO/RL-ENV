import numpy as np
import tensorflow as tf
import random

class GreedyActionSelector:
    def __init__(self, selector, epsilon_start=1.0, epsilon_final=0.02, epsilon_decay=1e5):
        self.selector = selector
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.step = tf.Variable(0.0)
        self.incrStep = tf.assign(self.step, self.step + 1)
        self.epsilon = tf.maximum(self.epsilon_final, self.epsilon_start - self.step / self.epsilon_decay)

    def __call__(self, values):
        session = tf.get_default_session()

        if random.random() <= session.run(self.epsilon):
            return np.random.randint(len(values[0]), size=len(values))
        else:
            return self.selector(values)

    def update_epsilon(self):
        session = tf.get_default_session()
        session.run(self.incrStep)
