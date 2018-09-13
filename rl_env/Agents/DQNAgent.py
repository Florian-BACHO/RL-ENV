import tensorflow as tf
from .AbstractAgent import *

class DQNAgent(AbstractAgent):
    def __init__(self, main_ann, target_ann, action_selector, learning_rate=1e-2):
        self.main_ann = main_ann
        self.target_ann = target_ann
        self.action_selector = action_selector

        self.y = tf.placeholder(tf.float32, main_ann.output.get_shape())
        loss = tf.losses.mean_squared_error(self.y, main_ann.output)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = optimizer.minimize(loss)

    def __call__(self, state):
        session = tf.get_default_session()

        outputs = self.main_ann(state)
        return self.action_selector(outputs)

    def train_full_tries(self, tries_experiences):
        pass

    def train_replay(self, replay_batch):
        self.action_selector.update_epsilon()

        if replay_batch is None:
            return
