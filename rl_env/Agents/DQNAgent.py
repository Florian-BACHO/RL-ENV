import tensorflow as tf
import numpy as np
from .AbstractAgent import *

class DQNAgent(AbstractAgent):
    def __init__(self, main_ann, target_ann, action_selector, learning_rate=1e-2, discount_rate=0.9, target_update_rate=1):
        self.main_ann = main_ann
        self.target_ann = target_ann
        self.action_selector = action_selector
        self.discount_rate = discount_rate
        self.target_update_rate = target_update_rate
        self.target_update_counter = 0

        self.expected_values = tf.placeholder(tf.float32, [None])

        self.actions = tf.placeholder(tf.int32, [None, 2])
        actions_tensors = tf.gather_nd(self.main_ann.output, self.actions)

        loss = tf.losses.mean_squared_error(labels=self.expected_values, predictions=actions_tensors)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = optimizer.minimize(loss)

    def __call__(self, state):
        outputs = self.main_ann(state)
        return self.action_selector(outputs)

    def train_full_tries(self, tries_experiences):
        pass

    def train_replay(self, replay_batch):
        self.action_selector.update_epsilon()

        if replay_batch is None:
            return

        states = [it.state for it in replay_batch]
        actions = [it.action[0] for it in replay_batch]
        rewards = [it.reward for it in replay_batch]
        dones = [it.done for it in replay_batch]
        new_states = [it.new_state for it in replay_batch]

        next_qvalues = self.target_ann(new_states).max(axis=1)
        next_qvalues[np.nonzero(dones)] = 0. # No next QValue for terminal state

        qvalues = next_qvalues * self.discount_rate + rewards

        actions_indices = np.array([[i, action] for i, action in enumerate(actions)])

        session = tf.get_default_session()

        session.run(self.training_op, feed_dict = {self.main_ann.input: states, \
                                                   self.expected_values: qvalues, \
                                                   self.actions: actions_indices})

        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_rate:
            self.target_ann.assign(self.main_ann)
            self.target_update_counter = 0
