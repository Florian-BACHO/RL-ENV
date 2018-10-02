from .ScalarTracker import *
import tensorflow as tf
import numpy as np

class EpsilonTracker(ScalarTracker):
    def __init__(self, time_step=1, dump=True, writer=None):
        super(MeanRewardTracker, self).__init__("Mean Reward", dump, writer)

        self.x = tf.Variable(0)
        self.incrX = tf.assign(self.x, self.x + 1)
        self.time_step = time_step

    def _convert_exp_to_rewards(self, try_exp):
        out = 0.
        for reward in try_exp:
            out += reward.reward
        return out

    def _calc_mean(self, all_exp):
        all_sum = [np.sum(self._convert_exp_to_rewards(it)) for it in all_exp]
        return np.mean(all_sum)

    def __call__(self, exp):
        session = tf.get_default_session()
        x = session.run(self.x)

        if x % self.time_step == 0:
            mean = self._calc_mean(exp)

            super(MeanRewardTracker, self).__call__(x, mean)

        session.run(self.incrX)

    def _dump(self, x, y):
        print("Epoch: %d, Mean reward: %f" % (x, y))
