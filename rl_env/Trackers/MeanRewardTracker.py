from .ScalarTracker import *
import numpy as np

class MeanRewardTracker(ScalarTracker):
    def __init__(self, time_step=1, dump=True, writer=None):
        super(MeanRewardTracker, self).__init__("Mean Reward", dump, writer)

        self.x = 0
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
        if self.x % self.time_step == 0:
            mean = self._calc_mean(exp)

            super(MeanRewardTracker, self).__call__(self.x, mean)

        self.x += 1

    def _dump(self, x, y):
        print("Epoch: %d, Mean reward: %f" % (x, y))
