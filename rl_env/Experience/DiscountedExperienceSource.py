from .ExperienceSource import *

class DiscountedExperienceSource(ExperienceSource):
    def __init__(self, env, agent, discount_rate=0.9):
        super(DiscountedExperienceSource, self).__init__(env, agent)
        self.discount_rate = discount_rate

    def do_episod(self):
        all_exp = super(DiscountedExperienceSource, self).do_episod()
        total = 0.
        for it in reversed(all_exp):
            total *= self.discount_rate
            total += it.reward
            it.reward = total
        return all_exp