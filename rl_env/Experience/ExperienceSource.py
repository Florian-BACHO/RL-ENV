from .Experience import *

class ExperienceSource:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def _step(self):
        state = self.env.state
        action = self.agent(state)
        reward = self.env(action)
        new_state = self.env.state

        return Experience(state, action, reward, new_state)

    def __call__(self):
        self.env.reset()

        all_exp = []
        while True:
            exp = self._step()
            all_exp.append(exp)

            if exp.new_state is None:
                break

        return all_exp
