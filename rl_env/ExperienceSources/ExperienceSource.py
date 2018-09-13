from .Experience import *

class ExperienceSource:
    def __init__(self, env, agent):
        self.env = env
        self.state = self.env.reset()
        self.agent = agent

    def __call__(self):
        state = self.state
        action = self.agent([state])
        new_state, reward, done, _ = self.env.step(action[0])
        if done:
            new_state = None
            self.state = self.env.reset()
        else:
            self.state = new_state

        return Experience(state, action, reward, new_state)
