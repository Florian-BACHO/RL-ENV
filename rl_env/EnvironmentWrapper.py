class EnvironmentWrapper:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.state = self.env.reset()

    def step(self, action):
        self.state, reward, done, _ = self.env.step(action)
        if done:
            self.state = None
        return reward
