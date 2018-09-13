import gym

class CumulateEnvWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(CumulateEnvWrapper, self).__init__(env)
        self.cumulated_reward = 0.

    def reset(self):
        out = self.env.reset()
        self.cumulated_reward = 0.

        return out

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.cumulated_reward += reward

        if done:
            reward_return = self.cumulated_reward
            self.cumulated_reward = 0.
            return state, reward_return, done, info
        else:
            return state, 0., done, info
