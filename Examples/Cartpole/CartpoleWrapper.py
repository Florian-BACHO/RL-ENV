import gym

class CartpoleWrapper(gym.Wrapper):
    def __init__(self):
        env = gym.make("CartPole-v0")
        super(CartpoleWrapper, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward = 0.
        reward -= abs(state[2]) # Punish angle

        return state, reward, done, info
