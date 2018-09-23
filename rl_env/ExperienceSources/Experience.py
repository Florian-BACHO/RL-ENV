class Experience:
    def __init__(self, state, action, reward, done, new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.new_state = new_state

    def __repr__(self):
        return "State:" + str(self.state) + ", Action:" + str(self.action) + \
            ", Reward:" + str(self.reward) + ", Done:" + str(self.done) + ", New State:" + str(self.new_state)
