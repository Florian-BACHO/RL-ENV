class Experience:
    def __init__(self, state, action, reward, new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state

    def __repr__(self):
        return "State:" + str(self.state) + ", Action:" + str(self.action) + \
            ", Reward:" + str(self.reward) + ", New State:" + str(self.new_state)
