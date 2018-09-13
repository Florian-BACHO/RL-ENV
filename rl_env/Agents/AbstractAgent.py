class AbstractAgent:
    def __init__(self):
        pass

    def __call__(self, state):
        return 0

    def train_full_tries(self, tries_experiences):
        pass

    def train_replay(self, replay_batch):
        pass
