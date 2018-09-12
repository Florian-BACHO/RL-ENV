class AbstractAgent:
    def __init__(self):
        pass

    def __call__(self, state):
        return 0

    def train(self, tries_experiences, replay_batch):
        pass
