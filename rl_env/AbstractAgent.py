class AbstractAgent:
    def __init__(self):
        pass

    def action(self, state):
        return 0

    def train(self, tries_experiences, replay_batch):
        print(tries_experiences)
        print(replay_batch)
        pass
