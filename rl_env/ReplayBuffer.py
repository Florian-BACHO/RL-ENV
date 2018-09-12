from queue import Queue
import random

class ReplayBuffer(Queue):
    # maxsize = 0 is infinite replay buffer
    def __init__(self, maxsize=10000):
        super(ReplayBuffer, self).__init__(maxsize=maxsize)

    def add_experiences(self, exp):
        for e in exp:
            if self.full():
                self.get()
            self.put(e)

    def __call__(self, sample_size):
        indices = random.sample(range(self.qsize()), sample_size)
        batch = [self.queue[i] for i in indices]

        return batch
