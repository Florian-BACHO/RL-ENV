import tensorflow as tf
from .Trackers import *

class Learner:
    def __init__(self, agent, experience_source, replay_buffer=None, log_dir=None):
        self.agent = agent
        self.exp_src = experience_source
        self.replay_buf = replay_buffer
        self.all_tries_exp = []

        if log_dir:
            self.summary_writer = tf.summary.FileWriter(log_dir)
            self.reward_tracker = MeanRewardTracker(writer=self.summary_writer, time_step=1)
        else:
            self.summary_writer = None

    def __del__(self):
        if self.summary_writer:
            self.summary_writer.close()

    def _update_summaries(self):
        if not self.summary_writer:
            return
        self.reward_tracker(self.all_tries_exp)

    def __call__(self, nb_replay, nb_tries=1):
        self.all_tries_exp = []

        for _ in range(nb_tries):
            done = False
            current_exp = []

            while not done:
                exp = self.exp_src()
                current_exp.append(exp)

                done = exp.done

                if self.replay_buf is not None:
                    self.replay_buf.append(exp)
                    replay = self.replay_buf(nb_replay)
                    self.agent.train_replay(replay)

            self.all_tries_exp.append(current_exp)

        self.agent.train_full_tries(self.all_tries_exp)
        self._update_summaries()
