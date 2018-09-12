import tensorflow as tf
from .Trackers import *

class BasicLearner:
    def __init__(self, agent, experience_source, replay_buffer=None, log_dir=None):
        self.agent = agent
        self.exp_src = experience_source
        self.replay_buf = replay_buffer
        self.all_tries_exp = []

        if log_dir:
            self.summary_writer = tf.summary.FileWriter(log_dir)
            self.reward_tracker = MeanRewardTracker(writer=self.summary_writer, time_step=10)
        else:
            self.summary_writer = None

    def __del__(self):
        if self.summary_writer:
            self.summary_writer.close()

    def _update_summaries(self):
        if not self.summary_writer:
            return
        self.reward_tracker(self.all_tries_exp)

    def __call__(self, nb_tries, nb_replay):
        self.all_tries_exp = []

        for _ in range(nb_tries):
            exps = self.exp_src()
            self.all_tries_exp.append(exps)
            self.replay_buf.add_experiences(exps)

        if self.replay_buf:
            replay = self.replay_buf(nb_replay)
        else:
            replay = None
        self.agent.train(self.all_tries_exp, replay)

        self._update_summaries()
