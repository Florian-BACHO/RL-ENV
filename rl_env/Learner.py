import tensorflow as tf
from .Trackers import *
from .ActionSelectors import *

class Learner:
    def __init__(self, agent, experience_source, replay_buffer=None, \
                 reward_tracker=None, loss_tracker=None, epsilon_tracker=None):
        self.agent = agent
        self.exp_src = experience_source
        self.replay_buf = replay_buffer
        self.all_tries_exp = []

        self.reward_tracker = reward_tracker
        self.loss_tracker = loss_tracker
        self.epsilon_tracker = epsilon_tracker

        self.epoch = tf.Variable(0)
        self.inc_epoch = tf.assign(self.epoch, self.epoch + 1)

    def _update_epoch_summaries(self, session, epoch, exp, loss):
        ep = session.run(epoch)
        if self.reward_tracker is not None:
            self.reward_tracker(ep, exp.reward)
        if self.loss_tracker is not None and loss is not None:
            self.loss_tracker(ep, loss)
        if self.epsilon_tracker is not None and \
           type(self.agent.action_selector) is GreedyActionSelector:
            epsilon = session.run(self.agent.action_selector.epsilon)
            self.epsilon_tracker(ep, epsilon)

    def __call__(self, nb_replay, nb_tries=1):
        session = tf.get_default_session()

        self.all_tries_exp = []

        for _ in range(nb_tries):
            done = False
            current_exp = []

            while not done:
                exp = self.exp_src()
                current_exp.append(exp)

                session.run(self.inc_epoch)

                done = exp.done
                loss = None

                if self.replay_buf is not None:
                    self.replay_buf.append(exp)
                    replay = self.replay_buf(nb_replay)
                    loss = self.agent.train_replay(replay)

                self._update_epoch_summaries(session, self.epoch, exp, loss)

            self.all_tries_exp.append(current_exp)

        self.agent.train_full_tries(self.all_tries_exp)
