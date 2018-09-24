import tensorflow as tf
import rl_env as rl

class DQNAnn(rl.Agents.AbstractANN):
    def __init__(self, nb_entry, nb_hiddens, nb_out, scope=""):
        self.nb_entry = nb_entry
        self.nb_hiddens = nb_hiddens
        self.nb_out = nb_out
        super(DQNAnn, self).__init__(scope)

    def _create_ann(self):
        self.input = tf.placeholder(tf.float32, [None, self.nb_entry])
        last = self.input
        for it in self.nb_hiddens:
            last = tf.layers.dense(last, it, tf.nn.relu)
        self.output = tf.layers.dense(last, self.nb_out)
