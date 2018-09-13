import tensorflow as tf

class AbstractANN:
    def __init__(self, scope=""):
        self.scope = scope

        with tf.variable_scope(self.scope):
            self._create_ann()

    def _create_ann(self):
        self.input = None
        self.output = None

    def __call__(self, entries):
        session = tf.get_default_session()

        return session.run(self.output, feed_dict={self.input: entries})

    def assign(self, otherANN):
        session = tf.get_default_session()
        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        all_other_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=otherANN.scope)
        for i, var in enumerate(all_variables):
            session.run(tf.assign(var, all_other_variables[i]))
