import tensorflow as tf

class ScalarTracker:
    def __init__(self, name, dump=True, writer=None):
        self.name = name
        self.dump = dump
        self.placeholder = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar(name, self.placeholder)
        self.writer = writer

    def __call__(self, x, y):
        session = tf.get_default_session()
        if self.dump:
            self._dump(x, y)
        if self.writer:
            sum = session.run(self.summary, feed_dict={self.placeholder: y})
            self.writer.add_summary(sum, x)

    def _dump(self, x, y):
        print("%d: %s = %f" % (x, self.name, y))
