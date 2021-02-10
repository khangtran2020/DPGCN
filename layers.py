import tensorflow as tf

class GraphConv(tf.keras.Layer):

    def __init__(self, units=32, activation=None):
        super(GraphConv, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[0][-1], self.units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs[1], tf.matmul(inputs[0], self.w)) + self.b)