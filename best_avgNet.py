import tensorflow as tf
import Config


cf = Config.Config()

class best_avgNet(object):
    def __init__(self, scope_name):
        self.action_dim = cf.actionDim
        self.s = tf.placeholder("float", [None, 80, 80, 4])
        self.a = tf.placeholder("float", [None, cf.actionDim])
        self.y = tf.placeholder("float", [None])
        self.name = scope_name
        with tf.variable_scope(self.name):
            self.best_q_t = self.createNetwork("target_net")
            self.best_q = self.createNetwork("source_net")

        self.source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/source_net")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/target_net")
        self.update_op = [self.target_vars[i].assign(self.source_vars[i]) for i in range(len(self.source_vars))]

        self.train_best_op = self.train_best()

    def train_best(self):
        q_action = tf.reduce_sum(tf.multiply(self.best_q, self.a), reduction_indices = 1)
        self.loss = tf.reduce_mean(tf.square(self.y - q_action))
        train_op = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(self.loss)
        return train_op

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def createNetwork(self, scope):
        with tf.variable_scope(scope):
            # network weights
            W_conv1 = self.weight_variable([8, 8, 4, 32])
            b_conv1 = self.bias_variable([32])

            W_conv2 = self.weight_variable([4, 4, 32, 64])
            b_conv2 = self.bias_variable([64])

            W_conv3 = self.weight_variable([3, 3, 64, 64])
            b_conv3 = self.bias_variable([64])

            W_fc1 = self.weight_variable([256, 256])
            b_fc1 = self.bias_variable([256])

            W_fc2 = self.weight_variable([256, self.action_dim])
            b_fc2 = self.bias_variable([self.action_dim])


            # hidden layers
            h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)

            h_conv3 = tf.nn.relu(self.conv2d(h_pool2,W_conv3, 1) + b_conv3)
            h_pool3 = self.max_pool_2x2(h_conv3)

            h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

            # readout layer
            readout = tf.matmul(h_fc1, W_fc2) + b_fc2

            return readout