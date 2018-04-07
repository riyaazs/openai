import tensorflow as tf

# Q network used by agent as function approximator to train and test.
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10, layer_size=2,
                 name='QNetwork'):
        self.name = name
        self.copy_op = None
        # state inputs to the Q-network
        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            self.layers=[]
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.layers.append(self.fc1)
            for i in range(layer_size-1):
                self.layers.append(tf.contrib.layers.fully_connected(
                    self.layers[-1], hidden_size))

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.layers[-1],
                                                            action_size, activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
