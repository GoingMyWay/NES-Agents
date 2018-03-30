# THIS IS SINGLE CPU VERSION
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nes import NES


class PongAgent(object):
    def __init__(self, img_shape=(40, 40), hist_len=4):
        self.env = gym.make('PongDeterministic-v0')
        self.es = NES(weights=self.weights,
                      reward_func=self.reward_func,
                      alpha=0.01,
                      sigma=0.01,
                      population_size=15)
        self._weights = []
        self._img_shape = img_shape
        self._hist_len = hist_len
        self._create_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def play(self):
        pass

    def train(self, n_iters=300, p_steps=20):
        with self.sess.as_default(), self.sess.graph.as_default():
            self.es.train(n_iters=n_iters, p_steps=p_steps)

    def predict_action(self, input_state):
        pass

    def reward_func(self):
        pass

    def _create_network(self, scope='network'):
        with tf.variable_scope(scope):
            inputs = tf.placeholder(shape=[None, *self.img_shape, self.hist_len], dtype=tf.float32)
            conv_1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=inputs, num_outputs=32,
                                 kernel_size=[8, 8], stride=4, padding='SAME')
            conv_2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=conv_1, num_outputs=64,
                                 kernel_size=[4, 4], stride=2, padding='SAME')
            conv_3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=conv_2, num_outputs=64,
                                 kernel_size=[3, 3], stride=1, padding='SAME')
            fc = slim.fully_connected(slim.flatten(conv_3), 64, activation_fn=tf.nn.elu)


    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def img_shape(self):
        return self._img_shape

    @img_shape.setter
    def img_shape(self, img_shape):
        self._img_shape = img_shape

    @property
    def hist_len(self):
        return self._hist_len

    @hist_len.setter
    def hist_len(self, hist_len):
        self._hist_len = hist_len
