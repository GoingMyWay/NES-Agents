# THIS IS SINGLE CPU VERSION
import random

import gym
import joblib
import numpy as np
import scipy.misc as sp
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nes import NES


class PongAgent(object):
    def __init__(self,
                 img_shape=(84, 84),
                 hist_len=4,
                 epsilon=0.0,
                 alpha=0.01,
                 sigma=0.01,
                 population_size=15,
                 env_name='PongDeterministic-v0',
                 gym_mode='rgb_array'):
        self.gym_mode = gym_mode
        self.env = gym.make(env_name)
        self.epsilon = epsilon
        self._weights = []
        self._img_shape = img_shape
        self._hist_len = hist_len
        self._create_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.es = NES(sess=self.sess,
                      weights=self._weights,
                      reward_func=self.reward_func,
                      alpha=alpha,
                      sigma=sigma,
                      population_size=population_size,
                      update_func=self.update_values)

    def play(self, n_episode=10):
        pass

    def train(self, n_iters=1000, p_steps=1):
        """
        Train NES
        """
        with self.sess.as_default(), self.sess.graph.as_default():
            self.es.train(n_iters=n_iters)

    def predict_action(self, input_state):
        outputs = self.sess.run([self.outputs], feed_dict={self.inputs: np.expand_dims(input_state, 0)})
        return np.argmax(outputs[0])

    def reward_func(self, weights_try, max_steps=300):
        # assign the weights
        self.update_values(self.sess, weights_try)
        # only run one episode
        the_last_action = None
        total_reward = 0
        max_steps, step = max_steps, 0
        ob = self.env.reset()
        st_s = process_frame(ob, self.img_shape)
        state = np.stack((st_s, st_s, st_s, st_s), axis=2)
        done = False

        while not done and step <= max_steps:
            self.env.render(self.gym_mode)
            if random.random() < self.epsilon:
                action = np.random.choice(range(self.env.action_space.n))
            else:
                action = self.predict_action(input_state=state)
            ob, reward, done, info = self.env.step(action)
            total_reward += reward + float(np.random.choice([-0.00001, 0.00001]))
            step += 1
            the_last_action = action

            img = np.expand_dims(process_frame(ob, self.img_shape), axis=2)
            state = np.append(img, state[:, :, :3], axis=2)

        return total_reward, the_last_action

    def _create_network(self, scope='network'):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, *self.img_shape, self.hist_len], dtype=tf.float32)
            self.conv_1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.inputs, num_outputs=16,
                                      kernel_size=[8, 8], stride=4, padding='SAME')
            self.conv_2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_1, num_outputs=64,
                                      kernel_size=[4, 4], stride=2, padding='SAME')
            self.fc = slim.fully_connected(slim.flatten(self.conv_2), 256, activation_fn=tf.nn.elu)
            self.outputs = slim.fully_connected(self.fc, self.env.action_space.n, activation_fn=tf.nn.softmax)
            # get all the weights
            self._weights = tf.trainable_variables()
            # set the placeholders of weights
            self.placeholders = [tf.placeholder(shape=w.get_shape().as_list(), dtype=tf.float32) for w in self._weights]
            # get assign ops
            self.assign_ops = [tf.assign(w, p) for w, p in zip(self._weights, self.placeholders)]

    def update_values(self, sess, real_weights):
        for idx, p in enumerate(self.placeholders):
            sess.run(self.assign_ops[idx], feed_dict={p: real_weights[idx]})

    def save_weights(self, file_name='weights.pkl'):
        joblib.dump(self.sess.run(self._weights), file_name)

    def load_weights(self, file_name='weights.pkl'):
        self.sess.run([tf_v.assign(v) for tf_v, v in zip(self._weights, joblib.load(file_name))])

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


def process_frame(frame, img_shape):
    img = sp.imresize(rgb2gray(frame[33:194, :, :]), img_shape)
    return img


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
