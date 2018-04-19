
import os
import re
import gym
import tensorflow as tf
import numpy as np


import Config

class ActorCriticNetwork:

    def __init__(self):

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('gpu:2'):

                self.build_model()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.LOAD:
                    variables = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in variables}, max_to_keep=0)

    def build_model(self):

        env = gym.make(Config.ATARI_GAME)
        self.num_actions = env.action_space.n
        env.close()

        self.x = tf.placeholder(tf.float32, [None, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES],
                                name='X')

        # As implemented in A3C paper
        self.conv1 = conv2d_layer(self.x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.conv2 = conv2d_layer(self.conv1, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        self.flatten = tf.reshape(self.conv2, shape=[-1, (self.conv2.get_shape()[1] * self.conv2.get_shape()[2] * self.conv2.get_shape()[3])._value])
        self.fc1 = dense_layer(self.flatten, 256, 'dense1')

        self.logits_v = tf.squeeze(dense_layer(self.fc1, 1, 'logits_v', func=None), axis=[1])
        self.logits_p = dense_layer(self.fc1, self.num_actions, 'logits_p', func=None)

        self.R = tf.placeholder(tf.float32, [None], name='Yr')
        self.action_one_hot = tf.placeholder(tf.float32, [None, self.num_actions])

        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.R - self.logits_v), axis=0)
        self.softmax_p = tf.nn.softmax(self.logits_p)
        self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_one_hot, axis=1)

        self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, Config.LOG_NOISE)) \
                        * (self.R - tf.stop_gradient(self.logits_v))
        self.cost_p_2 = -1 * Config.ENTROPY_BETA * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, Config.LOG_NOISE)) *
                                      self.softmax_p, axis=1)

        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)

        self.cost_all = self.cost_p + self.cost_v
        self.opt = tf.train.RMSPropOptimizer(
            learning_rate=Config.LEARNING_RATE,
            decay=Config.RMSPROP_DECAY,
            momentum=Config.RMSPROP_MOMENTUM,
            epsilon=Config.RMSPROP_EPSILON)

        self.minimizer = self.opt.minimize(self.cost_all)

    def predict(self, states):
        return self.actor_critic(states)

    def actor_single(self, state):
        return self.actor(state[None, :])[0]

    def actor_critic(self, states):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: states})

    def critic(self, states):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: states})
        return prediction

    def actor(self, states):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: states})
        return prediction

    def train(self, states, actions, rewards):
        self.sess.run(self.minimizer, feed_dict={self.x: states, self.R: rewards, self.action_one_hot: actions})

    def checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % ('network', episode)

    def get_episode_from_filename(self, filename):
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self.checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self.checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self.get_episode_from_filename(filename)

    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))


def dense_layer(input, out_dim, name, func=tf.nn.relu):
    in_dim = input.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.matmul(input, w) + b
        if func is not None:
            output = func(output)

    return output


def conv2d_layer(input, filter_size, out_dim, name, strides, func=tf.nn.relu):
    in_dim = input.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, in_dim, out_dim],
                            dtype=tf.float32,
                            initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
        if func is not None:
            output = func(output)

    return output