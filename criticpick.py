import numpy as np
import random
import math
import tensorflow as tf
import tensorflow.contrib as tc

class critic():
    def __init__(self, state_size, action_size, action_bound, sess, ac_lr = 0.00001, tau = 0.001, ini=False):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        
        self.lr = ac_lr
        self.batch_size = 128
        self.tau = tau

        if ini:
            self.sess = sess
            with tf.variable_scope('critic_pick_net'):
                self.input_critic, self.action_critic, self.value,  = self.build_net()
            self.cr_prams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'critic_pick_net')

            with tf.variable_scope('target_critic_pick_net'):
                self.input_target_critic, self.action_target_critic, self.target_value = self.build_net()
            self.target_cr_prams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_critic_pick_pram')

            self.update_critic_target_in = [self.target_cr_prams[i].assign ( tf.multiply(self.target_cr_prams[i], 0) + tf.multiply(self.cr_prams[i],1) ) for i in range(len(self.target_cr_prams))]

            self.update_critic_target = [self.target_cr_prams[i].assign ( tf.multiply(self.target_cr_prams[i], 1 - self.tau) + tf.multiply(self.cr_prams[i], self.tau) ) for i in range(len(self.target_cr_prams))]

            self.pred = tf.placeholder(tf.float32, [None, 1], name= 'pred_value')
            self.loss = tf.reduce_mean(tf.square(self.pred - self.value))
            self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.comment_grad = tf.gradients(self.value, self.action_critic)


    def build_net(self):
        inputs = tf.placeholder(tf.float32, [None, self.state_size])
        x = tc.layers.layer_norm(inputs, center=True, scale=True, begin_norm_axis=0)

        action = tf.placeholder(tf.float32, [None, self.action_size])
        h1 = tf.layers.dense(x, 400, activation = tf.nn.relu)
        h1 = tc.layers.layer_norm(h1, center=True, scale=True)
        h11 = tf.layers.dense(h1, 300,activation = tf.nn.relu)
        a1 = tf.layers.dense(action, 300)

        h1_ = tf.concat([h11,a1],axis = 1)
        h1_ = tc.layers.layer_norm(h1_, center=True, scale=True)

        h2 = tf.layers.dense(h1_, 200, activation=tf.nn.relu)
        h2 = tc.layers.layer_norm(h2, center=True, scale=True)
        h3 = tf.layers.dense(h2, 100, activation=tf.nn.relu)
        h3 = tc.layers.layer_norm(h3, center=True, scale=True)
        k_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        out_cr = tf.layers.dense(h3, 1,kernel_initializer=k_init)
        return inputs, action, out_cr

    def get_val(self,s,a):
        return self.sess.run(self.value,feed_dict={self.input_critic : s, self.action_critic : a})

    def update_critic_target_net(self):
        #print('------------++')
        self.sess.run(self.update_critic_target)

    def train_critic(self,s,a,tar):
        self.sess.run(self.optimize, feed_dict = {self.input_critic : s , self.action_critic : a, self.pred : tar})

    def get_val_target(self,s,a):
        return self.sess.run(self.target_value, feed_dict = {self.input_target_critic : s, self.action_target_critic: a})

    def get_grad(self,s,a):
        return self.sess.run(self.comment_grad, feed_dict = {self.input_critic : s, self.action_critic: a})

    def first_up(self):
        self.sess.run(self.update_critic_target_in)

