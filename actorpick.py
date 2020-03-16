import numpy as np
import random
import math
import tensorflow as tf
import tensorflow.contrib as tc

class actor():
    def __init__(self, state_size, action_size, action_bound, sess, ac_lr = 0.000005, tau = 0.001, ini=False):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.lr = ac_lr
        self.batch_size = 128
        self.tau = tau
        self.lam=0.1
        self.ini=ini
        if ini: 
            self.sess = sess
            with tf.variable_scope('actor_pick_net'):
               
               self.input_actor, self.out_, self.scaled_out = self.actor_model()

            self.ac_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_pick_net')
            with tf.variable_scope('actor_pick_target_net'):
               self.input_target_actor, self.target_out_, self.target_scaled_out = self.actor_model()

            self.ac_target_pram = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'actor_target_net')
            #print(len(self.ac_params))


            self.update_target_in = [self.ac_target_pram[i].assign ( tf.multiply(self.ac_target_pram[i], 0) + tf.multiply(self.ac_params[i],1) ) for i in range(len(self.ac_target_pram))]
            self.update_target = [self.ac_target_pram[i].assign ( tf.multiply(self.ac_target_pram[i], 1-self.tau) + tf.multiply(self.ac_params[i],self.tau) ) for i in range(len(self.ac_target_pram))]

            self.critic_grad = tf.placeholder(tf.float32,[None, self.action_size], name = 'critic_grad')

            self.actor_grad = tf.gradients(self.scaled_out, self.ac_params, -self.critic_grad)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.actor_grad))

            self.loss = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.ac_params))
        self.tauy = np.load('lam.npy')
        self.updateac = 0   
        

    def expand_action(self,action):
        action2 = np.empty(8)
        action2 = np.expand_dims(action2, axis=0)
        j=0
        for i in range(8):
           j = j%4
           action2[0][i] = action[0][j]
           j+=1
        action2[0][4] = (action[0][0]*-1)
        return action2

    def actor_model(self):
        inputs = tf.placeholder(tf.float32, [None, self.state_size])
        x = tc.layers.layer_norm(inputs, center=True, scale=True, begin_norm_axis=0)
        h1 =  tf.layers.dense(x, 400, activation = tf.nn.relu )
        h1 = tc.layers.layer_norm(h1, center=True, scale=True)
        h2 = tf.layers.dense(h1, 300, activation = tf.nn.relu )
        h2 = tc.layers.layer_norm(h2, center=True, scale=True)
        h3 = tf.layers.dense(h2, 200, activation = tf.nn.relu )
        h3 = tc.layers.layer_norm(h3, center=True, scale=True)
        h4 = tf.layers.dense(h3, 100, activation = tf.nn.relu )
        h4 = tc.layers.layer_norm(h4, center=True, scale=True)
        k_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        out = tf.layers.dense(h4, self.action_size, activation = tf.nn.tanh,kernel_initializer = k_init)
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out
  
    def optimize(self,a):
       z = []
       a=a.astype(float)
       for i in range(8):
          x = random.uniform(-1, 1)
          z.append(x)
       for i in range(8):
          b = float((z[i]*self.lam))
          c = float((a[i]*(1-self.lam)))
          a[i] = b+c
       return a 

    def get_action(self,s):
        return self.expand_action(self.sess.run(self.scaled_out, feed_dict = {self.input_actor : s}))

    def update_target_tar(self):
        #print('---------------')
        self.sess.run(self.update_target)
        #return True
    def get_action_target(self,s):
        return self.sess.run(self.target_scaled_out, feed_dict = {self.input_target_actor : s})

    def train_actor(self,s,grad):
        self.sess.run(self.loss, feed_dict = {self.input_actor : s, self.critic_grad : grad})

    def first_up(self):
        self.sess.run(self.update_target_in)

    def get_actions_(self,s):
       
        if self.updateac > -1 and self.updateac<120:
            ac = self.optimize(self.tauy[self.updateac])
            self.updateac= self.updateac+1
            return ac
        else:
            self.lam = self.lam*0.998
            self.updateac=-1
            if self.ini:
                return self.expand_action(self.sess.run(self.scaled_out, feed_dict = {self.input_actor : s}))[0]
            else:
                return [0,0,0,0,0,0,0,0]

    def get_actions(self,s):
        return self.sess.run(self.scaled_out, feed_dict = {self.input_actor : s})
