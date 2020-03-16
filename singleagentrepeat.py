#######version differer
import csv
import datetime
import numpy as np
import gym
import random
import MultiAgentEnv
import mujoco_py
import math
import tensorflow as tf
import tensorflow.contrib as tc
from collections import deque
env = gym.make('MultiAgentEnv-v0')
action_size = 4
state_size = 21
action_bound = env.action_space.high[:4]
print(action_bound)
batch_size = 128
import random
import matplotlib.pyplot as plt
###################seeding###################
seeding = 1234
np.random.seed(seeding)
tf.set_random_seed(seeding)
env.seed(seeding)

######################################

def mirror_action(action):
    action2 = np.empty(8)
    action2 = np.expand_dims(action2, axis=0)
    j=0
    for i in range(8):
       j = j%4
       action2[0][i] = action[0][j]
       j+=1
    action2[0][4] = (action[0][0]*-1)
    return action2

def cut_action_batch(batch):
    batch2 = np.empty([batch_size,action_size])
    for i in range(batch_size):
        batch2[i] = batch[i][:4]
    return batch2

class actor():
    def __init__(self, state_size, action_size, action_bound, sess, ac_lr = 0.0001, tau = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.sess = sess
        self.lr = ac_lr
        self.batch_size = 128
        self.tau = tau
        #self.input = tf.placeholder(tf.float32, [None, self.state_size], name = "State_actor_input")
        #self.target_input = tf.placeholder(tf.float32, [None, self.state_size], name = "State_target_actor_input")

        with tf.variable_scope('actor_net'):

            self.input_actor, self.out_, self.scaled_out = self.actor_model()

        self.ac_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_net')


        with tf.variable_scope('actor_target_net'):
            self.input_target_actor, self.target_out_, self.target_scaled_out = self.actor_model()

        self.ac_target_pram = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'actor_target_net')
        #print(len(self.ac_params))


        self.update_target_in = [self.ac_target_pram[i].assign ( tf.multiply(self.ac_target_pram[i], 0) + tf.multiply(self.ac_params[i],1) ) for i in range(len(self.ac_target_pram))]
        self.update_target = [self.ac_target_pram[i].assign ( tf.multiply(self.ac_target_pram[i], 1-self.tau) + tf.multiply(self.ac_params[i],self.tau) ) for i in range(len(self.ac_target_pram))]


        self.critic_grad = tf.placeholder(tf.float32,[None, self.action_size], name = 'critic_grad')

        self.actor_grad = tf.gradients(self.scaled_out, self.ac_params, -self.critic_grad)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.actor_grad))

        self.loss = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.ac_params))

    def actor_model(self):
        inputs = tf.placeholder(tf.float32, [None, self.state_size])
        x = tc.layers.layer_norm(inputs, center=True, scale=True, begin_norm_axis=0)
        h1 =  tf.layers.dense(x, 300, activation = tf.nn.relu )
        h1 = tc.layers.layer_norm(h1, center=True, scale=True)
        h2 = tf.layers.dense(h1, 200, activation = tf.nn.relu )
        h2 = tc.layers.layer_norm(h2, center=True, scale=True)
    
        k_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        out = tf.layers.dense(h2, self.action_size, activation = tf.nn.tanh,kernel_initializer = k_init)
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def get_action(self,s):
        return self.sess.run(self.scaled_out, feed_dict = {self.input_actor : s})

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


class critic():
    def __init__(self, state_size, action_size, action_bound, sess, ac_lr = 0.001, tau = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.sess = sess
        self.lr = ac_lr
        self.batch_size = 128
        self.tau = tau


        with tf.variable_scope('critic_net'):
            self.input_critic, self.action_critic, self.value,  = self.build_net()
        self.cr_prams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'critic_net')

        with tf.variable_scope('target_critic_net'):
            self.input_target_critic, self.action_target_critic, self.target_value = self.build_net()
        self.target_cr_prams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_critic_pram')

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
        h1 = tf.layers.dense(x, 300, activation = tf.nn.relu)
        h1 = tc.layers.layer_norm(h1, center=True, scale=True)
        h11 = tf.layers.dense(h1, 200,activation = tf.nn.relu)
        a1 = tf.layers.dense(action, 200)

        h1_ = tf.concat([h11,a1],axis = 1)
        h1_ = tc.layers.layer_norm(h1_, center=True, scale=True)

        h2 = tf.layers.dense(h1_, 200, activation=tf.nn.relu)
        h2 = tc.layers.layer_norm(h2, center=True, scale=True)
        k_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        out_cr = tf.layers.dense(h2, 1,kernel_initializer=k_init)
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



#############This noise code is copied from openai baseline #########OrnsteinUhlenbeckActionNoise############# Openai Code#########

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

#########################################################################################################

def HER(environment, achieved, info):
    return environment.compute_reward(s['achieved_goal'], s['achieved_goal'], info)

def store_sample(s,a,r,d,info ,s2):
    ob_1 = np.reshape(s['observation'],(1,18))
    ac_1 = np.reshape(s['achieved_goal'],(1,3))
    de_1 = np.reshape(s['desired_goal'],(1,3))
    ob_2 = np.reshape(s2['observation'],(1,18))
    ac_2 = np.reshape(s2['achieved_goal'],(1,3))
    de_2 = np.reshape(s2['desired_goal'],(1,3))
    s_1 = np.concatenate([ob_1,ac_1],axis=1)
    s2_1 = np.concatenate([ob_2,ac_1],axis=1)
    s_2 = np.concatenate([ob_1,de_1],axis=1)
    s2_2 = np.concatenate([ob_2,de_1],axis=1)
    substitute_goal = s['achieved_goal'].copy()
    
    substitute_reward = HER(env, s['achieved_goal'],info)
   
    replay_memory.append((s_2,a,r,d,s2_2))
    replay_memory.append((s_1,a,substitute_reward,True,s2_1))

def stg(s):
    #print(len(s))
    ob_1 = np.reshape(s['observation'],(1,18))
    de_1 = np.reshape(s['desired_goal'],(1,3))
    return np.concatenate([ob_1,de_1],axis=1)


def compute_dist(achieved_goal, goal):
      temp_desired = goal
      temp_achieved = achieved_goal
      eu_vector = []
      for i in range(len(temp_desired)):
            x = temp_desired[i]-temp_achieved[i]
            x = x*x
            eu_vector.append(x)
      sqr_sum = sum(eu_vector)
      sqrt = math.sqrt(sqr_sum)
      return sqrt


sess = tf.Session()
ac = actor(state_size, action_size, action_bound, sess)
cr = critic(state_size, action_size, action_bound, sess)
s = env.reset()

noice = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size+action_size))

scores_path = 'model/single_ddpg/scores/score-{date:%Y-%m-%d %H:%M:%S}.csv'.format( date=datetime.datetime.now() )

demo_actions = np.load('demoactions.npy')
demo_actions_counter = 0

demo_pick_actions = np.load('demopick.npy')
demo_pick_actions_counter = 0

demo_pick_flag=False
#model-2019-03-29 03:10:59.ckpt
sess.run(tf.global_variables_initializer())
save_path = 'model/single_ddpg/'

saver = tf.train.Saver(max_to_keep=100)
replay_memory = deque(maxlen = 100000)
max_ep = 50000
max_ep_len = 400
demo_ep_threshold=-1
pick_ep_threshold=9
gamma = 0.99
R_graph = deque(maxlen = 10)
R_graph_= []

#best_r = -(max_ep_len)
best_r = -280
#cr.first_up()
#ac.first_up()
#saver.restore(sess, "model/single_ddpg/model-2019-03-04 17:28:57.ckpt")
for ii in range(max_ep):
    env = env.unwrapped
    env.set_random_goal(False)
    s = env.reset()
    demo_ep = False
    demo_pick_ep = False
    cache_rand = random.randint(0,9)
    if cache_rand <= demo_ep_threshold:
        demo_ep = True
        demo_actions_counter = 0
    elif cache_rand <= pick_ep_threshold:
        demo_pick_ep = True
        demo_pick_actions_counter = 0

    #print(s.shape)
    #s = s[np.newaxis, :]
    R,r = 0,0
    for kk in range(max_ep_len):
        #print('++')
        ss = stg(s)
        if demo_ep and demo_actions_counter < demo_actions.shape[0]:
           a = demo_actions[demo_actions_counter]
           demo_actions_counter+=1
           demo_pick_flag = True
        elif demo_pick_ep and demo_pick_actions_counter < demo_pick_actions.shape[0]:
           a = demo_pick_actions[demo_pick_actions_counter]
           demo_pick_actions_counter+=1
        else:
           a = ac.get_action(ss)
           a = mirror_action(a)
           #print(a)
           a += noice()
           #print(a)
           a=a[0]
           demo_pick_flag=False
        #env.render()
        s2,r,d,info=env.step(a)
        if not demo_pick_flag:
           if kk == 0 and r == 0:
              print("reset")
              s = env.reset()
              break
           #print(s2['observation'][2])
           #if s2['observation'][2] < 0.42 and s2['observation'][2] > 0.416:
              #r = 0
              #print('touching table')
           #if s2['observation'][2] < 0.417:
              #r = -1
              #print('below table')

           #print(s2)
           #s2=s2[np.newaxis, :]
           r_2 = r
           r=r
           store_sample(s,a,r,d,info,s2)
           #replay_memory.append((s,a,r,d,s2))
           s = s2
           R += r_2

    if (R > best_r and ii > 0):
       if compute_dist(s['achieved_goal'],s['desired_goal']) < 0.15:
            time_path = 'model-{date:%Y-%m-%d %H:%M:%S}-reward-'.format(date=datetime.datetime.now())
            reward_path= '{}.cktp'.format(R)
            best_save_path = 'model/single_ddpg_best/'
            saver.save(sess, best_save_path+time_path+reward_path)
            best_r = R

    if batch_size < len(replay_memory):
          minibatch = random.sample(replay_memory, batch_size)
          s_batch, a_batch,r_batch, d_batch, s2_batch = [], [], [], [], []
          for s_, a_, r_, d_, s2_ in minibatch:
                   s_batch.append(s_)
                   s2_batch.append(s2_)
                   a_batch.append(a_)
                   r_batch.append(r_)
                   d_batch.append(d_)
          s_batch = np.squeeze(np.array(s_batch),axis=1)
          s2_batch = np.squeeze(np.array(s2_batch),axis=1)
          r_batch=np.reshape(np.array(r_batch),(len(r_batch),1))
          a_batch=np.array(a_batch)
          d_batch=np.reshape(np.array(d_batch)+0,(128,1))
          #print(d_batch)
          a2 = ac.get_action_target(s2_batch)
          #print(a2.shape)
          v2 = cr.get_val_target(s2_batch,a2)
          #print(v2.shape)
          #for
          tar= np.zeros((128,1))
          for o in range(128):
            tar[o] = r_batch[o] + gamma * v2[o]
          #print(tar.shape)
          a_batch = cut_action_batch(a_batch)

          cr.train_critic(s_batch,a_batch,tar)

          a_out = ac.get_action(s_batch)
          kk = cr.get_grad(s_batch,a_out)[0]
          #print(kk)
          ac.train_actor(s_batch, kk)
          cr.update_critic_target_net()
          ac.update_target_tar()
          #exit()
    
    R_graph.append(R)
    R_graph_.append(R)

    #print(ii, R)
    #if ii % 20 ==0 :
        #time_path = 'model-{date:%Y-%m-%d %H:%M:%S}.ckpt'.format( date=datetime.datetime.now() )
        #saver.save(sess, save_path+ time_path)
    print(ii, R, np.mean(np.array(R_graph)), np.max(np.array(R_graph)), compute_dist(s['achieved_goal'],s['desired_goal']))
    if ii%5 == 0:
        with open(scores_path, mode='a+') as score_file:
           score_writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
           score_writer.writerow([ii, np.mean(np.array(R_graph))])
    if ii%1000 == 0:
        sess.run(tf.global_variables_initializer())

    if (ii+1) % 100:
        plt.plot(np.array(R_graph_))
