import os
import mujoco_py as mj
import gym
import MultiAgentEnv 

env = gym.make("MultiAgentEnv-v0")
obs = env.reset()
done = False
while not done:
   a  = [0,0,1,-1,0,1,0,-1]
   #print(env.action_space.sample())
   obs,reward,done,info=env.step(env.action_space.sample())
   env.render()
   print(obs)
   if done:
       env.reset()
       done=False
