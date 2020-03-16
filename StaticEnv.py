import os
import mujoco_py as mj
import gym
import MultiAgentEnv 

env = gym.make("ReachEnv-v0")
obs = env.reset()
done = False
while not done:
   a  = [1,1,1]
   obs,reward,done,info=env.step(a)
   print(obs)
   env.render()
   if done:
       done=False
