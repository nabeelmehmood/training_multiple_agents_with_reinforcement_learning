import os
import mujoco_py as mj
import gym
import MultiAgentEnv 

env = gym.make("MultiAgentEnv-v0")
obs = env.reset()
done = False
act = []

a  = [1,0,1,1,1,0,1,1]
act.append(a)
a  = [1,0,1,1,1,0,1,1]
act.append(a)
a  = [1,0,1,1,1,0,1,1]
act.append(a)
a  = [1,0,1,1,1,0,1,1]
act.append(a)
a  = [-1,0,1,1,1,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,1,0,0,1,1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [0,0,1,-1,0,0,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,1,1,-1,-1,1,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
a  = [0.5,0,1,-1,-0.5,0,1,-1]
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
a  = [0.5,0,1,-1,-0.5,0,1,-1]
act.append(a)
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [1,0,1,-1,-1,0,1,-1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
a  = [0,1,-1,1,0,1,-1,1]
act.append(a)
i = 0

import numpy as np

act2 = np.load('demoactions.npy')
print(len(act2))
x = input()
act2 = act[:120]
while not done:
   #print(env.action_space.sample())
  
   if i >= len(act2):
      #a = act[len(act)-1]
      a=[0,0,0,0,0,0,0,0]
   else:
      a = act[i]
   obs,reward,done,info=env.step(a)
   env.render()
   if reward>-2:
      print(reward)
   i+=1
   if done:
       env.reset()
       i = 0
       done=False
