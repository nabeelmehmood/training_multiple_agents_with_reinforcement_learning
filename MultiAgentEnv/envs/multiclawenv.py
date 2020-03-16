from MultiAgentEnv.envs.mujocomultienv import MujocoMultiEnv
import random
import numpy as np
import math
from gym import utils

class MultiClawEnv(MujocoMultiEnv, utils.EzPickle): #Child class of MujocoMultiEnv
    
    def __init__(self):
        MujocoMultiEnv.__init__(self, 'arm_claw.xml', 5)
        utils.EzPickle.__init__(self)

    def get_screen(self): #Render environment
        return self.sim.render(512,512)

    def compute_dist(self,achieved_goal, goal): #Calculate distance between goal and object
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

    def compute_reward(self, achieved_goal, goal, info): #Calculate reward based on distance
        temp = self._get_obs() 
        temp_desired = goal
        temp_achieved = achieved_goal
        eu_vector = []
        for i in range(len(temp_desired)):
            x = temp_desired[i]-temp_achieved[i]
            x = x*x
            eu_vector.append(x)
        sqr_sum = sum(eu_vector)
        sqrt = math.sqrt(sqr_sum)
        if (sqrt < 0.15):
            return 0
        else:
            return -1  

    def recompute_action(self, a): #Move both claws at once
        for i in range(0,len(a)):
           if a[i]<0:
              a[i] = -1
           elif a[i]>0:
              a[i] = 1
        return a

    def step(self, a): #Perform action. This part is copied from Mujoco and OpenAI's samples
        xposbefore = self.get_body_com("upperarm")[0]
        sim_state_whole = self.sim.get_state()
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("upperarm")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        obs = self._get_obs()
        achieved = obs['achieved_goal']
        goal = obs['desired_goal']
        info = []
        reward = self.compute_reward(achieved, goal, info)
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)


    def _get_obs(self): #Create state vector
        obs = dict()
        goal = self.get_body_com("goal").tolist() #Get goal position
        achieved_goal = self.get_body_com("object").tolist() #Get object position
        dist = self.compute_dist(goal,achieved_goal)
      
        obs["observation"] = np.concatenate([ #Get arm position
               self.get_body_com("goal"), 
               self.get_body_com("claw"),
               self.get_body_com_vel("claw"),
               self.get_body_com("claw2"),
               self.get_body_com_vel("claw2"),
               self.get_body_com("object"),
            ])
        obs["observation"] = np.append(obs["observation"],dist)
        
        #Use achievedgoal and desiredgoal notation for easy HER implementation
        obs["desired_goal"] = np.concatenate([
            self.get_body_com("goal"),])
        obs["achieved_goal"] = np.concatenate([
            self.get_body_com("object"),])
        return obs

    def reset_model(self):
        qpos = self.init_qpos 
        if self.random_goal: #Randomize goal position
             rand_goal=random.uniform(-0.2,0.2)
             qpos[22] = rand_goal
        if self.random_object: #Randomize object position
             random_object=random.uniform(-0.05,-0.2)
             qpos[15] = random_object
        qvel = self.init_qvel # + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

