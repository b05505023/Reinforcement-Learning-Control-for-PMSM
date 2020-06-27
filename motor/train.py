import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
import env
import numpy as np
import math
import os
env_name = "buffer1e5_rewardOrigin_gamma99_3" #### 10 current #####
env = env.env(env_name)

def readfile(filepath):
        file = open(filepath,'rt')
        x=[]
        line = file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            line = line.replace('\n',"")
            y = line.split("    ")
            y = [float(i) for i in y]
            x.append(y[1])
        f=np.array(x)
        return x
# TD3 algorithm
def train():
    ######### Hyperparameters #########
    log_interval = 1           # print avg reward after interval
    random_seed = 8787
    gamma = 0.99                # discount for future rewards
    batch_size = 1500            # num of transitions sampled from replay buffer
    lr = 1e-4
    exploration_noise = 2
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 2          # target policy smoothing noise
    noise_clip = 2.5
    policy_delay = 20            # delayed policy updates parameter
    max_episodes = 1000         # max num of episodes
    max_timesteps = 2000        # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name) # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    begin = True                #Train a new model from begin or not.
    steps = 0
    start_steps = 30000
    start_epoch = 1
    if not os.path.isdir(directory):
        os.makedirs(directory)
    ###################################
    
    # env = env.env(env_name)
    state_dim = env.observation_space
    action_dim = env.action_space
    max_action = 30 #float(env.action_space.high[0])
    
    policy = TD3( state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(directory)
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        print(env_name)
        #env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open("./preTrained/{}/log.txt".format(env_name),"a")
    if begin == False:

        start_epoch = policy.load(directory, filename+'nf')
        replay_buffer.loadbuf()
        print("Restore training from episode {}".format(start_epoch))
        log_f = open("./preTrained/{}/log.txt".format(env_name),"a")
    else:
        log_f = open("./preTrained/{}/log.txt".format(env_name),"w")
    # training procedure:
    # ac_tmp = readfile('./30sec/360/current.txt')
    
    for episode in range(start_epoch, max_episodes+1):
        state = env.reset(episode)
        #f = open("./TestingData/"+str(episode+10),"wt")
        for t in range(max_timesteps):
            # select action and add exploration noise:
            if(((episode-1)*max_timesteps)+t < start_steps):
                action = np.array([np.random.uniform(-30,30)])
            else:
                EPS_DECAY = 0.99**episode
                noise = max(exploration_noise*EPS_DECAY,1.5)
                action = policy.select_action(state)
                action = action + np.random.normal(0, noise, size = None)
                action = action.clip(-30, 30)
            r_tmp = 0
            for i in range(10):
                
                next_state, reward, done = env.step(action)#(np.array([ac_tmp[t*10+i]]))#(action_list[0])
                r_tmp += reward

            

            replay_buffer.add((state, action, reward, next_state, float(done)))
            
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            #f.write(str(reward)+"\n")
            

            if replay_buffer.get_size() > 1000 and t % 1000 == 0 and episode > 1 :
                policy.update(replay_buffer, 100, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,t)

            # if episode == 1:
            #   if t>1000 and t%1000 == 0:
            #     policy.update(replay_buffer, 100, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,t)
            #   elif t == 1000:
            #     policy.update(replay_buffer, 100, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,t)
            # else:
            #   if t%1000 == 0:
            #     policy.update(replay_buffer, 100, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,t)
            # if episode is done then update policy:
            steps+=1

            if done or t==(max_timesteps-1):
                print("updating")
                replay_buffer.savebuf()
                policy.save(directory, filename+"nf", episode)            
                print('end save')
                #f.close()
                break
        
        # logging updates:
        print(episode)
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0
        
        # if avg reward > 300 then save and stop traning:
        
        
        
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

if __name__ == '__main__':
    train()
    
