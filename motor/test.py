import torch
import gym
import numpy as np
from TD3 import TD3
# from utils import ReplayBuffer
import env
import numpy as np
import pickle
from tqdm import tqdm
env_name = "buffer2e5_rewardV_gamma99"
env = env.env(env_name,train = False)

def readfile(filepath):
        file = open(filepath,'rt')
        x=[]
        line = file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            line = line.replace('\n',"")
            y = line.split("	")
            y = [float(i) for i in y]
            x.append(y[1])
        f=np.array(x)
        return x

def test():
    ######### Hyperparameters #########
    
    log_interval = 1           # print avg reward after interval
    random_seed = 5252
    max_episodes=3
    max_timesteps=9994
    directory = "./preTrained/{}".format(env_name) # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    ###################################
    
    #env = env.env()
    state_dim = env.observation_space
    action_dim = env.action_space
    max_action = 30 #float(env.action_space.high[0])
    
    policy = TD3(state_dim, action_dim, max_action)
    # replay_buffer = ReplayBuffer()
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        print(env_name)
        #env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open("log.txt","w+")
    policy.load(directory, filename+'nf')
    #replay_buffer.loadbuf()
    # training procedure:
    #ac_tmp = readfile('./30sec/360/current.txt')
    currents = []
    for episode in range(1, max_episodes+1):
        state = env.reset(episode)
        #f = open("./TestingData/"+str(episode+10),"wt")
        for t in tqdm(range(max_timesteps)):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action
            action = action.clip(-30, 30)
            currents.append(action.flatten()[0])
            # take action in env:
            for i in range(10):
               
                next_state, reward, done = env.step(action)#(np.array([ac_tmp[t*10+i]]))#(action_list[0])

            #next_state, reward, done = env.step(action)
            
            # replay_buffer.add((state, action, reward, next_state, float(done)))
            #print(str(action),end="")
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            #f.write(str(reward)+"\n")
            
            if done or t==(max_timesteps-1):
                print("updating")       
                #f.close()
                break
        
        # logging updates:
        print(episode)
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0
        with open('./current', 'wb') as f:
            pickle.dump(currents, f)
        # if avg reward > 300 then save and stop traning:
        
        
        
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

if __name__ == '__main__':
    test()
    
