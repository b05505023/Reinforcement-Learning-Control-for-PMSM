'''
@Author : wilber
IF you want to rnu the code please change the directory in 'def reset' and 'def step' function
'''

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
class env:

    def __init__(self,env_name,train = True):
        #
        self.observation_space = 5+5
        self.action_space = 1
        self.length = 60
        self.count = 0
        self.train = train
        if train:
            self.runfor = 20000
        else:
            self.runfor = 99940
        self.position_reference = []
        self.last_current = []
        self.velocity = []
        self.last_true_position = []
        self.last_true_spin = []
        self.allposition = []
        self.episode = 0
        self.ran = 0
        self.drawcurrent = []
        self.amplitude = 0
        self.env_name = env_name
        if not os.path.isdir("./pic/{}/route".format(env_name)):
           os.makedirs("./pic/{}/route".format(env_name))
        if not os.path.isdir("./pic/{}/current".format(env_name)):
           os.makedirs("./pic/{}/current".format(env_name))


    def readfile(self,filepath):
        file = open(filepath,'rt')
        x=[]
        line = file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            line = line.replace('\n',"")
            y = line.split("\t")
            y = [float(i) for i in y]
            x.append(y[1])
        f=np.array(x)
        return x
    
    def True_position_step(self,current,last_current,last_true_spin,last_true_position):
        kt = 0.49245
        tor=current*kt
        ltor=last_current*kt
        load = 4.2
        if self.count >0:
          if current > 0 and tor > load:
            tor = tor - load
          elif current < 0 and tor< -load:
            tor = tor + load
          else:
            tor = tor*0
          if last_current > 0 and ltor > load:
            ltor = ltor - load
          elif last_current < 0 and ltor< -load:
            ltor = ltor + load
          else:
            ltor = ltor*0
        #print(type(tor))
        '''else:
          if current > 0:
            tor = tor - 20.4
          elif current < 0:
            tor = tor + 20.4
          if last_current > 0:
            ltor = ltor - 20.4
          elif last_current < 0:
            ltor = ltor + 20.4'''
        
        true_spin = (0.9998699272*last_true_spin+(1/246.016)*(tor)+(1/246.016)*ltor) 
        #else:
        #  true_spin = (0.9998699272*last_true_spin+(1/246.016)*(current*kt+4.2)+(1/246.016)*last_current*kt)           
        true_position = last_true_position+0.00005*true_spin+0.00005*last_true_spin
        
        return true_position,true_spin,current,(true_position - last_true_position) #new state

    def reward_count(self,true_position,want_position):

        ## 變動gaussian
        #x = (true_position*720/(4*3.14159)-want_position)**2
        #devi = max(1000*(0.995**self.episode),200)
        #reward = 50*math.exp(-x/devi)

        ## 不動gaussian
        #x = (true_position*720/(4*3.14159)-want_position)**2
        #reward = 50*math.exp(-x/500)

        ## 尖尖那個
        #x = abs(true_position*720/(4*3.14159)-want_position)
        #reward = 50*math.exp(-x/10)

        ## 最原始那個
        reward = -abs(true_position*720/(4*3.14159)-want_position)
        
        # clip origin
        #reward = -abs(true_position*720/(4*3.14159)-want_position)+50
        #if reward < 0:
        #  reward = 0

        return reward,-abs(true_position*720/(4*3.14159)-want_position)

    def reset(self,episode):
        #random starting point
        if self.train: 
          if episode%4 == 0:
            self.ran = 0
          elif episode%4 == 1:
            self.ran = 10000
          elif episode%4 == 2:
            self.ran = 20000
          elif episode%4 == 3:
            self.ran = 30000
            #self.ran = random.randint(32,79999)
        else:
            self.ran = 0
        #self.ran = 0
        print("self.ran: "+str(self.ran))
        



        #different training environment
        if not self.train:
            if episode%3 == 0:
              self.position_reference = self.readfile('./30sec/360/position_reference.txt')
              self.amplitude = 360
            elif episode%3 == 1:
              self.position_reference = self.readfile('./30sec/T/position_reference.txt')
              self.amplitude = 360
            elif episode%3 == 2:
              self.position_reference = self.readfile('./30sec/V/position_reference.txt')
              self.amplitude = 360

            # if episode%13==1:
            #     self.position_reference = self.readfile('../../data_all/trap_72/position_reference.txt')
            #     self.amplitude = 72
            # elif episode%13==2:
            #     self.position_reference = self.readfile('../../data_all/trap_144/position_reference.txt')
            #     self.amplitude = 144
            # elif episode%13==3:
            #     self.position_reference = self.readfile('../../data_all/trap_288/position_reference.txt')
            #     self.amplitude = 288
            # elif episode%13==4:
            #     self.position_reference = self.readfile('../../data_all/trap_360/position_reference.txt')
            #     self.amplitude = 360
            # elif episode%13==5:
            #     self.position_reference = self.readfile('../../data_all/tri_72/position_reference.txt')
            #     self.amplitude = 72
            # elif episode%13==6:
            #     self.position_reference = self.readfile('../../data_all/tri_144/position_reference.txt')
            #     self.amplitude = 144
            # elif episode%13==7:
            #     self.position_reference = self.readfile('../../data_all/tri_288/position_reference.txt')
            #     self.amplitude = 288
            # elif episode%13==8:
            #     self.position_reference = self.readfile('../../data_all/tri_360/position_reference.txt')
            #     self.amplitude = 360
            # elif episode%13==9:
            #   self.position_reference = self.readfile('../../data_all/sin_72/position_reference.txt')
            #   self.amplitude = 72
            # elif episode%13==10:
            #   self.position_reference = self.readfile('../../data_all/sin_144/position_reference.txt')
            #   self.amplitude = 144
            # elif episode%13==11:
            #   self.position_reference = self.readfile('../../data_all/sin_216/position_reference.txt')
            #   self.amplitude = 216
            # elif episode%13==12:
            #   self.position_reference = self.readfile('../../data_all/sin_288/position_reference.txt')
            #   self.amplitude = 288
            # elif episode%13==0:
            #   self.position_reference = self.readfile('../../data_all/sin_360/position_reference.txt')
            #   self.amplitude = 360

        else:
            #different training environment
            if episode%5==1:
                self.position_reference = self.readfile('./30sec/72/position_reference.txt')
                self.amplitude = 72
            elif episode%5==2:
                self.position_reference = self.readfile('./30sec/144/position_reference.txt')
                self.amplitude = 144
            elif episode%5==3:
                self.position_reference = self.readfile('./30sec/216/position_reference.txt')
                self.amplitude = 288
            elif episode%5==4:
                self.position_reference = self.readfile('./30sec/288/position_reference.txt')
                self.amplitude = 288
            elif episode%5==0:
                self.position_reference = self.readfile('./30sec/360/position_reference.txt')
            #    self.amplitude = 360
            # if episode%13==1:
            #     self.position_reference = self.readfile('../../data_all/trap_72/position_reference.txt')
            #     self.amplitude = 72
            # elif episode%13==2:
            #     self.position_reference = self.readfile('../../data_all/trap_144/position_reference.txt')
            #     self.amplitude = 144
            # elif episode%13==3:
            #     self.position_reference = self.readfile('../../data_all/trap_288/position_reference.txt')
            #     self.amplitude = 288
            # elif episode%13==4:
            #     self.position_reference = self.readfile('../../data_all/trap_360/position_reference.txt')
            #     self.amplitude = 360
            # elif episode%13==5:
            #     self.position_reference = self.readfile('../../data_all/tri_72/position_reference.txt')
            #     self.amplitude = 72
            # elif episode%13==6:
            #     self.position_reference = self.readfile('../../data_all/tri_144/position_reference.txt')
            #     self.amplitude = 144
            # elif episode%13==7:
            #     self.position_reference = self.readfile('../../data_all/tri_288/position_reference.txt')
            #     self.amplitude = 288
            # elif episode%13==8:
            #     self.position_reference = self.readfile('../../data_all/tri_360/position_reference.txt')
            #     self.amplitude = 360
            # elif episode%13==9:
            #     self.position_reference = self.readfile('../../data_all/sin_72/position_reference.txt')
            #     self.amplitude = 72
            # elif episode%13==10:
            #     self.position_reference = self.readfile('../../data_all/sin_144/position_reference.txt')
            #     self.amplitude = 144
            # elif episode%13==11:
            #     self.position_reference = self.readfile('../../data_all/sin_216/position_reference.txt')
            #     self.amplitude = 216
            # elif episode%13==12:
            #     self.position_reference = self.readfile('../../data_all/sin_288/position_reference.txt')
            #     self.amplitude = 288
            # elif episode%13==0:
            #     self.position_reference = self.readfile('../../data_all/sin_360/position_reference.txt')
            #     self.amplitude = 360

        
       
        '''
        elif episode%10==7:
          self.position_reference = self.readfile('./TestingData/2.txt')
        elif episode%10==8:
          self.position_reference = self.readfile('./TestingData/3.txt')
        elif episode%10==9:
          self.position_reference = self.readfile('./TestingData/5.txt')
        '''
        self.last_current = []
        self.velocity = []
        self.last_true_position = []
        self.last_true_spin = []
        self.count = 0
        self.allposition = []
        self.episode = episode
        self.toteval = 0
        new_state = []
        self.allposition.append(0)
        for i in range(self.length):
            self.last_current.append(0)
            self.velocity.append(0)
            self.last_true_position.append(self.position_reference[self.ran+self.count]*(4*3.14159)/720)
            self.last_true_spin.append(0)
        for i in range(5):
            
            ############換input的話這邊要改###############
            # new_state.append(self.last_current[i])
            new_state.append(self.velocity[i])
            # new_state.append(0)
        for i in range(5):
            new_state.append(self.position_reference[self.ran + self.count + 10*(i+1) ]  - self.last_true_position[-1]*720/(4*3.14159))
        new_state = np.array(new_state)
        return new_state

    def step(self,action):
        #drawcurrent is just for drawing current
        self.drawcurrent.append(action)

        # exit()
        true_position,true_spin,current,vel = self.True_position_step(action,self.last_current[-1],self.last_true_spin[-1],self.last_true_position[-1])

        self.last_true_position = self.last_true_position[1:]
        self.last_true_position.append(true_position.tolist()[0])

        self.last_current = self.last_current[1:]
        self.last_current.append(current.tolist()[0])

        self.last_true_spin = self.last_true_spin[1:]
        self.last_true_spin.append(true_spin.tolist()[0])

        self.velocity = self.velocity[1:]
        self.velocity.append(vel.tolist()[0]*720/(4*3.14159))
        
        self.allposition.append(true_position.tolist()[0])

        
        want_position = self.position_reference[self.ran+self.count+1]
        self.count += 1
        done = False
        

        reward,evaluate = self.reward_count(true_position.tolist()[0],want_position)
        self.toteval += evaluate

        if self.count % self.runfor==0:
            print("working on nf"+str(self.count))

        #plot my route figure
        if self.train and (self.count % (self.runfor/10) == 0):
            plt.figure(figsize=(20,8))
            po=[]
            for i in range(self.ran):
              po.append(0)
            for i in range(len(self.allposition)):
              po.append(self.allposition[i]*720/(4*3.14159))
            plt.plot(po)            
            plt.plot(self.position_reference)
            plt.savefig("./pic/{}/route/".format(self.env_name)+str(self.episode)+'_360.png')
            
            plt.close()
            print("save fig 360 test"+str(self.episode))
            

        # # plot my route figure
        if self.train and (self.count  == self.runfor):
            #plt.figure(figsize=(20,8))
            plt.figure(figsize=(40,10))
            po=[]
            for i in range(self.ran):
              po.append(0)
            for i in range(len(self.allposition)):
              po.append(self.allposition[i]*720/(4*3.14159))
            plt.plot(po)            
            plt.plot(self.position_reference[:100000])
            print(len(self.position_reference))
            plt.savefig("./pic/{}/route/".format(self.env_name)+str(self.episode)+'_360.png')
            plt.clf() # 清图。
            plt.cla() # 清坐标轴。
            plt.plot(self.drawcurrent)
            plt.savefig("./pic/{}/current/".format(self.env_name)+str(self.episode)+'_360.png')
            self.drawcurrent = [] 
            plt.close()
            done = True
            print("eval")
            print(self.toteval)
            print("runfor")
            print(self.runfor)
        

        if (not self.train) and (self.count  == self.runfor):
            #plt.figure(figsize=(20,8))
            plt.figure(figsize=(40,10))
            po=[]
            for i in range(self.ran):
              po.append(0)
            for i in range(len(self.allposition)):
              po.append(self.allposition[i]*720/(4*3.14159))
            plt.plot(po)            
            plt.plot(self.position_reference[:100000])
            plt.title("position")
            plt.savefig("./preTrained/{}/route_".format(self.env_name)+str(self.episode)+"+_360.png")
            plt.clf() # 清图。
            plt.cla() # 清坐标轴。
            plt.plot(self.drawcurrent)
            plt.title("current")
            plt.savefig("./preTrained/{}/current_".format(self.env_name)+str(self.episode)+"+_360.png")
            self.drawcurrent = [] 
            plt.close()
            done = True
            print("eval")
            print(self.toteval)
            print("runfor")
            print(self.runfor)
        

        new_state=[]

        # making new state for next action to input
        for i in range(5):
            
            #new_state.append(self.position_reference[self.ran + self.count + i + 1]  - self.last_true_position[-1]*720/(4*3.14159))
            # new_state.append(self.last_current[i])
            new_state.append(self.velocity[11+i*10])
            '''if i < 12:
                new_state.append(self.velocity[11])
            elif i <22:
                new_state.append(self.velocity[21])
            elif i<32:
                new_state.append(self.velocity[31])'''
            # if  self.count - self.length + i + 1 >= 0:
            #   new_state.append(self.position_reference[self.ran + self.count - self.length + i + 1]  - self.last_true_position[i]*720/(4*3.14159))
            # else :
            #    new_state.append(0)
            # new_state.append(self.position_reference[self.ran + self.count + 1 + i] - self.last_true_position[-1]*720/(4*3.14159))
        # print(self.position_reference[self.ran + self.count + 1]  - self.last_true_position[-1]*720/(4*3.14159))
        for i in range(5):
            new_state.append(self.position_reference[self.ran + self.count + 10*(i+1) ]  - self.last_true_position[-1]*720/(4*3.14159))#+10
    
        new =  np.array(new_state)
       
        
        return new, reward, done

        
