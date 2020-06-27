import numpy as np
import pickle
class ReplayBuffer:
    def __init__(self, directory, max_size=1e5):
        self.directory = directory
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/10)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def savebuf(self):
        with open('{}/buffer3'.format(self.directory),'wb') as config_file:
          pickle.dump(self.buffer, config_file)
        with open('{}/size3'.format(self.directory),'wb') as config_file2:
          pickle.dump(self.size, config_file2)
          
    
    def loadbuf(self):
        with open('{}/buffer3'.format(self.directory),'rb') as config_file:
          self.buffer = pickle.load(config_file)
        with open('{}/size3'.format(self.directory),'rb') as config_file2:
          self.size = pickle.load(config_file2)

    def get_size(self):
        return self.size
          
