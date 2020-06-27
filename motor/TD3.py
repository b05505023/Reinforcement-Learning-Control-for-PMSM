import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)
  
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=10,
        num_layers=1, bidirectional=True, batch_first=True) # , dropout=rnn_dropout
        self.rnn2 = nn.GRU(input_size=1, hidden_size=10,
        num_layers=1, bidirectional=True, batch_first=True) # , dropout=rnn_dropout

        self.l1 = nn.Linear(40, 20)
        self.l2 = nn.Linear(20, action_dim)
        #self.l3 = nn.Linear(16, action_dim)
        
        self.max_action = max_action
        
        
        
    
    
    def forward(self, state):
        #print(state.size())
        out,_ = self.rnn(state[:,:5].unsqueeze(2))
        cat = torch.cat((out[:,-1,:10],out[:,0,10:]),dim = 1)
        out2,_ = self.rnn2(state[:,5:].unsqueeze(2))
        cat2 = torch.cat((out2[:,-1,:10],out2[:,0,10:]),dim = 1)
        
        state = torch.cat((cat,cat2),dim = -1)
        #print(state.size())
        a = selu(self.l1(state))
        
        a = self.l2(a)
        
        a = torch.clamp(a, min=-self.max_action, max=self.max_action)
        #print(a)
        return a
        
class Critic(nn.Module):

  
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim  + action_dim, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 1)
       
        
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        
        q = selu(self.l1(state_action))
        q = selu(self.l2(q))
        q = self.l3(q)
        return q
    
class TD3:
    def init_normal(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            #nn.init.normal_(m.weight)
            
            
    def __init__(self, state_dim, action_dim, max_action):
        torch.manual_seed(52)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=0.0001)
        
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.RMSprop(self.critic_1.parameters(), lr=0.0001)
        
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.RMSprop(self.critic_2.parameters(), lr=0.0001)
        
        self.max_action = max_action
        self.actor.apply(self.init_normal)
        self.actor_target.apply(self.init_normal)
        self.critic_1.apply(self.init_normal)
        self.critic_1_target.apply(self.init_normal)
        self.critic_2.apply(self.init_normal)
        self.critic_2_target.apply(self.init_normal)
        

    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,t):
        TF=True
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)            
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            

            if t % 10000 == 0 and TF == True:
              print(target_Q.mean())
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            loss_mse = torch.nn.MSELoss()
            if t % 10000 == 0 and TF == True:
              print('t_Q')
              print(target_Q.mean())
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            #if i % 10000 == 0:
            #  print('c_Q')
            #  print(current_Q1)
            loss_Q1 = loss_mse(current_Q1, target_Q)

            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)            
            loss_Q2 = loss_mse(current_Q2, target_Q)
            if t % 10000 == 0 and TF == True:
              '''for param in self.actor.parameters():
                print(param.data)'''
                
              print(current_Q1.mean())
              print('Q1'+str(loss_Q1))
              print('Q2'+str(loss_Q2))
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                if t % (10000) ==0 and TF == True:
                  #print(state)
                  #print(self.actor(state)*50)
                  print(actor_loss)
                  TF = False
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
                
    def save(self, directory, name, epoch):
        obj = {'actor':self.actor.state_dict(),
               'actor_target':self.actor_target.state_dict(),
               'critic_1':self.critic_1.state_dict(),
               'critic_1_target':self.critic_1_target.state_dict(),
               'critic_2':self.critic_2.state_dict(),
               'critic_2_target':self.critic_2_target.state_dict(),
               'critic_1_optimizer':self.critic_1_optimizer.state_dict(),
               'critic_2_optimizer':self.critic_2_optimizer.state_dict(),
               'actor_optimizer':self.actor_optimizer.state_dict(),
               'epoch':epoch+1}
        torch.save(obj,'%s/%s.pth'%(directory,name))

        # torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        # torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        # torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        # torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        # torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        # torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
        # torch.save(self.critic_1_optimizer.state_dict(),'%s/%s_crtic_optimizer_1.pth' % (directory, name))
        # torch.save(self.critic_2_optimizer.state_dict(),'%s/%s_crtic_optimizer_2.pth' % (directory, name))
        # torch.save(self.actor_optimizer.state_dict(),'%s/%s_actor_optimizer.pth' % (directory, name))
        
    def load(self, directory, name):

        ckpt = torch.load('%s/%s.pth'%(directory, name),  map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])

        self.critic_1.load_state_dict(ckpt['critic_1'])
        self.critic_1_target.load_state_dict(ckpt['critic_1_target'])

        self.critic_2.load_state_dict(ckpt['critic_2'])
        self.critic_2_target.load_state_dict(ckpt['critic_2_target'])

        self.critic_1_optimizer.load_state_dict(ckpt['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(ckpt['critic_2_optimizer'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.actor.train()
        self.actor_target.train()
        self.critic_1.eval()
        self.critic_1_target.eval()
        self.critic_2.eval()
        self.critic_2_target.eval()

        return ckpt['epoch']


        # self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        # self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        # self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        # self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        # self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        # self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        # self.critic_1_optimizer.load_state_dict(torch.load('%s/%s_crtic_optimizer_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        # self.critic_2_optimizer.load_state_dict(torch.load('%s/%s_crtic_optimizer_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        # self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, name), map_location=lambda storage, loc: storage))
        # self.actor.eval()
        # self.actor_target.eval()
        # self.critic_1.eval()
        # self.critic_1_target.eval()
        # self.critic_2.eval()
        # self.critic_2_target.eval()
            
        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
        
      
        
    def load_critic(self, directory, name):

        ckpt = torch.load('%s/%s.pth'%(directory, name),  map_location=lambda storage, loc: storage)
        

        self.critic_1.load_state_dict(ckpt['critic_1'])
        self.critic_1_target.load_state_dict(ckpt['critic_1_target'])

        self.critic_2.load_state_dict(ckpt['critic_2'])
        self.critic_2_target.load_state_dict(ckpt['critic_2_target'])

        self.critic_1_optimizer.load_state_dict(ckpt['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(ckpt['critic_2_optimizer'])
        
        self.critic_1.eval()
        self.critic_1_target.eval()
        self.critic_2.eval()
        self.critic_2_target.eval()

        return ckpt['epoch']
