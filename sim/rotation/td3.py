import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, max_torque):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.dir_head = nn.Linear(300, 2)  # unit‐vector force direction
        self.mag_head = nn.Linear(300, 1)  # force magnitude in [0,1]
        self.action_dim = action_dim
        self.tau_head = nn.Linear(300, 1) if action_dim >= 3 else None  # torque only when 3D action
        self.max_action = max_action
        self.max_torque = max_torque
        
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        # Force direction (unit 2D vector)
        direction = F.normalize(torch.tanh(self.dir_head(x)), dim=1, eps=1e-6)

        # Force magnitude [0,1]
        magnitude = torch.sigmoid(self.mag_head(x))

        # Scale to actual force [Fx, Fy]
        force = direction * magnitude
        if self.action_dim >= 3:
            tau = torch.tanh(self.tau_head(x))
            return torch.cat([force, tau], dim=1)
        return force
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class ReplayBuffer:
    def __init__(self, max_size=5e5, rng = None):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def add(self, transition):
        self.size += 1
        # transition is a tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # Delete 1/5th of the buffer when full.
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = self.rng.integers(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.asarray(s))
            action.append(np.asarray(a))
            reward.append(np.asarray(r))
            next_state.append(np.asarray(s_))
            done.append(np.asarray(d))
        
        return np.asarray(state), np.asarray(action), np.asarray(reward), np.asarray(next_state), np.asarray(done)
    

class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action, max_torque, torch_rng=None, device=None):
        
        # Resolve and store device (supports MPS, CUDA, or CPU)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.torch_rng = torch_rng or torch.Generator(device=self.device).manual_seed(0)
        self.actor = Actor(state_dim, action_dim, max_action, max_torque).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, max_torque).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.max_action = max_action
        self.max_torque = max_torque
    
    def select_action(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return self.actor(state_tensor).cpu().detach().numpy()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = torch.as_tensor(action_, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).reshape((batch_size, 1))
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device).reshape((batch_size, 1))
            
            # Select next action according to target policy:
            noise = torch.randn(action_.shape, generator=self.torch_rng, device=self.device) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            # next_action = (self.actor_target(next_state) + noise)
            # next_action = next_action.clamp(-self.max_action, self.max_action)

            next_action = (self.actor_target(next_state) + noise)
            force = next_action[:, :2]
            force_norm = torch.norm(force, dim=1, keepdim=True).clamp(min=1.0)
            force = force / force_norm
            if next_action.shape[1] >= 3:
                torque = next_action[:, 2:].clamp(-1.0, 1.0)
                next_action = torch.cat([force, torque], dim=1)
            else:
                next_action = force
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
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
                    
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_critic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_critic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_critic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_critic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
 