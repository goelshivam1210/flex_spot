# sim/td3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    # Action space is now 2D: [left_thrust_scale, right_thrust_scale] in [0,1]
    def __init__(self, state_dim, action_dim, max_action_val): # action_dim will be 2
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim) # Outputs 2 values for the two thrusts
        self.max_action = max_action_val # Should be 1.0 as env.action_space is Box(0,1)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Output actions scaled to [0, 1] using sigmoid, then scaled by max_action if needed.
        # If env.action_space is [0,1], max_action should be 1.0.
        return torch.sigmoid(self.l3(a)) * self.max_action

class Critic(nn.Module): # action_dim will be 2
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.l1_q1 = nn.Linear(state_dim + action_dim, 400)
        self.l2_q1 = nn.Linear(400, 300)
        self.l3_q1 = nn.Linear(300, 1)
        # Critic 2
        self.l1_q2 = nn.Linear(state_dim + action_dim, 400)
        self.l2_q2 = nn.Linear(400, 300)
        self.l3_q2 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1_q1(sa)); q1 = F.relu(self.l2_q1(q1)); q1 = self.l3_q1(q1)
        q2 = F.relu(self.l1_q2(sa)); q2 = F.relu(self.l2_q2(q2)); q2 = self.l3_q2(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1_q1(sa)); q1 = F.relu(self.l2_q1(q1)); q1 = self.l3_q1(q1)
        return q1

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.ptr = 0
        self.current_size = 0

    def add(self, transition):
        if self.current_size < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.current_size, size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        for i in indexes:
            s, a, r, s_prime, d = self.buffer[i]
            state.append(np.asarray(s, dtype=np.float32))
            action.append(np.asarray(a, dtype=np.float32)) # Actions are now 2D
            reward.append(np.asarray(r, dtype=np.float32))
            next_state.append(np.asarray(s_prime, dtype=np.float32))
            done.append(np.asarray(d, dtype=np.float32)) # This should be terminated flag
        return (np.array(state), np.array(action), 
                np.array(reward).reshape(-1,1), 
                np.array(next_state), np.array(done).reshape(-1,1))

class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action): # action_dim is 2
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = max_action # Should be 1.0 for actions scaled [0,1]
        self.policy_update_counter = 0 # For true policy delay

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten() # Returns 2D action

    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        for _ in range(n_iter): # This loop is for gradient steps per call to update
            state, action, reward, next_state, done_for_bellman = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done_for_bellman = torch.FloatTensor(done_for_bellman).to(device)

            with torch.no_grad():
                noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = self.actor_target(next_state) + noise
                next_action = next_action.clamp(0, self.max_action) 

                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1.0 - done_for_bellman) * gamma * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.policy_update_counter += 1

            if self.policy_update_counter % policy_delay == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)
    
    def save(self, directory, name):
        if not os.path.exists(directory): os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{name}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{name}_critic.pth"))
        # Saving target networks is optional if they are re-synced on load from main networks
        # torch.save(self.actor_target.state_dict(), os.path.join(directory, f"{name}_actor_target.pth"))
        # torch.save(self.critic_target.state_dict(), os.path.join(directory, f"{name}_critic_target.pth"))
        
    def load(self, directory, name): # For continuing training
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{name}_actor.pth"), map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict()) 
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{name}_critic.pth"), map_location=device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.actor_optimizer.load_state_dict(...) # Optional: load optimizer state
        # self.critic_optimizer.load_state_dict(...) # Optional: load optimizer state
            
    def load_actor(self, directory, name): # For deployment/playing
        actor_path = os.path.join(directory, f"{name}_actor.pth")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            self.actor.eval() 
        else:
            raise FileNotFoundError(f"Actor model not found at {actor_path}")