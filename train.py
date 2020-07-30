import gym
import random
import numpy as np
from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN, self).__init__()
        self.ln1 = nn.Linear(in_features=num_inputs, out_features=32)
        self.ln2 = nn.Linear(in_features=32, out_features=32)
        self.outputs = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        return self.outputs(x)

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory():
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.counter = 0

    def push(self, *value):
        if self.length() < self.size:
            self.memory.append(Experience(*value))
        else:
            self.memory[self.counter%self.length()] = Experience(*value)
        self.counter += 1

    def length(self):
        return len(self.memory)

    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class Agent():
    def __init__(self, batch_size = 128, gamma = 0.9, eps_max = 1, eps_min = 0.005, eps_decay = 300):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.steps_completed = 0
        self.env = gym.make('Blackjack-v0')
        self.memory = ReplayMemory(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = DQN(num_inputs = 3, num_outputs = 2).to(self.device)
        self.target_network = DQN(num_inputs = 3, num_outputs = 2)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(params=self.policy_network.parameters(), lr=0.1)

    def get_exploration_rate(self):
        return self.eps_min + (self.eps_max - self.eps_min) * math.exp(-1.0 * self.steps_completed / self.eps_decay)

    def get_action(self, state):
        if random.random() > self.get_exploration_rate():
            self.steps_completed += 1
            with torch.no_grad():
                return self.policy_network(state).argmax().item()
        else:
            self.steps_completed += 1
            return random.random() > 0.5

    def current_state_values(self, states, actions):
        return self.policy_network(states).gather(dim=-1, index=actions)

    def next_state_values(self, next_states, dones):
        return self.target_network(next_states).max(dim=1)[0].detach()

    def back_prop(self, curr_qvalues, target): 
        loss = F.mse_loss(target.unsqueeze(1), curr_qvalues)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        evaluation = {}
        total_reward = 0

        for episode in range(30000):
            done = False
            curr_state = self.env.reset()
            state = torch.FloatTensor(curr_state)
            if episode % 10 == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            if episode %5000 == 0:
                evaluation[episode] = total_reward/5000
                total_reward = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.memory.push(curr_state, action, reward, next_state, done)
                curr_state = next_state
                state = torch.FloatTensor(curr_state)
        
                if self.memory.length() > self.batch_size:
                    experiences = self.memory.random_sample(self.batch_size)
                    batch = Experience(*zip(*experiences))
                    states = torch.FloatTensor(batch.state)
                    actions = torch.LongTensor(batch.action).unsqueeze(1)
                    rewards = torch.FloatTensor(batch.reward)
                    next_states = torch.FloatTensor(batch.next_state)
                    dones = torch.BoolTensor(batch.done)
        
                    curr_qvalues = self.current_state_values(states, actions)
                    next_qvalues = self.next_state_values(next_states, dones)
                    target = (next_qvalues * self.gamma) + rewards
                    self.back_prop(curr_qvalues, target)
            if episode > 29950:
                print(reward)

        print(evaluation)




agent = Agent()
agent.run()
