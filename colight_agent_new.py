import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from colight_network import CoLightNet


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.data = []

    def push(self, s, a, r, ns, d):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append((s, a, r, ns, d))

    def sample(self, batch):
        idx = np.random.choice(len(self.data), batch, replace=False)
        return [self.data[i] for i in idx]

    def __len__(self):
        return len(self.data)


class CoLightAgent:
    def __init__(self, state_dim, action_dim, num_nodes, lr=0.0005):
        self.num_nodes = num_nodes

        self.q_net = CoLightNet(state_dim, action_dim, num_nodes)
        self.target_net = CoLightNet(state_dim, action_dim, num_nodes)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = 0.95

        self.epsilon = 1.0
        self.eps_min = 0.1
        self.eps_decay = 0.995

        self.buffer = ReplayBuffer()

    def select_actions(self, q_vals):
        actions = {}
        for i in range(self.num_nodes):
            if np.random.rand() < self.epsilon:
                actions[i] = np.random.randint(0, 2)
            else:
                actions[i] = q_vals[i].argmax().item()
        return actions

    def train_step(self, batch_size, adj):
        if len(self.buffer) < batch_size:
            return

        batch = self.buffer.sample(batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for s, a, r, ns, d in batch:
            states.append(s)  # (N,S)
            actions.append([a[i] for i in sorted(a.keys())])  # (N,)
            rewards.append(list(r))  # (N,)
            next_states.append(ns)
            dones.append(d)

        states = torch.FloatTensor(states)  # (B,N,S)
        actions = torch.LongTensor(actions)  # (B,N)
        rewards = torch.FloatTensor(rewards)  # (B,N)
        next_states = torch.FloatTensor(next_states)  # (B,N,S)
        dones = torch.FloatTensor(dones).unsqueeze(1)  # (B,1)

        B, N, S = states.shape

        # =========== Q(next) computation ===========
        with torch.no_grad():
            # input aggregated state across batch dimension (mean pooling)
            next_q_all = self.target_net(next_states.mean(0), adj)  # (N,A)

            # max over actions
            next_max = next_q_all.max(1)[0]  # (N,)

            # expand to (B,N)
            next_max_q = next_max.unsqueeze(0).repeat(B, 1)  # (B,N)

        # =========== Q(state) computation ===========
        q_val_all = self.q_net(states.mean(0), adj)  # (N,A)
        q_vals = q_val_all.unsqueeze(0).repeat(B, 1, 1)  # (B,N,A)

        # gather actual chosen action values
        q_taken = q_vals.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B,N)

        # =========== Bellman target ===========
        targets = rewards + self.gamma * next_max_q * (1 - dones)

        # =========== Loss ===========
        loss = (q_taken - targets).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
