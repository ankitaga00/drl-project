# train_independent_multi.py

import numpy as np
import torch
from multi_intersection_cstat import TrafficNetworkEnv
from dqn_agent_new import DQNAgent

env = TrafficNetworkEnv()

num_nodes = env.num_nodes
state_dim = 5
action_dim = 2

# Create agent per node
agents = {i: DQNAgent(state_dim, action_dim) for i in range(num_nodes)}

episodes = 200
batch_size = 32
target_update = 20

print("=== Training Independent RL on 4-node network ===")

for ep in range(episodes):

    state_dict = env.reset()
    done = False
    total_reward = 0

    while not done:

        actions = {}
        for i in range(num_nodes):
            actions[i] = agents[i].select_action(state_dict[i])

        next_state, reward_dict, done = env.step(actions)

        for i in range(num_nodes):
            agents[i].buffer.push(
                state_dict[i],
                actions[i],
                reward_dict[i],
                next_state[i],
                done
            )
            agents[i].train_step(batch_size)

        state_dict = next_state
        total_reward += sum(reward_dict.values())

    for i in range(num_nodes):
        agents[i].decay_epsilon()

        if ep % target_update == 0:
            agents[i].update_target()

    print(f"Episode {ep+1}/{episodes}, Total reward: {total_reward:.2f}")

# save all models
for i in range(num_nodes):
    torch.save(agents[i].q_net.state_dict(), f"indep_agent_{i}.pth")

print("Independent models saved.")
