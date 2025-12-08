from multi_intersection_cstat import TrafficNetworkEnv
from dqn_agent_new import DQNAgent
import torch
import numpy as np

env = TrafficNetworkEnv()

num_nodes = env.num_nodes
state_dim = 5
action_dim = 2

agents = {}

# Load trained models
for i in range(num_nodes):
    agent = DQNAgent(state_dim, action_dim)
    agent.q_net.load_state_dict(torch.load(f"indep_agent_{i}.pth"))
    agent.epsilon = 0.0  # greedy evaluation
    agents[i] = agent

episodes = 10
results = []

print("=== Evaluating Independent RL ===")

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        actions = {i: agents[i].select_action(state[i]) for i in range(num_nodes)}
        next_state, reward_dict, done = env.step(actions)
        total_reward += sum(reward_dict.values())
        state = next_state

    print(f"Episode {ep+1}, Total reward: {total_reward:.2f}")
    results.append(total_reward)

avg_reward= sum(results)/len(results)
print("Average:", avg_reward)
# Save evaluation result
with open("results_independent.txt", "w") as f:
    f.write(str(avg_reward))
print(f"Saved independent RL result: {avg_reward}")

