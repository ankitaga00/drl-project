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

episodes = 20
results = []
episode_rewards = []
switch_counts = []
avg_queues = []

print("=== Evaluating Independent RL ===")

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    min_green_steps = 3
    last_switch_time = {i: -min_green_steps for i in range(num_nodes)}
    step_counter = 0

    while not done:
        raw_actions = {i: agents[i].select_action(state[i]) for i in range(num_nodes)}
        actions = {}

        for i in range(num_nodes):
            if raw_actions[i] == 1 and (step_counter - last_switch_time[i]) < min_green_steps:
                actions[i] = 0
            else:
                actions[i] = raw_actions[i]
                if raw_actions[i] == 1:
                    last_switch_time[i] = step_counter

        next_state, reward_dict, done = env.step(actions)
        raw_sum = sum(reward_dict.values())
        switch_events = sum(1 for i in range(num_nodes) if actions[i] == 1)
        switch_penalty = 0.1
        total_reward += raw_sum - switch_penalty * switch_events

        state = next_state
        step_counter += 1

    avg_queue = env.total_queue / max(env.steps, 1)

    print(f"Episode {ep + 1}, Total reward: {total_reward:.2f}, "
          f"Switches: {env.switch_count}, Avg queue: {avg_queue:.2f}")
    results.append(total_reward)
    #episode_rewards.append(total_reward)
    switch_counts.append(env.switch_count)
    avg_queues.append(avg_queue)

avg_reward= sum(results)/len(results)
print("Average Reward:", avg_reward)
print("Average Switches:", np.mean(switch_counts))
print("Average Queue Length:", np.mean(avg_queues))

# Save evaluation result
with open("results_independent.txt", "w") as f:
    f.write(f"avg_reward={avg_reward}\n")
    f.write(f"avg_switches={np.mean(switch_counts)}\n")
    f.write(f"avg_queue={np.mean(avg_queues)}\n")
print(f"Saved independent RL result")

