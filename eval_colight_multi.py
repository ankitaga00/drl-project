import numpy as np
import torch
from multi_intersection_cstat import TrafficNetworkEnv
from colight_agent_new import CoLightAgent

print("=== Evaluating CoLight ===")

env = TrafficNetworkEnv()
adj = torch.FloatTensor(env.adj_matrix)

# Load trained model
agent = CoLightAgent(5, 2, env.num_nodes)
agent.q_net.load_state_dict(torch.load("colight_model.pth"))
agent.epsilon = 0.0  # greedy

episodes = 20
results = []
switch_counts = []
avg_queues = []

for ep in range(episodes):
    state = env.reset()
    state_matrix = np.array(list(state.values()))
    done = False
    total = 0
    min_green_steps = 3
    last_switch_time = {i: -min_green_steps for i in range(env.num_nodes)}
    step_counter = 0


    while not done:
        # Format input correctly
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0)

        q_vals = agent.q_net(state_tensor, adj)  # (1, N, A)
        q_vals = q_vals.squeeze(0)               # (N, A)

        # Choose best actions per intersection
        raw_actions = {i: q_vals[i].argmax().item() for i in range(env.num_nodes)}
        actions = {}

        for i in range(env.num_nodes):
            if raw_actions[i] == 1 and (step_counter - last_switch_time[i]) < min_green_steps:
                actions[i] = 0
            else:
                actions[i] = raw_actions[i]
                if raw_actions[i] == 1:
                    last_switch_time[i] = step_counter

        next_state, reward, done = env.step(actions)
        step_counter += 1
        raw_sum = sum(reward.values())

        switch_events = sum(1 for i in range(env.num_nodes) if actions[i] == 1)
        switch_penalty = 0.1  # tuneable

        total += raw_sum - switch_penalty * switch_events
        state_matrix = np.array(list(next_state.values()))


    avg_queue = env.total_queue / max(env.steps, 1)
    print(f"Episode {ep + 1}, Total reward:{total:.2f}, "
          f"Switches:{env.switch_count}, Avg queue:{avg_queue:.2f}")
    results.append(total)
    switch_counts.append(env.switch_count)
    avg_queues.append(avg_queue)

avg = sum(results) / len(results)
print(f"Average Reward: {avg:.2f}")
print("Average Switches:", np.mean(switch_counts))
print("Average Queue Length:", np.mean(avg_queues))

# Save evaluation result
with open("results_colight.txt", "w") as f:
    f.write(f"avg_reward={avg}\n")
    f.write(f"avg_switches={np.mean(switch_counts)}\n")
    f.write(f"avg_queue={np.mean(avg_queues)}\n")
print(f"Saved CoLight result")
