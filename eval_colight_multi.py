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

for ep in range(episodes):
    state = env.reset()
    state_matrix = np.array(list(state.values()))
    done = False
    total = 0

    while not done:
        # Format input correctly
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0)

        q_vals = agent.q_net(state_tensor, adj)  # (1, N, A)
        q_vals = q_vals.squeeze(0)               # (N, A)

        # Choose best actions per intersection
        actions = {i: q_vals[i].argmax().item() for i in range(env.num_nodes)}

        next_state, reward, done = env.step(actions)

        total += sum(reward.values())
        state_matrix = np.array(list(next_state.values()))

    print(f"Episode {ep+1}/{episodes}, Total reward: {total:.2f}")
    results.append(total)

avg = sum(results) / len(results)
print(f"Average Reward: {avg:.2f}")

# Save evaluation result
with open("results_colight.txt", "w") as f:
    f.write(str(avg))
print(f"Saved CoLight result: {avg}")
