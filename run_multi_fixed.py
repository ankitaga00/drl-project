# run_multi_fixed.py

from multi_intersection_cstat import TrafficNetworkEnv
import numpy as np

env = TrafficNetworkEnv()

episodes = 5
episode_totals = []
for ep in range(episodes):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        # Fixed-time: always EXTEND (action 0) at all intersections
        actions = {i: 0 for i in range(env.num_nodes)}
        next_state, rewards, done = env.step(actions)

        total_reward += sum(rewards.values())
        state = next_state

    print(f"[Fixed 4-node] Episode {ep+1}/{episodes}, Total reward: {total_reward:.2f}")
    episode_totals.append(total_reward)

avg_fixed = sum(episode_totals) / episodes

# Save to file
with open("results_fixed.txt", "w") as f:
    f.write(str(avg_fixed))

print(f"Saved Fixed controller result: {avg_fixed}")