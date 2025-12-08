# run_multi_fixed.py

from multi_intersection_cstat import TrafficNetworkEnv
import numpy as np

env = TrafficNetworkEnv()

episodes = 20
episode_totals = []
switch_counts = []
avg_queues = []

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

    avg_queue = env.total_queue / max(env.steps, 1)
    print(f"[Fixed 4-node] Episode {ep + 1}/{episodes}, "
          f"Reward: {total_reward:.2f}, "
          f"Switches: {env.switch_count}, "
          f"AvgQueue: {avg_queue:.2f}")
    episode_totals.append(total_reward)
    switch_counts.append(env.switch_count)
    avg_queues.append(avg_queue)

avg_fixed = sum(episode_totals) / episodes
print("Average Reward:", avg_fixed)
print("Average Switches:", np.mean(switch_counts))
print("Average Queue Length:", np.mean(avg_queues))

# Save to file
with open("results_fixed.txt", "w") as f:
    f.write(f"avg_reward={avg_fixed}\n")
    f.write(f"avg_switches={np.mean(switch_counts)}\n")
    f.write(f"avg_queue={np.mean(avg_queues)}\n")

print(f"Saved Fixed controller result")