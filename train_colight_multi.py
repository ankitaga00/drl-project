from multi_intersection_cstat import TrafficNetworkEnv
from colight_agent_new import CoLightAgent
import numpy as np
import torch

env = TrafficNetworkEnv()
adj = torch.FloatTensor(env.adj_matrix)

agent = CoLightAgent(5, 2, env.num_nodes)

episodes = 200
batch = 32
target_update = 20

print("=== Training CoLight ===")

for ep in range(episodes):
    state = env.reset()
    state_matrix = np.array(list(state.values()))
    done = False
    total = 0

    while not done:
        # state_matrix: (num_nodes, state_dim)
        state_tensor = torch.FloatTensor(state_matrix)  # [N, S]

        q_vals = agent.q_net(state_tensor, adj)  # [N, A]

        actions = agent.select_actions(q_vals)
        next_state, reward, done = env.step(actions)

        next_matrix = np.array(list(next_state.values()))
        reward_vec = np.array(list(reward.values()))

        agent.buffer.push(state_matrix, actions, reward_vec, next_matrix, done)
        agent.train_step(batch, adj)

        total += reward_vec.sum()
        state_matrix = next_matrix

    agent.decay_epsilon()

    if ep % target_update == 0:
        agent.update_target()

    print(f"Episode {ep+1}/{episodes}, Total reward:{total:.2f}, eps:{agent.epsilon:.3f}")

torch.save(agent.q_net.state_dict(), "colight_model.pth")
print("Model saved")
