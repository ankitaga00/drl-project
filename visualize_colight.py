import pygame
import numpy as np
import torch

from multi_intersection_cstat import TrafficNetworkEnv
from colight_agent_new import CoLightAgent

# ========== Pygame Setup ==========
pygame.init()
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CoLight – College Station Traffic Visualization")

WHITE = (245, 245, 245)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
GREEN = (60, 200, 60)
RED = (220, 50, 50)
BLUE = (50, 100, 220)

FONT = pygame.font.SysFont("Arial", 16)

# 4 intersections arranged in grid (same logic as before)
positions = [(200, 200), (600, 200), (200, 400), (600, 400)]

FPS = 30
STEP_EVERY = 5   # RL step frequency

# ========== Load Environment + Agent ==========
env = TrafficNetworkEnv()
adj = torch.FloatTensor(env.adj_matrix)

state_dim = 5
action_dim = 2
num_nodes = env.num_nodes

agent = CoLightAgent(state_dim, action_dim, num_nodes)
agent.q_net.load_state_dict(torch.load("colight_model.pth"))
agent.epsilon = 0.0  # use trained policy only

clock = pygame.time.Clock()

state_dict = env.reset()
state_matrix = np.array(list(state_dict.values()))
done = False
frame_counter = 0
min_green_steps = 3
last_switch_time = {i: -min_green_steps for i in range(env.num_nodes)}
step_counter = 0

running = True
while running:
    # ----- window + quit handling -----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    win.fill(WHITE)

    # ----- RL / environment step -----
    if not done and frame_counter % STEP_EVERY == 0:
        s_tensor = torch.FloatTensor(state_matrix).unsqueeze(0)  # (1,N,S)
        q_values = agent.q_net(s_tensor.squeeze(0), adj)

        raw_actions = agent.select_actions(q_values)
        actions = {}

        for i in range(num_nodes):
            # prevent switching unless minimum time passed
            if raw_actions[i] == 1 and (step_counter - last_switch_time[i]) < min_green_steps:
                actions[i] = 0
            else:
                actions[i] = raw_actions[i]
                if raw_actions[i] == 1:
                    last_switch_time[i] = step_counter
        step_counter += 1
        next_state_dict, rewards, done = env.step(actions)
        state_matrix = np.array(list(next_state_dict.values()))

    # ----- draw nodes -----
    for idx, pos in enumerate(positions):
        pygame.draw.rect(win, GRAY, (*pos, 80, 80), border_radius=6)

        # Extract state features
        q_ns = float(state_matrix[idx][0])
        q_ew = float(state_matrix[idx][1])

        phase0 = state_matrix[idx][2]
        phase1 = state_matrix[idx][3]
        phase = 0 if phase0 > phase1 else 1

        # ===== Traffic lights =====
        ns_color = GREEN if phase == 0 else RED
        pygame.draw.circle(win, ns_color, (pos[0] + 40, pos[1] - 10), 8)

        ew_color = GREEN if phase == 1 else RED
        pygame.draw.circle(win, ew_color, (pos[0] - 10, pos[1] + 40), 8)

        # ===== Queue bars =====
        max_bar = 60.0
        ns_h = min(max_bar, q_ns * 2.0)
        ew_w = min(max_bar, q_ew * 2.0)

        ns_rect = pygame.Rect(pos[0] + 25, pos[1] + 85 - ns_h, 10, ns_h)
        pygame.draw.rect(win, BLUE, ns_rect)

        ew_rect = pygame.Rect(pos[0] + 85, pos[1] + 25, ew_w, 10)
        pygame.draw.rect(win, BLUE, ew_rect)

        # ===== labels =====
        t1 = FONT.render(f"Node {idx}", True, BLACK)
        win.blit(t1, (pos[0] + 20, pos[1] - 30))

        t2 = FONT.render(f"NS:{q_ns:.1f} EW:{q_ew:.1f}", True, BLACK)
        win.blit(t2, (pos[0] - 20, pos[1] + 90))

    # ---- status text ----
    status = "Running" if not done else "Episode ended"
    info = FONT.render(status, True, BLACK)
    win.blit(info, (20, 20))

    pygame.display.update()
    clock.tick(FPS)
    frame_counter += 1

pygame.quit()
