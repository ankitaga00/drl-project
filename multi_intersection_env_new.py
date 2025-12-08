# multi_intersection_env.py

import numpy as np
from single_intersection import SingleIntersectionEnv


class TrafficNetworkEnv:
    """
    Simple 4-intersection traffic network built from SingleIntersectionEnv.

    - num_nodes: 4 (fixed 2x2 grid)
    - state: dict {i -> 5-dim numpy array}
    - actions: dict {i -> 0 or 1} (extend or switch)
    - rewards: dict {i -> local negative queue}
    - done: True when episode time reaches max_time
    """

    def __init__(
        self,
        num_nodes: int = 4,
        dt: float = 5.0,
        max_time: float = 3600.0,
        lam_ns: float = 0.3,
        lam_ew: float = 0.3,
        sat_rate: float = 0.8,
        max_green_time: float = 60.0,
        seed: int = 42,
    ):
        assert num_nodes == 4, "This simple env currently supports exactly 4 nodes."

        self.num_nodes = num_nodes
        self.dt = dt
        self.max_time = max_time

        # Fix RNG for reproducibility
        np.random.seed(seed)

        # Create 4 independent intersections with same parameters
        self.intersections = [
            SingleIntersectionEnv(
                dt=dt,
                max_time=max_time,
                lam_ns=lam_ns,
                lam_ew=lam_ew,
                sat_rate=sat_rate,
                max_green_time=max_green_time,
            )
            for _ in range(num_nodes)
        ]

        # 2x2 grid adjacency:
        #
        #   0 --- 1
        #   |     |
        #   2 --- 3
        #
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for i, j in edges:
            self.adj_matrix[i, j] = 1.0
            self.adj_matrix[j, i] = 1.0

    # ---------- Core API ----------

    def reset(self):
        """
        Reset all 4 intersections.
        Returns:
            state_dict: {i -> 5-dim numpy array}
        """
        state_dict = {}
        for i in range(self.num_nodes):
            s = self.intersections[i].reset()
            state_dict[i] = s
        return state_dict

    def step(self, actions):
        """
        Take a joint action dict over all intersections.

        Args:
            actions: dict {i -> 0 or 1}

        Returns:
            next_states: dict {i -> state}
            rewards: dict {i -> reward}
            done: bool
        """
        next_states = {}
        rewards = {}
        done_flags = []

        for i in range(self.num_nodes):
            a = actions.get(i, 0)  # default extend if missing
            s_next, r, done_i, info = self.intersections[i].step(a)
            next_states[i] = s_next
            rewards[i] = r
            done_flags.append(done_i)

        # They should all terminate at the same time since they share dt/max_time
        done = any(done_flags)

        return next_states, rewards, done
