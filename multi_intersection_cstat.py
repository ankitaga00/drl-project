import numpy as np
from single_intersection import SingleIntersectionEnv


class TrafficNetworkEnv:
    """
    4-intersection network modeled after a small College Station traffic region.

    Node mapping:
        0 = Texas Ave @ George Bush Dr
        1 = Texas Ave @ University Dr
        2 = Wellborn Rd @ George Bush Dr
        3 = Olsen Blvd @ Penberthy Rd

    State returned as:
        {i -> 5-dim numpy vector}
    """

    def __init__(
        self,
        num_nodes: int = 4,
        dt: float = 5.0,
        max_time: float = 3600.0,
        sat_rate: float = 0.8,
        max_green_time: float = 60.0,
        seed: int = 42,
    ):
        assert num_nodes == 4, "This environment is fixed to 4 College Station nodes."

        self.num_nodes = num_nodes
        self.dt = dt
        self.max_time = max_time

        # Fix RNG
        np.random.seed(seed)

        # === OPTIONAL Metadata for visualization/reporting ===
        self.node_coords = {
            0: (-96.33956, 30.61329),  # Texas Ave @ Bush
            1: (-96.34058, 30.61952),  # Texas Ave @ University
            2: (-96.35125, 30.61369),  # Wellborn @ Bush
            3: (-96.35610, 30.61084),  # Olsen @ Penberthy
        }

        # === Traffic demand settings ===
        # Slightly different arrival rates per intersection to mimic real imbalance
        lam_values = [
            (0.45, 0.40),  # busier north–south at Texas & Bush
            (0.35, 0.30),  # moderate demand at University
            (0.55, 0.25),  # heavier southbound arrivals at Wellborn
            (0.20, 0.10),  # quiet zone at Olsen
        ]

        # Create 4 single intersections with slightly different traffic patterns
        self.intersections = [
            SingleIntersectionEnv(
                dt=dt,
                max_time=max_time,
                lam_ns=lam_values[i][0],
                lam_ew=lam_values[i][1],
                sat_rate=sat_rate,
                max_green_time=max_green_time,
            )
            for i in range(num_nodes)
        ]

        # === College Station-inspired adjacency ===
        #
        #   (1) University Dr
        #        |
        #   Texas Ave
        #        |
        #   (0) Bush Dr --- (2) Wellborn Rd --- (3) Olsen Blvd
        #
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        edges = [
            (0, 1),  # Texas Ave @ Bush ↔ Texas Ave @ University
            (0, 2),  # Texas Ave @ Bush ↔ Wellborn @ Bush
            (2, 3),  # Wellborn @ Bush ↔ Olsen @ Penberthy area
        ]

        for i, j in edges:
            self.adj_matrix[i, j] = 1.0
            self.adj_matrix[j, i] = 1.0

    # -------- Core RL Environment API --------

    def reset(self):
        """
        Reset all intersections and return dict of 4 states.
        """
        return {i: self.intersections[i].reset() for i in range(self.num_nodes)}

    def step(self, actions):
        """
        Apply joint action dict {i -> 0/1} across all intersections.
        """
        next_states = {}
        rewards = {}
        done_flags = []

        for i in range(self.num_nodes):
            a = actions.get(i, 0)  # default: keep phase
            s_next, r, done_i, info = self.intersections[i].step(a)

            next_states[i] = s_next
            rewards[i] = r
            done_flags.append(done_i)

        done = any(done_flags)  # all nodes share same episode horizon

        return next_states, rewards, done