import torch
import torch.nn as nn
import torch.nn.functional as F

class CoLightNet(nn.Module):
    def __init__(self, state_dim, action_dim, num_nodes, embed_dim=32):
        super().__init__()

        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        # Per-intersection feature encoder
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Attention layers
        self.attn_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_k = nn.Linear(embed_dim, embed_dim, bias=False)

        # Action head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )

    def forward(self, state_matrix, adj):
        """
        state_matrix: Tensor [num_nodes, state_dim]
        adj: Tensor [num_nodes, num_nodes]
        """
        # If a dummy batch dim sneaks in (e.g., [1, N, S]), squeeze it
        if state_matrix.dim() == 3:
            state_matrix = state_matrix.squeeze(0)   # -> [N, S]

        # Encode per-node features
        h = self.state_mlp(state_matrix)            # [N, E]

        Q = self.attn_q(h)                          # [N, E]
        K = self.attn_k(h)                          # [N, E]

        # Attention scores across neighbors
        scores = torch.matmul(Q, K.T) / (self.embed_dim ** 0.5)  # [N, N]

        # Mask non-neighbors
        scores = scores.masked_fill(adj == 0, float('-inf'))

        # Normalize attention over neighbors
        attn = F.softmax(scores, dim=1)             # [N, N]

        # Aggregate neighbor features
        agg = torch.matmul(attn, h)                 # [N, E]

        # Per-node Q-values
        output = self.head(agg)                     # [N, action_dim]
        return output
