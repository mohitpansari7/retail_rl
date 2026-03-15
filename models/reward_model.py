"""
models/reward_model.py
───────────────────────
Phase 4 — Neural reward model trained with Bradley-Terry loss.

Maps a trajectory segment (K steps of obs + action) → scalar score.
Trained to assign higher scores to segments humans (or our proxy) prefer.

Architecture:
  Input  : segment of K steps, each step = concat(obs, action)
  Trunk  : Linear → LayerNorm → Tanh  (shared feature extraction)
  Pool   : mean over K time steps  (aggregate segment into one vector)
  Output : single scalar r_φ(segment)  (no activation — unbounded)

Training loss (Bradley-Terry):
  L = −mean[ log σ(r_φ(preferred) − r_φ(rejected)) ]

After training, r_φ replaces the hand-crafted reward function
as the signal used to train the PPO policy.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class RewardModel(nn.Module):
    """
    Segment-level reward model.

    Scores a trajectory segment (K steps) with a single scalar.
    Higher score = the segment represents better pricing behaviour.

    Args:
        obs_dim    : Observation dimension per step.
        action_dim : Action dimension per step.
        hidden_dim : Width of hidden layers (128 sufficient — simpler than policy net).
        n_layers   : Depth of trunk.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Input: concatenate obs and action at each step
        # WHY concat? The reward depends on BOTH what was observed AND what was done.
        # "Raising prices when inventory is low" is different from
        # "Raising prices when inventory is high" even if the action number is the same.
        input_dim = obs_dim + action_dim

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Output head: single scalar, no activation (reward is unbounded)
        self.head = nn.Linear(hidden_dim, 1)

        # Initialise weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of trajectory segments.

        Args:
            obs     : shape (B, K, obs_dim)    — B segments, K steps each
            actions : shape (B, K, action_dim)

        Returns:
            scores : shape (B,) — one scalar score per segment
        """
        # Concatenate obs and action at each timestep
        # x shape: (B, K, obs_dim + action_dim)
        x = torch.cat([obs, actions], dim=-1)

        # Process each step through the trunk
        # Reshape to (B*K, input_dim) for the linear layers,
        # then reshape back to (B, K, hidden_dim)
        B, K, D = x.shape
        x = x.view(B * K, D)
        features = self.trunk(x)                    # (B*K, hidden_dim)
        features = features.view(B, K, -1)          # (B, K, hidden_dim)

        # Mean pool over the K time steps
        # WHY mean pool? We want the AVERAGE quality of the segment,
        # not just the last step or a weighted sequence.
        # Mean pooling is simple, fast, and works well for this task.
        pooled = features.mean(dim=1)               # (B, hidden_dim)

        # Project to scalar
        scores = self.head(pooled).squeeze(-1)      # (B,)
        return scores

    def bradley_terry_loss(
        self,
        obs_a: torch.Tensor,
        actions_a: torch.Tensor,
        obs_b: torch.Tensor,
        actions_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Bradley-Terry preference loss.

        Loss = −mean[ log σ(r_φ(preferred) − r_φ(rejected)) ]

        When label = 0: segment_a is preferred → maximise r_φ(a) − r_φ(b)
        When label = 1: segment_b is preferred → maximise r_φ(b) − r_φ(a)

        We unify both cases:
          signed_diff = (1 − 2·label) × (r_a − r_b)
          If label=0: signed_diff = +1 × (r_a − r_b) → positive when a better
          If label=1: signed_diff = −1 × (r_a − r_b) → positive when b better

        Loss = −mean[ log σ(signed_diff) ]

        Args:
            obs_a, actions_a : shape (B, K, *) — preferred/rejected batches
            obs_b, actions_b : shape (B, K, *)
            labels           : shape (B,) — 0=a preferred, 1=b preferred

        Returns:
            loss     : scalar loss tensor (differentiable)
            accuracy : fraction of pairs ranked correctly (for monitoring)
        """
        score_a = self.forward(obs_a, actions_a)   # (B,)
        score_b = self.forward(obs_b, actions_b)   # (B,)

        # Sign: +1 if a preferred, −1 if b preferred
        # (1 − 2×label): label=0 → 1, label=1 → −1
        signed_diff = (1.0 - 2.0 * labels) * (score_a - score_b)

        # Bradley-Terry loss = negative log sigmoid
        # When signed_diff is large and positive → σ(large) ≈ 1 → log(1) ≈ 0 → loss ≈ 0 ✓
        # When signed_diff is negative (wrong ranking) → σ(negative) < 0.5 → large loss ✓
        loss = -torch.log(torch.sigmoid(signed_diff) + 1e-8).mean()

        # Accuracy: fraction of pairs where score ordering matches label
        with torch.no_grad():
            correct = ((score_a > score_b) == (labels < 0.5)).float()
            accuracy = correct.mean()

        return loss, accuracy


class RewardModelTrainer:
    """
    Wraps RewardModel with an optimiser and training loop.

    Keeps the training logic separate from the network definition —
    same principle as separating ReinforceAgent from PolicyNetwork.

    Args:
        model      : The RewardModel instance.
        lr         : Learning rate. 1e-3 works well for reward models.
        device     : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model: RewardModel,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimiser = optim.Adam(model.parameters(), lr=lr)
        self._train_log = []

    def train_step(
        self,
        obs_a: "np.ndarray",
        actions_a: "np.ndarray",
        obs_b: "np.ndarray",
        actions_b: "np.ndarray",
        labels: "np.ndarray",
    ) -> dict:
        """
        One minibatch training step.

        Args:
            obs_a, actions_a : (B, K, *) numpy arrays for segment A
            obs_b, actions_b : (B, K, *) numpy arrays for segment B
            labels           : (B,) numpy array — 0=a preferred, 1=b preferred

        Returns:
            dict with loss and accuracy for this batch.
        """
        import torch

        def to_t(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        obs_a_t     = to_t(obs_a)
        actions_a_t = to_t(actions_a)
        obs_b_t     = to_t(obs_b)
        actions_b_t = to_t(actions_b)
        labels_t    = to_t(labels)

        self.optimiser.zero_grad()
        loss, accuracy = self.model.bradley_terry_loss(
            obs_a_t, actions_a_t, obs_b_t, actions_b_t, labels_t
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()

        metrics = {
            "rm_loss":     loss.item(),
            "rm_accuracy": accuracy.item(),
        }
        self._train_log.append(metrics)
        return metrics

    @torch.no_grad()
    def score_segment(
        self,
        obs: "np.ndarray",
        actions: "np.ndarray",
    ) -> float:
        """
        Score a single trajectory segment.
        Used to replace hand-crafted reward during PPO training.

        Args:
            obs     : shape (K, obs_dim)
            actions : shape (K, action_dim)

        Returns:
            Scalar reward score.
        """
        import torch
        obs_t     = torch.tensor(obs[None],     dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions[None], dtype=torch.float32, device=self.device)
        return self.model(obs_t, actions_t).item()

    def save(self, path: str) -> None:
        torch.save({
            "model":     self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimiser.load_state_dict(ckpt["optimiser"])