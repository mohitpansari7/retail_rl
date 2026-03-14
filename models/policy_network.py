"""
models/policy_network.py
─────────────────────────
Phase 2 — Single-Agent Policy Gradient

The policy network maps state observations → action distribution.

Architecture choice — WHY a Gaussian policy?
────────────────────────────────────────────
Our action space is continuous: price change ∈ [-0.15, +0.15].
We can't use a discrete softmax (too many bins).
Instead we output a Gaussian distribution N(μ, σ):
    - μ (mean)   : the "best guess" price change
    - σ (std dev): how uncertain / exploratory the agent is

The agent samples an action from this distribution.
    - High σ early in training → lots of exploration
    - Low σ later → exploiting what it has learned

This is the standard approach for continuous action RL.

Design principles:
    - PolicyNetwork: pure PyTorch module, no RL logic
    - Separate value head for the critic/baseline (reused in Phase 3 PPO)
    - Deterministic evaluation mode (use mean, not sample)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network with shared trunk.

    Shared trunk → two heads:
        actor  : outputs (μ, log_σ) per action dimension
        critic : outputs V(s), the state value (used as baseline)

    WHY share the trunk?
        The early layers learn "what is the state of this store" —
        useful both for choosing actions AND for estimating value.
        Sharing = fewer parameters, faster learning.

    Args:
        obs_dim    : Total observation dimension (n_skus × 24)
        action_dim : Number of actions (n_skus)
        hidden_dim : Width of hidden layers
        n_layers   : Depth of shared trunk
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ── Shared trunk (feature extractor)
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),   # stabilises training vs BatchNorm
                nn.Tanh(),                  # bounded activations suit price domain
            ]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # ── Actor head: outputs mean of Gaussian per action
        self.actor_mean = nn.Linear(hidden_dim, action_dim)

        # log_std as a learnable parameter (not input-dependent)
        # Initialised to 0 → σ = 1 (high exploration at start)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Critic head: scalar state value
        self.critic = nn.Linear(hidden_dim, 1)

        # ── Weight initialisation (orthogonal → better gradient flow)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs : shape (batch, obs_dim)  — normalised observation

        Returns:
            dist  : Normal distribution over actions (for sampling / log_prob)
            value : shape (batch, 1)      — critic's estimate of V(s)
        """
        features = self.trunk(obs)

        # Actor
        mean = self.actor_mean(features)
        mean = torch.tanh(mean)   # squash to (-1, 1) → maps to ±15% price change

        # Clamp log_std for numerical stability: σ ∈ [e^-4, e^0.5] ≈ [0.018, 1.65]
        log_std = torch.clamp(self.log_std, min=-4.0, max=0.5)
        std = log_std.exp().expand_as(mean)

        dist = Normal(mean, std)

        # Critic
        value = self.critic(features)

        return dist, value

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or take mean) action from the policy.

        Args:
            obs          : shape (obs_dim,) or (batch, obs_dim)
            deterministic: if True, return mean (for evaluation / deployment)

        Returns:
            action   : shape (action_dim,)  — clipped to [-1, 1]
            log_prob : shape ()             — log π(a|s), needed for REINFORCE loss
            value    : shape (1,)           — critic estimate
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)   # add batch dim

        dist, value = self.forward(obs)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # reparameterisation trick for gradients

        # Clip to valid action range (redundant with tanh but explicit is better)
        action = torch.clamp(action, -1.0, 1.0)

        # Sum log_probs across action dimensions (independence assumption)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate old actions under current policy.
        Used in PPO (Phase 3) to compute importance ratios.

        Args:
            obs     : shape (batch, obs_dim)
            actions : shape (batch, action_dim)

        Returns:
            log_prob : shape (batch,)
            value    : shape (batch, 1)
            entropy  : shape (batch,)  — used for entropy bonus
        """
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy