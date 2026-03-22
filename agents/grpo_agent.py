"""
agents/grpo_agent.py
─────────────────────
Phase 8 — Group Relative Policy Optimisation (GRPO) with RLVR.

Key difference from PPO:
  PPO  : one action per state → advantage = G_t − V_critic(s_t)
  GRPO : G actions per state  → advantage = (r_i − mean_group) / std_group

No critic network. The group mean IS the baseline.
Works best with verifiable rewards (binary, exact, unhackable).

Algorithm:
  For each minibatch of states:
    1. Sample G actions from current policy for each state
    2. Score each action with verifiable reward checker
    3. Compute group-relative advantage: Â_i = (r_i − μ_g) / (σ_g + ε)
    4. PPO clip loss using group-relative advantages
    5. Update policy

GRPO loss:
  L = −(1/G) Σ_i min(r_i(θ)·Â_i, clip(r_i(θ), 1−ε, 1+ε)·Â_i)

  where r_i(θ) = π_θ(a_i|s) / π_old(a_i|s)  (importance ratio)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from config.settings import TrainingConfig
from models.policy_network import PolicyNetwork
from reward.verifiable_rewards import VerifiableRewardScorer, VerifiableResult


class GRPOAgent(BaseAgent):
    """
    Single-store GRPO agent with RLVR.

    Designed to be used on top of the existing environment.
    Can wrap any single-store RetailEnv or replace PPOAgent.

    Args:
        obs_dim      : Observation dimension.
        action_dim   : Action dimension (= n_skus).
        group_size   : G — number of actions to sample per state.
                       Larger G → more accurate baseline, more compute.
                       G=8 is standard (from GRPO paper).
        cfg          : Training hyperparameters.
        device       : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        group_size: int = 8,
        cfg: Optional[TrainingConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.group_size = group_size
        self.cfg        = cfg or TrainingConfig()
        self.device     = torch.device(device)

        # ── Policy network (actor only — NO critic head needed)
        # We still use PolicyNetwork but IGNORE the critic output.
        # WHY keep PolicyNetwork? Reuse the same architecture.
        # The critic head exists but its loss is never computed.
        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            n_layers=3,
        ).to(self.device)

        self.optimiser = optim.Adam(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
            eps=1e-5,
        )

        # ── Verifiable reward scorer
        self.scorer = VerifiableRewardScorer()

        # ── Experience storage
        # GRPO doesn't use a rollout buffer in the same way as PPO.
        # Instead we store (obs, verifiable_results) pairs and
        # re-sample the group at update time.
        self._obs_buffer:     List[np.ndarray]          = []
        self._results_buffer: List[List[VerifiableResult]] = []

        # For the BaseAgent interface (select_action / store_transition)
        self._pending_obs:     Optional[np.ndarray] = None
        self._pending_action:  Optional[np.ndarray] = None

        # Tracking
        self.update_count:  int = 0
        self.total_steps:   int = 0
        self._training_log: List[Dict] = []

    # ── BaseAgent interface ───────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Sample one action for environment interaction.

        During training we use this for rollout collection only.
        The actual GRPO update uses sample_group() to get G actions.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t, _, _ = self.policy.get_action(obs_t, deterministic=False)
        self._pending_obs    = obs
        self._pending_action = action_t.cpu().numpy()
        return self._pending_action

    def store_transition(
        self,
        action: np.ndarray,
        reward: float,
        verifiable_results: Optional[List[VerifiableResult]] = None,
    ) -> None:
        """
        Store observation for GRPO update.

        Unlike PPO, we don't store the reward directly — we store the
        verifiable results so we can recompute scores for each group sample.

        Args:
            action             : Action taken (stored but not used in GRPO update).
            reward             : Raw env reward (stored for logging only).
            verifiable_results : List of VerifiableResult from this step.
        """
        if self._pending_obs is not None:
            self._obs_buffer.append(self._pending_obs.copy())
            self._results_buffer.append(verifiable_results or [])
        self.total_steps += 1

    def buffer_full(self) -> bool:
        return len(self._obs_buffer) >= self.cfg.minibatch_size

    # ── Core GRPO methods ─────────────────────────────────────

    def sample_group(
        self, obs: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample G actions from the current policy for a single state.

        Implementation: repeat obs G times → one batched forward pass.

        WHY batch instead of loop?
        One GPU operation on (G, obs_dim) is 4-8× faster than
        G sequential operations on (1, obs_dim).

        Args:
            obs : single observation shape (obs_dim,)

        Returns:
            actions    : shape (G, action_dim)
            log_probs  : shape (G,)
            values     : shape (G,)   ← unused in GRPO, but PolicyNetwork outputs them
        """
        # Repeat obs G times: (obs_dim,) → (G, obs_dim)
        obs_repeated = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_batch    = obs_repeated.unsqueeze(0).expand(self.group_size, -1)

        dist, values = self.policy(obs_batch)

        # Sample G different actions from the same distribution
        actions   = dist.rsample()                           # (G, action_dim)
        actions   = actions.clamp(-1.0, 1.0)
        log_probs = dist.log_prob(actions).sum(dim=-1)       # (G,)

        return actions, log_probs, values.squeeze(-1)

    def compute_group_advantages(
        self, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute group-relative advantages from G rewards.

        Formula:
            μ_g = mean(r_1, ..., r_G)
            σ_g = std(r_1, ..., r_G)
            Â_i = (r_i − μ_g) / (σ_g + ε)

        Edge case: if all rewards are identical, σ_g = 0.
            → Â_i = 0 for all i
            → zero gradient — this group contributes nothing to learning
            → this is CORRECT behaviour (no information in uniform rewards)

        Args:
            rewards : shape (G,) — one reward per group member

        Returns:
            advantages : shape (G,) — normalised group-relative advantages
        """
        mean = rewards.mean()
        std  = rewards.std() + 1e-8   # ε prevents division by zero

        advantages = (rewards - mean) / std
        return advantages

    def score_action_with_verifiable_rewards(
        self,
        verifiable_results: List[VerifiableResult],
    ) -> float:
        """
        Score an action using verifiable reward checks.

        Returns weighted score in [-1, +1].
        Used to score each group member during GRPO update.
        """
        return self.scorer.score(verifiable_results)

    # ── GRPO update ───────────────────────────────────────────

    def update(self) -> Dict:
        """
        GRPO policy update using stored observations.

        For each stored observation:
          1. Sample G actions from current policy (batched)
          2. Score each action using verifiable reward scorer
             (We use stored verifiable_results as a proxy here.
              In a full implementation, each group action would be
              executed in the environment to get fresh scores.)
          3. Compute group-relative advantages
          4. PPO clip loss with group-relative advantages
          5. Backprop + clip gradients + Adam step

        Note on scoring: Since we can't re-execute G actions in the
        environment (that would require G parallel env instances),
        we use the stored verifiable_results from the single rollout
        action as a common score, then add noise to simulate group
        diversity. In production, you'd use parallel environments.

        Returns:
            Metrics dict for logging.
        """
        if not self._obs_buffer:
            return {}

        all_losses      = []
        all_advantages  = []
        all_group_stds  = []
        zero_grad_count = 0

        for obs, vr_results in zip(self._obs_buffer, self._results_buffer):

            # ── Step 1: Sample G actions
            with torch.no_grad():
                actions_g, old_log_probs_g, _ = self.sample_group(obs)

            # ── Step 2: Score each group member
            # Base score from verifiable results
            base_score = self.score_action_with_verifiable_rewards(vr_results)

            # Simulate score diversity across the group:
            # Different price multipliers → different margin/stockout outcomes.
            # In production: execute each action in a parallel env instance.
            rng = np.random.default_rng(self.total_steps)
            score_noise = rng.normal(0.0, 0.15, size=self.group_size)
            group_scores = np.clip(base_score + score_noise, -1.0, 1.0)
            rewards_g = torch.tensor(group_scores, dtype=torch.float32,
                                     device=self.device)

            # ── Step 3: Group-relative advantages
            advantages_g = self.compute_group_advantages(rewards_g)

            # Track zero-advantage groups (all scores identical)
            group_std = rewards_g.std().item()
            if group_std < 1e-6:
                zero_grad_count += 1
                all_group_stds.append(0.0)
                continue   # skip — zero gradient, no information

            all_group_stds.append(group_std)
            all_advantages.extend(advantages_g.tolist())

            # ── Step 4: GRPO clip loss
            # Recompute log_probs under current policy (may have changed)
            obs_batch = torch.tensor(obs, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            obs_batch = obs_batch.expand(self.group_size, -1)
            dist, _ = self.policy(obs_batch)
            new_log_probs_g = dist.log_prob(actions_g).sum(dim=-1)

            # Importance ratio
            log_ratio  = new_log_probs_g - old_log_probs_g.detach()
            ratio      = log_ratio.exp()

            # Clipped PPO objective with group-relative advantages
            adv = advantages_g.detach()
            obj1 = ratio * adv
            obj2 = ratio.clamp(
                1.0 - self.cfg.clip_epsilon,
                1.0 + self.cfg.clip_epsilon,
            ) * adv
            loss = -torch.min(obj1, obj2).mean()

            # ── Step 5: Backprop
            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimiser.step()

            all_losses.append(loss.item())

        # Clear buffer
        self._obs_buffer     = []
        self._results_buffer = []
        self.update_count   += 1

        metrics = {
            "update":          self.update_count,
            "grpo_loss":       float(np.mean(all_losses)) if all_losses else 0.0,
            "mean_advantage":  float(np.mean(all_advantages)) if all_advantages else 0.0,
            "mean_group_std":  float(np.mean(all_group_stds)) if all_group_stds else 0.0,
            "zero_grad_groups": zero_grad_count,
            "effective_updates": len(all_losses),
        }
        self._training_log.append(metrics)
        return metrics

    # ── BaseAgent persistence ─────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "policy":      self.policy.state_dict(),
            "optimiser":   self.optimiser.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.update_count = ckpt["update_count"]

    def get_training_log(self) -> List[Dict]:
        return self._training_log