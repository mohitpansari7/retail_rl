"""
agents/reinforce_agent.py
──────────────────────────
Phase 2 — REINFORCE with baseline.

Algorithm (Williams, 1992):
    For each episode:
      1. Collect full trajectory: (s_0,a_0,r_0), (s_1,a_1,r_1), ..., (s_T,a_T,r_T)
      2. Compute discounted returns G_t for each step t (backward pass)
      3. Compute advantage: A_t = G_t − V(s_t)   [baseline subtraction]
      4. Policy loss:  L_π = −mean_t[ log π_θ(a_t|s_t) · A_t ]
      5. Value loss:   L_V = mean_t[ (V(s_t) − G_t)² ]
      6. Total loss:   L   = L_π + c_v · L_V
      7. θ ← θ − α · ∇L   [Adam + gradient clip]
      8. Clear buffer

WHY wait for the full episode before updating?
    G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    At step t, future rewards haven't happened yet.
    We NEED them to compute G_t. So we wait.
    This is the core limitation REINFORCE → PPO will fix in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from config.settings import TrainingConfig
from models.policy_network import PolicyNetwork
from reward.reward_fn import compute_returns


# ─────────────────────────────────────────────────────────────
# Episode buffer
# ─────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """
    One complete (s, a, r, log_prob, value) record from a single step.

    WHY store log_prob at collection time?
    At update time the policy parameters θ have already changed slightly
    (or haven't yet — they change at the END of the episode).
    For REINFORCE specifically, we update once per episode so the stored
    log_prob matches the current θ exactly. In PPO (Phase 3) we'll use
    the stored log_prob to compute the importance ratio π_new/π_old.
    Storing it now makes that transition to PPO seamless.

    WHY store value (V(s_t)) at collection time?
    Same reason — we want the critic's estimate from the time of the
    decision, not a re-evaluation after learning has occurred.
    """
    obs: np.ndarray
    action: np.ndarray
    reward: float
    log_prob: float
    value: float


class EpisodeBuffer:
    """
    Stores all transitions for exactly one episode.

    Lifecycle:
        reset() → add() × T steps → get_*() in update() → clear() → repeat

    WHY a separate class (not a list in the agent)?
    The buffer has its own distinct lifecycle with clear responsibilities.
    Encapsulating it makes the agent cleaner and the buffer independently
    testable. We can verify "buffer accumulates correctly" without running
    a full training loop.
    """

    def __init__(self) -> None:
        self._data: List[Transition] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
    ) -> None:
        self._data.append(Transition(obs, action, reward, log_prob, value))

    def clear(self) -> None:
        self._data = []

    def __len__(self) -> int:
        return len(self._data)

    def get_rewards(self) -> List[float]:
        return [t.reward for t in self._data]

    def get_log_probs(self) -> torch.Tensor:
        """Stored log π(a|s) for each step — used directly in policy loss."""
        return torch.tensor([t.log_prob for t in self._data], dtype=torch.float32)

    def get_values(self) -> torch.Tensor:
        """Stored V(s_t) estimates — used as baseline in advantage computation."""
        return torch.tensor([t.value for t in self._data], dtype=torch.float32)

    def get_obs_tensor(self) -> torch.Tensor:
        """Stack all observations into a (T, obs_dim) tensor for value re-estimation."""
        return torch.tensor(
            np.array([t.obs for t in self._data], dtype=np.float32)
        )

    def get_actions_tensor(self) -> torch.Tensor:
        """Stack all actions into a (T, action_dim) tensor."""
        return torch.tensor(
            np.array([t.action for t in self._data], dtype=np.float32)
        )


# ─────────────────────────────────────────────────────────────
# REINFORCE Agent
# ─────────────────────────────────────────────────────────────

class ReinforceAgent(BaseAgent):
    """
    Single-store REINFORCE agent with baseline.

    Interacts with ONE RetailEnv instance.
    Phase 5 wraps N ReinforceAgents for multi-agent learning.

    Args:
        obs_dim    : env.observation_space.shape[0]
        action_dim : env.action_space.shape[0]  (= n_skus)
        cfg        : Training hyperparameters from settings.py
        device     : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: Optional[TrainingConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or TrainingConfig()
        self.device = torch.device(device)

        # ── Policy network (actor + critic shared trunk)
        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            n_layers=3,
        ).to(self.device)

        # ── Adam optimiser
        # WHY Adam and not vanilla SGD?
        # Adam maintains per-parameter adaptive learning rates.
        # In our 425-SKU action space, different SKU parameters learn
        # at very different rates. Adam adapts to each automatically.
        self.optimiser = optim.Adam(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
        )

        # ── Episode buffer
        self.buffer = EpisodeBuffer()

        # ── Tracking
        self.episode: int = 0
        self.total_steps: int = 0
        self._training_log: List[Dict] = []

        # Temporary storage between select_action() and store_transition()
        self._pending_log_prob: float = 0.0
        self._pending_value: float = 0.0
        self._pending_obs: Optional[np.ndarray] = None

    # ── Gymnasium interaction ─────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Sample an action from the current policy.

        @torch.no_grad() because this is INFERENCE — we don't need
        PyTorch to build a computation graph here. Building it would
        waste 2-3× memory and time. We compute gradients only in update().

        Steps:
          1. obs (numpy) → tensor
          2. Forward pass through policy network → distribution N(μ, σ)
          3. Sample action from distribution (exploration)
          4. Store log_prob and value for use in update()

        Args:
            obs : raw numpy observation from env.step()

        Returns:
            action : numpy array shape (action_dim,) in [-1, 1]
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t, log_prob_t, value_t = self.policy.get_action(obs_t, deterministic=False)

        # Stash for store_transition() which is called immediately after env.step()
        self._pending_log_prob = log_prob_t.item()
        self._pending_value = value_t.squeeze().item()
        self._pending_obs = obs

        return action_t.cpu().numpy()

    def store_transition(self, action: np.ndarray, reward: float) -> None:
        """
        Record what happened after we took an action.
        Called immediately after env.step() returns.

        WHY is this separate from select_action()?
        The environment computes the reward — the agent doesn't know it
        until AFTER env.step() returns. So we split:
          action = agent.select_action(obs)      ← before env.step
          obs, reward, ... = env.step(action)    ← env decides reward
          agent.store_transition(action, reward) ← agent records result
        """
        assert self._pending_obs is not None, "Call select_action() before store_transition()"
        self.buffer.add(
            obs=self._pending_obs,
            action=action,
            reward=reward,
            log_prob=self._pending_log_prob,
            value=self._pending_value,
        )
        self.total_steps += 1

    # ── Learning ──────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Run one REINFORCE update using the completed episode.

        Full update procedure:
          1. Compute discounted returns G_t (backward through rewards)
          2. Compute advantage A_t = G_t − V(s_t)   [baseline subtraction]
          3. Policy loss: L_π = −mean(log_π · A_t)  [negative = ascent via descent]
          4. Value loss:  L_V = MSE(V(s_t), G_t)    [train critic to predict G_t]
          5. Total loss:  L = L_π + c_v · L_V
          6. Backpropagate, clip gradients, Adam step
          7. Clear buffer for next episode

        Returns:
            Dict of scalar metrics for logging (losses, returns, episode stats).
        """
        if len(self.buffer) == 0:
            return {}

        # ── Step 1: Discounted returns G_t
        # normalise=True applies the baseline trick: zero-mean, unit-variance.
        # Now G_t>0 means "better than average this episode."
        rewards = self.buffer.get_rewards()
        returns_np = compute_returns(rewards, gamma=self.cfg.gamma, normalise=True)
        returns_t = torch.tensor(returns_np, dtype=torch.float32, device=self.device)

        # ── Step 2: Advantage A_t = G_t − V(s_t)
        # .detach(): we don't want gradients flowing back through V(s_t) here.
        # The critic is trained separately via value loss below.
        values_t = self.buffer.get_values().to(self.device)
        advantages_t = (returns_t - values_t).detach()

        # ── Step 3: Policy loss
        # log_probs were stored at collection time (current θ, since one update/episode)
        log_probs_t = self.buffer.get_log_probs().to(self.device)

        # The core REINFORCE loss:
        #   L_π = −mean_t[ log π_θ(a_t|s_t) · A_t ]
        # Negative because:
        #   - We WANT to MAXIMISE E[log π · A]
        #   - PyTorch MINIMISES the loss
        #   - Minimising −E[log π · A] = Maximising E[log π · A]  ✓
        policy_loss = -(log_probs_t * advantages_t).mean()

        # ── Step 4: Value loss
        # Re-run the CURRENT critic on all stored observations.
        # WHY re-run instead of using stored values?
        # We want to train the critic with current θ parameters.
        # The stored values were from before any update this episode.
        obs_t = self.buffer.get_obs_tensor().to(self.device)
        _, current_values, _ = self.policy.evaluate_actions(
            obs_t, self.buffer.get_actions_tensor().to(self.device)
        )
        value_loss = nn.functional.mse_loss(current_values.squeeze(), returns_t)

        # ── Step 5: Combined loss
        total_loss = policy_loss + self.cfg.value_loss_coeff * value_loss

        # ── Step 6: Backprop + update
        self.optimiser.zero_grad()    # clear gradients from previous update
        total_loss.backward()         # compute ∂L/∂θ for all parameters

        # Gradient clipping: if ||∇L|| > 0.5, scale it down.
        # Prevents a single bad batch from causing a catastrophic weight update.
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

        self.optimiser.step()         # θ ← θ − α · ∇L  (Adam)

        # ── Step 7: Cleanup
        self.buffer.clear()
        self.episode += 1

        metrics = {
            "episode": self.episode,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": float(np.mean(rewards)),
            "total_reward": float(np.sum(rewards)),
            "episode_length": len(rewards),
        }
        self._training_log.append(metrics)
        return metrics

    # ── BaseAgent interface ───────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "episode": self.episode,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.episode = ckpt["episode"]
        self.total_steps = ckpt["total_steps"]

    def get_training_log(self) -> List[Dict]:
        return self._training_log