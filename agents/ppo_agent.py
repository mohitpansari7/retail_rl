"""
agents/ppo_agent.py
────────────────────
Phase 3 — Proximal Policy Optimisation (PPO-Clip)

Fixes REINFORCE's two problems:
  Problem 1 — Wasted experience: one update per episode, then discard.
  Fix: collect N steps, run K=10 update epochs, THEN discard.

  Problem 2 — Catastrophic updates: no bound on policy change per step.
  Fix: clip importance ratio to [1−ε, 1+ε]. Gradient zeroed if ratio
       goes outside this range — automatic trust region.

Algorithm:
  loop:
    1. Collect N steps with π_old  → RolloutBuffer
    2. Compute GAE advantages       → buffer.compute_advantages_and_returns()
    3. For K epochs:
         For each minibatch:
           a. Re-evaluate actions under current π_θ → new_log_prob, value, entropy
           b. ratio = exp(new_log_prob − old_log_prob)
           c. L^CLIP = −mean( min(ratio·Â, clip(ratio,1−ε,1+ε)·Â) )
           d. L^VF   = MSE(V(s), returns)
           e. L^ent  = −mean(H[π])
           f. L = L^CLIP + c_v·L^VF − c_e·L^ent
           g. backprop + clip_grad_norm + Adam step
    4. Clear buffer, repeat
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
from training.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """
    Single-store PPO agent.

    Same interface as ReinforceAgent (inherits BaseAgent) —
    the training loop doesn't need to change at all to switch agents.

    Key differences from ReinforceAgent:
      - Uses RolloutBuffer (N steps) instead of EpisodeBuffer (1 episode)
      - update() runs K epochs of minibatch updates, not 1 full-batch update
      - Loss uses clipped importance ratio instead of raw log_prob × advantage
      - GAE instead of raw discounted returns

    Args:
        obs_dim    : env.observation_space.shape[0]
        action_dim : env.action_space.shape[0]
        cfg        : TrainingConfig — all hyperparameters
        n_steps    : Rollout length before each update. 2048 standard.
        device     : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: Optional[TrainingConfig] = None,
        n_steps: int = 2048,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or TrainingConfig()
        self.device = torch.device(device)
        self.n_steps = n_steps

        # ── Same policy network as REINFORCE — architecture unchanged
        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            n_layers=3,
        ).to(self.device)

        # ── Adam optimiser — same learning rate as REINFORCE
        self.optimiser = optim.Adam(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
            eps=1e-5,   # slightly larger than default for stability
        )

        # ── Rollout buffer — stores N steps across episode boundaries
        self.buffer = RolloutBuffer(
            n_steps=n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        # ── Tracking
        self.update_count: int = 0
        self.total_steps: int = 0
        self._training_log: List[Dict] = []

        # Pending values between select_action → store_transition
        self._pending_log_prob: float = 0.0
        self._pending_value: float = 0.0
        self._pending_obs: Optional[np.ndarray] = None

    # ── Gymnasium interaction ─────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Sample action from current policy.

        Identical contract to ReinforceAgent.select_action() —
        this is why BaseAgent matters.

        @torch.no_grad(): inference only, no gradient graph needed.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t, log_prob_t, value_t = self.policy.get_action(obs_t, deterministic=False)

        self._pending_log_prob = log_prob_t.item()
        self._pending_value    = value_t.squeeze().item()
        self._pending_obs      = obs

        return action_t.cpu().numpy()

    def store_transition(self, action: np.ndarray, reward: float, done: bool = False) -> None:
        """
        Store one completed transition in the rollout buffer.

        NOTE: PPO's store_transition takes an extra `done` flag vs REINFORCE.
        This is necessary for correct GAE computation across episode boundaries.

        Args:
            action : action taken this step.
            reward : reward received.
            done   : True if this step ended an episode.
        """
        assert self._pending_obs is not None
        self.buffer.add(
            obs=self._pending_obs,
            action=action,
            reward=reward,
            value=self._pending_value,
            log_prob=self._pending_log_prob,
            done=done,
        )
        self.total_steps += 1

    def buffer_full(self) -> bool:
        """True when N steps have been collected — time to call update()."""
        return self.buffer.is_full()

    # ── Learning ──────────────────────────────────────────────

    @torch.no_grad()
    def _get_last_value(self, obs: np.ndarray) -> float:
        """
        Get critic's value estimate for the state AFTER the last rollout step.

        Used to bootstrap returns for incomplete episodes.
        If the rollout ended mid-episode, we need V(s_{N+1}) to estimate
        the return for the last few steps.

        @torch.no_grad(): inference only.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        _, _, value_t = self.policy.get_action(obs_t, deterministic=True)
        return value_t.squeeze().item()

    def update(self, last_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Run K epochs of PPO updates on the collected rollout.

        Full procedure:
          1. Bootstrap last value for GAE computation
          2. Compute GAE advantages and returns (once, before any epoch)
          3. For K epochs:
               For each minibatch (shuffled):
                 a. Re-evaluate actions under CURRENT policy
                 b. Compute importance ratio r_t = exp(log_π_new − log_π_old)
                 c. Clipped policy loss
                 d. Value function loss
                 e. Entropy bonus
                 f. Combined loss → backprop → clip grads → Adam step
          4. Clear buffer

        Args:
            last_obs : The observation AFTER the last stored step.
                       Used to bootstrap V(s_{N+1}) for GAE.
                       Pass None if the last step was terminal (done=True).

        Returns:
            Dict of mean metrics across all minibatch updates this call.
        """
        assert self.buffer.is_full(), "Call update() only when buffer is full"

        # ── Step 1: Bootstrap last value
        last_value = self._get_last_value(last_obs) if last_obs is not None else 0.0

        # ── Step 2: GAE advantages and returns (one pass, before any epoch)
        self.buffer.compute_advantages_and_returns(last_value)

        # Accumulators for logging
        all_policy_losses, all_value_losses, all_entropies, all_ratios = [], [], [], []

        # ── Step 3: K epochs of minibatch updates
        for epoch in range(self.cfg.n_epochs_per_update):
            for batch in self.buffer.get_minibatches(self.cfg.minibatch_size):
                obs_np, actions_np, old_log_probs_np, advantages_np, returns_np = batch

                # Convert to tensors
                obs_t          = torch.tensor(obs_np,          device=self.device)
                actions_t      = torch.tensor(actions_np,      device=self.device)
                old_log_probs_t = torch.tensor(old_log_probs_np, device=self.device)
                advantages_t   = torch.tensor(advantages_np,   device=self.device)
                returns_t      = torch.tensor(returns_np,      device=self.device)

                # ── Step 3a: Re-evaluate under current policy
                # This is what makes PPO "on-policy with replays" —
                # we reuse the obs/actions but recompute log_probs with
                # the CURRENT θ (which has been updated by previous epochs).
                new_log_probs_t, values_t, entropy_t = self.policy.evaluate_actions(
                    obs_t, actions_t
                )

                # ── Step 3b: Importance ratio
                # Using log difference for numerical stability:
                #   log(π_new/π_old) = log(π_new) − log(π_old)
                #   ratio = exp(log_diff)
                # This avoids computing tiny probabilities that might underflow.
                log_ratio = new_log_probs_t - old_log_probs_t
                ratio = log_ratio.exp()

                # ── Step 3c: Clipped policy loss
                # obj1: unclipped — standard policy gradient
                # obj2: clipped  — ratio bounded to [1−ε, 1+ε]
                # We take the MIN — the pessimistic estimate.
                # This is the key PPO inequality that prevents large updates.
                obj1 = ratio * advantages_t
                obj2 = ratio.clamp(
                    1.0 - self.cfg.clip_epsilon,
                    1.0 + self.cfg.clip_epsilon
                ) * advantages_t
                policy_loss = -torch.min(obj1, obj2).mean()

                # ── Step 3d: Value loss
                # Train critic to predict the GAE returns.
                # .squeeze(): values_t shape is (batch, 1) → (batch,)
                value_loss = nn.functional.mse_loss(
                    values_t.squeeze(), returns_t
                )

                # ── Step 3e: Entropy bonus
                # entropy_t = H[π_θ(·|s)] — how spread out is the distribution?
                # Higher entropy = more exploration.
                # We SUBTRACT entropy from loss → maximise entropy → stay exploratory.
                entropy_loss = -entropy_t.mean()

                # ── Step 3f: Combined loss
                total_loss = (
                    policy_loss
                    + self.cfg.value_loss_coeff * value_loss
                    + self.cfg.entropy_coeff * entropy_loss
                )

                # ── Backprop
                self.optimiser.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimiser.step()

                # Log
                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(-entropy_loss.item())
                all_ratios.append(ratio.mean().item())

        # ── Step 4: Clear buffer for next rollout
        self.buffer.clear()
        self.update_count += 1

        metrics = {
            "update": self.update_count,
            "policy_loss":   float(np.mean(all_policy_losses)),
            "value_loss":    float(np.mean(all_value_losses)),
            "entropy":       float(np.mean(all_entropies)),
            "mean_ratio":    float(np.mean(all_ratios)),
            "total_steps":   self.total_steps,
        }
        self._training_log.append(metrics)
        return metrics

    # ── BaseAgent interface ───────────────────────────────────

    def episode(self) -> int:
        return self.update_count

    def save(self, path: str) -> None:
        torch.save({
            "policy":      self.policy.state_dict(),
            "optimiser":   self.optimiser.state_dict(),
            "update_count": self.update_count,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.update_count = ckpt["update_count"]
        self.total_steps  = ckpt["total_steps"]

    def get_training_log(self) -> List[Dict]:
        return self._training_log