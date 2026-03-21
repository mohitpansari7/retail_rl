"""
agents/mappo_agent.py
──────────────────────
Phase 6 — MAPPO: Multi-Agent PPO with Centralized Training,
          Decentralized Execution (CTDE).

Solves the two remaining Phase 5 problems:

  Problem 1 — Non-stationarity:
    Centralized critic V_φ(global_state) is stationary because it
    conditions on the FULL joint state — not just one agent's view.
    As other agents update, the critic simply re-evaluates the new
    joint state. No staleness.

  Problem 2 — Credit assignment:
    Advantage Â_t^i = G_t − V_φ(s_global) correctly attributes
    outcomes to their causes. If Store 3 caused Store 7's bad step,
    V_φ knows that — so Store 7's advantage is adjusted accordingly.

Architecture:
  N actor networks    : π_{θ_i}(a_i | local_obs_i)   one per store
  1 critic network    : V_φ(global_state)             shared

Training loop:
  1. Collect N_steps with decentralized actors + centralized value estimates
  2. Compute GAE advantages using global state values
  3. K epochs of minibatch updates:
       - Each actor: PPO clip loss using local obs + shared advantage
       - Shared critic: MSE loss against GAE returns

MAPPO loss for agent i:
  L^i = L^CLIP(θ_i) + c_v·L^VF(φ) − c_e·H(π_{θ_i})
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from config.settings import TrainingConfig
from models.policy_network import PolicyNetwork
from models.centralized_critic import CentralizedCritic


# ─────────────────────────────────────────────────────────────
# CTDE Rollout Buffer
# ─────────────────────────────────────────────────────────────

@dataclass
class CTDETransition:
    """
    One step of joint experience across all agents.

    Stores both:
      - local_obs per agent    (for actor updates)
      - global_state           (for critic updates)

    WHY store global_state here and not reconstruct from local_obs?
    At update time we need global_state for every step.
    Reconstructing from local_obs requires all agents' buffers to be
    synchronized step-by-step — fragile. Storing directly is cleaner.
    """
    local_obs:    Dict[str, np.ndarray]   # {store_id: obs}
    actions:      Dict[str, np.ndarray]   # {store_id: action}
    log_probs:    Dict[str, float]        # {store_id: log π(a|s)}
    global_state: np.ndarray              # all stores concatenated
    joint_reward: float                   # R_total this step
    value:        float                   # V_φ(global_state)
    done:         bool


class CTDERolloutBuffer:
    """
    Stores N joint transitions for MAPPO update.

    Similar to RolloutBuffer (Phase 3) but stores joint experience
    instead of single-agent experience.

    GAE is computed using the SHARED critic's global-state values —
    this is the key difference from Phase 5's per-agent buffers.
    """

    def __init__(
        self,
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._data: List[CTDETransition] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns:    Optional[np.ndarray] = None

    def add(self, transition: CTDETransition) -> None:
        assert len(self._data) < self.n_steps
        self._data.append(transition)

    def is_full(self) -> bool:
        return len(self._data) >= self.n_steps

    def __len__(self) -> int:
        return len(self._data)

    def compute_gae(self, last_value: float) -> None:
        """
        Compute GAE advantages using the CENTRALIZED critic's values.

        The centralized V_φ(global_state) gives accurate advantage estimates
        because it accounts for ALL agents' contributions — solving the
        credit assignment problem from Phase 5.

        Backward pass identical to Phase 3, but values come from
        the shared critic's global-state estimates.
        """
        T = len(self._data)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            trans = self._data[t]
            next_value = (
                last_value if t == T - 1
                else self._data[t + 1].value
            )
            non_terminal = 1.0 - float(trans.done)

            delta = (
                trans.joint_reward
                + self.gamma * next_value * non_terminal
                - trans.value
            )
            last_gae = (
                delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            )
            advantages[t] = last_gae

        self.returns = advantages + np.array(
            [t.value for t in self._data], dtype=np.float32
        )

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        self.advantages = (advantages - adv_mean) / adv_std

    def get_batches(
        self, batch_size: int, store_ids: List[str]
    ):
        """
        Yield shuffled minibatches for K epochs.

        Each batch contains:
          - Per-agent local obs and actions    (for actor losses)
          - Global states                      (for critic loss)
          - Shared advantages and returns      (same for all agents)
        """
        assert self.advantages is not None, "Call compute_gae() first"
        T = len(self._data)
        indices = np.random.permutation(T)

        for start in range(0, T, batch_size):
            idx = indices[start:start + batch_size]

            # Per-agent local data
            local_obs_batch = {
                sid: np.array([self._data[i].local_obs[sid]  for i in idx],
                               dtype=np.float32)
                for sid in store_ids
            }
            actions_batch = {
                sid: np.array([self._data[i].actions[sid]    for i in idx],
                               dtype=np.float32)
                for sid in store_ids
            }
            old_log_probs_batch = {
                sid: np.array([self._data[i].log_probs[sid]  for i in idx],
                               dtype=np.float32)
                for sid in store_ids
            }

            # Shared global data
            global_states = np.array(
                [self._data[i].global_state for i in idx], dtype=np.float32
            )
            advantages = self.advantages[idx]
            returns    = self.returns[idx]

            yield (
                local_obs_batch,
                actions_batch,
                old_log_probs_batch,
                global_states,
                advantages,
                returns,
            )

    def clear(self) -> None:
        self._data = []
        self.advantages = None
        self.returns = None


# ─────────────────────────────────────────────────────────────
# MAPPO Agent
# ─────────────────────────────────────────────────────────────

class MAPPOAgent:
    """
    Multi-Agent PPO with Centralized Training, Decentralized Execution.

    Manages:
      - N actor networks (one per store, separate weights)
      - 1 shared centralized critic (global state → value)
      - 1 CTDE rollout buffer (joint experience)
      - N actor optimizers + 1 critic optimizer

    Args:
        store_ids       : List of store identifiers.
        obs_dim         : Local observation dimension per store.
        action_dim      : Action dimension per store.
        global_state_dim: Total global state dim (n_stores × obs_dim).
        cfg             : Training hyperparameters.
        n_steps         : Rollout length before each update.
        device          : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        store_ids: List[str],
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        cfg: Optional[TrainingConfig] = None,
        n_steps: int = 512,
        device: str = "cpu",
    ) -> None:
        self.store_ids = store_ids
        self.n_stores  = len(store_ids)
        self.obs_dim   = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg or TrainingConfig()
        self.device = torch.device(device)
        self.n_steps = n_steps

        # ── N actor networks (separate weights per store)
        # WHY separate? Urban/rural/suburban stores have different
        # optimal pricing strategies. Shared weights would average them out.
        self.actors: Dict[str, PolicyNetwork] = {
            sid: PolicyNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=256,
                n_layers=3,
            ).to(self.device)
            for sid in store_ids
        }

        # ── ONE shared centralized critic
        # WHY shared? "How good is this joint situation?" is the
        # same question for all agents — shared parameters converge faster.
        self.critic = CentralizedCritic(
            global_state_dim=global_state_dim,
            hidden_dim=512,
            n_layers=2,
        ).to(self.device)

        # ── Separate optimizer per actor + one for critic
        # Separate optimizers: each actor may converge at different rates
        # (urban stores have more volatile gradients than rural ones).
        self.actor_optimizers: Dict[str, optim.Adam] = {
            sid: optim.Adam(
                self.actors[sid].parameters(),
                lr=self.cfg.learning_rate,
                eps=1e-5,
            )
            for sid in store_ids
        }
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.cfg.learning_rate,
            eps=1e-5,
        )

        # ── CTDE rollout buffer
        self.buffer = CTDERolloutBuffer(
            n_steps=n_steps,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        # Tracking
        self.update_count: int = 0
        self.total_steps:  int = 0
        self._training_log: List[Dict] = []

    # ── Collection phase (decentralized) ─────────────────────

    @torch.no_grad()
    def select_actions(
        self,
        local_obs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Each actor selects its action using ONLY its local observation.

        This is the DECENTRALIZED part of CTDE — no store ever sees
        another store's observation during action selection.

        Returns:
            actions   : Dict[store_id → action_array]

        Also stores log_probs internally for the update step.
        """
        self._pending_log_probs: Dict[str, float] = {}
        actions: Dict[str, np.ndarray] = {}

        for sid in self.store_ids:
            obs_t = torch.tensor(
                local_obs[sid], dtype=torch.float32, device=self.device
            )
            action_t, log_prob_t, _ = self.actors[sid].get_action(
                obs_t, deterministic=False
            )
            actions[sid] = action_t.cpu().numpy()
            self._pending_log_probs[sid] = log_prob_t.item()

        return actions

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        """
        Get centralized critic's value estimate for the current global state.

        Called once per step during collection — provides the value
        estimate stored in the buffer for GAE computation.
        """
        gs_t = torch.tensor(
            global_state[None], dtype=torch.float32, device=self.device
        )
        return self.critic.forward(gs_t).squeeze().item()

    def store_transition(
        self,
        local_obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        global_state: np.ndarray,
        joint_reward: float,
        done: bool,
    ) -> None:
        """
        Store one joint transition in the CTDE buffer.

        Called after env.step() returns. Stores BOTH local obs (for actors)
        AND global state (for critic) in a single synchronized record.
        This is what enables the centralized update in update().
        """
        value = self.get_value(global_state)

        self.buffer.add(CTDETransition(
            local_obs=local_obs,
            actions=actions,
            log_probs=self._pending_log_probs.copy(),
            global_state=global_state,
            joint_reward=joint_reward / self.n_stores,   # normalise
            value=value,
            done=done,
        ))
        self.total_steps += 1

    def buffer_full(self) -> bool:
        return self.buffer.is_full()

    # ── Update phase (centralized) ────────────────────────────

    def update(self, last_global_state: np.ndarray) -> Dict:
        """
        MAPPO update: K epochs of minibatch updates across all actors + critic.

        This is the CENTRALIZED part of CTDE:
          - GAE uses global-state values → correct credit assignment
          - Critic loss trains on global state → learns joint dynamics
          - Actor losses use shared advantage → aligned gradient signals

        Procedure:
          1. Bootstrap last value from centralized critic
          2. Compute GAE advantages using global-state values
          3. K epochs × minibatches:
               For each agent i:
                 a. Recompute log π_i(a_i | local_obs_i) under current θ_i
                 b. Importance ratio r_i = exp(new_log_prob - old_log_prob)
                 c. Clipped actor loss L^CLIP_i
                 d. Entropy bonus
                 Update actor_i
               For shared critic:
                 e. Recompute V_φ(global_state)
                 f. Value loss MSE(V_φ, returns)
                 Update critic
          4. Clear buffer

        Args:
            last_global_state : Global state after the last stored step,
                                used to bootstrap V(s_{T+1}).

        Returns:
            Metrics dict for logging.
        """
        assert self.buffer.is_full()

        # ── Step 1: Bootstrap last value
        last_value = self.get_value(last_global_state)

        # ── Step 2: GAE with centralized critic values
        self.buffer.compute_gae(last_value)

        # Accumulators
        actor_losses  = {sid: [] for sid in self.store_ids}
        critic_losses = []
        entropies     = []
        ratios        = []

        # ── Step 3: K epochs
        for _ in range(self.cfg.n_epochs_per_update):
            for batch in self.buffer.get_batches(
                self.cfg.minibatch_size, self.store_ids
            ):
                (
                    local_obs_b,
                    actions_b,
                    old_log_probs_b,
                    global_states_b,
                    advantages_b,
                    returns_b,
                ) = batch

                adv_t = torch.tensor(advantages_b, device=self.device)
                ret_t = torch.tensor(returns_b,    device=self.device)
                gs_t  = torch.tensor(global_states_b, device=self.device)

                # ── Update each actor independently
                # Key CTDE insight: actors use LOCAL obs but SHARED advantage.
                # The shared advantage (from centralized critic) corrects for
                # what the other agents were doing — clean credit assignment.
                for sid in self.store_ids:
                    obs_t = torch.tensor(local_obs_b[sid],      device=self.device)
                    act_t = torch.tensor(actions_b[sid],        device=self.device)
                    old_lp_t = torch.tensor(old_log_probs_b[sid], device=self.device)

                    # Recompute log_prob under current actor weights
                    new_lp_t, _, entropy_t = self.actors[sid].evaluate_actions(
                        obs_t, act_t
                    )

                    # Importance ratio
                    ratio = (new_lp_t - old_lp_t).exp()

                    # PPO clipped objective (same as Phase 3)
                    obj1 = ratio * adv_t
                    obj2 = ratio.clamp(
                        1.0 - self.cfg.clip_epsilon,
                        1.0 + self.cfg.clip_epsilon,
                    ) * adv_t
                    actor_loss = -torch.min(obj1, obj2).mean()
                    entropy_loss = -entropy_t.mean()

                    total_actor_loss = (
                        actor_loss
                        + self.cfg.entropy_coeff * entropy_loss
                    )

                    self.actor_optimizers[sid].zero_grad()
                    total_actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[sid].parameters(), max_norm=0.5
                    )
                    self.actor_optimizers[sid].step()

                    actor_losses[sid].append(actor_loss.item())
                    entropies.append(-entropy_loss.item())
                    ratios.append(ratio.mean().item())

                # ── Update shared critic
                # Critic uses GLOBAL STATE — this is what makes it centralized.
                # V_φ(global_state) sees all stores simultaneously.
                values_t = self.critic.forward(gs_t).squeeze()
                critic_loss = nn.functional.mse_loss(values_t, ret_t)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=0.5
                )
                self.critic_optimizer.step()
                critic_losses.append(critic_loss.item())

        # ── Cleanup
        self.buffer.clear()
        self.update_count += 1

        metrics = {
            "update":       self.update_count,
            "critic_loss":  float(np.mean(critic_losses)),
            "mean_entropy": float(np.mean(entropies)),
            "mean_ratio":   float(np.mean(ratios)),
            "total_steps":  self.total_steps,
            "actor_losses": {
                sid: float(np.mean(losses))
                for sid, losses in actor_losses.items()
            },
        }
        self._training_log.append(metrics)
        return metrics

    # ── Persistence ───────────────────────────────────────────

    def save(self, dir_path: str) -> None:
        import os
        os.makedirs(dir_path, exist_ok=True)
        for sid, actor in self.actors.items():
            torch.save(actor.state_dict(),
                       os.path.join(dir_path, f"actor_{sid}.pt"))
        torch.save(self.critic.state_dict(),
                   os.path.join(dir_path, "critic_shared.pt"))

    def load(self, dir_path: str) -> None:
        import os
        for sid, actor in self.actors.items():
            path = os.path.join(dir_path, f"actor_{sid}.pt")
            if os.path.exists(path):
                actor.load_state_dict(
                    torch.load(path, map_location=self.device)
                )
        critic_path = os.path.join(dir_path, "critic_shared.pt")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )

    def get_training_log(self) -> List[Dict]:
        return self._training_log