"""
training/rollout_buffer.py
───────────────────────────
Phase 3 — PPO Rollout Buffer

Stores exactly N environment steps (across episode boundaries),
computes GAE advantages, and yields random minibatches for K epochs.

The three stages of its lifecycle:
  1. COLLECT : add() called N times as the agent interacts with the env
  2. PROCESS : compute_advantages_and_returns() called once after collection
  3. CONSUME  : get_minibatches() called K times during the update loop
  4. RESET    : clear() — ready for the next rollout

WHY store across episode boundaries?
  A rollout of N=2048 steps may span several episodes.
  When one episode ends (done=True), we reset the env and keep collecting.
  The done flag tells GAE: "don't bootstrap across this boundary."
"""

from __future__ import annotations

from typing import Generator, Tuple
import numpy as np


class RolloutBuffer:
    """
    Fixed-size buffer storing N environment steps for PPO.

    Args:
        n_steps    : Rollout length. How many env steps before an update.
                     2048 is standard for continuous control problems.
        obs_dim    : Flat observation dimension.
        action_dim : Action dimension (= n_skus).
        gamma      : Discount factor γ (0.99).
        gae_lambda : GAE smoothing parameter λ (0.95).
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate all arrays — faster than growing lists
        # and makes the memory footprint predictable
        self.obs      = np.zeros((n_steps, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.rewards  = np.zeros(n_steps,               dtype=np.float32)
        self.values   = np.zeros(n_steps,               dtype=np.float32)
        self.log_probs = np.zeros(n_steps,              dtype=np.float32)
        self.dones    = np.zeros(n_steps,               dtype=np.float32)

        # Computed in compute_advantages_and_returns()
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

        self._pos = 0          # current write position
        self._full = False     # True once n_steps collected

    # ── Stage 1: Collect ─────────────────────────────────────

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Store one transition.

        Called once per env step during rollout collection.
        Overwrites position _pos in the pre-allocated arrays.

        Args:
            obs      : Observation s_t from the environment.
            action   : Action a_t taken by the agent.
            reward   : Reward r_t returned by the environment.
            value    : V(s_t) estimated by critic at time of action.
            log_prob : log π_old(a_t|s_t) — stored for importance ratio.
            done     : True if this step ended an episode.
        """
        assert self._pos < self.n_steps, "Buffer is full — call clear() first"

        self.obs[self._pos]       = obs
        self.actions[self._pos]   = action
        self.rewards[self._pos]   = reward
        self.values[self._pos]    = value
        self.log_probs[self._pos] = log_prob
        self.dones[self._pos]     = float(done)

        self._pos += 1
        if self._pos == self.n_steps:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    # ── Stage 2: Process — GAE computation ───────────────────

    def compute_advantages_and_returns(self, last_value: float) -> None:
        """
        Compute GAE advantages and discounted returns for all stored steps.

        GAE formula:
            δ_t   = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
            Â_t   = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + ...

        Equivalently (iterated backwards):
            Â_T   = δ_T
            Â_t   = δ_t + γλ · (1 − done_t) · Â_{t+1}

        WHY (1 − done_t)?
            If step t ends an episode, there is no s_{t+1}.
            The next observation is a fresh episode start.
            We must NOT bootstrap from that next state's value.
            Multiplying by (1 − done) zeros out the future term.

        WHY iterate backwards?
            Â_t depends on Â_{t+1}, exactly like G_t in REINFORCE.
            Same backward-pass trick: O(N) instead of O(N²).

        Args:
            last_value : V(s_{N+1}) — the critic's estimate of the state
                         AFTER the last collected step. Used to bootstrap
                         the return for incomplete episodes.
                         Pass 0.0 if the last step was terminal.
        """
        last_gae = 0.0

        for t in reversed(range(self.n_steps)):
            # Bootstrap: what's the value of the next state?
            if t == self.n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            # TD residual δ_t: how much better/worse was this step
            # than the critic predicted?
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE: accumulate weighted TD residuals
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values  (used to train the critic)
        # WHY? Â_t = G_t − V(s_t)  →  G_t = Â_t + V(s_t)
        self.returns = self.advantages + self.values

        # Normalise advantages: zero mean, unit variance
        # Same reason as normalising returns in REINFORCE —
        # prevents one unusually large advantage dominating the batch
        adv_mean = self.advantages.mean()
        adv_std  = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    # ── Stage 3: Consume — minibatch generator ────────────────

    def get_minibatches(
        self, minibatch_size: int
    ) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Yield random minibatches for K epochs of PPO updates.

        Shuffles ALL indices then splits into chunks of minibatch_size.
        Called K=10 times (once per epoch) — same shuffle each epoch
        is fine because the policy changes minimally per epoch.

        WHY shuffle?
            Sequential minibatches have temporal correlation —
            steps 0-63 all happened at the start of a rollout,
            steps 64-127 later, etc. Shuffling breaks this correlation
            so each minibatch is a random sample of the full rollout.
            Better gradient estimates, more stable training.

        Yields:
            Tuple of (obs, actions, old_log_probs, advantages, returns)
            all as numpy arrays of shape (minibatch_size, *).
        """
        assert self._full, "Buffer not full yet — collect more steps"

        indices = np.random.permutation(self.n_steps)

        for start in range(0, self.n_steps, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            yield (
                self.obs[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
            )

    # ── Stage 4: Reset ────────────────────────────────────────

    def clear(self) -> None:
        """
        Reset buffer for the next rollout.
        Does NOT zero the arrays — they'll be overwritten on next add().
        Just resets the position pointer and full flag.
        """
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self._pos