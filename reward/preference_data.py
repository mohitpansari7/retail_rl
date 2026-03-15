"""
reward/preference_data.py
──────────────────────────
Phase 4 — Preference dataset generation.

Generates synthetic pairwise preferences from trajectory segments.
In production these would come from human expert annotations.
Here we use our hand-crafted reward as a proxy for expert judgement.

Key concept: a PreferencePair is two trajectory segments + a label.
  label = 0  →  segment_a is preferred
  label = 1  →  segment_b is preferred

The reward model trains on these pairs using Bradley-Terry loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from reward.reward_fn import compute_returns


@dataclass
class TrajectorySegment:
    """
    A slice of an agent's interaction with the environment.

    Stores observations, actions, and rewards for a contiguous window
    of K steps. The reward model learns to score these segments.

    WHY store obs AND actions?
    The reward model needs BOTH to score a decision:
    - obs alone: "what was the market situation?" — not enough
    - action alone: "what price was set?" — not enough
    - obs + action: "given this market, was this price decision good?" — complete

    Attributes:
        obs     : shape (K, obs_dim)    — state at each step
        actions : shape (K, action_dim) — action taken at each step
        rewards : shape (K,)            — actual rewards received
        total_return : discounted G_0 for this segment (used for labelling)
    """
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    total_return: float = field(init=False)

    def __post_init__(self):
        # Compute discounted return for the whole segment
        # This is what we use to determine which segment is "better"
        # when generating synthetic preference labels
        returns = compute_returns(
            list(self.rewards), gamma=0.99, normalise=False
        )
        self.total_return = float(returns[0])  # G_0 for the segment


@dataclass
class PreferencePair:
    """
    One labelled preference comparison between two trajectory segments.

    label = 0: segment_a is preferred (higher return)
    label = 1: segment_b is preferred (higher return)
    label = 0.5: tie (returns within noise threshold) — rare

    The Bradley-Terry loss uses these labels directly.
    """
    segment_a: TrajectorySegment
    segment_b: TrajectorySegment
    label: float          # 0.0, 0.5, or 1.0

    @property
    def preferred(self) -> TrajectorySegment:
        """Return the preferred segment."""
        return self.segment_a if self.label == 0.0 else self.segment_b

    @property
    def rejected(self) -> TrajectorySegment:
        """Return the rejected segment."""
        return self.segment_b if self.label == 0.0 else self.segment_a


class PreferenceDataset:
    """
    Generates and stores preference pairs from rollout data.

    Workflow:
      1. add_trajectory() — store raw (obs, action, reward) step data
      2. generate_pairs() — sample random pairs, label by return
      3. get_batch()      — yield minibatches for reward model training

    Args:
        segment_length : Steps per segment. 25–50 is standard in RLHF.
                         Short enough to evaluate, long enough to show behaviour.
        noise_prob     : Probability of flipping a preference label.
                         Models human annotator inconsistency (~10%).
        max_pairs      : Maximum pairs to store. Older pairs dropped when full.
    """

    def __init__(
        self,
        segment_length: int = 25,
        noise_prob: float = 0.1,
        max_pairs: int = 5_000,
        seed: int = 42,
    ) -> None:
        self.segment_length = segment_length
        self.noise_prob = noise_prob
        self.max_pairs = max_pairs
        self._rng = np.random.default_rng(seed)

        # Raw trajectory storage: list of (obs, action, reward) per step
        self._traj_obs:     List[np.ndarray] = []
        self._traj_actions: List[np.ndarray] = []
        self._traj_rewards: List[float]      = []

        # Labelled pairs for training
        self._pairs: List[PreferencePair] = []

    # ── Data collection ───────────────────────────────────────

    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
    ) -> None:
        """
        Record one environment step into the raw trajectory buffer.
        Called once per step during rollout collection.
        """
        self._traj_obs.append(obs.copy())
        self._traj_actions.append(action.copy())
        self._traj_rewards.append(reward)

    def _extract_segment(self, start_idx: int) -> Optional[TrajectorySegment]:
        """
        Extract a TrajectorySegment of length segment_length starting at start_idx.
        Returns None if not enough data.
        """
        end_idx = start_idx + self.segment_length
        if end_idx > len(self._traj_rewards):
            return None

        return TrajectorySegment(
            obs=np.array(self._traj_obs[start_idx:end_idx],     dtype=np.float32),
            actions=np.array(self._traj_actions[start_idx:end_idx], dtype=np.float32),
            rewards=np.array(self._traj_rewards[start_idx:end_idx], dtype=np.float32),
        )

    # ── Pair generation ───────────────────────────────────────

    def generate_pairs(self, n_pairs: int) -> int:
        """
        Sample random pairs of segments and label by discounted return.

        Labelling logic:
          - if return_a > return_b + threshold : label = 0 (a preferred)
          - if return_b > return_a + threshold : label = 1 (b preferred)
          - otherwise                          : label = 0.5 (tie — skip)

        The threshold prevents labelling nearly-identical segments,
        which would add noise without useful information.

        Label noise: with probability noise_prob, flip the label.
        This models human annotation inconsistency.

        Args:
            n_pairs : How many pairs to try to generate.

        Returns:
            Number of pairs actually added (some attempts may fail).
        """
        n_steps = len(self._traj_rewards)
        if n_steps < 2 * self.segment_length:
            return 0   # not enough data yet

        max_start = n_steps - self.segment_length
        added = 0
        threshold = 0.1   # minimum return difference to count as preference

        for _ in range(n_pairs * 3):  # try 3× to account for ties
            if added >= n_pairs:
                break

            # Sample two non-overlapping start indices
            start_a = int(self._rng.integers(0, max_start))
            start_b = int(self._rng.integers(0, max_start))

            # Ensure segments don't overlap
            if abs(start_a - start_b) < self.segment_length:
                continue

            seg_a = self._extract_segment(start_a)
            seg_b = self._extract_segment(start_b)
            if seg_a is None or seg_b is None:
                continue

            # Label by return comparison
            diff = seg_a.total_return - seg_b.total_return
            if abs(diff) < threshold:
                continue   # too similar — skip (tie)

            label = 0.0 if diff > 0 else 1.0

            # Apply label noise — models human annotation inconsistency
            if self._rng.uniform() < self.noise_prob:
                label = 1.0 - label   # flip

            pair = PreferencePair(segment_a=seg_a, segment_b=seg_b, label=label)
            self._pairs.append(pair)
            added += 1

            # Cap total pairs (drop oldest when full)
            if len(self._pairs) > self.max_pairs:
                self._pairs = self._pairs[-self.max_pairs:]

        return added

    # ── Training data access ──────────────────────────────────

    def get_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random minibatch of preference pairs.

        Returns 5 arrays ready for the reward model's forward pass:
          obs_a      : shape (B, K, obs_dim)    — segment A observations
          actions_a  : shape (B, K, action_dim) — segment A actions
          obs_b      : shape (B, K, obs_dim)    — segment B observations
          actions_b  : shape (B, K, action_dim) — segment B actions
          labels     : shape (B,)               — 0.0 = A preferred, 1.0 = B preferred
        """
        assert len(self._pairs) >= batch_size, (
            f"Not enough pairs: {len(self._pairs)} < {batch_size}"
        )

        idx = self._rng.choice(len(self._pairs), size=batch_size, replace=False)
        batch = [self._pairs[i] for i in idx]

        obs_a     = np.array([p.segment_a.obs     for p in batch], dtype=np.float32)
        actions_a = np.array([p.segment_a.actions for p in batch], dtype=np.float32)
        obs_b     = np.array([p.segment_b.obs     for p in batch], dtype=np.float32)
        actions_b = np.array([p.segment_b.actions for p in batch], dtype=np.float32)
        labels    = np.array([p.label             for p in batch], dtype=np.float32)

        return obs_a, actions_a, obs_b, actions_b, labels

    def __len__(self) -> int:
        return len(self._pairs)

    def n_steps_collected(self) -> int:
        return len(self._traj_rewards)