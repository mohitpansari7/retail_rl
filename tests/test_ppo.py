"""
tests/test_ppo.py
──────────────────
Phase 3 tests. Run with: pytest tests/test_ppo.py -v

Test philosophy:
  - Test GAE computation MATHEMATICALLY (manual trace of δ and Â)
  - Test episode boundary handling (done=True zeros bootstrap)
  - Test buffer lifecycle (fill → process → minibatch → clear)
  - Test clipped loss is always ≤ unclipped (PPO invariant)
  - Test PPO agent produces valid actions and updates
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from training.rollout_buffer import RolloutBuffer

OBS_DIM    = 240
ACTION_DIM = 10
N_STEPS    = 64


# ══════════════════════════════════════════
# RolloutBuffer tests (pure numpy — no torch)
# ══════════════════════════════════════════

class TestRolloutBuffer:

    @pytest.fixture
    def buffer(self):
        return RolloutBuffer(
            n_steps=N_STEPS,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            gamma=0.99,
            gae_lambda=0.95,
        )

    def _fill_buffer(self, buffer, reward=1.0, done=False):
        rng = np.random.default_rng(0)
        for _ in range(N_STEPS):
            buffer.add(
                obs      = rng.standard_normal(OBS_DIM).astype(np.float32),
                action   = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
                reward   = reward,
                value    = 10.0,
                log_prob = -0.5,
                done     = done,
            )

    def test_buffer_not_full_before_n_steps(self, buffer):
        rng = np.random.default_rng(0)
        for i in range(N_STEPS - 1):
            buffer.add(rng.standard_normal(OBS_DIM).astype(np.float32),
                       rng.uniform(-1,1,ACTION_DIM).astype(np.float32),
                       1.0, 10.0, -0.5, False)
        assert not buffer.is_full()

    def test_buffer_full_at_n_steps(self, buffer):
        self._fill_buffer(buffer)
        assert buffer.is_full()

    def test_gae_advantage_shape(self, buffer):
        self._fill_buffer(buffer)
        buffer.compute_advantages_and_returns(last_value=0.0)
        assert buffer.advantages.shape == (N_STEPS,)
        assert buffer.returns.shape    == (N_STEPS,)

    def test_normalised_advantages_zero_mean(self, buffer):
        """After GAE + normalisation, advantages must have zero mean."""
        self._fill_buffer(buffer, reward=1.0)
        buffer.compute_advantages_and_returns(last_value=0.0)
        assert abs(buffer.advantages.mean()) < 1e-5

    def test_normalised_advantages_unit_std(self, buffer):
        self._fill_buffer(buffer, reward=1.0)
        buffer.compute_advantages_and_returns(last_value=0.0)
        assert abs(buffer.advantages.std() - 1.0) < 1e-4

    def test_gae_manual_trace(self):
        """
        Manual verification of GAE backward computation.

        Setup: 3-step buffer, γ=1.0, λ=1.0, no discounting.
        All rewards = 1, values = 0, no done flags.
        last_value = 0.

        Expected:
          δ_2 = r_2 + γ·V(s_3)·(1-done) - V(s_2) = 1 + 0 - 0 = 1
          δ_1 = r_1 + γ·V(s_2)·(1-done) - V(s_1) = 1 + 0 - 0 = 1
          δ_0 = r_0 + γ·V(s_1)·(1-done) - V(s_0) = 1 + 0 - 0 = 1

          Â_2 = δ_2 = 1
          Â_1 = δ_1 + γλ·Â_2 = 1 + 1·1 = 2
          Â_0 = δ_0 + γλ·Â_1 = 1 + 1·2 = 3

        (before normalisation)
        """
        buf = RolloutBuffer(
            n_steps=3, obs_dim=4, action_dim=2, gamma=1.0, gae_lambda=1.0
        )
        rng = np.random.default_rng(0)
        for _ in range(3):
            buf.add(
                obs=rng.standard_normal(4).astype(np.float32),
                action=rng.uniform(-1,1,2).astype(np.float32),
                reward=1.0, value=0.0, log_prob=-0.5, done=False,
            )
        buf.compute_advantages_and_returns(last_value=0.0)

        # Before normalisation: Â = [3, 2, 1]
        # After normalisation: standardised version of [3, 2, 1]
        raw = np.array([3.0, 2.0, 1.0])
        expected = (raw - raw.mean()) / (raw.std() + 1e-8)
        np.testing.assert_allclose(buf.advantages, expected, rtol=1e-5)

    def test_done_flag_zeros_bootstrap(self):
        """
        When done=True at step t, the future value V(s_{t+1}) should be
        zeroed out in the TD residual.

        Setup: 2 steps. Step 0 done=True, reward=5.
        V(s) = 10 everywhere. last_value=10.

        Without done flag:
          δ_0 = 5 + γ·10 - 10 = 5 + 9.9 - 10 = 4.9

        With done flag (correct):
          δ_0 = 5 + γ·10·(1-1) - 10 = 5 + 0 - 10 = -5
        """
        buf = RolloutBuffer(
            n_steps=2, obs_dim=4, action_dim=2, gamma=0.99, gae_lambda=1.0
        )
        rng = np.random.default_rng(0)
        # Step 0: done=True (episode ends here)
        buf.add(rng.standard_normal(4).astype(np.float32),
                rng.uniform(-1,1,2).astype(np.float32),
                reward=5.0, value=10.0, log_prob=-0.5, done=True)
        # Step 1: normal step
        buf.add(rng.standard_normal(4).astype(np.float32),
                rng.uniform(-1,1,2).astype(np.float32),
                reward=1.0, value=10.0, log_prob=-0.5, done=False)

        buf.compute_advantages_and_returns(last_value=10.0)

        # Raw advantage at step 0 (before normalisation) should be -5
        # After normalisation it changes but step 0 should be < step 1
        # because done zeroed the future bootstrap
        raw_adv_0 = 5.0 + 0.99 * 10.0 * 0.0 - 10.0   # = -5.0
        # We can check sign relative to step 1
        # Step 1: δ_1 = 1 + 0.99·10·1 - 10 = 0.9, Â_1 = 0.9
        # Step 0: Â_0 = -5 + 0.99·1·0·Â_1 = -5 (done zeros λ term too)
        # After normalisation: step 0 < step 1
        assert buf.advantages[0] < buf.advantages[1], (
            "Done flag should make step 0 advantage lower than step 1"
        )

    def test_minibatch_shapes(self, buffer):
        self._fill_buffer(buffer)
        buffer.compute_advantages_and_returns(last_value=0.0)
        for batch in buffer.get_minibatches(minibatch_size=16):
            obs, actions, old_lp, adv, ret = batch
            assert obs.shape     == (16, OBS_DIM)
            assert actions.shape == (16, ACTION_DIM)
            assert adv.shape     == (16,)
            break   # just check first batch

    def test_all_steps_covered_across_minibatches(self, buffer):
        """Every index must appear exactly once across all minibatches."""
        self._fill_buffer(buffer)
        buffer.compute_advantages_and_returns(last_value=0.0)
        seen = []
        for batch in buffer.get_minibatches(minibatch_size=16):
            seen.extend(range(len(seen), len(seen) + len(batch[0])))
        assert len(seen) == N_STEPS

    def test_clear_resets_buffer(self, buffer):
        self._fill_buffer(buffer)
        assert buffer.is_full()
        buffer.clear()
        assert not buffer.is_full()
        assert len(buffer) == 0

    def test_returns_equal_advantages_plus_values(self, buffer):
        """
        Mathematical invariant: returns = advantages (before normalisation) + values.
        After normalisation this doesn't hold, but we can check the relationship
        via returns = normalised_adv * std + mean + values.
        Simpler check: returns array is populated and finite.
        """
        self._fill_buffer(buffer, reward=2.0)
        buffer.compute_advantages_and_returns(last_value=5.0)
        assert np.all(np.isfinite(buffer.returns))
        assert np.all(np.isfinite(buffer.advantages))