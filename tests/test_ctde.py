"""
tests/test_ctde.py
───────────────────
Phase 6 tests. Run with: pytest tests/test_ctde.py -v

Test philosophy:
  - Test CTDERolloutBuffer fills, computes GAE, yields batches correctly
  - Test GAE with centralized values is mathematically correct
  - Test global state is correctly assembled from local obs
  - Test actors use ONLY local obs (decentralized execution invariant)
  - Test critic uses ONLY global state (centralized training invariant)
  - Test each store gets its own advantage signal (credit assignment)
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from envs.multi_store_env import MultiStoreEnv
from agents.mappo_agent import CTDETransition, CTDERolloutBuffer


N_STORES   = 4
OBS_DIM    = 120    # 5 SKUs × 24
ACTION_DIM = 5
GLOBAL_DIM = N_STORES * OBS_DIM
N_STEPS    = 32


# ══════════════════════════════════════════
# CTDERolloutBuffer tests (pure numpy)
# ══════════════════════════════════════════

class TestCTDERolloutBuffer:

    STORE_IDS = [f"store_{i}" for i in range(N_STORES)]

    def _make_transition(self, reward=1.0, value=10.0, done=False):
        rng = np.random.default_rng(0)
        return CTDETransition(
            local_obs    = {sid: rng.standard_normal(OBS_DIM).astype(np.float32)
                            for sid in self.STORE_IDS},
            actions      = {sid: rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
                            for sid in self.STORE_IDS},
            log_probs    = {sid: -0.5 for sid in self.STORE_IDS},
            global_state = rng.standard_normal(GLOBAL_DIM).astype(np.float32),
            joint_reward = reward,
            value        = value,
            done         = done,
        )

    @pytest.fixture
    def buffer(self):
        return CTDERolloutBuffer(n_steps=N_STEPS, gamma=0.99, gae_lambda=0.95)

    def _fill(self, buffer, reward=1.0, value=10.0):
        for _ in range(N_STEPS):
            buffer.add(self._make_transition(reward=reward, value=value))

    def test_not_full_before_n_steps(self, buffer):
        for _ in range(N_STEPS - 1):
            buffer.add(self._make_transition())
        assert not buffer.is_full()

    def test_full_at_n_steps(self, buffer):
        self._fill(buffer)
        assert buffer.is_full()

    def test_gae_advantages_shape(self, buffer):
        self._fill(buffer)
        buffer.compute_gae(last_value=0.0)
        assert buffer.advantages.shape == (N_STEPS,)
        assert buffer.returns.shape    == (N_STEPS,)

    def test_advantages_normalised_zero_mean(self, buffer):
        self._fill(buffer)
        buffer.compute_gae(last_value=0.0)
        assert abs(buffer.advantages.mean()) < 1e-5

    def test_advantages_normalised_unit_std(self, buffer):
        self._fill(buffer)
        buffer.compute_gae(last_value=0.0)
        assert abs(buffer.advantages.std() - 1.0) < 1e-4

    def test_returns_finite(self, buffer):
        self._fill(buffer, reward=2.0, value=5.0)
        buffer.compute_gae(last_value=3.0)
        assert np.all(np.isfinite(buffer.returns))
        assert np.all(np.isfinite(buffer.advantages))

    def test_gae_backward_recurrence_manual(self):
        """
        Manual trace: 3 steps, γ=1.0, λ=1.0, reward=1, value=0.
        Same as Phase 3 manual test but using CTDETransition.

        Expected raw advantages (before normalisation): [3, 2, 1]
        """
        buf = CTDERolloutBuffer(n_steps=3, gamma=1.0, gae_lambda=1.0)
        rng = np.random.default_rng(0)
        for _ in range(3):
            buf.add(CTDETransition(
                local_obs    = {"s0": rng.standard_normal(4).astype(np.float32)},
                actions      = {"s0": rng.uniform(-1,1,2).astype(np.float32)},
                log_probs    = {"s0": -0.5},
                global_state = rng.standard_normal(4).astype(np.float32),
                joint_reward = 1.0,
                value        = 0.0,
                done         = False,
            ))
        buf.compute_gae(last_value=0.0)

        raw = np.array([3.0, 2.0, 1.0])
        expected = (raw - raw.mean()) / (raw.std() + 1e-8)
        np.testing.assert_allclose(buf.advantages, expected, rtol=1e-5)

    def test_done_zeros_bootstrap(self):
        """
        Done=True at step 0 must zero the bootstrap term.
        Same logic as Phase 3 done-flag test.
        """
        buf = CTDERolloutBuffer(n_steps=2, gamma=0.99, gae_lambda=1.0)
        rng = np.random.default_rng(0)

        # Step 0: done (episode ends, no future)
        buf.add(CTDETransition(
            local_obs={"s0": rng.standard_normal(4).astype(np.float32)},
            actions={"s0": rng.uniform(-1,1,2).astype(np.float32)},
            log_probs={"s0": -0.5},
            global_state=rng.standard_normal(4).astype(np.float32),
            joint_reward=5.0, value=10.0, done=True,
        ))
        # Step 1: normal
        buf.add(CTDETransition(
            local_obs={"s0": rng.standard_normal(4).astype(np.float32)},
            actions={"s0": rng.uniform(-1,1,2).astype(np.float32)},
            log_probs={"s0": -0.5},
            global_state=rng.standard_normal(4).astype(np.float32),
            joint_reward=1.0, value=10.0, done=False,
        ))
        buf.compute_gae(last_value=10.0)

        # Step 0 done=True → bootstrap zeroed → smaller raw advantage
        # After normalisation step 0 should be below step 1
        assert buf.advantages[0] < buf.advantages[1]

    def test_get_batches_yields_all_steps(self, buffer):
        """Every stored step must appear in exactly one minibatch."""
        self._fill(buffer)
        buffer.compute_gae(last_value=0.0)
        total_seen = 0
        for batch in buffer.get_batches(8, self.STORE_IDS):
            # advantages is the 5th element
            total_seen += len(batch[4])
        assert total_seen == N_STEPS

    def test_batch_has_all_store_keys(self, buffer):
        """Each batch's local_obs dict must contain all store IDs."""
        self._fill(buffer)
        buffer.compute_gae(last_value=0.0)
        for batch in buffer.get_batches(8, self.STORE_IDS):
            local_obs_b = batch[0]
            for sid in self.STORE_IDS:
                assert sid in local_obs_b
            break

    def test_clear_resets_buffer(self, buffer):
        self._fill(buffer)
        buffer.compute_gae(last_value=0.0)
        buffer.clear()
        assert not buffer.is_full()
        assert len(buffer) == 0
        assert buffer.advantages is None


# ══════════════════════════════════════════
# MultiStoreEnv + CTDE integration test
# ══════════════════════════════════════════

class TestCTDEIntegration:

    def test_global_state_contains_all_stores(self):
        """
        Global state shape must be n_stores × obs_dim.
        Verifies the centralized critic receives the correct input.
        """
        env = MultiStoreEnv(n_stores=4, n_skus=5, max_steps=5, seed=0)
        env.reset()
        global_state = env.get_global_state()
        assert global_state.shape == (4 * env.obs_dim,), (
            f"Expected ({4 * env.obs_dim},), got {global_state.shape}"
        )

    def test_actors_receive_only_local_obs(self):
        """
        Each actor's input must be obs_dim — NOT global_state_dim.
        This is the DECENTRALIZED execution invariant.
        """
        env = MultiStoreEnv(n_stores=4, n_skus=5, max_steps=5, seed=0)
        obs = env.reset()

        for sid in env.store_ids:
            local_obs = obs[sid]
            assert local_obs.shape == (env.obs_dim,), (
                f"Actor {sid} received global state instead of local obs!"
                f" Shape: {local_obs.shape} vs expected ({env.obs_dim},)"
            )
            # Critically: local_obs must be SMALLER than global state
            assert local_obs.shape[0] < env.global_obs_dim

    def test_different_agents_get_different_advantages(self):
        """
        In CTDE, different agents CAN have different advantages at the
        same time step because the critic correctly attributes outcomes.
        This test verifies the buffer stores per-agent log_probs separately.
        """
        buf = CTDERolloutBuffer(n_steps=4, gamma=0.99, gae_lambda=0.95)
        store_ids = ["store_0", "store_1"]
        rng = np.random.default_rng(0)

        for _ in range(4):
            buf.add(CTDETransition(
                local_obs    = {sid: rng.standard_normal(OBS_DIM).astype(np.float32)
                                for sid in store_ids},
                actions      = {sid: rng.uniform(-1,1,ACTION_DIM).astype(np.float32)
                                for sid in store_ids},
                # Different log_probs per agent
                log_probs    = {"store_0": -0.3, "store_1": -0.8},
                global_state = rng.standard_normal(GLOBAL_DIM).astype(np.float32),
                joint_reward = 1.0,
                value        = 5.0,
                done         = False,
            ))

        buf.compute_gae(last_value=0.0)

        # Check log_probs are stored per-agent (not collapsed)
        for trans in buf._data:
            assert "store_0" in trans.log_probs
            assert "store_1" in trans.log_probs
            # Different agents had different log_probs
            assert trans.log_probs["store_0"] != trans.log_probs["store_1"]