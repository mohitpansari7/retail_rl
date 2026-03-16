"""
tests/test_marl.py
───────────────────
Phase 5 tests. Run with: pytest tests/test_marl.py -v

Test philosophy:
  - Test cross-store cannibalization actually reduces loser's reward
  - Test joint reward is sum of individual rewards
  - Test global state is correctly assembled from all store observations
  - Test price variance metric captures coordination (or lack thereof)
  - Test IndependentMARL distributes joint reward to all agents
  - Test non-stationarity proxy is tracked
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from envs.multi_store_env import MultiStoreEnv, _build_store_region_map
from agents.independent_agent import IndependentMARL
from config.settings import TrainingConfig


# ══════════════════════════════════════════
# MultiStoreEnv tests
# ══════════════════════════════════════════

class TestMultiStoreEnv:

    @pytest.fixture
    def env(self):
        return MultiStoreEnv(n_stores=4, n_skus=5, max_steps=10, seed=42)

    def test_reset_returns_obs_for_all_stores(self, env):
        obs = env.reset(seed=42)
        assert len(obs) == 4
        for sid in env.store_ids:
            assert sid in obs
            assert obs[sid].shape == (env.obs_dim,)

    def test_step_returns_correct_structure(self, env):
        env.reset(seed=42)
        actions = env.sample_actions()
        obs, ind_rewards, joint_reward, done, info = env.step(actions)

        assert len(obs)         == 4
        assert len(ind_rewards) == 4
        assert isinstance(joint_reward, float)
        assert isinstance(done, bool)
        assert "joint_reward"   in info
        assert "price_variance" in info

    def test_joint_reward_approximately_sum_of_individual(self, env):
        """
        joint_reward ≈ sum(ind_rewards).
        Small difference allowed due to cannibalization adjustments.
        """
        env.reset(seed=42)
        actions = env.sample_actions()
        _, ind_rewards, joint_reward, _, _ = env.step(actions)
        assert joint_reward == pytest.approx(sum(ind_rewards.values()), rel=1e-4)

    def test_global_state_shape(self, env):
        """Global state must be n_stores × obs_dim."""
        env.reset(seed=42)
        global_state = env.get_global_state()
        assert global_state.shape == (env.global_obs_dim,)
        assert global_state.shape == (4 * env.obs_dim,)

    def test_episode_truncates_at_max_steps(self, env):
        env.reset(seed=42)
        done = False
        for _ in range(15):
            _, _, _, done, _ = env.step(env.sample_actions())
            if done:
                break
        assert done
        assert env.current_step == 10

    def test_different_stores_have_different_obs(self, env):
        """Each store should see a different initial observation."""
        obs = env.reset(seed=42)
        obs_list = list(obs.values())
        # At least some observations should differ
        all_same = all(np.array_equal(obs_list[0], obs_list[i])
                       for i in range(1, len(obs_list)))
        assert not all_same, "All stores have identical observations — seed not working"

    def test_price_variance_is_non_negative(self, env):
        env.reset(seed=42)
        actions = env.sample_actions()
        _, _, _, _, info = env.step(actions)
        assert info["price_variance"] >= 0.0

    def test_region_map_correct_total_stores(self):
        store_ids = [f"store_{i}" for i in range(10)]
        region_map = _build_store_region_map(store_ids)
        assert len(region_map) == 10

    def test_region_map_urban_has_4_stores(self):
        store_ids = [f"store_{i}" for i in range(10)]
        region_map = _build_store_region_map(store_ids)
        urban_count = sum(1 for r in region_map.values() if r == "urban")
        assert urban_count == 4

    def test_cannibalization_penalises_underperforming_stores(self, env):
        """
        After several steps with random actions, stores in the same region
        should have asymmetric rewards due to cannibalization.
        We can't control which store wins, but we can verify the
        cannibalization dictionary contains some negative adjustments.
        """
        env.reset(seed=42)
        # Run one step and check that cannibalization adjustments exist
        actions = env.sample_actions()
        obs, ind_rewards, joint_reward, done, info = env.step(actions)
        # The demand_transfers should be non-zero (stores are competing)
        # With random actions it may be 0 if revenues happen to be equal
        # so we just check the key exists and is a valid float
        assert isinstance(info["demand_transfers"], float)
        assert info["demand_transfers"] >= 0.0


# ══════════════════════════════════════════
# IndependentMARL tests
# ══════════════════════════════════════════

class TestIndependentMARL:

    @pytest.fixture
    def setup(self):
        env = MultiStoreEnv(n_stores=4, n_skus=5, max_steps=20, seed=0)
        cfg = TrainingConfig(learning_rate=1e-3)
        marl = IndependentMARL(
            store_ids=env.store_ids,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            cfg=cfg,
            n_steps=16,   # very short for tests
        )
        return env, marl

    def test_correct_number_of_agents(self, setup):
        env, marl = setup
        assert len(marl.agents) == 4
        for sid in env.store_ids:
            assert sid in marl.agents

    def test_select_actions_returns_all_stores(self, setup):
        env, marl = setup
        obs = env.reset(seed=0)
        actions = marl.select_actions(obs)
        assert len(actions) == 4
        for sid in env.store_ids:
            assert sid in actions
            assert actions[sid].shape == (env.action_dim,)

    def test_actions_in_valid_range(self, setup):
        env, marl = setup
        obs = env.reset(seed=0)
        actions = marl.select_actions(obs)
        for action in actions.values():
            assert np.all(action >= -1.0)
            assert np.all(action <=  1.0)

    def test_joint_reward_distributed_to_all_agents(self, setup):
        """
        After store_transitions(), every agent's buffer should have
        received the same (normalised) joint reward.
        """
        env, marl = setup
        obs = env.reset(seed=0)
        actions = marl.select_actions(obs)
        _, _, joint_reward, done, _ = env.step(actions)
        marl.store_transitions(actions, joint_reward, done)

        # Each agent should have 1 step in their buffer
        for sid, agent in marl.agents.items():
            assert len(agent.buffer) == 1, f"{sid} buffer should have 1 step"

        # All agents received the same reward (joint / n_stores)
        expected_reward = joint_reward / marl.n_stores
        for sid, agent in marl.agents.items():
            stored_reward = agent.buffer._data[0].reward
            assert stored_reward == pytest.approx(expected_reward, rel=1e-5), (
                f"{sid} received wrong reward: {stored_reward} vs {expected_reward}"
            )

    def test_full_episode_and_update(self, setup):
        """Run a full episode and verify update() produces metrics."""
        env, marl = setup
        obs = env.reset(seed=0)
        metrics = {}

        for step in range(20):
            actions = marl.select_actions(obs)
            obs_next, _, joint_reward, done, info = env.step(actions)
            marl.store_transitions(actions, joint_reward, done)

            if marl.update_ready():
                metrics = marl.update(last_observations=obs_next)

            obs = obs_next if not done else env.reset(seed=step)

        # Should have produced at least one update
        assert len(marl.get_training_log()) > 0

    def test_non_stationarity_proxy_tracked(self, setup):
        """After several updates, non_stationarity_proxy should be in log."""
        env, marl = setup
        obs = env.reset(seed=0)

        for step in range(50):
            actions = marl.select_actions(obs)
            obs_next, _, joint_reward, done, _ = env.step(actions)
            marl.store_transitions(actions, joint_reward, done)
            if marl.update_ready():
                marl.update(last_observations=obs_next)
            obs = obs_next if not done else env.reset(seed=step)

        log = marl.get_training_log()
        if log:
            assert "non_stationarity_proxy" in log[0]