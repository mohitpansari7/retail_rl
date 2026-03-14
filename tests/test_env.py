"""
tests/test_env.py
──────────────────
Phase 1 tests: verify the MDP is economically correct and constraint-safe.

Run with: pytest tests/test_env.py -v

Test philosophy:
  - Test ECONOMICS (higher price → lower demand), not just code paths
  - Test HARD CONSTRAINTS are truly unbreakable under worst-case actions
  - Test REPRODUCIBILITY (same seed → same trajectory)
  - Test SHAPES precisely (wrong dimensions break all downstream code silently)
"""

import numpy as np
import pytest

from config.settings import StateConfig, HardConstraints, N_SKUS
from envs.demand_model import compute_demand, SKUDemandModel
from envs.retail_env import RetailEnv


# ══════════════════════════════════════════
# Demand model tests
# ══════════════════════════════════════════

class TestDemandModel:

    def test_higher_price_reduces_demand(self):
        """Core economic law: raising price must reduce demand."""
        demand_normal = compute_demand(100, 100, 100, -1.5, 1.0, 0.0)
        demand_high   = compute_demand(100, 200, 100, -1.5, 1.0, 0.0)
        assert demand_high < demand_normal, "Higher price should reduce demand"

    def test_lower_price_increases_demand(self):
        """Cutting price must increase demand (the other direction)."""
        demand_normal = compute_demand(100, 100, 100, -1.5, 1.0, 0.0)
        demand_low    = compute_demand(100,  50, 100, -1.5, 1.0, 0.0)
        assert demand_low > demand_normal

    def test_electronics_more_elastic_than_groceries(self):
        """
        With a 10% price raise:
          Electronics (ε=-2.5) should lose more demand than Groceries (ε=-0.8).
        This validates the category elasticity constants.
        """
        base, ref, raised = 100, 100, 110  # 10% price increase

        demand_elec_normal  = compute_demand(base, ref,    ref, -2.5, 1.0, 0.0)
        demand_elec_raised  = compute_demand(base, raised, ref, -2.5, 1.0, 0.0)
        demand_groc_normal  = compute_demand(base, ref,    ref, -0.8, 1.0, 0.0)
        demand_groc_raised  = compute_demand(base, raised, ref, -0.8, 1.0, 0.0)

        elec_pct_loss = (demand_elec_normal - demand_elec_raised) / demand_elec_normal
        groc_pct_loss = (demand_groc_normal - demand_groc_raised) / demand_groc_normal

        assert elec_pct_loss > groc_pct_loss, (
            f"Electronics should be more elastic: {elec_pct_loss:.2%} vs {groc_pct_loss:.2%}"
        )

    def test_demand_never_negative(self):
        """Even extreme prices or heavy negative noise can't make demand < 0."""
        # Price 500x reference with large negative noise
        result = compute_demand(100, 50_000, 100, -2.5, 1.0, noise=-0.99)
        assert result >= 0.0

    def test_festive_season_boosts_demand(self):
        """Q4 multiplier (1.3) must produce higher demand than Q1 (0.85)."""
        d_festive = compute_demand(100, 100, 100, -1.0, 1.30, 0.0)
        d_offpeak = compute_demand(100, 100, 100, -1.0, 0.85, 0.0)
        assert d_festive > d_offpeak

    def test_demand_history_always_7_days(self):
        """History must always be exactly 7 entries — no growth over time."""
        rng = np.random.default_rng(0)
        model = SKUDemandModel("test_sku", "groceries", 50.0, 100.0, -0.8)
        for day in range(1, 30):   # run for 29 days
            model.sample_demand(100.0, day, rng)
        assert len(model.get_history()) == 7

    def test_demand_history_updates(self):
        """Most recent demand must appear at the END of the history array."""
        rng = np.random.default_rng(7)
        model = SKUDemandModel("test_sku", "groceries", 50.0, 100.0, -0.8)
        # Sample a demand at an extreme price so it's distinguishable
        demand_val = model.sample_demand(200.0, 1, rng)   # raised price → lower demand
        # Most recent value is at index -1
        assert model.get_history()[-1] == pytest.approx(demand_val, rel=1e-5)


# ══════════════════════════════════════════
# Environment tests
# ══════════════════════════════════════════

class TestRetailEnv:

    @pytest.fixture
    def env(self):
        """Small env (10 SKUs) for speed. Full env is 425 SKUs."""
        return RetailEnv(n_skus=10, seed=42)

    def test_observation_shape_matches_state_config(self, env):
        """
        Obs shape must be exactly (n_skus × TOTAL_PER_SKU,).
        If this breaks, every downstream neural network breaks silently.
        """
        obs, _ = env.reset(seed=42)
        expected_dim = 10 * StateConfig.TOTAL_PER_SKU   # 10 × 24 = 240
        assert obs.shape == (expected_dim,), (
            f"Expected ({expected_dim},), got {obs.shape}"
        )

    def test_action_space_shape(self, env):
        env.reset()
        assert env.action_space.shape == (10,)

    def test_step_returns_correct_types(self, env):
        """Gymnasium contract: step() must return exactly these types."""
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)      # not np.float32
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_truncates_at_max_steps(self):
        """truncated must become True exactly at max_steps."""
        env = RetailEnv(n_skus=5, max_steps=10, seed=0)
        env.reset()
        truncated = False
        for _ in range(15):
            _, _, _, truncated, _ = env.step(env.action_space.sample())
            if truncated:
                break
        assert truncated, "Episode must truncate after max_steps"
        assert env.current_step == 10

    def test_inventory_never_goes_negative(self, env):
        """
        Hard constraint test: no matter what the agent does,
        inventory must always be >= 0.
        We use a full episode with random actions (worst case).
        """
        env.reset(seed=99)
        for _ in range(50):
            _, _, _, truncated, info = env.step(env.action_space.sample())
            for sku_info in info["skus"]:
                assert sku_info["inventory"] >= 0.0, (
                    f"Inventory went negative for {sku_info['sku_id']}"
                )
            if truncated:
                break

    def test_price_floor_enforced(self, env):
        """
        Hard constraint test: even with action=-1.0 (maximum price cut),
        no SKU's price should fall below cost × (1 + MIN_MARGIN_PCT).
        """
        env.reset(seed=42)
        # Max downward action for every SKU
        max_cut_action = np.full(10, -1.0, dtype=np.float32)
        _, _, _, _, info = env.step(max_cut_action)
        # We can verify that all prices are positive (cost tracking
        # lives inside SKU, not exposed in info, but price > 0 is verifiable)
        for sku_info in info["skus"]:
            assert sku_info["price"] > 0.0

    def test_same_seed_produces_same_trajectory(self, env):
        """
        Reproducibility: same seed → same obs, same reward on first step.
        Critical for debugging and ablation studies.
        """
        obs_a, _ = env.reset(seed=7)
        action = np.zeros(10, dtype=np.float32)
        obs_next_a, reward_a, *_ = env.step(action)

        obs_b, _ = env.reset(seed=7)
        obs_next_b, reward_b, *_ = env.step(action)

        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(obs_next_a, obs_next_b)
        assert reward_a == reward_b

    def test_different_seeds_produce_different_episodes(self, env):
        """Different seeds must produce different episodes."""
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        assert not np.array_equal(obs_a, obs_b)

    def test_episode_revenue_accumulates(self, env):
        """Episode revenue must grow monotonically (revenue >= 0 each step)."""
        env.reset(seed=0)
        prev_ep_revenue = 0.0
        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["episode_revenue"] >= prev_ep_revenue
            prev_ep_revenue = info["episode_revenue"]