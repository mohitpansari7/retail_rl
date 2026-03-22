"""
tests/test_grpo.py
───────────────────
Phase 8 tests. Run with: pytest tests/test_grpo.py -v

Test philosophy:
  - Test group-relative advantage math precisely (manual traces)
  - Test zero-gradient edge case (all rewards identical → advantages = 0)
  - Test verifiable reward checkers are exact (no approximation)
  - Test VerifiableRewardScorer weighted aggregation
  - Test hard constraint detection (margin, price_change)
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from reward.verifiable_rewards import (
    check_margin_constraint,
    check_stockout_constraint,
    check_revenue_target,
    check_price_change_constraint,
    check_cross_store_price_variance,
    VerifiableRewardScorer,
    VerifiableResult,
)


# ══════════════════════════════════════════
# Individual verifiable reward checker tests
# ══════════════════════════════════════════

class TestMarginConstraint:

    def test_passes_above_floor(self):
        # cost=100, floor=105, price=110 → pass
        r = check_margin_constraint(price=110, cost=100)
        assert r.passed
        assert r.score == pytest.approx(+1.0)

    def test_fails_below_floor(self):
        # cost=100, floor=105, price=103 → fail
        r = check_margin_constraint(price=103, cost=100)
        assert not r.passed
        assert r.score == pytest.approx(-1.0)

    def test_fails_exactly_at_cost(self):
        r = check_margin_constraint(price=100, cost=100)
        assert not r.passed

    def test_passes_exactly_at_floor(self):
        # price = cost × 1.05 exactly → pass
        r = check_margin_constraint(price=105.0, cost=100.0)
        assert r.passed

    def test_detail_contains_sku_id(self):
        r = check_margin_constraint(price=110, cost=100, sku_id="elec_001")
        assert "elec_001" in r.detail

    def test_name_is_margin_constraint(self):
        r = check_margin_constraint(price=110, cost=100)
        assert r.name == "margin_constraint"


class TestStockoutConstraint:

    def test_passes_when_inventory_sufficient(self):
        r = check_stockout_constraint(inventory=50, units_demanded=30)
        assert r.passed
        assert r.score == pytest.approx(+1.0)

    def test_fails_when_inventory_zero(self):
        r = check_stockout_constraint(inventory=0, units_demanded=10)
        assert not r.passed
        assert r.score == pytest.approx(-1.0)

    def test_passes_when_demand_exactly_met(self):
        r = check_stockout_constraint(inventory=10, units_demanded=10)
        assert r.passed


class TestRevenueTarget:

    def test_passes_above_target(self):
        r = check_revenue_target(actual_revenue=1200, target_revenue=1000)
        assert r.passed
        assert r.score == pytest.approx(+1.0)

    def test_fails_below_target(self):
        r = check_revenue_target(actual_revenue=800, target_revenue=1000)
        assert not r.passed
        assert r.score == pytest.approx(-1.0)

    def test_passes_exactly_at_target(self):
        r = check_revenue_target(actual_revenue=1000, target_revenue=1000)
        assert r.passed


class TestPriceChangeConstraint:

    def test_passes_within_limit(self):
        # 10% change, limit is 15%
        r = check_price_change_constraint(new_price=110, old_price=100)
        assert r.passed

    def test_fails_above_limit(self):
        # 20% change, limit is 15%
        r = check_price_change_constraint(new_price=120, old_price=100)
        assert not r.passed

    def test_passes_exactly_at_limit(self):
        # exactly 15%
        r = check_price_change_constraint(new_price=115.0, old_price=100.0)
        assert r.passed


class TestCrossStorePriceVariance:

    def test_passes_low_variance(self):
        # All prices within 5% of each other
        r = check_cross_store_price_variance([100, 102, 99, 101])
        assert r.passed

    def test_fails_high_variance(self):
        # Large spread — CV will exceed 10%
        r = check_cross_store_price_variance([100, 200, 50, 150])
        assert not r.passed

    def test_passes_single_store(self):
        # Can't have variance with only one store
        r = check_cross_store_price_variance([100.0])
        assert r.passed


# ══════════════════════════════════════════
# VerifiableRewardScorer tests
# ══════════════════════════════════════════

class TestVerifiableRewardScorer:

    @pytest.fixture
    def scorer(self):
        return VerifiableRewardScorer()

    def test_all_pass_gives_positive_score(self, scorer):
        results = [
            VerifiableResult.pass_("margin_constraint"),
            VerifiableResult.pass_("stockout_constraint"),
            VerifiableResult.pass_("revenue_target"),
        ]
        score = scorer.score(results)
        assert score == pytest.approx(+1.0)

    def test_all_fail_gives_negative_score(self, scorer):
        results = [
            VerifiableResult.fail_("margin_constraint"),
            VerifiableResult.fail_("stockout_constraint"),
            VerifiableResult.fail_("revenue_target"),
        ]
        score = scorer.score(results)
        assert score == pytest.approx(-1.0)

    def test_score_in_valid_range(self, scorer):
        """Mixed results must produce score in [-1, +1]."""
        results = [
            VerifiableResult.pass_("margin_constraint"),
            VerifiableResult.fail_("stockout_constraint"),
        ]
        score = scorer.score(results)
        assert -1.0 <= score <= 1.0

    def test_hard_constraint_weighted_higher(self, scorer):
        """
        Margin (weight=2.0) failure should hurt more than
        stockout (weight=1.0) failure when both fail.
        Test: margin pass + stockout fail vs margin fail + stockout pass.
        """
        pass_margin_fail_stockout = scorer.score([
            VerifiableResult.pass_("margin_constraint"),
            VerifiableResult.fail_("stockout_constraint"),
        ])
        fail_margin_pass_stockout = scorer.score([
            VerifiableResult.fail_("margin_constraint"),
            VerifiableResult.pass_("stockout_constraint"),
        ])
        # Failing margin (high weight) should produce a LOWER score
        assert fail_margin_pass_stockout < pass_margin_fail_stockout

    def test_empty_results_returns_zero(self, scorer):
        assert scorer.score([]) == pytest.approx(0.0)

    def test_all_hard_constraints_pass_when_all_pass(self, scorer):
        results = [
            VerifiableResult.pass_("margin_constraint"),
            VerifiableResult.pass_("price_change_constraint"),
        ]
        assert scorer.all_hard_constraints_pass(results)

    def test_all_hard_constraints_fail_when_margin_fails(self, scorer):
        results = [
            VerifiableResult.fail_("margin_constraint"),
            VerifiableResult.pass_("stockout_constraint"),
        ]
        assert not scorer.all_hard_constraints_pass(results)

    def test_get_failures_returns_correct_names(self, scorer):
        results = [
            VerifiableResult.pass_("margin_constraint"),
            VerifiableResult.fail_("stockout_constraint"),
            VerifiableResult.fail_("revenue_target"),
        ]
        failures = scorer.get_failures(results)
        assert "stockout_constraint" in failures
        assert "revenue_target"      in failures
        assert "margin_constraint"   not in failures


# ══════════════════════════════════════════
# Group advantage math tests (pure numpy)
# ══════════════════════════════════════════

class TestGroupAdvantages:
    """
    Test the group-relative advantage formula directly.
    These tests validate the GRPO math without needing torch.
    """

    def _compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """Replicate GRPOAgent.compute_group_advantages() in numpy."""
        mean = rewards.mean()
        std  = rewards.std() + 1e-8
        return (rewards - mean) / std

    def test_advantages_have_zero_mean(self):
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        adv = self._compute_advantages(rewards)
        assert abs(adv.mean()) < 1e-6

    def test_best_action_has_highest_advantage(self):
        rewards = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.3, -0.3, 0.8])
        adv = self._compute_advantages(rewards)
        best_idx = np.argmax(rewards)
        assert np.argmax(adv) == best_idx

    def test_worst_action_has_lowest_advantage(self):
        rewards = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.3, -0.3, 0.8])
        adv = self._compute_advantages(rewards)
        worst_idx = np.argmin(rewards)
        assert np.argmin(adv) == worst_idx

    def test_zero_gradient_when_all_rewards_equal(self):
        """
        If all G rewards are identical, std=0, advantages=0.
        Zero advantages → zero gradient. Policy doesn't update.
        This is the correct behavior — uniform reward = no information.
        """
        rewards = np.array([0.5] * 8)   # all identical
        adv = self._compute_advantages(rewards)
        # std ≈ 0, so (rewards - mean) / (std + ε) ≈ 0 / ε ≈ 0
        assert np.all(np.abs(adv) < 0.01)

    def test_manual_trace(self):
        """
        Manual calculation:
          rewards = [1, 3]
          mean = 2, std = 1
          adv_0 = (1-2)/(1+ε) ≈ -1
          adv_1 = (3-2)/(1+ε) ≈ +1
        """
        rewards = np.array([1.0, 3.0])
        adv = self._compute_advantages(rewards)
        assert adv[0] < 0   # below mean → negative advantage
        assert adv[1] > 0   # above mean → positive advantage
        assert abs(adv[0] + adv[1]) < 1e-5  # they sum to ~0

    def test_shape_preserved(self):
        rewards = np.random.randn(8).astype(np.float32)
        adv = self._compute_advantages(rewards)
        assert adv.shape == (8,)