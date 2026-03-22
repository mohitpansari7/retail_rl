"""
tests/test_safety.py
─────────────────────
Phase 9 tests. Run with: pytest tests/test_safety.py -v

Test philosophy:
  - Test ConstraintProjector correctly flags boundary exploitation
  - Test margin violation logging is accurate
  - Test Lagrangian λ increases on violation, decreases on satisfaction
  - Test λ is capped at lambda_max (no collapse)
  - Test λ never goes negative (max(0,...) enforced)
  - Test each hacking detector fires on the right pattern
  - Test normal behavior produces no alerts
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from safety.constraints import (
    ConstraintProjector, LagrangianConstraintOptimizer,
    SoftConstraintConfig, ConstraintViolation,
)
from safety.reward_hacking import (
    RewardHackingDetector, AlertLevel, HackingAlert,
)


# ══════════════════════════════════════════
# ConstraintProjector tests
# ══════════════════════════════════════════

class TestConstraintProjector:

    @pytest.fixture
    def proj(self):
        return ConstraintProjector()

    def test_clean_action_no_violation(self, proj):
        action = np.array([0.1, -0.2, 0.3, 0.0, -0.1])
        safe, violations = proj.project_price_action(action)
        assert len(violations) == 0

    def test_boundary_exploitation_detected(self, proj):
        """More than 50% of actions at ≥90% boundary → violation flagged."""
        action = np.array([0.95, -0.92, 0.98, -0.91, 0.94])   # all at boundary
        safe, violations = proj.project_price_action(action)
        assert len(violations) == 1
        assert violations[0].constraint == "boundary_exploitation"

    def test_action_clipped_to_minus_one_plus_one(self, proj):
        action = np.array([1.5, -1.5, 2.0, -0.5])
        safe, _ = proj.project_price_action(action)
        assert np.all(safe <= 1.0)
        assert np.all(safe >= -1.0)

    def test_violation_logged(self, proj):
        action = np.array([0.95, 0.97, 0.96, 0.98, 0.99])
        proj.project_price_action(action)
        assert proj.get_violation_count() == 1

    def test_clean_action_not_logged(self, proj):
        action = np.array([0.1, 0.2, -0.1, 0.0])
        proj.project_price_action(action)
        assert proj.get_violation_count() == 0

    def test_margin_violation_logged(self, proj):
        prices = np.array([103.0, 110.0])   # 103 < 105 (floor for cost=100)
        costs  = np.array([100.0, 100.0])
        violations = proj.check_margin_constraint(prices, costs)
        assert len(violations) == 1
        assert violations[0].constraint == "margin_constraint"

    def test_margin_no_violation_above_floor(self, proj):
        prices = np.array([110.0, 120.0])
        costs  = np.array([100.0, 100.0])
        violations = proj.check_margin_constraint(prices, costs)
        assert len(violations) == 0

    def test_consecutive_boundary_hits_tracked(self, proj):
        action = np.array([0.95, 0.97, 0.96, 0.98, 0.99])
        proj.project_price_action(action, store_id="store_0")
        proj.project_price_action(action, store_id="store_0")
        assert proj.get_consecutive_boundary_hits("store_0") == 2

    def test_consecutive_hits_reset_on_clean_action(self, proj):
        bad_action   = np.array([0.95, 0.97, 0.96, 0.98, 0.99])
        clean_action = np.array([0.1, 0.2, -0.1, 0.0, 0.05])
        proj.project_price_action(bad_action,   store_id="store_0")
        proj.project_price_action(clean_action, store_id="store_0")
        assert proj.get_consecutive_boundary_hits("store_0") == 0

    def test_reset_log_clears_violations(self, proj):
        action = np.array([0.95, 0.97, 0.96, 0.98, 0.99])
        proj.project_price_action(action)
        proj.reset_log()
        assert proj.get_violation_count() == 0


# ══════════════════════════════════════════
# LagrangianConstraintOptimizer tests
# ══════════════════════════════════════════

class TestLagrangianOptimizer:

    @pytest.fixture
    def optimizer(self):
        return LagrangianConstraintOptimizer(constraints=[
            SoftConstraintConfig(
                name="stockout_rate",
                threshold=0.02,
                lambda_init=0.0,
                lambda_lr=0.1,
                lambda_max=10.0,
            ),
            SoftConstraintConfig(
                name="price_smoothness",
                threshold=0.05,
                lambda_init=0.0,
                lambda_lr=0.1,
                lambda_max=10.0,
            ),
        ])

    def test_lambda_increases_on_violation(self, optimizer):
        """Constraint violated → λ must increase."""
        old_lambda = optimizer.lambdas["stockout_rate"]
        optimizer.update_lambdas({"stockout_rate": 0.05})   # 0.05 > threshold 0.02
        assert optimizer.lambdas["stockout_rate"] > old_lambda

    def test_lambda_decreases_on_satisfaction(self, optimizer):
        """Constraint satisfied → λ must decrease (or stay at 0)."""
        # First raise lambda
        optimizer.lambdas["stockout_rate"] = 5.0
        old_lambda = optimizer.lambdas["stockout_rate"]
        optimizer.update_lambdas({"stockout_rate": 0.01})   # 0.01 < threshold 0.02
        assert optimizer.lambdas["stockout_rate"] < old_lambda

    def test_lambda_never_goes_negative(self, optimizer):
        """max(0,...) must hold — λ must always be ≥ 0."""
        optimizer.lambdas["stockout_rate"] = 0.0
        # Large satisfaction → would push λ negative without max(0)
        for _ in range(20):
            optimizer.update_lambdas({"stockout_rate": 0.0})
        assert optimizer.lambdas["stockout_rate"] >= 0.0

    def test_lambda_capped_at_lambda_max(self, optimizer):
        """min(λ_max,...) must hold — λ must never exceed lambda_max."""
        for _ in range(100):
            optimizer.update_lambdas({"stockout_rate": 1.0})   # constant severe violation
        assert optimizer.lambdas["stockout_rate"] <= 10.0   # lambda_max=10

    def test_penalty_zero_when_all_satisfied(self, optimizer):
        """No violation → penalty should be 0."""
        penalty = optimizer.compute_penalty({
            "stockout_rate": 0.01,    # below threshold 0.02
            "price_smoothness": 0.03, # below threshold 0.05
        })
        assert penalty == pytest.approx(0.0)

    def test_penalty_positive_when_violated(self, optimizer):
        """Violation → penalty must be positive."""
        optimizer.lambdas["stockout_rate"] = 5.0   # non-zero lambda
        penalty = optimizer.compute_penalty({
            "stockout_rate": 0.10,    # well above threshold 0.02
        })
        assert penalty > 0.0

    def test_higher_lambda_means_higher_penalty(self, optimizer):
        """Doubling λ should double the penalty for the same violation."""
        optimizer.lambdas["stockout_rate"] = 1.0
        p1 = optimizer.compute_penalty({"stockout_rate": 0.10})
        optimizer.lambdas["stockout_rate"] = 2.0
        p2 = optimizer.compute_penalty({"stockout_rate": 0.10})
        assert p2 == pytest.approx(2 * p1, rel=1e-5)

    def test_active_constraints_returns_high_lambda_names(self, optimizer):
        optimizer.lambdas["stockout_rate"]   = 5.0
        optimizer.lambdas["price_smoothness"] = 0.5
        active = optimizer.get_active_constraints(threshold=1.0)
        assert "stockout_rate"    in active
        assert "price_smoothness" not in active

    def test_manual_lambda_update_trace(self, optimizer):
        """
        Manual trace:
          λ_0 = 0, α = 0.1, C = 0.05, d = 0.02
          violation = 0.05 - 0.02 = 0.03
          λ_1 = max(0, min(10, 0 + 0.1 × 0.03)) = 0.003
        """
        optimizer.update_lambdas({"stockout_rate": 0.05})
        expected = max(0.0, min(10.0, 0.0 + 0.1 * (0.05 - 0.02)))
        assert optimizer.lambdas["stockout_rate"] == pytest.approx(expected, rel=1e-5)

    def test_history_recorded(self, optimizer):
        optimizer.update_lambdas({"stockout_rate": 0.05})
        optimizer.update_lambdas({"stockout_rate": 0.01})
        history = optimizer.get_lambda_history()
        assert len(history) == 2
        assert "lambdas" in history[0]
        assert "constraint_values" in history[0]


# ══════════════════════════════════════════
# RewardHackingDetector tests
# ══════════════════════════════════════════

class TestRewardHackingDetector:

    @pytest.fixture
    def detector(self):
        return RewardHackingDetector(
            smoothness_threshold=0.10,
            drain_threshold=0.05,
            revenue_cv_min=0.02,
            window=10,
        )

    def test_no_alerts_on_normal_behavior(self, detector):
        """Normal pricing behavior should not trigger any alerts."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            old_p = rng.uniform(90, 110, 20)
            new_p = old_p * rng.uniform(0.98, 1.02, 20)   # ±2% changes
            detector.record_price_changes(old_p, new_p)

            inv_before = rng.uniform(80, 100, 20)
            sold       = rng.uniform(5, 10, 20)
            inv_after  = np.maximum(0, inv_before - sold)
            detector.record_inventory(inv_before, inv_after, sold)

            revenues = {"s0": 1000, "s1": 900, "s2": 1100, "s3": 850}
            detector.record_revenues(revenues)

        alerts = detector.check_all()
        assert len(alerts) == 0

    def test_boundary_exploitation_detected(self, detector):
        """Daily ±15% price changes should trigger smoothness alert."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            old_p = rng.uniform(90, 110, 20)
            new_p = old_p * rng.uniform(1.13, 1.15, 20)   # 13-15% changes EVERY day
            detector.record_price_changes(old_p, new_p)
        alerts = detector.check_all()
        assert any(a.hack_type == "boundary_exploitation" for a in alerts)

    def test_inventory_drain_detected(self, detector):
        """Inventory falling much faster than units_sold → drain alert."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            inv_before = np.full(20, 100.0)
            sold       = np.full(20, 5.0)
            # Inventory drops 30 units but only 5 were sold → 25 "vanished"
            inv_after  = np.full(20, 70.0)
            detector.record_inventory(inv_before, inv_after, sold)
        alerts = detector.check_all()
        assert any(a.hack_type == "inventory_drain" for a in alerts)

    def test_revenue_equalization_detected(self, detector):
        """Suspiciously uniform revenues across stores → equalization warning."""
        for _ in range(10):
            # All stores earn exactly the same — highly suspicious
            revenues = {"s0": 1000.0, "s1": 1000.0, "s2": 1000.0, "s3": 1000.0}
            detector.record_revenues(revenues)
        alerts = detector.check_all()
        assert any(a.hack_type == "revenue_equalization" for a in alerts)

    def test_critical_level_on_severe_smoothness(self, detector):
        """2× threshold breach should produce CRITICAL alert."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            old_p = rng.uniform(90, 110, 20)
            new_p = old_p * 1.25   # 25% daily change — way above 10% threshold
            detector.record_price_changes(old_p, new_p)
        alerts = detector.check_all()
        smoothness_alerts = [a for a in alerts if a.hack_type == "boundary_exploitation"]
        assert any(a.level == AlertLevel.CRITICAL for a in smoothness_alerts)

    def test_alert_count_correct(self, detector):
        rng = np.random.default_rng(0)
        for _ in range(10):
            old_p = rng.uniform(90, 110, 5)
            new_p = old_p * 1.14
            detector.record_price_changes(old_p, new_p)
        detector.check_all()
        detector.check_all()   # call twice
        assert detector.get_alert_count(hack_type="boundary_exploitation") == 2

    def test_summary_returns_all_metrics(self, detector):
        summary = detector.summary()
        assert "smoothness_score" in summary
        assert "drain_rate"       in summary
        assert "revenue_cv"       in summary
        assert "total_alerts"     in summary

    def test_smoothness_score_correct(self, detector):
        """10% daily changes → smoothness score ≈ 0.10."""
        for _ in range(10):
            old_p = np.array([100.0, 100.0])
            new_p = np.array([110.0, 110.0])   # exactly 10%
            detector.record_price_changes(old_p, new_p)
        score = detector.compute_smoothness_score()
        assert score == pytest.approx(0.10, rel=1e-3)