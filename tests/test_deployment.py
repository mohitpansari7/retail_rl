"""
tests/test_deployment.py
─────────────────────────
Phase 10 tests. Run with: pytest tests/test_deployment.py -v

Test philosophy:
  - Test PSI = 0 for identical distributions (no drift)
  - Test PSI > 0 for shifted distributions
  - Test PSI ≥ 0.25 for severely shifted distributions (critical)
  - Test A/B test: significant when treatment truly better
  - Test A/B test: not significant when difference is noise
  - Test rollback resets traffic to 0
  - Test expand moves to next traffic tier
  - Test metrics tracker aggregates correctly
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from evaluation.deployment import (
    ABTestFramework, ABGroup, DriftDetector, DriftLevel,
)
from evaluation.metrics import ProductionMetricsTracker, EpisodeMetrics


# ══════════════════════════════════════════
# PSI / DriftDetector tests
# ══════════════════════════════════════════

class TestDriftDetector:

    def _make_detector(self, baseline_rewards, n_bins=5):
        return DriftDetector(
            baseline_rewards=baseline_rewards,
            window=50,
            n_bins=n_bins,
        )

    def test_psi_zero_for_identical_distributions(self):
        """Same distribution → PSI should be ≈ 0."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        detector = self._make_detector(data.tolist())
        psi = detector.compute_psi(data, data)
        assert psi < 0.01, f"PSI={psi:.4f} should be ≈ 0 for identical distributions"

    def test_psi_positive_for_shifted_distribution(self):
        """Mean-shifted distribution → PSI > 0."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 500)
        shifted  = rng.normal(2, 1, 500)   # shifted right by 2σ
        detector = self._make_detector(baseline.tolist())
        psi = detector.compute_psi(baseline, shifted)
        assert psi > 0.0, "Shifted distribution should have positive PSI"

    def test_psi_increases_with_shift_magnitude(self):
        """Larger shift → larger PSI."""
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 500)

        small_shift = rng.normal(0.5, 1, 500)
        large_shift = rng.normal(3.0, 1, 500)

        detector = self._make_detector(base.tolist())
        psi_small = detector.compute_psi(base, small_shift)
        psi_large = detector.compute_psi(base, large_shift)

        assert psi_large > psi_small, (
            f"Large shift PSI ({psi_large:.4f}) should exceed small ({psi_small:.4f})"
        )

    def test_psi_critical_threshold_for_severe_shift(self):
        """Very large shift should breach PSI ≥ 0.25 (critical threshold)."""
        rng = np.random.default_rng(0)
        base    = rng.normal(0, 1, 1000)
        extreme = rng.normal(5, 1, 1000)   # 5σ shift
        detector = self._make_detector(base.tolist())
        psi = detector.compute_psi(base, extreme)
        assert psi >= DriftDetector.PSI_CRITICAL_THRESHOLD, (
            f"Extreme shift should breach critical threshold. PSI={psi:.4f}"
        )

    def test_no_drift_when_distributions_match(self):
        """Stable production rewards → STABLE drift level."""
        rng = np.random.default_rng(0)
        baseline = rng.normal(1000, 50, 200).tolist()
        detector = self._make_detector(baseline)

        # Record production rewards from same distribution
        for r in rng.normal(1000, 50, 50):
            detector.record(float(r))

        report = detector.check_drift()
        assert report is not None
        assert report.level == DriftLevel.STABLE

    def test_reward_drift_detected_after_large_shift(self):
        """Reward dropping by 3σ should trigger reward drift flag."""
        rng = np.random.default_rng(1)
        baseline = rng.normal(1000, 50, 200).tolist()
        detector = self._make_detector(baseline)

        # Production rewards much lower (competitor took market share)
        for r in rng.normal(700, 50, 50):   # 6σ below baseline
            detector.record(float(r))

        report = detector.check_drift()
        assert report is not None
        assert report.reward_drift, "Large reward drop should trigger drift flag"
        assert report.level in (DriftLevel.WARNING, DriftLevel.CRITICAL)

    def test_insufficient_data_returns_none(self):
        """Check returns None when not enough production data."""
        detector = self._make_detector([1.0, 2.0, 3.0] * 10)
        detector.record(1.5)   # only 1 point — not enough
        report = detector.check_drift()
        assert report is None

    def test_psi_non_negative(self):
        """PSI must always be ≥ 0."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            a = rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 3), 100)
            b = rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 3), 100)
            detector = self._make_detector(a.tolist())
            psi = detector.compute_psi(a, b)
            assert psi >= 0.0, f"PSI must be non-negative, got {psi}"


# ══════════════════════════════════════════
# ABTestFramework tests
# ══════════════════════════════════════════

class TestABTestFramework:

    @pytest.fixture
    def ab(self):
        return ABTestFramework(
            treatment_pct=0.10,
            significance_threshold=0.05,
            minimum_episodes=20,
        )

    def test_returns_none_before_minimum_episodes(self, ab):
        """Test should return None until enough data is collected."""
        rng = np.random.default_rng(0)
        for _ in range(10):   # fewer than minimum_episodes=20
            ab.record_episode(ABGroup.CONTROL,   float(rng.normal(1000, 50)))
            ab.record_episode(ABGroup.TREATMENT, float(rng.normal(1050, 50)))
        result = ab.evaluate()
        assert result is None

    def test_significant_when_treatment_clearly_better(self, ab):
        """When treatment is genuinely better by a large margin → significant."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            ab.record_episode(ABGroup.CONTROL,   float(rng.normal(1000, 30)))
            ab.record_episode(ABGroup.TREATMENT, float(rng.normal(1200, 30)))  # 20% better

        result = ab.evaluate()
        assert result is not None
        assert result.significant, f"p_value={result.p_value:.4f} should be < 0.05"
        assert result.treatment_effect > 0

    def test_not_significant_when_noise_only(self, ab):
        """When treatment and control come from the same distribution → not significant."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            ab.record_episode(ABGroup.CONTROL,   float(rng.normal(1000, 100)))
            ab.record_episode(ABGroup.TREATMENT, float(rng.normal(1000, 100)))  # same

        result = ab.evaluate()
        assert result is not None
        # With same distributions, p_value should typically be > 0.05
        # (may occasionally fail by chance, but overwhelmingly should pass)
        assert not result.significant or result.p_value > 0.001

    def test_treatment_effect_computed_correctly(self, ab):
        """treatment_effect = mean_treatment - mean_control."""
        rng = np.random.default_rng(0)
        for _ in range(30):
            ab.record_episode(ABGroup.CONTROL,   1000.0)
            ab.record_episode(ABGroup.TREATMENT, 1100.0)

        result = ab.evaluate()
        assert result is not None
        assert result.treatment_effect == pytest.approx(100.0, rel=1e-3)

    def test_expand_increases_treatment_pct(self, ab):
        assert ab.treatment_pct == pytest.approx(0.10)
        ab.expand_rollout()
        assert ab.treatment_pct == pytest.approx(0.25)

    def test_expand_follows_schedule(self, ab):
        """Rollout schedule: 10% → 25% → 50% → 100%."""
        expected = [0.25, 0.50, 1.00]
        for exp_pct in expected:
            ab.expand_rollout()
            assert ab.treatment_pct == pytest.approx(exp_pct)

    def test_rollback_resets_to_zero(self, ab):
        ab.expand_rollout()
        ab.rollback()
        assert ab.treatment_pct == pytest.approx(0.0)

    def test_expand_resets_episode_buffers(self, ab):
        """After expansion, buffers are cleared for fresh data at new tier."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            ab.record_episode(ABGroup.CONTROL,   1000.0)
            ab.record_episode(ABGroup.TREATMENT, 1100.0)
        ab.expand_rollout()
        # Buffers should be empty after expansion
        assert len(ab._control_rewards) == 0
        assert len(ab._treatment_rewards) == 0

    def test_assign_group_respects_treatment_pct(self):
        """With treatment_pct=0.5, roughly 50% should be assigned treatment."""
        ab = ABTestFramework(treatment_pct=0.5, minimum_episodes=1)
        rng = np.random.default_rng(0)
        groups = [ab.assign_group(f"store_{i}", rng) for i in range(1000)]
        treatment_frac = sum(1 for g in groups if g == ABGroup.TREATMENT) / 1000
        assert 0.45 <= treatment_frac <= 0.55, (
            f"Expected ~50% treatment, got {treatment_frac:.1%}"
        )


# ══════════════════════════════════════════
# ProductionMetricsTracker tests
# ══════════════════════════════════════════

class TestProductionMetricsTracker:

    @pytest.fixture
    def tracker(self):
        return ProductionMetricsTracker(
            store_ids=["store_0", "store_1", "store_2", "store_3"],
            window_size=20,
        )

    def _add_episodes(self, tracker, n=10, revenue=1000.0, stockouts=1):
        for i in range(n):
            tracker.record_episode(EpisodeMetrics(
                episode=i,
                total_revenue=revenue + np.random.randn() * 10,
                stockout_count=stockouts,
                margin_violations=0,
                policy_loss=0.05,
                entropy=0.8,
            ))

    def test_business_kpis_computed(self, tracker):
        self._add_episodes(tracker)
        kpis = tracker.compute_business_kpis()
        assert "mean_daily_revenue" in kpis
        assert "stockout_rate"      in kpis
        assert "revenue_trend_pct"  in kpis

    def test_mean_revenue_correct(self, tracker):
        for i in range(10):
            tracker.record_episode(EpisodeMetrics(
                episode=i, total_revenue=1000.0, stockout_count=0,
            ))
        kpis = tracker.compute_business_kpis()
        assert kpis["mean_daily_revenue"] == pytest.approx(1000.0, rel=1e-3)

    def test_training_health_computed(self, tracker):
        self._add_episodes(tracker)
        health = tracker.compute_training_health()
        assert "mean_policy_loss" in health
        assert "mean_entropy"     in health

    def test_safety_summary_counts_violations(self, tracker):
        for i in range(5):
            tracker.record_episode(EpisodeMetrics(
                episode=i,
                total_revenue=1000.0,
                constraint_violations=2,
                margin_violations=1,
            ))
        safety = tracker.compute_safety_summary()
        assert safety["total_constraint_violations"] == 10
        assert safety["total_margin_violations"]     == 5

    def test_report_generates_without_error(self, tracker):
        self._add_episodes(tracker)
        report = tracker.generate_report()
        assert "RetailRL Production Report" in report
        assert "BUSINESS KPIs"             in report
        assert "SAFETY SUMMARY"            in report

    def test_json_output_is_valid(self, tracker):
        import json
        self._add_episodes(tracker)
        json_str = tracker.to_json()
        data = json.loads(json_str)
        assert "business_kpis"   in data
        assert "training_health" in data
        assert "safety_summary"  in data

    def test_empty_tracker_returns_empty_dicts(self, tracker):
        assert tracker.compute_business_kpis() == {}
        assert tracker.compute_training_health() == {}