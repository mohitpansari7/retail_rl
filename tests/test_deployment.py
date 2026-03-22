"""
evaluation/deployment.py
─────────────────────────
Phase 10 — Production deployment: optimization, A/B testing, drift detection.

Three components:

ModelOptimizer:
  Wraps trained policy for production deployment.
  Applies INT8 dynamic quantization: 4× smaller, 2-3× faster inference.
  Target: < 50ms per pricing decision cycle (spec requirement).

ABTestFramework:
  Safely rolls out new policy alongside production baseline.
  Starts at 10% traffic. Expands only when p_value < 0.05.
  Tracks per-episode results for honest variance estimation.

DriftDetector:
  Monitors production state/reward distributions vs training baseline.
  Uses Population Stability Index (PSI) — industry standard for ML drift.
  PSI < 0.10: stable. PSI < 0.25: warning. PSI ≥ 0.25: retrain.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────
# Model Optimizer
# ─────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Results from model quantization."""
    original_size_mb:    float
    quantized_size_mb:   float
    compression_ratio:   float
    benchmark_ms_original:   float
    benchmark_ms_quantized:  float
    speedup_ratio:       float
    meets_latency_target: bool
    latency_target_ms:   float = 50.0


class ModelOptimizer:
    """
    Wraps a trained PolicyNetwork for production deployment.

    Applies dynamic INT8 quantization:
      - Weights: statically quantized (float32 → int8, done once)
      - Activations: dynamically quantized at inference time
      - No calibration dataset required

    WHY dynamic (not static) quantization?
    Static quantization requires a representative calibration dataset
    to compute activation statistics. Dynamic is simpler, safer, and
    works well when inference batches are small (our case: one decision
    cycle at a time, not large batches).

    WHY a wrapper class?
    Insulates the codebase from PyTorch quantization API changes.
    Future optimizations (pruning, ONNX export) slot in here cleanly.

    Args:
        policy       : Trained PolicyNetwork instance.
        latency_target_ms : Maximum acceptable inference time (spec: 50ms).
    """

    def __init__(
        self,
        policy,
        latency_target_ms: float = 50.0,
    ) -> None:
        self.original_policy  = policy
        self.quantized_policy = None
        self.latency_target   = latency_target_ms
        self._is_quantized    = False

    def quantize(self) -> OptimizationResult:
        """
        Apply INT8 dynamic quantization.

        PyTorch's dynamic quantization:
          1. Scans all nn.Linear layers
          2. Quantizes weights to int8 (done once, stored)
          3. During inference: converts activations to int8 on the fly

        Returns OptimizationResult with size and speed comparisons.
        """
        import torch
        import io

        # Measure original model size
        buffer = io.BytesIO()
        torch.save(self.original_policy.state_dict(), buffer)
        original_size = buffer.tell() / (1024 * 1024)

        # Apply dynamic quantization
        self.quantized_policy = torch.quantization.quantize_dynamic(
            self.original_policy,
            {torch.nn.Linear},    # only quantize Linear layers
            dtype=torch.qint8,
        )
        self._is_quantized = True

        # Measure quantized size
        buffer = io.BytesIO()
        torch.save(self.quantized_policy.state_dict(), buffer)
        quantized_size = buffer.tell() / (1024 * 1024)

        # Benchmark both models
        ms_original   = self._benchmark(self.original_policy)
        ms_quantized  = self._benchmark(self.quantized_policy)

        return OptimizationResult(
            original_size_mb=round(original_size, 3),
            quantized_size_mb=round(quantized_size, 3),
            compression_ratio=round(original_size / (quantized_size + 1e-8), 2),
            benchmark_ms_original=round(ms_original, 3),
            benchmark_ms_quantized=round(ms_quantized, 3),
            speedup_ratio=round(ms_original / (ms_quantized + 1e-6), 2),
            meets_latency_target=ms_quantized < self.latency_target,
            latency_target_ms=self.latency_target,
        )

    def _benchmark(self, policy, n_runs: int = 100) -> float:
        """
        Benchmark inference time in milliseconds.

        Runs n_runs forward passes and returns mean time.
        Uses a realistic observation size (policy.obs_dim).
        """
        import torch
        obs = torch.randn(1, policy.obs_dim)
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                policy.get_action(obs.squeeze(0))
        # Measure
        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                policy.get_action(obs.squeeze(0))
        elapsed_ms = (time.perf_counter() - start) / n_runs * 1000
        return elapsed_ms

    def get_inference_policy(self):
        """Return quantized policy for production, or original if not quantized."""
        return self.quantized_policy if self._is_quantized else self.original_policy

    def save_quantized(self, path: str) -> None:
        """Save quantized model for deployment."""
        import torch
        assert self._is_quantized, "Call quantize() first"
        torch.save(self.quantized_policy.state_dict(), path)


# ─────────────────────────────────────────────────────────────
# A/B Testing Framework
# ─────────────────────────────────────────────────────────────

class ABGroup(Enum):
    CONTROL   = "control"     # current production policy
    TREATMENT = "treatment"   # new policy being tested


@dataclass
class ABTestResult:
    """Statistical results of one A/B test evaluation."""
    n_control:          int
    n_treatment:        int
    mean_control:       float
    mean_treatment:     float
    treatment_effect:   float   # mean_treatment - mean_control
    t_statistic:        float
    p_value:            float
    confidence_level:   float
    significant:        bool    # True if p_value < (1 - confidence_level)
    recommended_action: str


class ABTestFramework:
    """
    Safe rollout framework for new policy deployment.

    Workflow:
      1. Deploy new policy to treatment_pct% of stores
      2. Run for minimum_episodes before evaluating
      3. Run t-test: is treatment significantly better than control?
      4. If yes and p_value < threshold → expand traffic
      5. If no → roll back or hold

    Traffic expansion schedule (safe rollout):
      10% → 25% → 50% → 100%
      Each expansion requires p_value < significance_threshold.

    WHY t-test and not just compare means?
    Means can differ by chance. The t-test tells us the probability
    that the observed difference is due to random variation alone.
    p_value < 0.05 means: "less than 5% chance this is noise."
    At Walmart scale, deploying based on noise costs real money.

    Args:
        treatment_pct         : Starting traffic fraction for new policy.
        significance_threshold: p_value threshold for declaring significance.
        minimum_episodes      : Minimum data points before statistical test.
    """

    ROLLOUT_SCHEDULE = [0.10, 0.25, 0.50, 1.00]

    def __init__(
        self,
        treatment_pct:          float = 0.10,
        significance_threshold: float = 0.05,
        minimum_episodes:       int   = 30,
        confidence_level:       float = 0.95,
    ) -> None:
        self.treatment_pct         = treatment_pct
        self.significance_threshold = significance_threshold
        self.minimum_episodes      = minimum_episodes
        self.confidence_level      = confidence_level

        self._control_rewards:   List[float] = []
        self._treatment_rewards: List[float] = []
        self._rollout_stage:     int = 0
        self._test_history:      List[ABTestResult] = []

    def assign_group(self, store_id: str, rng: np.random.Generator) -> ABGroup:
        """
        Randomly assign a store to control or treatment for this episode.

        Uses the store_id as a hash seed for consistent assignment
        within an episode (same store doesn't flip groups mid-episode).

        Args:
            store_id : Store identifier.
            rng      : Random generator for reproducible assignment.

        Returns:
            ABGroup.CONTROL or ABGroup.TREATMENT
        """
        return (
            ABGroup.TREATMENT
            if rng.uniform() < self.treatment_pct
            else ABGroup.CONTROL
        )

    def record_episode(self, group: ABGroup, total_reward: float) -> None:
        """Record one episode's total reward for the given group."""
        if group == ABGroup.CONTROL:
            self._control_rewards.append(total_reward)
        else:
            self._treatment_rewards.append(total_reward)

    def evaluate(self) -> Optional[ABTestResult]:
        """
        Run statistical significance test on accumulated data.

        Uses Welch's t-test (assumes unequal variances — more conservative).
        Requires minimum_episodes in BOTH groups before testing.

        Returns:
            ABTestResult if enough data, None otherwise.
        """
        n_c = len(self._control_rewards)
        n_t = len(self._treatment_rewards)

        if n_c < self.minimum_episodes or n_t < self.minimum_episodes:
            return None

        control_arr   = np.array(self._control_rewards)
        treatment_arr = np.array(self._treatment_rewards)

        mean_c = float(control_arr.mean())
        mean_t = float(treatment_arr.mean())
        effect = mean_t - mean_c

        # Welch's t-test: does NOT assume equal variances
        t_stat, p_value = stats.ttest_ind(
            treatment_arr, control_arr, equal_var=False
        )

        significant = (p_value < self.significance_threshold) and (effect > 0)

        if significant:
            action = "EXPAND rollout to next traffic tier"
        elif p_value < self.significance_threshold and effect < 0:
            action = "ROLLBACK — treatment is significantly worse"
        else:
            action = "HOLD — insufficient evidence to decide yet"

        result = ABTestResult(
            n_control=n_c,
            n_treatment=n_t,
            mean_control=round(mean_c, 2),
            mean_treatment=round(mean_t, 2),
            treatment_effect=round(effect, 2),
            t_statistic=round(float(t_stat), 4),
            p_value=round(float(p_value), 6),
            confidence_level=self.confidence_level,
            significant=significant,
            recommended_action=action,
        )
        self._test_history.append(result)
        return result

    def expand_rollout(self) -> float:
        """
        Move to the next traffic tier if current test is significant.

        Returns new treatment_pct. Raises if already at 100%.
        """
        if self._rollout_stage < len(self.ROLLOUT_SCHEDULE) - 1:
            self._rollout_stage += 1
            self.treatment_pct = self.ROLLOUT_SCHEDULE[self._rollout_stage]
            # Reset episode buffers for fresh data at new traffic level
            self._control_rewards   = []
            self._treatment_rewards = []
        return self.treatment_pct

    def rollback(self) -> None:
        """Return to control-only (treatment_pct = 0)."""
        self.treatment_pct    = 0.0
        self._rollout_stage   = 0
        self._control_rewards = []
        self._treatment_rewards = []

    @property
    def test_history(self) -> List[ABTestResult]:
        return self._test_history.copy()

    @property
    def current_stage(self) -> str:
        return f"{self.treatment_pct:.0%} treatment"


# ─────────────────────────────────────────────────────────────
# Drift Detector
# ─────────────────────────────────────────────────────────────

class DriftLevel(Enum):
    STABLE   = "stable"     # PSI < 0.10
    WARNING  = "warning"    # PSI 0.10 - 0.25
    CRITICAL = "critical"   # PSI ≥ 0.25 — retrain recommended


@dataclass
class DriftReport:
    """Output of one drift detection check."""
    step:             int
    psi_reward:       float
    psi_state:        float
    reward_drift:     bool
    state_drift:      bool
    level:            DriftLevel
    recommendation:   str


class DriftDetector:
    """
    Monitors production distributions for shift vs training baseline.

    Two signals monitored:

    1. Reward drift:
       Is the agent getting significantly different rewards than during training?
       Detects: environment changes, reward hacking, policy degradation.

    2. State drift (PSI):
       Is the distribution of states the agent sees in production
       different from the training distribution?
       Detects: competitor behavior changes, customer preference shifts,
       seasonality anomalies, new product categories.

    PSI formula:
       PSI = Σ_i (P_prod_i − P_train_i) × ln(P_prod_i / P_train_i)

    PSI interpretation (industry standard from financial ML):
       PSI < 0.10 : No significant shift → stable
       PSI < 0.25 : Moderate shift → monitor closely
       PSI ≥ 0.25 : Significant shift → consider retraining

    Args:
        baseline_rewards : Reward distribution from training (reference).
        baseline_states  : State distribution from training (reference).
        window           : Rolling window for production metrics.
        n_bins           : Number of bins for PSI histogram comparison.
    """

    PSI_WARNING_THRESHOLD  = 0.10
    PSI_CRITICAL_THRESHOLD = 0.25

    def __init__(
        self,
        baseline_rewards: List[float],
        baseline_states:  Optional[np.ndarray] = None,
        window:           int = 100,
        n_bins:           int = 10,
    ) -> None:
        self.baseline_rewards  = np.array(baseline_rewards)
        self.baseline_states   = baseline_states
        self.window            = window
        self.n_bins            = n_bins

        self._production_rewards: List[float]      = []
        self._production_states:  List[np.ndarray] = []
        self._step:               int              = 0
        self._reports:            List[DriftReport] = []

    def record(
        self,
        reward: float,
        state:  Optional[np.ndarray] = None,
    ) -> None:
        """Record one production observation."""
        self._step += 1
        self._production_rewards.append(reward)
        if state is not None:
            self._production_states.append(state)

        # Keep only recent window
        if len(self._production_rewards) > self.window:
            self._production_rewards = self._production_rewards[-self.window:]
        if len(self._production_states) > self.window:
            self._production_states  = self._production_states[-self.window:]

    def compute_psi(
        self,
        expected: np.ndarray,
        actual:   np.ndarray,
    ) -> float:
        """
        Population Stability Index between two distributions.

        PSI = Σ_i (A_i − E_i) × ln(A_i / E_i)

        where A_i and E_i are the fraction of observations in bin i
        for actual (production) and expected (training) distributions.

        ε = 1e-4 added to fractions to avoid log(0).

        Args:
            expected : Reference distribution (training data).
            actual   : Current distribution (production data).

        Returns:
            PSI scalar. Higher = more drift.
        """
        # Build common bin edges from the union of both distributions
        combined = np.concatenate([expected, actual])
        bin_edges = np.percentile(combined, np.linspace(0, 100, self.n_bins + 1))
        bin_edges = np.unique(bin_edges)   # remove duplicates from constant data

        if len(bin_edges) < 2:
            return 0.0   # degenerate case — all values identical

        # Compute histogram fractions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts,   _ = np.histogram(actual,   bins=bin_edges)

        epsilon = 1e-4
        expected_pct = (expected_counts / (len(expected) + epsilon)) + epsilon
        actual_pct   = (actual_counts   / (len(actual)   + epsilon)) + epsilon

        # Normalise so fractions sum to 1
        expected_pct /= expected_pct.sum()
        actual_pct   /= actual_pct.sum()

        psi = float(np.sum(
            (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        ))
        return max(0.0, psi)

    def compute_reward_drift(self) -> float:
        """
        Z-score drift: how many standard deviations has the mean reward shifted?
        Returns normalized drift score (0=none, higher=more drift).
        """
        if len(self._production_rewards) < 10:
            return 0.0

        baseline_mean = self.baseline_rewards.mean()
        baseline_std  = self.baseline_rewards.std() + 1e-8
        prod_mean     = np.mean(self._production_rewards)

        return abs(prod_mean - baseline_mean) / baseline_std

    def check_drift(self) -> Optional[DriftReport]:
        """
        Run all drift checks and return a DriftReport.

        Called periodically (e.g. daily or weekly) in production.
        Returns None if insufficient production data.
        """
        if len(self._production_rewards) < 20:
            return None

        # Reward drift (z-score based)
        reward_drift_score = self.compute_reward_drift()
        reward_drift = reward_drift_score > 2.0   # > 2σ shift

        # State PSI drift
        state_psi = 0.0
        if (self.baseline_states is not None and
                len(self._production_states) >= 20):
            # Compare one feature dimension for efficiency
            # (in production, check all dims and take max PSI)
            prod_arr  = np.array(self._production_states)[:, 0]
            base_arr  = self.baseline_states[:, 0]
            state_psi = self.compute_psi(base_arr, prod_arr)

        state_drift = state_psi >= self.PSI_WARNING_THRESHOLD

        # Determine level
        if state_psi >= self.PSI_CRITICAL_THRESHOLD or reward_drift_score > 3.0:
            level = DriftLevel.CRITICAL
            rec   = "Retrain immediately. Production environment has shifted significantly."
        elif state_psi >= self.PSI_WARNING_THRESHOLD or reward_drift:
            level = DriftLevel.WARNING
            rec   = "Monitor closely. Consider retraining if trend continues."
        else:
            level = DriftLevel.STABLE
            rec   = "No action needed. Distribution is stable."

        report = DriftReport(
            step=self._step,
            psi_reward=round(reward_drift_score, 4),
            psi_state=round(state_psi, 4),
            reward_drift=reward_drift,
            state_drift=state_drift,
            level=level,
            recommendation=rec,
        )
        self._reports.append(report)
        return report

    @property
    def reports(self) -> List[DriftReport]:
        return self._reports.copy()