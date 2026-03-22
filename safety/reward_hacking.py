"""
safety/reward_hacking.py
─────────────────────────
Phase 9 — Layer 3: Reward hacking detection and intervention.

Three metrics that reliably detect the documented reward hacking patterns:

  Hack 1 — Boundary exploitation:
    Agent hits ±15% price limit every day.
    Detector: price_smoothness_score = mean(|Δp_t / p_{t-1}|)
    Alert when: score > 0.10 (avg daily change > 10%)

  Hack 2 — Inventory drain:
    Agent liquidates stock for short-term revenue.
    Detector: drain_rate = (I_0 − I_T) / (I_0 × T)
    Alert when: drain_rate > 0.05 (losing >5% stock/day beyond sales)

  Hack 3 — Artificial metric gaming:
    Agent equalizes store revenues artificially.
    Detector: revenue_cv = std(revenues) / mean(revenues)
    Alert when: cv < 0.02 (suspiciously uniform revenues)

WHY monitor outputs, not weights?
  1. Weights have no business meaning — outputs do.
  2. Hacking is emergent behavior, distributed across all weights.
     No single weight looks suspicious; the output pattern is unmistakable.
  3. Business analysts can read output metrics, not weight tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class AlertLevel(Enum):
    NORMAL   = "normal"
    WARNING  = "warning"    # metric drifting toward threshold
    ALERT    = "alert"      # threshold breached — investigate
    CRITICAL = "critical"   # severe, automatic intervention triggered


@dataclass
class HackingAlert:
    """
    One reward hacking alert event.

    Stored in alert log and (in production) sent to monitoring system.
    """
    step:         int
    hack_type:    str
    level:        AlertLevel
    metric_value: float
    threshold:    float
    detail:       str
    recommended_action: str


class RewardHackingDetector:
    """
    Monitors agent behavior for the three documented reward hacking patterns.

    Usage:
        detector = RewardHackingDetector()

        # After each step:
        detector.record_price_changes(old_prices, new_prices, store_id)
        detector.record_inventory(inventory_levels, units_sold, store_id)
        detector.record_revenues(revenues_per_store)

        # After each episode:
        alerts = detector.check_all()
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                # Trigger intervention
    """

    def __init__(
        self,
        smoothness_threshold: float = 0.10,   # avg daily change > 10% = suspicious
        drain_threshold:      float = 0.05,   # >5%/day inventory loss = suspicious
        revenue_cv_min:       float = 0.02,   # cv < 2% = suspiciously uniform
        window:               int   = 30,     # days to look back
    ) -> None:
        self.smoothness_threshold = smoothness_threshold
        self.drain_threshold      = drain_threshold
        self.revenue_cv_min       = revenue_cv_min
        self.window               = window

        # Rolling buffers (one entry per step)
        self._price_changes:    List[float] = []
        self._inventory_deltas: List[float] = []
        self._units_sold:       List[float] = []
        self._revenues:         List[List[float]] = []   # per store per step

        self._step: int = 0
        self._alert_log: List[HackingAlert] = []

    # ── Data recording ────────────────────────────────────────

    def record_price_changes(
        self,
        old_prices: np.ndarray,
        new_prices: np.ndarray,
        store_id: str = "store_0",
    ) -> None:
        """
        Record absolute price change fraction this step.
        Uses mean over all SKUs so outliers don't dominate.
        """
        self._step += 1
        old = np.array(old_prices, dtype=float)
        new = np.array(new_prices, dtype=float)
        pct_changes = np.abs(new - old) / (old + 1e-8)
        self._price_changes.append(float(pct_changes.mean()))

        # Keep only recent window
        if len(self._price_changes) > self.window:
            self._price_changes = self._price_changes[-self.window:]

    def record_inventory(
        self,
        inventory_before: np.ndarray,
        inventory_after:  np.ndarray,
        units_sold:       np.ndarray,
    ) -> None:
        """
        Record inventory delta BEYOND what was sold.

        If inventory fell more than units_sold, something is wrong
        (artificial transfers, data manipulation, etc.).
        """
        before = np.array(inventory_before, dtype=float)
        after  = np.array(inventory_after,  dtype=float)
        sold   = np.array(units_sold,       dtype=float)

        actual_delta   = float((before - after).sum())
        expected_delta = float(sold.sum())

        # How much inventory fell beyond legitimate sales?
        excess_drain = max(0.0, actual_delta - expected_delta)
        total_inv    = float(before.sum()) + 1e-8
        drain_rate   = excess_drain / total_inv

        self._inventory_deltas.append(drain_rate)
        self._units_sold.append(float(sold.sum()))

        if len(self._inventory_deltas) > self.window:
            self._inventory_deltas = self._inventory_deltas[-self.window:]
            self._units_sold       = self._units_sold[-self.window:]

    def record_revenues(self, revenues_per_store: Dict[str, float]) -> None:
        """
        Record per-store revenues this step.
        Used to detect artificial revenue equalization.
        """
        self._revenues.append(list(revenues_per_store.values()))
        if len(self._revenues) > self.window:
            self._revenues = self._revenues[-self.window:]

    # ── Detection ─────────────────────────────────────────────

    def compute_smoothness_score(self) -> float:
        """
        Average absolute daily price change over the recent window.

        Normal operation: 2-5% average (occasional large moves)
        Boundary exploitation: 10-15% consistently
        """
        if not self._price_changes:
            return 0.0
        return float(np.mean(self._price_changes))

    def compute_drain_rate(self) -> float:
        """
        Average excess inventory drain rate (beyond legitimate sales).

        Normal: 0-2% (reordering lags, spoilage, etc.)
        Drain hacking: 5%+ consistently
        """
        if not self._inventory_deltas:
            return 0.0
        return float(np.mean(self._inventory_deltas))

    def compute_revenue_cv(self) -> float:
        """
        Coefficient of variation of per-store revenues.

        Normal: 5-20% (stores genuinely differ in performance)
        Artificial equalization: < 2% (suspiciously uniform)

        WHY LOW CV is suspicious (not high)?
        Stores in different regions SHOULD have different revenues.
        Urban stores earn more than rural. If all stores suddenly earn
        the same amount — that's the signal of artificial balancing.
        """
        if not self._revenues:
            return 1.0   # unknown → assume normal
        recent = np.array(self._revenues[-self.window:])
        means  = recent.mean(axis=0)    # mean revenue per store
        overall_mean = means.mean() + 1e-8
        cv = float(means.std() / overall_mean)
        return cv

    def check_all(self) -> List[HackingAlert]:
        """
        Run all three detectors and return any alerts raised.

        Called after each episode or at regular intervals.
        Returns list of HackingAlert — empty if no issues detected.
        """
        alerts = []

        # ── Detector 1: Boundary exploitation
        smoothness = self.compute_smoothness_score()
        if smoothness > self.smoothness_threshold:
            level = (
                AlertLevel.CRITICAL if smoothness > self.smoothness_threshold * 2
                else AlertLevel.ALERT
            )
            alert = HackingAlert(
                step=self._step,
                hack_type="boundary_exploitation",
                level=level,
                metric_value=smoothness,
                threshold=self.smoothness_threshold,
                detail=f"Avg daily price change={smoothness:.1%} > {self.smoothness_threshold:.0%}",
                recommended_action=(
                    "Boost price_smoothness Lagrange multiplier. "
                    "Check if agent is hitting ±15% limit daily."
                ),
            )
            alerts.append(alert)
            self._alert_log.append(alert)

        # ── Detector 2: Inventory drain
        drain = self.compute_drain_rate()
        if drain > self.drain_threshold:
            level = (
                AlertLevel.CRITICAL if drain > self.drain_threshold * 2
                else AlertLevel.ALERT
            )
            alert = HackingAlert(
                step=self._step,
                hack_type="inventory_drain",
                level=level,
                metric_value=drain,
                threshold=self.drain_threshold,
                detail=f"Excess drain rate={drain:.1%} > {self.drain_threshold:.0%}",
                recommended_action=(
                    "Boost inventory_health Lagrange multiplier. "
                    "Check for short-term liquidation behavior."
                ),
            )
            alerts.append(alert)
            self._alert_log.append(alert)

        # ── Detector 3: Artificial revenue equalization
        cv = self.compute_revenue_cv()
        if cv < self.revenue_cv_min:
            alert = HackingAlert(
                step=self._step,
                hack_type="revenue_equalization",
                level=AlertLevel.WARNING,
                metric_value=cv,
                threshold=self.revenue_cv_min,
                detail=f"Revenue CV={cv:.3f} < {self.revenue_cv_min:.3f} (suspiciously uniform)",
                recommended_action=(
                    "Audit inter-store inventory transfers. "
                    "Verify actual sales data matches reported revenue."
                ),
            )
            alerts.append(alert)
            self._alert_log.append(alert)

        return alerts

    def get_alert_log(self) -> List[HackingAlert]:
        return self._alert_log.copy()

    def get_alert_count(
        self, hack_type: Optional[str] = None,
        level: Optional[AlertLevel] = None,
    ) -> int:
        count = 0
        for a in self._alert_log:
            if hack_type and a.hack_type != hack_type:
                continue
            if level and a.level != level:
                continue
            count += 1
        return count

    def summary(self) -> Dict[str, float]:
        """Current state of all three metrics — for dashboard display."""
        return {
            "smoothness_score": round(self.compute_smoothness_score(), 4),
            "drain_rate":       round(self.compute_drain_rate(), 4),
            "revenue_cv":       round(self.compute_revenue_cv(), 4),
            "total_alerts":     len(self._alert_log),
        }