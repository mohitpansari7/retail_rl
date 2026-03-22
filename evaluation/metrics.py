"""
evaluation/metrics.py
──────────────────────
Phase 10 — Production metrics tracker and reporting.

Aggregates metrics from all system components into one unified view.
Produces business-readable reports that non-ML stakeholders can interpret.

Metric categories tracked:
  Business  : revenue, stockouts, margin violations, customer satisfaction proxy
  Training  : policy loss, value loss, entropy, update count
  Safety    : constraint violations, Lagrange multiplier values, hacking alerts
  LLM       : strategy call count, strategy distributions
  Deployment: latency, A/B test results, drift scores, Elo ratings

WHY a central tracker?
  Each agent knows its own metrics. The tracker sees ALL components.
  Revenue from env + safety violations from Phase 9 + LLM calls from Phase 7
  + drift from Phase 10 — no single agent has this full picture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for one complete episode across all stores."""
    episode:          int
    timestamp:        str = field(default_factory=lambda: datetime.now().isoformat())

    # Business metrics
    total_revenue:    float = 0.0
    stockout_count:   int   = 0
    margin_violations:int   = 0
    price_variance:   float = 0.0

    # Training metrics
    policy_loss:      float = 0.0
    value_loss:       float = 0.0
    entropy:          float = 0.0

    # Safety metrics
    constraint_violations: int   = 0
    active_lambdas:        List[str] = field(default_factory=list)

    # Deployment metrics
    inference_ms:     float = 0.0
    llm_calls:        int   = 0
    ab_group:         str   = "control"


class ProductionMetricsTracker:
    """
    Central metrics aggregator for the full RetailRL system.

    Tracks all metrics across episodes and produces human-readable
    reports for both engineering and business audiences.

    Args:
        store_ids    : List of store identifiers being tracked.
        window_size  : Rolling window for trend computations.
    """

    def __init__(
        self,
        store_ids:   List[str],
        window_size: int = 50,
    ) -> None:
        self.store_ids   = store_ids
        self.window_size = window_size

        self._episodes: List[EpisodeMetrics] = []

        # Rolling buffers for trend detection
        self._recent_revenues:   List[float] = []
        self._recent_stockouts:  List[int]   = []
        self._recent_losses:     List[float] = []

    def record_episode(self, metrics: EpisodeMetrics) -> None:
        """Store one episode's metrics."""
        self._episodes.append(metrics)

        self._recent_revenues.append(metrics.total_revenue)
        self._recent_stockouts.append(metrics.stockout_count)
        if metrics.policy_loss > 0:
            self._recent_losses.append(metrics.policy_loss)

        # Keep rolling windows
        if len(self._recent_revenues) > self.window_size:
            self._recent_revenues  = self._recent_revenues[-self.window_size:]
            self._recent_stockouts = self._recent_stockouts[-self.window_size:]
            self._recent_losses    = self._recent_losses[-self.window_size:]

    def compute_business_kpis(self) -> Dict:
        """
        Compute business KPIs over the recent window.
        These are the numbers a business stakeholder cares about.
        """
        if not self._recent_revenues:
            return {}

        revenues  = np.array(self._recent_revenues)
        stockouts = np.array(self._recent_stockouts)

        return {
            "mean_daily_revenue":    round(float(revenues.mean()), 2),
            "revenue_trend_pct":     round(self._compute_trend(revenues), 2),
            "stockout_rate":         round(float(stockouts.mean() / max(1, len(self.store_ids))), 4),
            "revenue_consistency":   round(float(1 - revenues.std() / (revenues.mean() + 1e-8)), 4),
            "n_episodes_tracked":    len(self._episodes),
        }

    def compute_training_health(self) -> Dict:
        """Training metrics for engineering audience."""
        if not self._episodes:
            return {}

        recent = self._episodes[-min(10, len(self._episodes)):]
        losses = [e.policy_loss for e in recent if e.policy_loss > 0]

        return {
            "mean_policy_loss":  round(float(np.mean(losses)), 6) if losses else None,
            "loss_trend":        round(self._compute_trend(np.array(losses)), 4) if len(losses) > 2 else None,
            "mean_entropy":      round(float(np.mean([e.entropy for e in recent])), 4),
            "total_updates":     len([e for e in self._episodes if e.policy_loss > 0]),
        }

    def compute_safety_summary(self) -> Dict:
        """Safety metrics — violations and constraint activity."""
        if not self._episodes:
            return {}

        total_violations = sum(e.constraint_violations for e in self._episodes)
        total_margin_viol = sum(e.margin_violations for e in self._episodes)
        recent = self._episodes[-self.window_size:]

        # Which Lagrange multipliers are currently active (λ > 1)?
        all_active = []
        for e in recent:
            all_active.extend(e.active_lambdas)
        lambda_counts = {}
        for name in all_active:
            lambda_counts[name] = lambda_counts.get(name, 0) + 1

        return {
            "total_constraint_violations": total_violations,
            "total_margin_violations":     total_margin_viol,
            "violations_per_episode":      round(total_violations / max(1, len(self._episodes)), 3),
            "most_active_constraint":      max(lambda_counts, key=lambda_counts.get) if lambda_counts else None,
        }

    def _compute_trend(self, values: np.ndarray) -> float:
        """
        Linear trend as percentage per episode.
        Positive = improving, negative = degrading.
        """
        if len(values) < 3:
            return 0.0
        x     = np.arange(len(values))
        slope = float(np.polyfit(x, values, 1)[0])
        mean  = float(values.mean()) + 1e-8
        return (slope / abs(mean)) * 100   # percentage change per episode

    def generate_report(self) -> str:
        """
        Generate a human-readable production report.
        Suitable for a weekly operations review.
        """
        biz    = self.compute_business_kpis()
        train  = self.compute_training_health()
        safety = self.compute_safety_summary()

        lines = [
            "=" * 60,
            "RetailRL Production Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Stores monitored: {len(self.store_ids)}",
            "=" * 60,
            "",
            "BUSINESS KPIs",
            "-" * 40,
        ]

        if biz:
            lines += [
                f"  Mean daily revenue    : ₹{biz['mean_daily_revenue']:>12,.2f}",
                f"  Revenue trend         : {biz['revenue_trend_pct']:>+8.2f}% per episode",
                f"  Stockout rate         : {biz['stockout_rate']:.2%}  (target < 2%)",
                f"  Revenue consistency   : {biz['revenue_consistency']:.2%}",
                f"  Episodes tracked      : {biz['n_episodes_tracked']}",
            ]

        lines += ["", "TRAINING HEALTH", "-" * 40]
        if train:
            loss_str = f"{train['mean_policy_loss']:.6f}" if train.get("mean_policy_loss") else "N/A"
            trend_str = f"{train['loss_trend']:+.4f}" if train.get("loss_trend") else "N/A"
            lines += [
                f"  Mean policy loss      : {loss_str}",
                f"  Loss trend            : {trend_str}",
                f"  Mean entropy          : {train.get('mean_entropy', 'N/A')}",
                f"  Total policy updates  : {train.get('total_updates', 0)}",
            ]

        lines += ["", "SAFETY SUMMARY", "-" * 40]
        if safety:
            lines += [
                f"  Constraint violations : {safety['total_constraint_violations']}",
                f"  Margin violations     : {safety['total_margin_violations']}",
                f"  Violations/episode    : {safety['violations_per_episode']:.3f}",
                f"  Most active constraint: {safety['most_active_constraint'] or 'none'}",
            ]

        lines += ["", "=" * 60]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize all metrics as JSON for dashboard ingestion."""
        return json.dumps({
            "business_kpis":    self.compute_business_kpis(),
            "training_health":  self.compute_training_health(),
            "safety_summary":   self.compute_safety_summary(),
            "n_episodes":       len(self._episodes),
        }, indent=2)

    @property
    def episodes(self) -> List[EpisodeMetrics]:
        return self._episodes.copy()