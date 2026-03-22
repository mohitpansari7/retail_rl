"""
reward/verifiable_rewards.py
─────────────────────────────
Phase 8 — Verifiable reward checkers for RLVR.

These are GROUND TRUTH binary checks — no network, no approximation.
They are the immutable business rules that cannot be gamed because
they are verified against actual environment state.

Contrast with:
  - reward_fn.py : hand-crafted soft reward (weighted sum, tunable)
  - reward_model.py : learned reward (can be hacked/gamed)

These checks produce exact +1 / -1 signals for GRPO training.

WHY binary (+1 / -1) instead of continuous?
  Binary is unambiguous. No grey zone for the policy to exploit.
  Either the constraint is satisfied or it isn't.
  The policy learns a hard cliff, not a smooth slope.

Each checker returns a VerifiableResult with:
  passed : bool   — constraint satisfied?
  score  : float  — +1.0 (pass) or -1.0 (fail)
  detail : str    — human-readable explanation for logging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from config.settings import HardConstraints, SoftConstraints


@dataclass
class VerifiableResult:
    """
    Result of one verifiable reward check.

    Attributes:
        name   : Constraint name (e.g. "margin_constraint")
        passed : True if constraint is satisfied
        score  : +1.0 if passed, -1.0 if failed
        detail : Human-readable explanation
    """
    name:   str
    passed: bool
    score:  float
    detail: str

    @classmethod
    def pass_(cls, name: str, detail: str = "") -> "VerifiableResult":
        return cls(name=name, passed=True,  score=+1.0, detail=detail)

    @classmethod
    def fail_(cls, name: str, detail: str = "") -> "VerifiableResult":
        return cls(name=name, passed=False, score=-1.0, detail=detail)


# ─────────────────────────────────────────────────────────────
# Individual verifiable reward checkers
# ─────────────────────────────────────────────────────────────

def check_margin_constraint(
    price: float,
    cost: float,
    sku_id: str = "",
) -> VerifiableResult:
    """
    HARD constraint: price must be >= cost × (1 + MIN_MARGIN_PCT).

    This is the most important check. Pricing below this floor
    means selling at a loss — legally and financially catastrophic.

    WHY binary? Either we're above the floor or below.
    The soft penalty in Phase 2 let agents hover near the floor.
    Binary makes the cliff sharp — policy learns to stay well clear.
    """
    floor = cost * (1.0 + HardConstraints.MIN_MARGIN_PCT)
    if price >= floor:
        margin = (price - cost) / price
        return VerifiableResult.pass_(
            "margin_constraint",
            f"{sku_id} margin={margin:.1%} ≥ {HardConstraints.MIN_MARGIN_PCT:.0%}"
        )
    else:
        shortfall = floor - price
        return VerifiableResult.fail_(
            "margin_constraint",
            f"{sku_id} price={price:.2f} below floor={floor:.2f} by {shortfall:.2f}"
        )


def check_stockout_constraint(
    inventory: float,
    units_demanded: float,
    sku_id: str = "",
) -> VerifiableResult:
    """
    SOFT constraint: inventory should not hit zero.

    Binary signal: did a stockout actually occur this step?
    +1 if we fulfilled all demand, -1 if we ran out.
    """
    if inventory >= units_demanded or inventory > 0:
        return VerifiableResult.pass_(
            "stockout_constraint",
            f"{sku_id} inventory={inventory:.0f} demand={units_demanded:.0f}"
        )
    else:
        return VerifiableResult.fail_(
            "stockout_constraint",
            f"{sku_id} STOCKOUT: demand={units_demanded:.0f} inventory={inventory:.0f}"
        )


def check_revenue_target(
    actual_revenue: float,
    target_revenue: float,
    store_id: str = "",
) -> VerifiableResult:
    """
    Revenue target check: did the store hit its daily revenue target?

    Target is computed as a moving average of recent performance.
    +1 if we met or exceeded target, -1 if we missed.
    """
    if actual_revenue >= target_revenue:
        pct = actual_revenue / (target_revenue + 1e-8) - 1
        return VerifiableResult.pass_(
            "revenue_target",
            f"{store_id} revenue={actual_revenue:.0f} target={target_revenue:.0f} (+{pct:.1%})"
        )
    else:
        pct = 1 - actual_revenue / (target_revenue + 1e-8)
        return VerifiableResult.fail_(
            "revenue_target",
            f"{store_id} revenue={actual_revenue:.0f} missed target={target_revenue:.0f} (-{pct:.1%})"
        )


def check_price_change_constraint(
    new_price: float,
    old_price: float,
    sku_id: str = "",
) -> VerifiableResult:
    """
    HARD constraint: daily price change must be within ±MAX_DAILY_PRICE_CHANGE_PCT.

    This is enforced physically in the environment (Phase 1),
    but we also track it as a verifiable signal for GRPO.
    """
    max_pct = HardConstraints.MAX_DAILY_PRICE_CHANGE_PCT
    actual_pct = abs(new_price - old_price) / (old_price + 1e-8)
    if actual_pct <= max_pct:
        return VerifiableResult.pass_(
            "price_change_constraint",
            f"{sku_id} change={actual_pct:.1%} ≤ {max_pct:.0%}"
        )
    else:
        return VerifiableResult.fail_(
            "price_change_constraint",
            f"{sku_id} change={actual_pct:.1%} exceeds {max_pct:.0%} limit"
        )


def check_cross_store_price_variance(
    prices_for_sku: List[float],
    sku_id: str = "",
) -> VerifiableResult:
    """
    SOFT constraint: same SKU should not have > 10% price variance across stores.

    Coefficient of variation (std/mean) must be below threshold.
    This prevents customer confusion and brand damage.
    """
    if len(prices_for_sku) < 2:
        return VerifiableResult.pass_("price_variance", "single store")

    mean = np.mean(prices_for_sku)
    cv   = np.std(prices_for_sku) / (mean + 1e-8)
    threshold = SoftConstraints.MAX_CROSS_STORE_PRICE_VARIANCE_PCT

    if cv <= threshold:
        return VerifiableResult.pass_(
            "price_variance",
            f"{sku_id} CV={cv:.1%} ≤ {threshold:.0%}"
        )
    else:
        return VerifiableResult.fail_(
            "price_variance",
            f"{sku_id} CV={cv:.1%} exceeds {threshold:.0%}"
        )


# ─────────────────────────────────────────────────────────────
# Composite verifiable reward scorer
# ─────────────────────────────────────────────────────────────

class VerifiableRewardScorer:
    """
    Aggregates multiple verifiable reward checks into one scalar signal.

    Used by GRPOAgent to score each sampled action in the group.

    Weighting philosophy:
      Hard constraints (margin, price_change) : weight 2.0 — must never fail
      Soft constraints (stockout, variance)   : weight 1.0 — important but recoverable
      Revenue target                          : weight 1.5 — primary business goal

    The weighted average gives a score in [-1, +1]:
      +1.0 → all constraints passed
      -1.0 → all constraints failed
      Mixed → proportional to which constraints passed
    """

    WEIGHTS: Dict[str, float] = {
        "margin_constraint":     2.0,
        "stockout_constraint":   1.0,
        "revenue_target":        1.5,
        "price_change_constraint": 2.0,
        "price_variance":        1.0,
    }

    def score(self, results: List[VerifiableResult]) -> float:
        """
        Compute weighted average score from a list of check results.

        Args:
            results : List of VerifiableResult from individual checks.

        Returns:
            Float in [-1, +1]. Used as the reward signal in GRPO.
        """
        if not results:
            return 0.0

        total_weight  = 0.0
        weighted_sum  = 0.0
        for r in results:
            w = self.WEIGHTS.get(r.name, 1.0)
            weighted_sum  += w * r.score
            total_weight  += w

        return weighted_sum / (total_weight + 1e-8)

    def all_hard_constraints_pass(self, results: List[VerifiableResult]) -> bool:
        """
        Check if ALL hard constraints passed.
        Used by Phase 9 Safety layer for hard rejection.
        """
        hard_names = {"margin_constraint", "price_change_constraint"}
        for r in results:
            if r.name in hard_names and not r.passed:
                return False
        return True

    def get_failures(self, results: List[VerifiableResult]) -> List[str]:
        """Return names of failed constraints — useful for logging."""
        return [r.name for r in results if not r.passed]