"""
reward/reward_fn.py
────────────────────
Phase 2 — Reward Function Design

WHY separate the reward from the environment?
    In Phase 4 we replace this hand-crafted reward with a LEARNED reward
    model (Bradley-Terry). Keeping reward logic here means we swap ONE
    file, not rip apart the environment.

The reward has four components (from the project spec):

    R = w_rev  × Revenue
      − w_hold × Holding cost
      − w_so   × Stockout penalty
      − w_mv   × Margin violation penalty

Weight intuition:
    - Revenue weight = 1.0       (the primary objective)
    - Holding cost = 0.001       (small — holding stock is mildly bad)
    - Stockout = 10.0            (10× revenue weight — painful)
    - Margin violation = 100.0   (hard constraint proxy — very painful)

The high stockout and margin weights teach the agent:
"don't sacrifice safety for short-term revenue gains."
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from config.settings import HardConstraints, RewardWeights


@dataclass
class RewardComponents:
    """Named container for one step's reward breakdown. Useful for logging."""
    revenue: float = 0.0
    holding_cost: float = 0.0
    stockout_penalty: float = 0.0
    margin_violation: float = 0.0

    @property
    def total(self) -> float:
        return (
            RewardWeights.REVENUE * self.revenue
            - self.holding_cost
            - self.stockout_penalty
            - self.margin_violation
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "revenue": round(self.revenue, 4),
            "holding_cost": round(self.holding_cost, 6),
            "stockout_penalty": round(self.stockout_penalty, 4),
            "margin_violation": round(self.margin_violation, 4),
            "total": round(self.total, 4),
        }


def compute_sku_reward(
    price: float,
    cost: float,
    units_sold: float,
    inventory: float,
    is_stockout: bool,
) -> RewardComponents:
    """
    Compute reward components for a single SKU in one time step.

    Args:
        price      : The price set by the agent (INR).
        cost       : The procurement cost of this SKU (INR).
        units_sold : Units actually sold (demand capped by inventory).
        inventory  : Remaining inventory AFTER selling (units).
        is_stockout: True if inventory hit zero before meeting demand.

    Returns:
        RewardComponents with all four components filled in.
    """
    # ── Revenue: what we earned this step
    revenue = price * units_sold

    # ── Holding cost: paying to store unsold stock
    holding_cost = RewardWeights.HOLDING_COST * inventory

    # ── Stockout penalty: we failed a customer, lost future loyalty
    stockout_penalty = RewardWeights.STOCKOUT_PENALTY if is_stockout else 0.0

    # ── Margin violation: price too close to (or below) cost
    min_allowed_price = cost * (1.0 + HardConstraints.MIN_MARGIN_PCT)
    margin_shortfall = max(0.0, min_allowed_price - price)
    margin_violation = RewardWeights.MARGIN_VIOLATION * margin_shortfall

    return RewardComponents(
        revenue=revenue,
        holding_cost=holding_cost,
        stockout_penalty=stockout_penalty,
        margin_violation=margin_violation,
    )


def compute_store_reward(sku_rewards: List[RewardComponents]) -> float:
    """
    Aggregate per-SKU rewards into a single scalar for the agent.

    Design note: we normalise by number of SKUs so the reward magnitude
    doesn't blow up as n_skus grows. This keeps gradients stable.
    """
    if not sku_rewards:
        return 0.0
    total = sum(r.total for r in sku_rewards)
    return total / len(sku_rewards)


def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute discounted returns G_t for a full episode trajectory.

    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T

    WHY compute backwards?
        G_T = r_T
        G_{T-1} = r_{T-1} + γ·G_T
        G_{T-2} = r_{T-2} + γ·G_{T-1}
        ...
    A single backward pass is O(T) — efficient and numerically exact.

    Args:
        rewards   : List of per-step rewards [r_0, r_1, ..., r_T]
        gamma     : Discount factor (0 < γ ≤ 1)
        normalise : Standardise returns → zero mean, unit variance.
                    This is the "baseline" trick for REINFORCE — reduces
                    gradient variance significantly.

    Returns:
        returns : np.ndarray shape (T,)
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    running_return = 0.0

    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    if normalise and len(returns) > 1:
        mean = returns.mean()
        std = returns.std() + 1e-8   # ε prevents division by zero
        returns = (returns - mean) / std

    return returns