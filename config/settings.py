"""
config/settings.py
──────────────────
Single source of truth for every business constant and hyperparameter.

Rule: Never hard-code a number anywhere else in the codebase.
      Always import from here. This makes tuning and experimentation clean.
"""

from dataclasses import dataclass
from typing import Dict


# ─────────────────────────────────────────────────────────────
# Store topology
# ─────────────────────────────────────────────────────────────

REGION_CONFIG: Dict[str, Dict] = {
    "urban":    {"n_stores": 4, "foot_traffic_multiplier": 1.4, "price_sensitivity": 1.2, "competitor_volatility": 0.15},
    "suburban": {"n_stores": 4, "foot_traffic_multiplier": 1.0, "price_sensitivity": 1.0, "competitor_volatility": 0.08},
    "rural":    {"n_stores": 2, "foot_traffic_multiplier": 0.7, "price_sensitivity": 0.8, "competitor_volatility": 0.04},
}

N_STORES: int = sum(r["n_stores"] for r in REGION_CONFIG.values())  # = 10

CATEGORY_CONFIG: Dict[str, Dict] = {
    "electronics":  {"n_skus": 50,  "price_min": 500.0,  "price_max": 50_000.0, "elasticity": -2.5, "base_daily_demand": 8},
    "groceries":    {"n_skus": 200, "price_min": 10.0,   "price_max": 500.0,    "elasticity": -0.8, "base_daily_demand": 60},
    "apparel":      {"n_skus": 100, "price_min": 200.0,  "price_max": 5_000.0,  "elasticity": -1.5, "base_daily_demand": 15},
    "home_kitchen": {"n_skus": 75,  "price_min": 100.0,  "price_max": 10_000.0, "elasticity": -1.2, "base_daily_demand": 20},
}

N_SKUS: int = sum(c["n_skus"] for c in CATEGORY_CONFIG.values())  # = 425


class HardConstraints:
    """Must NEVER be violated. Enforced physically inside the environment."""
    MIN_MARGIN_PCT: float = 0.05
    MAX_DAILY_PRICE_CHANGE_PCT: float = 0.15
    MAX_PREDATORY_DAYS: int = 3
    MIN_INVENTORY: int = 0


class SoftConstraints:
    """Optimised via reward shaping. Agent learns to avoid violations."""
    MIN_CUSTOMER_SATISFACTION: float = 4.0
    INVENTORY_TURNOVER_MIN: float = 8.0
    INVENTORY_TURNOVER_MAX: float = 12.0
    MAX_CROSS_STORE_PRICE_VARIANCE_PCT: float = 0.10
    MAX_STOCKOUT_RATE: float = 0.02


class StateConfig:
    """
    Exact dimensions of the per-SKU observation vector.
    Must stay in sync with retail_env._build_observation().
    Change here → everything adapts automatically.
    """
    DIM_PRICE: int = 1
    DIM_COST: int = 1
    DIM_INVENTORY: int = 1
    DIM_DEMAND_HISTORY: int = 7
    DIM_COMPETITOR_PRICES: int = 3
    DIM_SEASONALITY: int = 4
    DIM_WEATHER: int = 2
    DIM_CUSTOMER_SEGMENT: int = 5
    TOTAL_PER_SKU: int = 1+1+1+7+3+4+2+5  # = 24


class RewardWeights:
    """
    Weight design: Revenue=1.0 (baseline), Stockout=10x (very bad),
    Margin violation=100x (near-illegal, catastrophic).
    Agent learns the same priority the business has.
    """
    REVENUE: float = 1.0
    HOLDING_COST: float = 0.001
    STOCKOUT_PENALTY: float = 10.0
    MARGIN_VIOLATION: float = 100.0


@dataclass
class TrainingConfig:
    seed: int = 42
    n_episodes: int = 1_000
    max_steps_per_episode: int = 365
    gamma: float = 0.99
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    n_epochs_per_update: int = 10
    minibatch_size: int = 64
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    grpo_group_size: int = 8
    device: str = "cpu"
    log_interval: int = 10
    checkpoint_interval: int = 100