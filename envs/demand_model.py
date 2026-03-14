"""
envs/demand_model.py
─────────────────────
Phase 1 — The Transition Function P(s'|s,a)

The demand model answers: "if the agent sets this price today,
how many units will customers actually buy?"

Core equation:
    D_t = D_base × (p_t / p_ref)^ε × S_t^season × (1 + η_t)

Design: pure function (compute_demand) + stateful wrapper (SKUDemandModel).
  - Pure function  → trivially unit-testable, no hidden state
  - Stateful class → owns the 7-day history window and rng calls
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Seasonality: demand multiplier by quarter
# Q4 (festive Oct-Dec) is 1.3× baseline. Q1 (post-holiday) is 0.85×.
SEASONALITY = {"Q1": 0.85, "Q2": 1.00, "Q3": 0.95, "Q4": 1.30}

# Map month number → quarter string
MONTH_TO_QUARTER = {**dict.fromkeys([1,2,3], "Q1"),
                    **dict.fromkeys([4,5,6], "Q2"),
                    **dict.fromkeys([7,8,9], "Q3"),
                    **dict.fromkeys([10,11,12], "Q4")}


def compute_demand(
    base_demand: float,
    current_price: float,
    reference_price: float,
    elasticity: float,
    season_multiplier: float,
    noise: float,
) -> float:
    """
    Pure function: compute demand for one SKU at one time step.

    WHY pure? No side effects means we can test it in isolation,
    run it in parallel, and reason about it mathematically.

    The price elasticity term (p_t/p_ref)^ε:
      - When price equals reference: (1.0)^ε = 1.0 → no change in demand
      - When price is raised: ratio > 1, ε is negative → result < 1 → demand drops
      - When price is cut:   ratio < 1, ε is negative → result > 1 → demand rises
      - Higher |ε| = steeper response (electronics vs groceries)

    Args:
        base_demand      : Expected units/day at reference price, normal season.
        current_price    : Price the agent chose this step.
        reference_price  : The "normal" price this SKU was calibrated at.
        elasticity       : ε (negative). How sharply demand reacts to price.
        season_multiplier: Seasonal adjustment (e.g. 1.3 in Q4).
        noise            : Pre-sampled η_t from caller's rng. Keeps this pure.

    Returns:
        Realised demand (float ≥ 0). Floored at 0 — demand cannot be negative.
    """
    price_effect = (current_price / reference_price) ** elasticity
    demand = base_demand * price_effect * season_multiplier * (1.0 + noise)
    return max(0.0, demand)


@dataclass
class SKUDemandModel:
    """
    Stateful demand model for one (store, SKU) pair.

    Holds the 7-day rolling demand history — this is what makes our
    state vector approximately Markov. Without this history, the agent
    can't distinguish "demand dropped because we raised price" from
    "demand dropped because of an external shock."

    One instance per SKU per store → 425 × 10 = 4,250 instances total.
    They're lightweight (just a few floats + 7-element array).
    """
    sku_id: str
    category: str
    base_demand: float
    reference_price: float
    elasticity: float
    noise_std: float = 0.10          # σ for η_t ~ N(0, σ²). 10% demand noise.

    # Rolling 7-day history — initialised in __post_init__
    demand_history: np.ndarray = field(init=False)

    def __post_init__(self):
        # Start at base_demand — will be overwritten on first reset()
        self.demand_history = np.full(7, self.base_demand, dtype=np.float32)

    def sample_demand(
        self,
        current_price: float,
        day_of_year: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Sample realised demand for this time step, then update history.

        WHY does the caller pass in `rng` instead of us creating our own?
        Because the environment needs to control the random stream for
        reproducibility. Passing rng in makes this fully deterministic
        given the same seed — critical for debugging and unit tests.

        Args:
            current_price : Price set by agent this step.
            day_of_year   : 1–365. Used to compute which quarter we're in.
            rng           : Caller's random generator (env owns this).

        Returns:
            Realised demand as a float. (Inventory capping happens in the env.)
        """
        # Figure out season
        month = min(((day_of_year - 1) // 30) + 1, 12)
        season_mult = SEASONALITY[MONTH_TO_QUARTER[month]]

        # Sample noise — this is the only random call in this method
        noise = rng.normal(0.0, self.noise_std)

        demand = compute_demand(
            base_demand=self.base_demand,
            current_price=current_price,
            reference_price=self.reference_price,
            elasticity=self.elasticity,
            season_multiplier=season_mult,
            noise=noise,
        )

        # Shift history: drop oldest day, append today's demand
        # np.roll shifts right → element 0 becomes element 1, etc.
        # Then we overwrite the last slot with today's value.
        self.demand_history = np.roll(self.demand_history, -1)
        self.demand_history[-1] = demand

        return float(demand)

    def get_history(self) -> np.ndarray:
        """Return 7-day history (oldest → newest). Copy to prevent mutation."""
        return self.demand_history.copy()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        Reset history at episode start.
        If rng provided, warm-start with slightly noisy history (more realistic
        than a flat line of base_demand — avoids a "cold start" artifact).
        """
        if rng is not None:
            noise = rng.normal(0.0, self.noise_std, size=7)
            self.demand_history = np.clip(
                self.base_demand * (1.0 + noise), 0.0, None
            ).astype(np.float32)
        else:
            self.demand_history = np.full(7, self.base_demand, dtype=np.float32)