"""
envs/retail_env.py
───────────────────
Phase 1 — The complete MDP as a Gymnasium environment.

MDP tuple recap:
  S : per-SKU 24-dim vector × n_skus  (flat float32 array)
  A : price change fraction per SKU   ∈ [-1, 1] → scaled to ±15%
  P : demand model + inventory update (in step())
  R : revenue − holding cost − stockout penalty − margin violation
  γ : set in TrainingConfig (0.99)

Gymnasium API contract:
  reset() → (obs, info)
  step(action) → (obs, reward, terminated, truncated, info)

This is a SINGLE-STORE environment.
Phase 5 wraps N of these into a MultiStoreEnv for MARL.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import (
    CATEGORY_CONFIG, N_SKUS,
    HardConstraints, RewardWeights, StateConfig,
)
from envs.demand_model import SKUDemandModel


# ─────────────────────────────────────────────────────────────
# SKU runtime state
# ─────────────────────────────────────────────────────────────

@dataclass
class SKU:
    """
    Runtime state for one product in one store.

    WHY a dataclass and not just a dict?
    Type safety + dot-access syntax (sku.current_price vs sku['current_price']).
    Bugs caught at definition time, not runtime.
    """
    sku_id: str
    category: str
    cost: float                   # procurement cost (INR)
    reference_price: float        # "normal" market price — demand model baseline
    current_price: float          # price set by agent this step
    inventory: float              # units on hand
    demand_model: SKUDemandModel

    # Hard constraint tracker: how many consecutive days below cost?
    consecutive_below_cost_days: int = 0

    def is_margin_violated(self) -> bool:
        """True if price is below cost + minimum margin."""
        return self.current_price < self.cost * (1 + HardConstraints.MIN_MARGIN_PCT)


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

class RetailEnv(gym.Env):
    """
    Single-store QuickMart environment.

    Observation space : Box(n_skus × 24,)   — flat float32
    Action space      : Box(n_skus,)         — values in [-1, 1]

    Action interpretation:
        action[i] = +1.0  →  raise SKU i price by 15%
        action[i] = -1.0  →  lower SKU i price by 15%
        action[i] =  0.0  →  keep price unchanged

    Why clip actions to [-1, 1] at the gym level?
    This is the standard normalisation. The 15% business constraint is
    applied INSIDE step() as a scaling. Keeps the action space
    dimensionless and consistent across all price ranges.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        store_id: str = "store_0",
        n_skus: int = N_SKUS,
        max_steps: int = 365,
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.store_id = store_id
        self.n_skus = n_skus
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._base_seed = seed

        # ── Observation space: n_skus × 24 flat float vector
        obs_dim = n_skus * StateConfig.TOTAL_PER_SKU
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space: one continuous value per SKU in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_skus,), dtype=np.float32
        )

        # State — populated in reset()
        self.skus: List[SKU] = []
        self.current_step: int = 0
        self._episode_revenue: float = 0.0
        self._rng = np.random.default_rng(seed)

    # ── Private: environment setup ────────────────────────────

    def _build_skus(self) -> List[SKU]:
        """
        Instantiate SKU objects from the category config.

        WHY randomise reference prices within the category range?
        In reality, not all electronics cost ₹500 or all ₹50,000.
        Random sampling within the range gives us product diversity,
        making the agent learn a general pricing strategy rather than
        memorizing a single price point.

        WHY set cost as 55–75% of reference price?
        Typical retail gross margin is 25–45%. This means:
          cost = ref_price × U(0.55, 0.75)
          initial margin = (ref - cost) / ref ≈ 25–45%
        """
        skus = []
        sku_idx = 0
        for category, cfg in CATEGORY_CONFIG.items():
            for i in range(cfg["n_skus"]):
                if sku_idx >= self.n_skus:
                    break
                sku_id = f"{category}_{i:03d}"
                ref_price = float(self._rng.uniform(cfg["price_min"], cfg["price_max"]))
                cost = ref_price * float(self._rng.uniform(0.55, 0.75))
                inventory = float(self._rng.integers(50, 200))

                demand_model = SKUDemandModel(
                    sku_id=sku_id,
                    category=category,
                    base_demand=float(cfg["base_daily_demand"]),
                    reference_price=ref_price,
                    elasticity=float(cfg["elasticity"]),
                )
                demand_model.reset(self._rng)

                skus.append(SKU(
                    sku_id=sku_id,
                    category=category,
                    cost=cost,
                    reference_price=ref_price,
                    current_price=ref_price,   # start at reference price
                    inventory=inventory,
                    demand_model=demand_model,
                ))
                sku_idx += 1
            if sku_idx >= self.n_skus:
                break
        return skus

    def _get_weather(self) -> np.ndarray:
        """
        Simulate temperature and precipitation.
        Temperature follows a sinusoidal annual cycle (cold in Jan, hot in Jun).
        WHY include weather? Some SKUs (umbrellas, ACs, warm clothing) have
        weather-dependent demand. The agent must learn these correlations.
        """
        day = self.current_step % 365
        temp = 20.0 + 15.0 * np.sin(2 * np.pi * day / 365)  # 5°C–35°C range
        temp_norm = (temp - 20.0) / 15.0                      # normalise to ~[-1,1]
        precip = float(self._rng.uniform(0, 1) < 0.3)         # 30% chance of rain
        return np.array([temp_norm, precip], dtype=np.float32)

    def _get_competitor_prices(self, sku: SKU) -> np.ndarray:
        """
        Simulate 3 competitor prices near this SKU's reference price.
        WHY noise around reference_price? Competitors generally price
        in the same ballpark — the agent must learn to react to their
        small deviations (undercut, match, premium).
        """
        noise = self._rng.normal(0.0, 0.05, size=3)          # ±5% noise
        comp_prices = sku.reference_price * (1.0 + noise)
        # Normalise by reference price → dimensionless ratio
        return (comp_prices / sku.reference_price).astype(np.float32)

    def _get_seasonality_onehot(self) -> np.ndarray:
        """
        One-hot encode the current quarter.
        WHY one-hot and not a single integer 0–3?
        Neural networks handle categorical inputs better as one-hot.
        Integer encoding implies an ordinal relationship (Q4 > Q3 > Q2)
        which isn't meaningful here.
        """
        quarter = min((self.current_step % 365) // 91, 3)   # 0=Q1, 1=Q2, 2=Q3, 3=Q4
        oh = np.zeros(4, dtype=np.float32)
        oh[quarter] = 1.0
        return oh

    def _get_customer_segments(self) -> np.ndarray:
        """
        5-dim Dirichlet sample representing customer mix this step.
        Segments: [price_seeker, brand_loyal, occasional, bulk_buyer, impulse]
        WHY Dirichlet? It always sums to 1.0 (a proper distribution).
        The mix varies slightly each step — e.g. weekends may have more
        impulse buyers. The agent learns to exploit these patterns.
        """
        return self._rng.dirichlet(np.ones(5)).astype(np.float32)

    def _build_observation(self) -> np.ndarray:
        """
        Assemble the full flat observation vector.

        Structure: [sku_0_vec (24 dims), sku_1_vec (24 dims), ..., sku_N_vec]
        Total shape: (n_skus × 24,)

        WHY compute weather/season once per step (not per SKU)?
        Weather and season are store-level signals — same for all SKUs.
        Computing once and reusing is O(1) vs O(n_skus). At 425 SKUs,
        this matters.

        WHY normalise everything?
        Neural networks are sensitive to input scale. A price of ₹50,000
        and an inventory of 150 are on completely different scales. If we
        fed raw values, the network's gradients would be dominated by the
        large-magnitude features. Normalising makes all features comparable.
        """
        weather = self._get_weather()
        season = self._get_seasonality_onehot()
        segments = self._get_customer_segments()

        sku_vecs = []
        for sku in self.skus:
            comp_prices_norm = self._get_competitor_prices(sku)

            # Each feature normalised to a dimensionless ratio or bounded range
            p_norm   = np.array([sku.current_price / sku.reference_price], dtype=np.float32)
            c_norm   = np.array([sku.cost / sku.reference_price],          dtype=np.float32)
            inv_norm = np.array([sku.inventory / 200.0],                   dtype=np.float32)
            hist_norm = (sku.demand_model.get_history()
                         / max(sku.demand_model.base_demand, 1.0)).astype(np.float32)

            sku_vec = np.concatenate([
                p_norm,           # 1  — current price ratio
                c_norm,           # 1  — cost ratio
                inv_norm,         # 1  — inventory level
                hist_norm,        # 7  — 7-day demand history
                comp_prices_norm, # 3  — competitor price ratios
                season,           # 4  — quarter one-hot
                weather,          # 2  — temperature + precipitation
                segments,         # 5  — customer segment mix
            ])  # total = 24

            # Sanity check — catches mismatches during development
            assert len(sku_vec) == StateConfig.TOTAL_PER_SKU, (
                f"State vector is {len(sku_vec)}, expected {StateConfig.TOTAL_PER_SKU}"
            )
            sku_vecs.append(sku_vec)

        return np.concatenate(sku_vecs, dtype=np.float32)

    # ── Private: reward calculation ───────────────────────────

    def _compute_reward(self, sku: SKU, units_sold: float) -> Tuple[float, Dict]:
        """
        Compute scalar reward for one SKU this step.

        WHY return a breakdown dict too?
        During training, we log each component separately. This lets us
        diagnose WHY the agent earns what it earns. If margin violations
        are high, we know the agent is pricing recklessly. If holding cost
        dominates, we know it's over-ordering. The total alone hides this.
        """
        revenue = sku.current_price * units_sold
        holding_cost = RewardWeights.HOLDING_COST * sku.inventory
        stockout_penalty = RewardWeights.STOCKOUT_PENALTY if sku.inventory <= 0 else 0.0

        # Margin violation: how far below the floor are we? (0 if fine)
        min_allowed = sku.cost * (1 + HardConstraints.MIN_MARGIN_PCT)
        margin_violation = (
            RewardWeights.MARGIN_VIOLATION * max(0.0, min_allowed - sku.current_price)
            if sku.is_margin_violated() else 0.0
        )

        total = (RewardWeights.REVENUE * revenue
                 - holding_cost
                 - stockout_penalty
                 - margin_violation)

        return total, {
            "revenue": revenue,
            "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty,
            "margin_violation": margin_violation,
        }

    # ── Public: Gymnasium API ─────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset to start of a new episode.

        WHY reset the rng here and not in __init__ only?
        Each episode should be independently reproducible.
        reset(seed=42) will always produce the same episode,
        regardless of what happened in prior episodes.
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed if seed is not None else self._base_seed)
        self.current_step = 0
        self._episode_revenue = 0.0
        self.skus = self._build_skus()
        obs = self._build_observation()
        return obs, {"store_id": self.store_id, "n_skus": self.n_skus}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply price actions → simulate demand → update inventory → compute reward.

        The 5-step inner loop per SKU:
          1. Apply price action (enforce hard constraint: price floor)
          2. Sample demand from demand model
          3. Compute units sold = min(demand, inventory)
          4. Update inventory (sell units, weekly reorder)
          5. Compute reward

        WHY enforce price floor here (not in the agent)?
        The environment is the ground truth. If hard constraints lived
        in the agent, any agent bug could violate them. Inside the env,
        they're physically impossible to bypass.

        Returns:
            obs        : next state observation
            reward     : scalar total reward this step
            terminated : False — no natural terminal state for pricing
            truncated  : True when current_step reaches max_steps
            info       : diagnostic dict (step, revenue, per-SKU breakdown)
        """
        total_reward = 0.0
        total_revenue = 0.0
        stockouts = 0
        sku_infos = []

        for idx, sku in enumerate(self.skus):

            # ── Step 1: Apply price action
            # action[idx] ∈ [-1, 1] → scale to ±MAX_DAILY_PRICE_CHANGE_PCT
            raw_action = float(np.clip(action[idx], -1.0, 1.0))
            price_change = raw_action * HardConstraints.MAX_DAILY_PRICE_CHANGE_PCT
            new_price = sku.current_price * (1.0 + price_change)

            # Hard constraint: price cannot go below cost + min margin
            min_allowed = sku.cost * (1 + HardConstraints.MIN_MARGIN_PCT)
            new_price = max(new_price, min_allowed)
            sku.current_price = new_price

            # Track predatory pricing days (below cost, ignoring margin)
            if new_price < sku.cost:
                sku.consecutive_below_cost_days += 1
            else:
                sku.consecutive_below_cost_days = 0

            # ── Step 2: Sample demand
            demand = sku.demand_model.sample_demand(
                current_price=sku.current_price,
                day_of_year=(self.current_step % 365) + 1,
                rng=self._rng,
            )

            # ── Step 3: Units sold = demand capped by available stock
            units_sold = min(demand, sku.inventory)
            sku.inventory = max(0.0, sku.inventory - units_sold)
            is_stockout = sku.inventory <= 0

            if is_stockout:
                stockouts += 1

            # ── Step 4: Weekly reorder (simple fixed policy for now)
            # In Phase 7, the LLM agent will make smarter replenishment decisions.
            # For now: every 7 days, top up to 150 units.
            if self.current_step % 7 == 0:
                reorder = max(0.0, 150.0 - sku.inventory)
                sku.inventory += reorder

            # ── Step 5: Reward
            sku_reward, breakdown = self._compute_reward(sku, units_sold)
            total_reward += sku_reward
            total_revenue += breakdown["revenue"]
            sku_infos.append({"sku_id": sku.sku_id,
                               "price": round(sku.current_price, 2),
                               "units_sold": round(units_sold, 1),
                               "inventory": round(sku.inventory, 1),
                               **{k: round(v, 4) for k, v in breakdown.items()}})

        self._episode_revenue += total_revenue
        self.current_step += 1

        obs = self._build_observation()
        info = {
            "step": self.current_step,
            "total_revenue": round(total_revenue, 2),
            "episode_revenue": round(self._episode_revenue, 2),
            "stockouts": stockouts,
            "skus": sku_infos,
        }
        return obs, float(total_reward), False, self.current_step >= self.max_steps, info

    def render(self) -> None:
        if self.render_mode == "human":
            print(f"Step {self.current_step:>4} | "
                  f"Revenue: ₹{self._episode_revenue:>12,.0f} | "
                  f"SKUs: {self.n_skus}")