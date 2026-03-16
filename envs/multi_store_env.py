"""
envs/multi_store_env.py
────────────────────────
Phase 5 — Multi-store environment for MARL.

Wraps N RetailEnv instances and adds cross-store interaction:
  - Demand cannibalization: lower-priced store in same region
    steals demand from higher-priced stores
  - Inventory transfer: excess stock can move between stores
  - Joint reward: all agents share R_total = sum of individual rewards

WHY wrap RetailEnv instead of rebuilding?
  RetailEnv is fully tested. MultiStoreEnv ONLY adds inter-store logic.
  Each store's MDP remains encapsulated — clean separation of concerns.

Observation structure:
  Dict[store_id → local_obs]     (each agent sees only its own store)
  The global state (all stores) is assembled here for the CTDE critic in Phase 6.

Action structure:
  Dict[store_id → action_array]  (each agent controls only its own prices)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from envs.retail_env import RetailEnv
from config.settings import REGION_CONFIG, N_STORES, TrainingConfig


# Store-to-region assignment: stores 0-3 urban, 4-7 suburban, 8-9 rural
def _build_store_region_map(store_ids: List[str]) -> Dict[str, str]:
    """Assign each store_id to a region based on config."""
    region_map = {}
    idx = 0
    for region, cfg in REGION_CONFIG.items():
        for _ in range(cfg["n_stores"]):
            if idx < len(store_ids):
                region_map[store_ids[idx]] = region
                idx += 1
    return region_map


class MultiStoreEnv:
    """
    Multi-store QuickMart environment for MARL.

    Manages N RetailEnv instances and simulates cross-store effects.

    Args:
        n_stores              : Number of stores (default 10).
        n_skus                : SKUs per store.
        max_steps             : Episode length.
        cannibalization_alpha : How strongly lower price in same region
                                steals demand from other stores.
                                0.1 = 10% price gap → 1% demand transfer.
        seed                  : Base random seed (each store gets seed+i).
    """

    def __init__(
        self,
        n_stores: int = N_STORES,
        n_skus: int = 20,           # small default for fast training
        max_steps: int = 365,
        cannibalization_alpha: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_stores = n_stores
        self.n_skus = n_skus
        self.max_steps = max_steps
        self.cannibalization_alpha = cannibalization_alpha
        self.base_seed = seed
        self.current_step = 0

        # Build store IDs and region map
        self.store_ids = [f"store_{i}" for i in range(n_stores)]
        self.region_map = _build_store_region_map(self.store_ids)

        # Group stores by region (for cannibalization calculation)
        self.region_to_stores: Dict[str, List[str]] = {}
        for sid, region in self.region_map.items():
            self.region_to_stores.setdefault(region, []).append(sid)

        # Instantiate one RetailEnv per store
        # Each gets a different seed so they start with different prices/inventory
        self.envs: Dict[str, RetailEnv] = {
            sid: RetailEnv(
                store_id=sid,
                n_skus=n_skus,
                max_steps=max_steps,
                seed=seed + i,
            )
            for i, sid in enumerate(self.store_ids)
        }

        # Cache last observations and rewards for cross-store effects
        self._last_obs: Dict[str, np.ndarray] = {}
        self._last_rewards: Dict[str, float] = {}
        self._episode_rewards: Dict[str, float] = {sid: 0.0 for sid in self.store_ids}

    # ── Properties ────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Observation dimension for one store."""
        return self.envs[self.store_ids[0]].observation_space.shape[0]

    @property
    def action_dim(self) -> int:
        """Action dimension for one store."""
        return self.envs[self.store_ids[0]].action_space.shape[0]

    @property
    def global_obs_dim(self) -> int:
        """
        Global state dimension: all stores concatenated.
        Used by the centralized critic in Phase 6.
        """
        return self.obs_dim * self.n_stores

    # ── Gymnasium-style API ───────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset all store environments.

        Returns:
            Dict mapping store_id → initial observation.
        """
        self.current_step = 0
        self._episode_rewards = {sid: 0.0 for sid in self.store_ids}

        observations = {}
        for i, (sid, env) in enumerate(self.envs.items()):
            s = seed + i if seed is not None else self.base_seed + i
            obs, _ = env.reset(seed=s)
            observations[sid] = obs

        self._last_obs = observations.copy()
        return observations

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],   # next observations
        Dict[str, float],        # individual rewards
        float,                   # joint reward (sum)
        bool,                    # done
        Dict,                    # info
    ]:
        """
        Step all stores simultaneously.

        Procedure:
          1. Each store takes its action independently
          2. Compute cross-store demand cannibalization adjustments
          3. Apply adjustments to rewards
          4. Compute joint reward = sum of adjusted individual rewards
          5. Log cross-store metrics (price variance, demand transfers)

        Args:
            actions : Dict[store_id → action_array]

        Returns:
            observations  : Dict[store_id → next_obs]
            ind_rewards   : Dict[store_id → adjusted individual reward]
            joint_reward  : scalar — sum of all individual rewards
            done          : True when max_steps reached
            info          : diagnostic dict
        """
        observations = {}
        raw_rewards  = {}
        store_infos  = {}
        done = False

        # ── Step 1: Independent steps for each store
        for sid, env in self.envs.items():
            action = actions.get(sid, env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            observations[sid] = obs
            raw_rewards[sid]  = reward
            store_infos[sid]  = info
            done = done or terminated or truncated

        # ── Step 2: Cross-store demand cannibalization
        # For each store, check if any same-region competitor priced lower.
        # If so, adjust the reward downward to reflect stolen demand.
        cannibal_adjustments = self._compute_cannibalization(store_infos)

        # ── Step 3: Apply adjustments
        ind_rewards = {}
        for sid in self.store_ids:
            adjusted = raw_rewards[sid] + cannibal_adjustments[sid]
            ind_rewards[sid] = adjusted
            self._episode_rewards[sid] += adjusted

        # ── Step 4: Joint reward
        joint_reward = sum(ind_rewards.values())

        # ── Step 5: Cross-store metrics
        price_variance = self._compute_price_variance(store_infos)
        demand_transfers = sum(abs(v) for v in cannibal_adjustments.values())

        self._last_obs = observations
        self._last_rewards = ind_rewards
        self.current_step += 1

        info = {
            "step":               self.current_step,
            "joint_reward":       round(joint_reward, 2),
            "ind_rewards":        {k: round(v, 2) for k, v in ind_rewards.items()},
            "episode_rewards":    {k: round(v, 2) for k, v in self._episode_rewards.items()},
            "price_variance":     round(price_variance, 4),
            "demand_transfers":   round(demand_transfers, 2),
            "store_infos":        store_infos,
        }
        return observations, ind_rewards, joint_reward, done, info

    # ── Cross-store effects ───────────────────────────────────

    def _compute_cannibalization(
        self, store_infos: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Compute demand cannibalization adjustments for each store.

        Logic: for each store i in region R, look at all other stores j
        also in region R. If j priced LOWER than i for the same SKU category,
        some of i's demand shifted to j.

        Adjustment is negative for the losing store (demand stolen from it)
        and neutral for the gaining store (the raw env already captured that).

        WHY only penalise the loser and not reward the winner?
        The winner's raw env reward already includes the extra demand.
        Double-counting would inflate rewards artificially.

        Formula per store i:
          adj_i = −α × Σ_{j≠i, same_region} max(0, avg_p_j_rev − avg_p_i_rev)
                × base_revenue_i

        where p_rev = revenue / max_revenue (relative pricing signal)
        """
        adjustments = {sid: 0.0 for sid in self.store_ids}

        for region, region_stores in self.region_to_stores.items():
            if len(region_stores) < 2:
                continue

            # Extract average revenue per store as price-level proxy
            revenues = {}
            for sid in region_stores:
                info = store_infos.get(sid, {})
                revenues[sid] = info.get("total_revenue", 0.0)

            max_rev = max(revenues.values()) + 1e-8

            for sid in region_stores:
                for other_sid in region_stores:
                    if other_sid == sid:
                        continue
                    # If other store earned more revenue (higher throughput),
                    # it may have "stolen" demand via lower prices
                    rev_diff = revenues[other_sid] - revenues[sid]
                    if rev_diff > 0:
                        # Penalise sid for being outcompeted in same region
                        cannibal = (
                            self.cannibalization_alpha
                            * (rev_diff / max_rev)
                            * revenues[sid]
                            * 0.01   # scale to not dominate raw reward
                        )
                        adjustments[sid] -= cannibal

        return adjustments

    def _compute_price_variance(
        self, store_infos: Dict[str, Dict]
    ) -> float:
        """
        Compute cross-store price variance for the same SKU across stores.

        Uses coefficient of variation (std/mean) averaged across all SKUs
        that appear in multiple stores.

        This tracks the soft constraint: cross-store variance < 10%.
        Lower is better — agents that coordinate on pricing converge here.
        """
        # Collect prices by SKU ID across all stores
        sku_prices: Dict[str, List[float]] = {}
        for sid, info in store_infos.items():
            for sku_info in info.get("skus", []):
                sku_id = sku_info["sku_id"]
                sku_prices.setdefault(sku_id, []).append(sku_info["price"])

        if not sku_prices:
            return 0.0

        variances = []
        for sku_id, prices in sku_prices.items():
            if len(prices) > 1:
                mean = np.mean(prices)
                if mean > 0:
                    cv = np.std(prices) / mean   # coefficient of variation
                    variances.append(cv)

        return float(np.mean(variances)) if variances else 0.0

    # ── Global state (for Phase 6 centralized critic) ─────────

    def get_global_state(self) -> np.ndarray:
        """
        Concatenate all store observations into one global state vector.

        Shape: (n_stores × obs_dim,)

        Used ONLY by the centralized critic during training (Phase 6).
        Individual actors never see this — they only see their local obs.
        """
        return np.concatenate(
            [self._last_obs[sid] for sid in self.store_ids],
            dtype=np.float32,
        )

    def sample_actions(self) -> Dict[str, np.ndarray]:
        """Sample random actions for all stores — used for testing."""
        return {
            sid: env.action_space.sample()
            for sid, env in self.envs.items()
        }