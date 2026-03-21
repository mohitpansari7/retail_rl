"""
agents/hierarchical_agent.py
─────────────────────────────
Phase 7 — Hierarchical agent: LLM strategic layer + MAPPO tactical layer.

Wires together:
  LLMStrategicAgent  →  StrategyVector
  StrategyVector     →  augmented observation
  augmented obs      →  MAPPOAgent actors

The observation augmentation is the ONLY interface between layers.
Neither layer knows the internal details of the other.

Augmented observation:
  original_local_obs (obs_dim,) + strategy_vector (10,) = (obs_dim+10,)

WHY augment at this wrapper level?
  - MAPPOAgent stays clean (unaware of LLM) — reusable without LLM
  - LLMStrategicAgent stays clean (unaware of RL internals) — swappable
  - The wrapper is the only place that knows about both
  This is the Adapter pattern — clean composition without coupling.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from agents.llm_agent import LLMStrategicAgent, RetailTools, StrategyVector
from agents.mappo_agent import MAPPOAgent
from config.settings import TrainingConfig


class HierarchicalPricingAgent:
    """
    Two-layer pricing agent: LLM strategy + MAPPO execution.

    The LLM runs every K steps and produces a StrategyVector.
    The StrategyVector is appended to each store's local observation.
    MAPPO actors receive the augmented observation and output prices.

    Args:
        store_ids         : List of store identifiers.
        obs_dim           : Original local observation dimension.
        action_dim        : Action dimension.
        global_state_dim  : Global state dim for centralized critic.
                            Uses AUGMENTED obs: n_stores × (obs_dim + STRATEGY_DIM)
        cfg               : Training config.
        strategy_interval : Steps between LLM calls (24 = daily).
        llm_backend       : "mock" for training, "anthropic" for eval.
        n_steps           : Rollout length for MAPPO buffer.
        device            : 'cpu' or 'cuda'.
    """

    STRATEGY_DIM: int = StrategyVector.STRATEGY_DIM   # = 10

    def __init__(
        self,
        store_ids: List[str],
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        cfg: Optional[TrainingConfig] = None,
        strategy_interval: int = 24,
        llm_backend: str = "mock",
        n_steps: int = 512,
        device: str = "cpu",
    ) -> None:
        self.store_ids   = store_ids
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.n_stores    = len(store_ids)

        # Augmented observation dimension — what MAPPO actually sees
        self.aug_obs_dim = obs_dim + self.STRATEGY_DIM

        # Augmented global state dim for centralized critic
        self.aug_global_dim = self.aug_obs_dim * self.n_stores

        # ── LLM strategic layer
        self.llm_agent = LLMStrategicAgent(
            strategy_interval=strategy_interval,
            backend=llm_backend,
        )

        # ── MAPPO tactical layer
        # NOTE: MAPPO is initialised with AUGMENTED obs_dim
        # so its neural networks are sized correctly from the start.
        self.mappo = MAPPOAgent(
            store_ids=store_ids,
            obs_dim=self.aug_obs_dim,          # augmented!
            action_dim=action_dim,
            global_state_dim=self.aug_global_dim,  # augmented!
            cfg=cfg,
            n_steps=n_steps,
            device=device,
        )

        # Current strategy (shared across all stores this step)
        self._current_strategy = StrategyVector.neutral()
        self._env_state: Dict = {}

        # ── Tool interface (injected with env state accessor)
        self.tools = RetailTools(env_state_fn=lambda: self._env_state)

        # Tracking
        self._total_steps: int = 0
        self._llm_calls:   int = 0
        self._training_log: List[Dict] = []

    # ── Observation augmentation ──────────────────────────────

    def _augment_obs(
        self,
        local_obs: Dict[str, np.ndarray],
        strategy_vec: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Append strategy vector to each store's local observation.

        Augmentation: (obs_dim,) + (STRATEGY_DIM,) = (aug_obs_dim,)

        WHY the same strategy vector for all stores?
        The LLM reasons about the whole chain, not individual stores.
        "It's Diwali, boost electronics" applies equally to all stores.
        Store-specific adjustments happen through the RL actor's own
        learned behavior given the shared strategic context.

        Args:
            local_obs    : {store_id: (obs_dim,) array}
            strategy_vec : (STRATEGY_DIM,) strategy encoding

        Returns:
            {store_id: (aug_obs_dim,) augmented array}
        """
        return {
            sid: np.concatenate([obs, strategy_vec], dtype=np.float32)
            for sid, obs in local_obs.items()
        }

    def _build_augmented_global_state(
        self,
        aug_obs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Build augmented global state for centralized critic.
        Shape: (n_stores × aug_obs_dim,)
        """
        return np.concatenate(
            [aug_obs[sid] for sid in self.store_ids], dtype=np.float32
        )

    # ── Main interface ────────────────────────────────────────

    def update_env_state(self, env_state: Dict) -> None:
        """
        Update the environment state visible to the LLM tools.

        Called after each env.step() with the latest info dict.
        The tools query this state when the LLM calls them.
        """
        self._env_state = env_state

    def select_actions(
        self,
        local_obs: Dict[str, np.ndarray],
        env_state: Optional[Dict] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Full hierarchical action selection.

        Pipeline:
          1. Update env state for tools
          2. Get LLM strategy (cached or fresh call)
          3. Augment all store observations with strategy vector
          4. Pass augmented obs to MAPPO actors
          5. Return actions

        The LLM is only called every strategy_interval steps.
        All other steps use the cached strategy at zero cost.

        Args:
            local_obs : {store_id: local observation}
            env_state : Current environment state for LLM tools.

        Returns:
            {store_id: action array}
        """
        if env_state:
            self.update_env_state(env_state)
        self._env_state["day_of_year"] = (self._total_steps % 365) + 1

        # LLM strategy (fresh or cached)
        strategy = self.llm_agent.get_strategy(self._env_state, self.tools)
        strategy_vec = strategy.to_array()

        if self.llm_agent.should_update() or self._total_steps == 0:
            self._llm_calls += 1

        self._current_strategy = strategy

        # Augment observations
        aug_obs = self._augment_obs(local_obs, strategy_vec)

        # MAPPO actor selection (decentralized, each on its own aug obs)
        actions = self.mappo.select_actions(aug_obs)

        self._total_steps += 1
        return actions

    def store_transition(
        self,
        local_obs: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        global_state: np.ndarray,
        joint_reward: float,
        done: bool,
    ) -> None:
        """
        Store transition with AUGMENTED observations in MAPPO buffer.

        We store the augmented obs and augmented global state —
        not the raw ones — because MAPPO's networks expect augmented input.
        """
        strategy_vec = self._current_strategy.to_array()
        aug_obs      = self._augment_obs(local_obs, strategy_vec)
        aug_global   = self._build_augmented_global_state(aug_obs)

        self.mappo.store_transition(
            local_obs=aug_obs,
            actions=actions,
            global_state=aug_global,
            joint_reward=joint_reward,
            done=done,
        )

    def buffer_full(self) -> bool:
        return self.mappo.buffer_full()

    def update(self, last_local_obs: Dict[str, np.ndarray]) -> Dict:
        """
        Update MAPPO using augmented observations.
        LLM parameters are NOT updated here (no gradient through LLM).
        """
        strategy_vec = self._current_strategy.to_array()
        aug_last_obs = self._augment_obs(last_local_obs, strategy_vec)
        aug_last_global = self._build_augmented_global_state(aug_last_obs)

        metrics = self.mappo.update(last_global_state=aug_last_global)
        metrics["llm_calls_total"] = self._llm_calls
        metrics["current_strategy"] = self._current_strategy.to_array().tolist()

        self._training_log.append(metrics)
        return metrics

    @property
    def strategy_history(self) -> List[Dict]:
        return self.llm_agent.strategy_history

    def get_training_log(self) -> List[Dict]:
        return self._training_log