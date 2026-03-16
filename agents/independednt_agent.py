"""
agents/independent_agent.py
─────────────────────────────
Phase 5 — Independent learners with joint reward.

Each store has its own PPOAgent. They act independently (no communication)
but all receive the joint reward R_total = sum of all store rewards.

WHY this is Phase 5 and not the final solution:
  Joint reward aligns incentives (prevents price wars) but doesn't solve
  the credit assignment problem. Each agent still can't tell whether its
  reward came from its own actions or the other 9 agents' actions.
  This causes slow, noisy learning — which CTDE (Phase 6) fixes.

IndependentMARL:
  - Owns N PPOAgents, one per store
  - Collects actions from all agents each step
  - Distributes JOINT reward to all agents (key difference from IL)
  - Updates all agents when their individual buffers are full
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from agents.ppo_agent import PPOAgent
from config.settings import TrainingConfig


class IndependentMARL:
    """
    Coordinator for N independent PPO agents sharing a joint reward.

    This is the Phase 5 baseline. It demonstrates:
      ✓ Joint reward prevents price wars (incentive alignment)
      ✗ Credit assignment problem remains (can't tell whose action caused reward)
      ✗ Non-stationarity remains (each agent's environment still shifts)

    Phase 6 (CTDE) solves the remaining two problems.

    Args:
        store_ids  : List of store identifiers.
        obs_dim    : Observation dimension per store.
        action_dim : Action dimension per store.
        cfg        : Training config (shared across all agents).
        n_steps    : Rollout length per agent before update.
        device     : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        store_ids: List[str],
        obs_dim: int,
        action_dim: int,
        cfg: Optional[TrainingConfig] = None,
        n_steps: int = 512,       # shorter rollout per agent for faster experiments
        device: str = "cpu",
    ) -> None:
        self.store_ids = store_ids
        self.cfg = cfg or TrainingConfig()
        self.n_stores = len(store_ids)

        # One PPOAgent per store — identical architecture, independent weights
        # WHY independent weights? Each store has different regional characteristics
        # (urban vs suburban vs rural). Separate weights let each agent specialise.
        # Phase 6 will show how a shared critic helps despite separate actors.
        self.agents: Dict[str, PPOAgent] = {
            sid: PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                cfg=cfg,
                n_steps=n_steps,
                device=device,
            )
            for sid in store_ids
        }

        self.total_steps: int = 0
        self.update_count: int = 0
        self._training_log: List[Dict] = []

        # Track non-stationarity: measure how much agents' policies change
        # per update. High variance = high non-stationarity.
        self._policy_change_history: List[float] = []

    # ── Acting ────────────────────────────────────────────────

    def select_actions(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Select actions for all stores simultaneously.

        Each agent acts independently using only its own observation.
        No agent sees what the others are doing — pure decentralized execution.

        Args:
            observations : Dict[store_id → local_obs]

        Returns:
            Dict[store_id → action_array]
        """
        return {
            sid: self.agents[sid].select_action(observations[sid])
            for sid in self.store_ids
        }

    def store_transitions(
        self,
        actions: Dict[str, np.ndarray],
        joint_reward: float,
        done: bool,
    ) -> None:
        """
        Store transitions for all agents using the JOINT reward.

        This is the key Phase 5 mechanism:
        Every agent receives the SAME reward — the total chain revenue.
        This aligns incentives: actions that hurt other stores also
        hurt the agent that took them.

        WHY joint reward instead of individual?
        Individual reward → Store 3 maximises its own revenue even at
        Store 7's expense (price wars, demand cannibalization).
        Joint reward → Store 3's policy gradient penalises it for
        harming Store 7 because Store 7's loss is Store 3's loss too.

        Args:
            actions      : Dict[store_id → action taken]
            joint_reward : R_total = sum of all stores' rewards this step
            done         : Episode end flag
        """
        for sid in self.store_ids:
            self.agents[sid].store_transition(
                action=actions[sid],
                reward=joint_reward / self.n_stores,  # normalise by n_stores
                done=done,
            )
        self.total_steps += 1

    # ── Learning ──────────────────────────────────────────────

    def update_ready(self) -> bool:
        """True when at least one agent's buffer is full."""
        return any(agent.buffer_full() for agent in self.agents.values())

    def update(
        self, last_observations: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Update all agents whose buffers are full.

        Each agent updates independently — no shared information.
        This is the limitation Phase 6 will address.

        Returns:
            Dict[store_id → metrics] for agents that updated.
        """
        all_metrics = {}

        for sid, agent in self.agents.items():
            if agent.buffer_full():
                metrics = agent.update(last_obs=last_observations[sid])
                all_metrics[sid] = metrics

        if all_metrics:
            # Track non-stationarity: average policy loss change across agents
            # High variance in policy losses = agents are changing rapidly
            # = high non-stationarity for each other
            losses = [m.get("policy_loss", 0) for m in all_metrics.values()]
            self._policy_change_history.append(float(np.std(losses)))
            self.update_count += 1

            summary = {
                "update": self.update_count,
                "n_agents_updated": len(all_metrics),
                "mean_policy_loss": float(np.mean([
                    m.get("policy_loss", 0) for m in all_metrics.values()
                ])),
                "policy_loss_std": float(np.std([
                    m.get("policy_loss", 0) for m in all_metrics.values()
                ])),
                "non_stationarity_proxy": self._policy_change_history[-1],
            }
            self._training_log.append(summary)
            return summary

        return {}

    # ── Diagnostics ───────────────────────────────────────────

    def non_stationarity_trend(self) -> float:
        """
        Returns the trend in policy change variance over recent updates.

        Positive → non-stationarity increasing (agents diverging)
        Negative → non-stationarity decreasing (agents stabilising)
        Near 0   → stable (good sign, or stagnation)

        This metric is what Phase 6 (CTDE) should improve dramatically.
        """
        if len(self._policy_change_history) < 10:
            return 0.0
        recent = self._policy_change_history[-10:]
        # Linear trend: positive slope = increasing non-stationarity
        x = np.arange(len(recent))
        slope = float(np.polyfit(x, recent, 1)[0])
        return slope

    def get_training_log(self) -> List[Dict]:
        return self._training_log

    # ── Persistence ───────────────────────────────────────────

    def save(self, dir_path: str) -> None:
        """Save all agent weights to a directory."""
        import os
        os.makedirs(dir_path, exist_ok=True)
        for sid, agent in self.agents.items():
            agent.save(os.path.join(dir_path, f"{sid}.pt"))

    def load(self, dir_path: str) -> None:
        """Load all agent weights from a directory."""
        import os
        for sid, agent in self.agents.items():
            path = os.path.join(dir_path, f"{sid}.pt")
            if os.path.exists(path):
                agent.load(path)