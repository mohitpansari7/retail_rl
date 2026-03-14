"""
agents/
───────
All RL agents. Each phase adds one:
  Phase 2 : ReinforceAgent  — REINFORCE with baseline
  Phase 3 : PPOAgent        — Proximal Policy Optimisation
  Phase 6 : MAPPOAgent      — Multi-Agent PPO (CTDE)
  Phase 7 : LLMAgent        — LLM-augmented hierarchical agent
  Phase 8 : GRPOAgent       — Group Relative Policy Optimisation

All inherit from BaseAgent and implement the same interface:
  select_action(obs) → action
  store_transition(action, reward)
  update() → metrics dict
"""

from agents.base_agent import BaseAgent

__all__ = ["BaseAgent"]