"""
safety/constraints.py
──────────────────────
Phase 9 — Safety constraint enforcement and Lagrangian optimization.

Layer 1 — ConstraintProjector:
  Pre-flight check. Clips actions to hard constraint bounds.
  Logs every violation attempt — early warning system.

Layer 2 — LagrangianConstraintOptimizer:
  Soft constraint enforcement via self-tuning Lagrange multipliers.
  L(θ,λ) = J(θ) − Σ_i λ_i·(C_i(θ) − d_i)
  λ_i increases when constraint violated, decreases when satisfied.

λ update rule:
  λ_i ← max(0, min(λ_max, λ_i + α_λ · (C_i − d_i)))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import HardConstraints, SoftConstraints


@dataclass
class ConstraintViolation:
    """Record of one constraint violation attempt — for monitoring dashboards."""
    step:          int
    store_id:      str
    constraint:    str
    severity:      float     # 0=borderline, 1=severe
    action_before: float
    action_after:  float
    detail:        str


class ConstraintProjector:
    """
    Layer 1: Projects proposed actions onto feasible set before env sees them.

    WHY log violations even when clipped?
    Frequent proposals of illegal actions = policy drifting toward boundaries.
    Logging gives operators an early warning BEFORE it becomes a crisis.
    The environment clips anyway — this layer provides the audit trail.
    """

    def __init__(self) -> None:
        self._violation_log: List[ConstraintViolation] = []
        self._step: int = 0
        self._consecutive_boundary_hits: Dict[str, int] = {}

    def project_price_action(
        self,
        action: np.ndarray,
        store_id: str = "store_0",
    ) -> Tuple[np.ndarray, List[ConstraintViolation]]:
        """
        Clip price change actions to [-1, 1] (the ±15% hard limit).
        Flag boundary exploitation when >50% of actions saturate at the boundary.

        Returns:
            safe_action : clipped action array
            violations  : list of ConstraintViolation (empty if clean)
        """
        self._step += 1
        violations = []

        boundary_threshold = 0.90   # 90% of max = "at the boundary"
        at_boundary = np.abs(action) >= boundary_threshold
        n_boundary  = int(at_boundary.sum())
        safe_action = np.clip(action, -1.0, 1.0)

        if n_boundary > len(action) * 0.5:
            severity = n_boundary / len(action)
            v = ConstraintViolation(
                step=self._step,
                store_id=store_id,
                constraint="boundary_exploitation",
                severity=severity,
                action_before=float(np.abs(action).mean()),
                action_after=float(np.abs(safe_action).mean()),
                detail=f"{n_boundary}/{len(action)} actions at ≥90% boundary"
            )
            violations.append(v)
            self._violation_log.append(v)
            key = f"{store_id}_boundary"
            self._consecutive_boundary_hits[key] = (
                self._consecutive_boundary_hits.get(key, 0) + 1
            )
        else:
            self._consecutive_boundary_hits[f"{store_id}_boundary"] = 0

        return safe_action, violations

    def check_margin_constraint(
        self,
        prices: np.ndarray,
        costs: np.ndarray,
        store_id: str = "store_0",
    ) -> List[ConstraintViolation]:
        """
        Log any prices below cost × (1 + MIN_MARGIN_PCT).
        Env enforces this physically; we add the audit trail here.
        """
        violations = []
        floors   = costs * (1.0 + HardConstraints.MIN_MARGIN_PCT)
        failing  = prices < floors

        if failing.any():
            n_fail   = int(failing.sum())
            shortfall = float((floors[failing] - prices[failing]).max())
            severity  = min(1.0, shortfall / (costs.mean() + 1e-8))
            v = ConstraintViolation(
                step=self._step,
                store_id=store_id,
                constraint="margin_constraint",
                severity=severity,
                action_before=float(prices[failing].mean()),
                action_after=float(floors[failing].mean()),
                detail=f"{n_fail} SKUs below floor. Max shortfall: {shortfall:.2f}"
            )
            violations.append(v)
            self._violation_log.append(v)

        return violations

    def get_violation_log(self) -> List[ConstraintViolation]:
        return self._violation_log.copy()

    def get_violation_count(self, constraint: Optional[str] = None) -> int:
        if constraint:
            return sum(1 for v in self._violation_log if v.constraint == constraint)
        return len(self._violation_log)

    def get_consecutive_boundary_hits(self, store_id: str) -> int:
        return self._consecutive_boundary_hits.get(f"{store_id}_boundary", 0)

    def reset_log(self) -> None:
        self._violation_log = []


@dataclass
class SoftConstraintConfig:
    """
    Configuration for one soft constraint in the Lagrangian optimizer.

    threshold  : Allowed violation level (d_i). Below this → no penalty.
    lambda_lr  : How fast the multiplier responds to violations (α_λ).
    lambda_max : Hard cap on λ. Prevents training collapse when agent
                 repeatedly violates — without cap λ → ∞ and revenue
                 objective disappears completely.
    """
    name:        str
    threshold:   float
    lambda_init: float = 0.0
    lambda_lr:   float = 0.01
    lambda_max:  float = 50.0


class LagrangianConstraintOptimizer:
    """
    Layer 2: Self-tuning soft constraint enforcement.

    Lagrangian relaxation:
        L(θ,λ) = J(θ) − Σ_i λ_i · (C_i(θ) − d_i)

    θ updated via gradient descent (minimise policy loss).
    λ updated via gradient ascent (maximise Lagrangian over λ):
        λ_i ← max(0, min(λ_max, λ_i + α_λ · (C_i − d_i)))

    The self-tuning property:
        Constraint violated → λ increases → penalty heavier → agent avoids it
        Constraint satisfied → λ decreases → penalty lighter → agent has freedom
    No manual weight tuning needed.
    """

    def __init__(
        self,
        constraints: Optional[List[SoftConstraintConfig]] = None,
    ) -> None:
        self.constraints = constraints or [
            SoftConstraintConfig(
                name="stockout_rate",
                threshold=SoftConstraints.MAX_STOCKOUT_RATE,
                lambda_lr=0.02, lambda_max=30.0,
            ),
            SoftConstraintConfig(
                name="price_variance",
                threshold=SoftConstraints.MAX_CROSS_STORE_PRICE_VARIANCE_PCT,
                lambda_lr=0.01, lambda_max=20.0,
            ),
            SoftConstraintConfig(
                name="price_smoothness",
                threshold=0.05,   # avg daily change < 5%
                lambda_lr=0.015,  lambda_max=25.0,
            ),
            SoftConstraintConfig(
                name="inventory_health",
                threshold=0.15,   # depletion rate < 15%/week
                lambda_lr=0.01,   lambda_max=20.0,
            ),
        ]
        self.lambdas: Dict[str, float] = {
            c.name: c.lambda_init for c in self.constraints
        }
        self._history: List[Dict] = []

    def compute_penalty(self, constraint_values: Dict[str, float]) -> float:
        """
        Total Lagrangian penalty to ADD to policy loss.

        penalty = Σ_i λ_i · max(0, C_i − d_i)

        Only penalises actual violations (max with 0).
        Satisfied constraints contribute nothing.
        """
        total = 0.0
        for c in self.constraints:
            current   = constraint_values.get(c.name, 0.0)
            violation = max(0.0, current - c.threshold)
            total    += self.lambdas[c.name] * violation
        return total

    def update_lambdas(
        self, constraint_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update multipliers after each policy update.

        λ_i ← max(0, min(λ_max, λ_i + α_λ · (C_i − d_i)))

        max(0)     : λ must be non-negative (negative = reward violations)
        min(λ_max) : cap prevents collapse (λ→∞ kills revenue objective)

        Returns updated λ dict for logging.
        """
        updated = {}
        for c in self.constraints:
            current    = constraint_values.get(c.name, 0.0)
            violation  = current - c.threshold           # positive = violated
            new_lambda = self.lambdas[c.name] + c.lambda_lr * violation
            new_lambda = max(0.0, min(c.lambda_max, new_lambda))
            self.lambdas[c.name] = new_lambda
            updated[c.name] = new_lambda

        self._history.append({
            "lambdas": updated.copy(),
            "constraint_values": constraint_values.copy(),
        })
        return updated

    def get_active_constraints(self, threshold: float = 1.0) -> List[str]:
        """
        Constraints with λ > threshold are actively being enforced.
        High λ = agent is currently struggling with that constraint.
        Useful dashboard metric for operators.
        """
        return [n for n, lam in self.lambdas.items() if lam > threshold]

    def get_lambda_history(self) -> List[Dict]:
        return self._history.copy()

    def reset(self) -> None:
        for c in self.constraints:
            self.lambdas[c.name] = c.lambda_init
        self._history = []