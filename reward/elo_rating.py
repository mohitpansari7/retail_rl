"""
reward/elo_rating.py
─────────────────────
Phase 4 — Elo rating system for pricing strategies.

Tracks strategy-level quality using the same Elo system used in chess.
Every strategy starts at 1500. After each head-to-head comparison,
ratings update based on the actual vs expected outcome.

WHY Elo and not just average return?
  Average return is a point estimate with no sense of uncertainty.
  Elo is self-calibrating: a win against a strong opponent counts more
  than a win against a weak one. After enough comparisons, ratings
  converge to a stable ranking that reflects true relative quality.

Formulas:
  Expected score:  E_A = 1 / (1 + 10^((R_B − R_A) / 400))
  Rating update:   R'_A = R_A + K × (S_A − E_A)
    where S_A = 1 (win), 0.5 (draw), 0 (loss)
    and   K   = update speed (32 standard, higher = more volatile)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class StrategyRecord:
    """
    Tracks the Elo rating and match history for one pricing strategy.

    A "strategy" is identified by a string key — typically:
      - A policy checkpoint name: "ppo_ep100", "ppo_ep200"
      - A reward type: "hand_crafted", "learned_rm"
      - An experiment tag: "high_lr", "low_entropy"
    """
    name: str
    rating: float = 1500.0
    wins:   int = 0
    losses: int = 0
    draws:  int = 0
    rating_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.rating_history.append(self.rating)

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


class EloTracker:
    """
    Manages Elo ratings for a collection of pricing strategies.

    Usage:
        tracker = EloTracker(k_factor=32)
        tracker.register("ppo_v1")
        tracker.register("ppo_v2")

        # After comparing episodes:
        tracker.record_match("ppo_v1", "ppo_v2", winner="ppo_v1")

        # Get leaderboard:
        print(tracker.leaderboard())

    Args:
        k_factor       : Controls rating volatility.
                         32  = standard chess (fast convergence)
                         16  = more stable (slower convergence)
                         Higher K → bigger rating swings per match.
        initial_rating : Starting Elo for every new strategy.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
    ) -> None:
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self._strategies: Dict[str, StrategyRecord] = {}

    # ── Registration ──────────────────────────────────────────

    def register(self, name: str) -> None:
        """
        Add a new strategy to the tracker.
        Safe to call multiple times — ignores duplicates.
        """
        if name not in self._strategies:
            self._strategies[name] = StrategyRecord(
                name=name, rating=self.initial_rating
            )

    def get_rating(self, name: str) -> float:
        """Return current Elo rating for a strategy."""
        return self._strategies[name].rating

    # ── Core Elo math ─────────────────────────────────────────

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Expected score (win probability) for player A against player B.

        E_A = 1 / (1 + 10^((R_B − R_A) / 400))

        The 400 is a scale factor from chess tradition:
          - 400-point difference → 10:1 odds in favour of higher-rated
          - 200-point difference → ~3:1 odds
          - Equal ratings → 0.5 (50/50)

        Returns:
            Float in (0, 1) — probability A wins.
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _update_rating(
        self,
        current_rating: float,
        expected_score: float,
        actual_score: float,
    ) -> float:
        """
        Update Elo rating after one match.

        R' = R + K × (S − E)
          S = actual outcome (1=win, 0.5=draw, 0=loss)
          E = expected outcome (probability of winning)
          K = k_factor (how much one match can change the rating)

        If actual > expected (upset win): rating increases more
        If actual < expected (expected win): rating increases less
        """
        return current_rating + self.k_factor * (actual_score - expected_score)

    # ── Match recording ───────────────────────────────────────

    def record_match(
        self,
        name_a: str,
        name_b: str,
        winner: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Record a head-to-head match and update both ratings.

        Args:
            name_a  : First strategy's name.
            name_b  : Second strategy's name.
            winner  : name_a, name_b, or None (draw).

        Returns:
            (new_rating_a, new_rating_b) after the update.
        """
        # Auto-register unknown strategies
        self.register(name_a)
        self.register(name_b)

        rec_a = self._strategies[name_a]
        rec_b = self._strategies[name_b]

        # Determine actual scores
        if winner == name_a:
            s_a, s_b = 1.0, 0.0
            rec_a.wins   += 1
            rec_b.losses += 1
        elif winner == name_b:
            s_a, s_b = 0.0, 1.0
            rec_a.losses += 1
            rec_b.wins   += 1
        else:   # draw
            s_a, s_b = 0.5, 0.5
            rec_a.draws += 1
            rec_b.draws += 1

        # Expected scores
        e_a = self._expected_score(rec_a.rating, rec_b.rating)
        e_b = self._expected_score(rec_b.rating, rec_a.rating)

        # Update ratings
        rec_a.rating = self._update_rating(rec_a.rating, e_a, s_a)
        rec_b.rating = self._update_rating(rec_b.rating, e_b, s_b)

        # Track history (for plotting learning curves)
        rec_a.rating_history.append(rec_a.rating)
        rec_b.rating_history.append(rec_b.rating)

        return rec_a.rating, rec_b.rating

    def record_match_by_returns(
        self,
        name_a: str,
        name_b: str,
        return_a: float,
        return_b: float,
        draw_threshold: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Record a match where the winner is determined by episode returns.

        Convenience method: compares return_a vs return_b and calls
        record_match() with the appropriate winner.

        Args:
            return_a, return_b  : Total episode returns for each strategy.
            draw_threshold      : If |return_a − return_b| / max < threshold,
                                  declare a draw. Prevents noise from mattering.

        Returns:
            (new_rating_a, new_rating_b)
        """
        max_return = max(abs(return_a), abs(return_b), 1e-8)
        relative_diff = abs(return_a - return_b) / max_return

        if relative_diff < draw_threshold:
            winner = None   # draw
        elif return_a > return_b:
            winner = name_a
        else:
            winner = name_b

        return self.record_match(name_a, name_b, winner=winner)

    # ── Leaderboard ───────────────────────────────────────────

    def leaderboard(self) -> List[StrategyRecord]:
        """
        Return all strategies sorted by rating (highest first).
        """
        return sorted(
            self._strategies.values(),
            key=lambda r: r.rating,
            reverse=True,
        )

    def summary(self) -> str:
        """
        Human-readable leaderboard string for logging.
        """
        lines = ["Elo Leaderboard:", "-" * 50]
        for rank, rec in enumerate(self.leaderboard(), 1):
            lines.append(
                f"  {rank:>2}. {rec.name:<25} "
                f"Rating: {rec.rating:>7.1f}  "
                f"W/L/D: {rec.wins}/{rec.losses}/{rec.draws}  "
                f"Win%: {rec.win_rate:.1%}"
            )
        return "\n".join(lines)