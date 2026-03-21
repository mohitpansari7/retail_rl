"""
agents/llm_agent.py
────────────────────
Phase 7 — LLM Strategic Layer.

The LLM acts as a strategic reasoning layer above the RL tactical layer.
It observes high-level market conditions, calls tools to gather information,
and outputs a StrategyVector that modifies the RL agent's behavior.

Architecture:
  LLM (runs every K steps) → StrategyVector → augmented RL observation

Tool-use loop (ReAct style):
  1. LLM receives market summary as text
  2. LLM calls tools to gather specific data
  3. LLM reasons over tool outputs
  4. LLM outputs StrategyVector
  5. StrategyVector appended to RL agent's observation

Interface contract:
  StrategyVector is a FIXED-SIZE float32 array — clean interface
  that doesn't change the RL network architecture.
  LLM text never touches the RL agent directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import json


# ─────────────────────────────────────────────────────────────
# Strategy Vector — the LLM → RL interface contract
# ─────────────────────────────────────────────────────────────

@dataclass
class StrategyVector:
    """
    Structured encoding of the LLM's strategic recommendation.

    This is the ONLY channel through which the LLM communicates
    with the RL agent. All fields are floats in [-1, 1] or [0, 1].

    WHY a fixed-size vector?
    The RL agent's neural network takes fixed-size input.
    Text is variable-length and discrete — incompatible.
    A float32 vector slots directly into the observation space
    without changing the network architecture at all.

    WHY these specific fields?
    They correspond to the major pricing levers a human expert
    would reason about. The LLM maps its reasoning onto these
    levers — the RL agent then interprets them as guidance.

    Dimension: 10 (fixed — part of the RL observation space contract)
    """
    # Pricing posture: -1=aggressive cuts, 0=neutral, +1=raise prices
    price_aggression: float = 0.0

    # How much to prioritise avoiding stockouts over revenue
    stockout_priority: float = 0.5

    # Promotion activity: 0=none, 1=run heavy discounts
    promotion_intensity: float = 0.0

    # Per-category relative multipliers (electronics, groceries, apparel, home)
    category_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )

    # Market signals: competitor pressure, demand forecast
    competitor_pressure: float = 0.0   # 0=low threat, 1=intense competition
    demand_forecast:     float = 0.5   # 0=falling, 1=surging

    VECTOR_DIM: int = 10  # 1+1+1+4+1+1 = 9... we add urgency below
    urgency: float = 0.0  # 0=routine, 1=crisis (stockout imminent etc.)

    def validate_and_clip(self) -> "StrategyVector":
        """
        Clip all values to valid ranges before passing to RL agent.

        WHY validate here?
        LLM outputs can be miscalibrated. This is the safety gate
        that prevents bad LLM strategy from destabilising the RL agent.
        No matter what the LLM says, the RL agent only sees safe values.
        """
        self.price_aggression    = float(np.clip(self.price_aggression,    -1.0, 1.0))
        self.stockout_priority   = float(np.clip(self.stockout_priority,    0.0, 1.0))
        self.promotion_intensity = float(np.clip(self.promotion_intensity,  0.0, 1.0))
        self.category_weights    = [float(np.clip(w, 0.0, 2.0)) for w in self.category_weights]
        self.competitor_pressure = float(np.clip(self.competitor_pressure,  0.0, 1.0))
        self.demand_forecast     = float(np.clip(self.demand_forecast,      0.0, 1.0))
        self.urgency             = float(np.clip(self.urgency,              0.0, 1.0))
        return self

    def to_array(self) -> np.ndarray:
        """
        Convert to fixed-size float32 array for RL observation augmentation.
        Shape: (10,) — appended to the RL agent's local observation.
        """
        return np.array([
            self.price_aggression,
            self.stockout_priority,
            self.promotion_intensity,
            *self.category_weights,     # 4 values
            self.competitor_pressure,
            self.demand_forecast,
            self.urgency,
        ], dtype=np.float32)            # total = 10

    @classmethod
    def neutral(cls) -> "StrategyVector":
        """Default strategy: no opinion, let RL decide freely."""
        return cls()

    @classmethod
    def from_dict(cls, d: Dict) -> "StrategyVector":
        """Parse from LLM JSON output."""
        return cls(
            price_aggression    = d.get("price_aggression",    0.0),
            stockout_priority   = d.get("stockout_priority",   0.5),
            promotion_intensity = d.get("promotion_intensity", 0.0),
            category_weights    = d.get("category_weights",    [1.0]*4),
            competitor_pressure = d.get("competitor_pressure", 0.0),
            demand_forecast     = d.get("demand_forecast",     0.5),
            urgency             = d.get("urgency",             0.0),
        ).validate_and_clip()

    STRATEGY_DIM: int = 10


# ─────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────

class RetailTools:
    """
    Tool implementations available to the LLM strategic layer.

    Each tool takes structured input, queries the environment state,
    and returns a structured dict. The LLM receives this dict as
    text (JSON) and reasons over it.

    WHY structured dicts and not text?
    Unambiguous — no parsing errors, no hallucinated numbers.
    The LLM extracts exactly what it needs and moves on.
    Consistent format regardless of which LLM backend is used.
    """

    def __init__(self, env_state_fn) -> None:
        """
        Args:
            env_state_fn: Callable that returns current environment state dict.
                          Injected dependency — tools don't own the env.
        """
        self._get_state = env_state_fn
        self._call_log: List[Dict] = []

    def analyze_demand_trend(self, category: str) -> Dict:
        """
        Get 7-day demand trend for a product category.

        Returns trend direction, magnitude, and seasonality context.
        LLM uses this to decide whether to stimulate or constrain demand.
        """
        state = self._get_state()
        category_demands = state.get("demand_history", {}).get(category, [1.0]*7)

        recent = np.array(category_demands[-7:], dtype=float)
        trend  = float(np.polyfit(range(len(recent)), recent, 1)[0])
        mean   = float(recent.mean())
        cv     = float(recent.std() / (mean + 1e-8))

        result = {
            "category":         category,
            "7day_mean_demand": round(mean, 2),
            "trend_slope":      round(trend, 3),   # positive=growing, negative=falling
            "volatility_cv":    round(cv, 3),       # coefficient of variation
            "interpretation":   (
                "surging"   if trend > 0.5 else
                "growing"   if trend > 0.0 else
                "stable"    if abs(trend) < 0.1 else
                "declining"
            ),
        }
        self._call_log.append({"tool": "analyze_demand_trend", "input": category, "output": result})
        return result

    def check_competitor_prices(self, category: str) -> Dict:
        """
        Get competitive pricing landscape for a category.

        Returns our prices vs competitor prices and the gap.
        LLM uses this to decide competitive response strategy.
        """
        state = self._get_state()
        our_avg   = state.get("our_avg_prices", {}).get(category, 100.0)
        comp_avg  = state.get("competitor_avg_prices", {}).get(category, 100.0)
        gap_pct   = (our_avg - comp_avg) / (comp_avg + 1e-8)

        result = {
            "category":           category,
            "our_avg_price":      round(our_avg, 2),
            "competitor_avg":     round(comp_avg, 2),
            "price_gap_pct":      round(gap_pct * 100, 2),  # positive = we're higher
            "competitive_stance": (
                "premium"    if gap_pct >  0.10 else
                "parity"     if gap_pct > -0.05 else
                "competitive"if gap_pct > -0.15 else
                "undercutting"
            ),
        }
        self._call_log.append({"tool": "check_competitor_prices", "input": category, "output": result})
        return result

    def get_inventory_status(self, store_id: Optional[str] = None) -> Dict:
        """
        Check stock levels — globally or for a specific store.

        Returns stockout risk level and which SKUs are critical.
        LLM uses this to set stockout_priority in the strategy.
        """
        state = self._get_state()
        inventory = state.get("inventory_levels", {})

        if store_id:
            inventory = {k: v for k, v in inventory.items()
                         if k.startswith(store_id)}

        levels = list(inventory.values()) if inventory else [100.0]
        critical_count = sum(1 for v in levels if v < 20)
        low_count      = sum(1 for v in levels if 20 <= v < 50)

        result = {
            "store_id":        store_id or "all_stores",
            "mean_inventory":  round(float(np.mean(levels)), 1),
            "critical_skus":   critical_count,   # < 20 units
            "low_skus":        low_count,         # 20-50 units
            "stockout_risk":   (
                "critical" if critical_count > 5 else
                "high"     if critical_count > 0 else
                "moderate" if low_count > 10    else
                "low"
            ),
        }
        self._call_log.append({"tool": "get_inventory_status", "input": store_id, "output": result})
        return result

    def get_call_log(self) -> List[Dict]:
        return self._call_log.copy()

    def reset_log(self) -> None:
        self._call_log = []


# ─────────────────────────────────────────────────────────────
# LLM Strategic Agent
# ─────────────────────────────────────────────────────────────

class LLMStrategicAgent:
    """
    Strategic reasoning layer that wraps an LLM backend.

    Runs every K steps (not every step — LLM calls are expensive).
    Produces a StrategyVector that gets appended to the RL agent's observation.

    Backends:
      - "mock"     : Rule-based logic. Used during training. Fast, free, deterministic.
      - "anthropic": Real Claude API. Used for evaluation/deployment.

    WHY dependency injection for the backend?
    Training uses mock → fast and cheap.
    Evaluation uses real API → accurate strategic reasoning.
    Same interface → training loop doesn't change at all.

    Args:
        strategy_interval : Steps between LLM calls. 24 = once per day.
        backend           : "mock" or "anthropic".
        api_key           : Required for "anthropic" backend.
    """

    def __init__(
        self,
        strategy_interval: int = 24,
        backend: str = "mock",
        api_key: Optional[str] = None,
    ) -> None:
        self.strategy_interval = strategy_interval
        self.backend = backend
        self.api_key = api_key
        self._step_count = 0
        self._current_strategy = StrategyVector.neutral()
        self._strategy_history: List[Dict] = []

    def should_update(self) -> bool:
        """True when it's time for a new LLM strategy call."""
        return self._step_count % self.strategy_interval == 0

    def get_strategy(
        self,
        env_state: Dict,
        tools: RetailTools,
        force: bool = False,
    ) -> StrategyVector:
        """
        Get (possibly cached) strategy for the current step.

        Only calls the LLM when should_update() is True or force=True.
        Otherwise returns the cached strategy from the last call.

        WHY cache? LLM call every 24 steps means 364/365 steps use
        the cached strategy at zero cost. The strategy stays valid for
        24 steps — enough for the RL agent to act on it meaningfully.

        Args:
            env_state : Current state dict from environment.
            tools     : RetailTools instance for this environment.
            force     : Override interval and call LLM now.

        Returns:
            StrategyVector (validated and clipped).
        """
        self._step_count += 1

        if force or self.should_update():
            if self.backend == "mock":
                strategy = self._mock_strategy(env_state, tools)
            elif self.backend == "anthropic":
                strategy = self._anthropic_strategy(env_state, tools)
            else:
                strategy = StrategyVector.neutral()

            strategy.validate_and_clip()
            self._current_strategy = strategy
            self._strategy_history.append({
                "step":     self._step_count,
                "strategy": strategy.to_array().tolist(),
            })

        return self._current_strategy

    def _mock_strategy(
        self, env_state: Dict, tools: RetailTools
    ) -> StrategyVector:
        """
        Rule-based strategy for training. Mimics what a reasonable LLM would do.

        Rules:
          - If inventory is critical → max stockout priority, cut prices to clear
          - If competitors are much cheaper → moderate price cuts
          - If demand is surging → raise prices slightly, protect margins
          - If Q4 (festive) → boost all categories, push promotions

        WHY rules and not a simpler random strategy?
        Random strategies produce nonsensical training signal.
        Rule-based mock produces strategies that correlate with
        good outcomes — the RL agent learns to respond to strategy
        vectors that actually mean something.
        """
        strategy = StrategyVector.neutral()

        # Tool 1: Check inventory risk
        inv_status = tools.get_inventory_status()
        if inv_status["stockout_risk"] in ("critical", "high"):
            strategy.stockout_priority   = 0.9
            strategy.price_aggression    = -0.2   # slight cuts to move stock
            strategy.urgency             = 0.8

        # Tool 2: Check competitor pressure
        comp_check = tools.check_competitor_prices("electronics")
        gap = comp_check["price_gap_pct"]
        if gap > 10:   # we're 10%+ above competitors
            strategy.competitor_pressure = 0.8
            strategy.price_aggression    = max(strategy.price_aggression, -0.3)
        elif gap < -10:  # we're 10%+ below competitors
            strategy.competitor_pressure = 0.2
            strategy.price_aggression    = min(strategy.price_aggression, +0.2)

        # Tool 3: Check demand trend
        demand_trend = tools.analyze_demand_trend("groceries")
        if demand_trend["interpretation"] == "surging":
            strategy.demand_forecast  = 0.8
            strategy.price_aggression = min(strategy.price_aggression + 0.1, 0.5)
        elif demand_trend["interpretation"] == "declining":
            strategy.demand_forecast      = 0.2
            strategy.promotion_intensity  = 0.4   # run promotions to stimulate

        # Seasonality: Q4 festive boost
        day_of_year = env_state.get("day_of_year", 180)
        if day_of_year > 274:   # Oct onwards (Q4)
            strategy.category_weights    = [1.3, 1.1, 1.2, 1.1]
            strategy.promotion_intensity = max(strategy.promotion_intensity, 0.3)

        return strategy

    def _anthropic_strategy(
        self, env_state: Dict, tools: RetailTools
    ) -> StrategyVector:
        """
        Real Claude API call for production/evaluation use.

        Constructs a structured prompt with market context,
        calls tools, and parses the JSON strategy output.

        The prompt follows the ReAct pattern:
          Thought → Action (tool call) → Observation → Thought → Final Answer
        """
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            # Gather tool outputs first
            inv_status   = tools.get_inventory_status()
            comp_elec    = tools.check_competitor_prices("electronics")
            comp_groc    = tools.check_competitor_prices("groceries")
            demand_elec  = tools.analyze_demand_trend("electronics")
            demand_groc  = tools.analyze_demand_trend("groceries")

            # Build prompt
            market_summary = json.dumps({
                "day_of_year":         env_state.get("day_of_year", 180),
                "inventory_status":    inv_status,
                "competitor_pricing":  {"electronics": comp_elec, "groceries": comp_groc},
                "demand_trends":       {"electronics": demand_elec, "groceries": demand_groc},
            }, indent=2)

            prompt = f"""You are a retail pricing strategist for QuickMart, a 10-store retail chain.

Current market situation:
{market_summary}

Based on this data, provide a pricing strategy as a JSON object with these exact fields:
- price_aggression: float in [-1.0, 1.0]. -1=aggressive cuts, 0=neutral, +1=raise prices
- stockout_priority: float in [0.0, 1.0]. How much to prioritise avoiding stockouts
- promotion_intensity: float in [0.0, 1.0]. How actively to run discounts
- category_weights: list of 4 floats [electronics, groceries, apparel, home_kitchen]
- competitor_pressure: float in [0.0, 1.0]. Intensity of competitive threat
- demand_forecast: float in [0.0, 1.0]. 0=demand falling, 1=demand surging
- urgency: float in [0.0, 1.0]. 0=routine, 1=crisis requiring immediate action

Respond with ONLY the JSON object, no explanation."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_json = response.content[0].text.strip()
            strategy_dict = json.loads(raw_json)
            return StrategyVector.from_dict(strategy_dict)

        except Exception as e:
            # Fallback to neutral strategy on any API error
            # Production systems must never fail hard due to LLM unavailability
            print(f"[LLMStrategicAgent] API error, using neutral strategy: {e}")
            return StrategyVector.neutral()

    def get_strategy_array(self, env_state: Dict, tools: RetailTools) -> np.ndarray:
        """
        Convenience method: get strategy as numpy array for observation augmentation.
        """
        return self.get_strategy(env_state, tools).to_array()

    @property
    def strategy_history(self) -> List[Dict]:
        return self._strategy_history.copy()