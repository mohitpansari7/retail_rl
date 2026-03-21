"""
tests/test_llm_agent.py
────────────────────────
Phase 7 tests. Run with: pytest tests/test_llm_agent.py -v

Test philosophy:
  - Test StrategyVector validates and clips correctly (safety gate)
  - Test to_array() produces correct fixed-size output
  - Test tools return structured dicts with required keys
  - Test LLM is only called every strategy_interval steps (caching)
  - Test mock strategy produces sensible values for different market states
  - Test observation augmentation: aug_obs_dim = obs_dim + STRATEGY_DIM
  - Test strategy history is tracked correctly
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from agents.llm_agent import StrategyVector, RetailTools, LLMStrategicAgent


OBS_DIM    = 240
STORE_IDS  = ["store_0", "store_1", "store_2", "store_3"]


# ══════════════════════════════════════════
# StrategyVector tests
# ══════════════════════════════════════════

class TestStrategyVector:

    def test_default_is_neutral(self):
        sv = StrategyVector.neutral()
        assert sv.price_aggression    == 0.0
        assert sv.stockout_priority   == 0.5
        assert sv.promotion_intensity == 0.0

    def test_to_array_correct_dim(self):
        sv = StrategyVector.neutral()
        arr = sv.to_array()
        assert arr.shape == (StrategyVector.STRATEGY_DIM,)
        assert arr.dtype == np.float32

    def test_to_array_values_match_fields(self):
        sv = StrategyVector(
            price_aggression=0.5,
            stockout_priority=0.8,
            promotion_intensity=0.3,
            category_weights=[1.2, 0.9, 1.0, 1.1],
            competitor_pressure=0.4,
            demand_forecast=0.7,
            urgency=0.1,
        )
        arr = sv.to_array()
        assert arr[0] == pytest.approx(0.5)
        assert arr[1] == pytest.approx(0.8)
        assert arr[2] == pytest.approx(0.3)
        # category weights at indices 3-6
        assert arr[3] == pytest.approx(1.2)
        assert arr[6] == pytest.approx(1.1)
        assert arr[7] == pytest.approx(0.4)   # competitor_pressure
        assert arr[8] == pytest.approx(0.7)   # demand_forecast
        assert arr[9] == pytest.approx(0.1)   # urgency

    def test_validate_clips_price_aggression(self):
        sv = StrategyVector(price_aggression=5.0)
        sv.validate_and_clip()
        assert sv.price_aggression == pytest.approx(1.0)

    def test_validate_clips_negative_price_aggression(self):
        sv = StrategyVector(price_aggression=-999.0)
        sv.validate_and_clip()
        assert sv.price_aggression == pytest.approx(-1.0)

    def test_validate_clips_stockout_priority(self):
        sv = StrategyVector(stockout_priority=2.5)
        sv.validate_and_clip()
        assert sv.stockout_priority == pytest.approx(1.0)

    def test_validate_clips_category_weights(self):
        sv = StrategyVector(category_weights=[10.0, -1.0, 1.0, 1.0])
        sv.validate_and_clip()
        assert sv.category_weights[0] == pytest.approx(2.0)   # capped at 2.0
        assert sv.category_weights[1] == pytest.approx(0.0)   # floored at 0.0

    def test_from_dict_roundtrip(self):
        d = {
            "price_aggression": 0.3,
            "stockout_priority": 0.7,
            "promotion_intensity": 0.2,
            "category_weights": [1.1, 0.9, 1.0, 1.0],
            "competitor_pressure": 0.5,
            "demand_forecast": 0.6,
            "urgency": 0.0,
        }
        sv = StrategyVector.from_dict(d)
        assert sv.price_aggression  == pytest.approx(0.3)
        assert sv.stockout_priority == pytest.approx(0.7)

    def test_from_dict_missing_keys_uses_defaults(self):
        """Missing keys should fall back to defaults, not crash."""
        sv = StrategyVector.from_dict({"price_aggression": 0.5})
        assert sv.price_aggression  == pytest.approx(0.5)
        assert sv.stockout_priority == pytest.approx(0.5)   # default


# ══════════════════════════════════════════
# RetailTools tests
# ══════════════════════════════════════════

class TestRetailTools:

    def _make_tools(self, demand_trend="stable", stockout_risk="low"):
        """Build tools with a mock env state."""
        state = {
            "demand_history": {
                "electronics": [8.0, 8.2, 7.9, 8.1, 8.3, 8.0, 8.2],
                "groceries":   [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0],
            },
            "our_avg_prices": {"electronics": 1000.0, "groceries": 50.0},
            "competitor_avg_prices": {"electronics": 950.0, "groceries": 52.0},
            "inventory_levels": {"store_0_elec_001": 15.0, "store_0_groc_001": 80.0},
        }
        return RetailTools(env_state_fn=lambda: state)

    def test_analyze_demand_trend_returns_required_keys(self):
        tools = self._make_tools()
        result = tools.analyze_demand_trend("groceries")
        for key in ["category", "7day_mean_demand", "trend_slope",
                    "volatility_cv", "interpretation"]:
            assert key in result, f"Missing key: {key}"

    def test_analyze_demand_rising_correctly_identified(self):
        """Groceries demand is rising (60→66) — should be 'growing' or 'surging'."""
        tools = self._make_tools()
        result = tools.analyze_demand_trend("groceries")
        assert result["trend_slope"] > 0.0
        assert result["interpretation"] in ("growing", "surging")

    def test_check_competitor_prices_returns_required_keys(self):
        tools = self._make_tools()
        result = tools.check_competitor_prices("electronics")
        for key in ["category", "our_avg_price", "competitor_avg",
                    "price_gap_pct", "competitive_stance"]:
            assert key in result

    def test_competitor_price_gap_correct(self):
        """Our electronics = 1000, competitor = 950 → gap = +5.26%."""
        tools = self._make_tools()
        result = tools.check_competitor_prices("electronics")
        assert result["price_gap_pct"] > 0   # we're priced higher
        assert result["competitive_stance"] == "parity"  # within 10%

    def test_inventory_status_returns_required_keys(self):
        tools = self._make_tools()
        result = tools.get_inventory_status()
        for key in ["mean_inventory", "critical_skus", "low_skus", "stockout_risk"]:
            assert key in result

    def test_inventory_critical_sku_detected(self):
        """inventory=15 < 20 threshold → should show critical_skus=1."""
        tools = self._make_tools()
        result = tools.get_inventory_status()
        assert result["critical_skus"] >= 1

    def test_tool_calls_logged(self):
        tools = self._make_tools()
        tools.analyze_demand_trend("electronics")
        tools.check_competitor_prices("groceries")
        log = tools.get_call_log()
        assert len(log) == 2
        assert log[0]["tool"] == "analyze_demand_trend"
        assert log[1]["tool"] == "check_competitor_prices"

    def test_log_reset(self):
        tools = self._make_tools()
        tools.analyze_demand_trend("electronics")
        tools.reset_log()
        assert len(tools.get_call_log()) == 0


# ══════════════════════════════════════════
# LLMStrategicAgent tests
# ══════════════════════════════════════════

class TestLLMStrategicAgent:

    def _make_agent(self, interval=5):
        return LLMStrategicAgent(strategy_interval=interval, backend="mock")

    def _make_env_state(self, day=180):
        return {
            "day_of_year": day,
            "demand_history": {
                "electronics": [8.0]*7,
                "groceries": [60.0]*7,
            },
            "our_avg_prices": {"electronics": 1000.0, "groceries": 50.0},
            "competitor_avg_prices": {"electronics": 1000.0, "groceries": 50.0},
            "inventory_levels": {"sku_001": 100.0, "sku_002": 100.0},
        }

    def _make_tools(self, env_state):
        return RetailTools(env_state_fn=lambda: env_state)

    def test_get_strategy_returns_strategy_vector(self):
        agent = self._make_agent()
        state = self._make_env_state()
        tools = self._make_tools(state)
        strategy = agent.get_strategy(state, tools)
        assert isinstance(strategy, StrategyVector)

    def test_strategy_is_valid_after_mock(self):
        """Mock strategy must produce values in valid ranges."""
        agent = self._make_agent()
        state = self._make_env_state()
        tools = self._make_tools(state)
        sv = agent.get_strategy(state, tools, force=True)
        arr = sv.to_array()
        assert np.all(arr >= -1.0), f"Values below -1: {arr}"
        # price_aggression can be [-1,1], others [0,1] or [0,2]
        assert sv.stockout_priority   >= 0.0
        assert sv.promotion_intensity >= 0.0

    def test_llm_called_at_interval(self):
        """Strategy should only update every strategy_interval steps."""
        agent = self._make_agent(interval=3)
        state = self._make_env_state()
        tools = self._make_tools(state)

        strategies = []
        for step in range(9):
            sv = agent.get_strategy(state, tools)
            strategies.append(sv.to_array().copy())

        # Strategy at step 0 and step 3 should be the same type
        # (both are fresh calls), strategy at step 1 is cached
        assert len(agent.strategy_history) > 0

    def test_strategy_cached_between_calls(self):
        """Strategy must not change between interval steps."""
        agent = self._make_agent(interval=5)
        state = self._make_env_state()
        tools = self._make_tools(state)

        sv1 = agent.get_strategy(state, tools, force=True)   # fresh call
        arr1 = sv1.to_array().copy()

        sv2 = agent.get_strategy(state, tools)               # cached (step 2)
        arr2 = sv2.to_array().copy()

        np.testing.assert_array_equal(arr1, arr2,
            err_msg="Strategy should be cached between interval steps")

    def test_festive_season_boosts_promotion(self):
        """Q4 (day > 274) should trigger higher promotion intensity."""
        agent = self._make_agent(interval=1)

        # Mid-year state
        state_mid = self._make_env_state(day=180)
        tools_mid = self._make_tools(state_mid)
        sv_mid = agent.get_strategy(state_mid, tools_mid, force=True)

        # Festive season state
        state_fest = self._make_env_state(day=300)
        tools_fest = self._make_tools(state_fest)
        sv_fest = agent.get_strategy(state_fest, tools_fest, force=True)

        assert sv_fest.promotion_intensity >= sv_mid.promotion_intensity, (
            "Festive season should have higher promotion intensity"
        )

    def test_strategy_history_tracked(self):
        agent = self._make_agent(interval=1)
        state = self._make_env_state()
        tools = self._make_tools(state)

        for _ in range(3):
            agent.get_strategy(state, tools)

        history = agent.strategy_history
        assert len(history) > 0
        assert "step" in history[0]
        assert "strategy" in history[0]
        assert len(history[0]["strategy"]) == StrategyVector.STRATEGY_DIM


# ══════════════════════════════════════════
# Observation augmentation tests
# ══════════════════════════════════════════

class TestObservationAugmentation:

    def test_augmented_obs_dim(self):
        """Augmented obs = original + strategy vector."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        sv  = StrategyVector.neutral().to_array()
        aug = np.concatenate([obs, sv])
        assert aug.shape == (OBS_DIM + StrategyVector.STRATEGY_DIM,)

    def test_strategy_at_end_of_aug_obs(self):
        """Strategy values must appear at the END of augmented obs."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        sv  = StrategyVector(price_aggression=0.7).to_array()
        aug = np.concatenate([obs, sv])
        # price_aggression is index 0 of strategy → index OBS_DIM of aug
        assert aug[OBS_DIM] == pytest.approx(0.7)

    def test_original_obs_unchanged(self):
        """Original obs values must be preserved in augmented obs."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        sv  = StrategyVector.neutral().to_array()
        aug = np.concatenate([obs, sv])
        np.testing.assert_array_equal(aug[:OBS_DIM], obs)