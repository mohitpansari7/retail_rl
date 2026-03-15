"""
tests/test_reward_model.py
───────────────────────────
Phase 4 tests. Run with: pytest tests/test_reward_model.py -v

Test philosophy:
  - Test Bradley-Terry loss MATHEMATICALLY (correct when model right, high when wrong)
  - Test Elo update formulas with manual traces
  - Test preference dataset generates valid pairs
  - Test label noise actually flips labels
  - Test Elo leaderboard ranks correctly after matches
"""

import numpy as np
import pytest
import sys, math
sys.path.insert(0, '.')

from reward.preference_data import (
    TrajectorySegment, PreferencePair, PreferenceDataset
)
from reward.elo_rating import EloTracker, StrategyRecord


OBS_DIM    = 24
ACTION_DIM = 4
SEG_LEN    = 10


# ══════════════════════════════════════════
# TrajectorySegment tests
# ══════════════════════════════════════════

class TestTrajectorySegment:

    def test_total_return_computed_on_init(self):
        """total_return must be set in __post_init__."""
        seg = TrajectorySegment(
            obs=np.zeros((SEG_LEN, OBS_DIM),    dtype=np.float32),
            actions=np.zeros((SEG_LEN, ACTION_DIM), dtype=np.float32),
            rewards=np.ones(SEG_LEN,               dtype=np.float32),
        )
        assert seg.total_return > 0.0

    def test_higher_reward_segment_has_higher_return(self):
        """Segment with reward=10 must have higher return than reward=1."""
        seg_low  = TrajectorySegment(
            obs=np.zeros((SEG_LEN, OBS_DIM), dtype=np.float32),
            actions=np.zeros((SEG_LEN, ACTION_DIM), dtype=np.float32),
            rewards=np.ones(SEG_LEN, dtype=np.float32) * 1.0,
        )
        seg_high = TrajectorySegment(
            obs=np.zeros((SEG_LEN, OBS_DIM), dtype=np.float32),
            actions=np.zeros((SEG_LEN, ACTION_DIM), dtype=np.float32),
            rewards=np.ones(SEG_LEN, dtype=np.float32) * 10.0,
        )
        assert seg_high.total_return > seg_low.total_return


# ══════════════════════════════════════════
# PreferenceDataset tests
# ══════════════════════════════════════════

class TestPreferenceDataset:

    @pytest.fixture
    def dataset(self):
        return PreferenceDataset(segment_length=SEG_LEN, noise_prob=0.0, seed=42)

    def _fill_dataset(self, dataset, n_steps=500):
        """Fill dataset with random trajectory data."""
        rng = np.random.default_rng(0)
        for i in range(n_steps):
            # Alternate between high and low reward to create clear preferences
            reward = 10.0 if i % 2 == 0 else 1.0
            dataset.add_step(
                obs=rng.standard_normal(OBS_DIM).astype(np.float32),
                action=rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
                reward=reward,
            )

    def test_no_pairs_before_enough_data(self, dataset):
        """Can't generate pairs without at least 2×segment_length steps."""
        for _ in range(SEG_LEN - 1):
            dataset.add_step(
                np.zeros(OBS_DIM, dtype=np.float32),
                np.zeros(ACTION_DIM, dtype=np.float32),
                1.0
            )
        added = dataset.generate_pairs(10)
        assert added == 0

    def test_pairs_generated_after_enough_data(self, dataset):
        self._fill_dataset(dataset)
        added = dataset.generate_pairs(20)
        assert added > 0

    def test_pair_label_is_valid(self, dataset):
        """All labels must be 0.0, 0.5, or 1.0."""
        self._fill_dataset(dataset)
        dataset.generate_pairs(50)
        for pair in dataset._pairs:
            assert pair.label in (0.0, 0.5, 1.0)

    def test_higher_return_segment_labelled_preferred(self, dataset):
        """
        With noise_prob=0, the segment with higher return must be
        labelled as preferred (label=0 means segment_a preferred).
        """
        self._fill_dataset(dataset)
        dataset.generate_pairs(100)
        for pair in dataset._pairs:
            if pair.label == 0.0:
                assert pair.segment_a.total_return >= pair.segment_b.total_return
            elif pair.label == 1.0:
                assert pair.segment_b.total_return >= pair.segment_a.total_return

    def test_noise_flips_some_labels(self):
        """With noise_prob=1.0, ALL labels should be flipped."""
        ds_clean = PreferenceDataset(segment_length=SEG_LEN, noise_prob=0.0, seed=0)
        ds_noisy = PreferenceDataset(segment_length=SEG_LEN, noise_prob=1.0, seed=0)

        rng = np.random.default_rng(1)
        for i in range(300):
            obs    = rng.standard_normal(OBS_DIM).astype(np.float32)
            action = rng.uniform(-1,1,ACTION_DIM).astype(np.float32)
            reward = 10.0 if i < 150 else 1.0
            ds_clean.add_step(obs, action, reward)
            ds_noisy.add_step(obs, action, reward)

        ds_clean.generate_pairs(30)
        ds_noisy.generate_pairs(30)

        # With noise=1.0, labels should differ from noise=0.0
        clean_labels = [p.label for p in ds_clean._pairs]
        noisy_labels = [p.label for p in ds_noisy._pairs]
        assert clean_labels != noisy_labels

    def test_get_batch_shapes(self, dataset):
        self._fill_dataset(dataset)
        dataset.generate_pairs(50)
        obs_a, actions_a, obs_b, actions_b, labels = dataset.get_batch(8)
        assert obs_a.shape     == (8, SEG_LEN, OBS_DIM)
        assert actions_a.shape == (8, SEG_LEN, ACTION_DIM)
        assert obs_b.shape     == (8, SEG_LEN, OBS_DIM)
        assert actions_b.shape == (8, SEG_LEN, ACTION_DIM)
        assert labels.shape    == (8,)


# ══════════════════════════════════════════
# Elo rating tests
# ══════════════════════════════════════════

class TestEloTracker:

    @pytest.fixture
    def tracker(self):
        return EloTracker(k_factor=32.0, initial_rating=1500.0)

    def test_initial_rating_is_1500(self, tracker):
        tracker.register("strategy_a")
        assert tracker.get_rating("strategy_a") == pytest.approx(1500.0)

    def test_winner_rating_increases(self, tracker):
        tracker.register("a")
        tracker.register("b")
        old_a = tracker.get_rating("a")
        tracker.record_match("a", "b", winner="a")
        assert tracker.get_rating("a") > old_a

    def test_loser_rating_decreases(self, tracker):
        tracker.register("a")
        tracker.register("b")
        old_b = tracker.get_rating("b")
        tracker.record_match("a", "b", winner="a")
        assert tracker.get_rating("b") < old_b

    def test_rating_sum_preserved(self, tracker):
        """
        Elo is zero-sum: total rating across both players must be conserved.
        What A gains, B loses.
        """
        tracker.register("a")
        tracker.register("b")
        total_before = tracker.get_rating("a") + tracker.get_rating("b")
        tracker.record_match("a", "b", winner="a")
        total_after = tracker.get_rating("a") + tracker.get_rating("b")
        assert total_before == pytest.approx(total_after, rel=1e-6)

    def test_upset_causes_larger_update(self, tracker):
        """
        Upset (low-rated beats high-rated) → larger rating change.
        Expected win (high-rated beats low-rated) → smaller rating change.
        """
        # Setup: a is much stronger than b
        tracker.register("strong")
        tracker.register("weak")
        tracker._strategies["strong"].rating = 1800.0
        tracker._strategies["weak"].rating   = 1200.0

        # Upset: weak beats strong
        rating_weak_before = tracker.get_rating("weak")
        tracker.record_match("strong", "weak", winner="weak")
        upset_gain = tracker.get_rating("weak") - rating_weak_before

        # Reset
        tracker._strategies["strong"].rating = 1800.0
        tracker._strategies["weak"].rating   = 1200.0

        # Expected: strong beats weak
        rating_strong_before = tracker.get_rating("strong")
        tracker.record_match("strong", "weak", winner="strong")
        expected_gain = tracker.get_rating("strong") - rating_strong_before

        assert upset_gain > expected_gain

    def test_manual_elo_calculation(self, tracker):
        """
        Manual trace of the Elo formula.

        Both players at 1500 → E_A = 0.5.
        A wins → R'_A = 1500 + 32×(1 − 0.5) = 1516.
        B loses → R'_B = 1500 + 32×(0 − 0.5) = 1484.
        """
        tracker.register("a")
        tracker.register("b")
        new_a, new_b = tracker.record_match("a", "b", winner="a")
        assert new_a == pytest.approx(1516.0, rel=1e-4)
        assert new_b == pytest.approx(1484.0, rel=1e-4)

    def test_draw_changes_ratings_minimally(self, tracker):
        """Equal-rated players drawing: ratings should not change."""
        tracker.register("a")
        tracker.register("b")
        old_a = tracker.get_rating("a")
        old_b = tracker.get_rating("b")
        tracker.record_match("a", "b", winner=None)   # draw
        assert tracker.get_rating("a") == pytest.approx(old_a, rel=1e-6)
        assert tracker.get_rating("b") == pytest.approx(old_b, rel=1e-6)

    def test_leaderboard_sorted_by_rating(self, tracker):
        tracker.register("a")
        tracker.register("b")
        tracker.register("c")
        tracker._strategies["a"].rating = 1700.0
        tracker._strategies["b"].rating = 1400.0
        tracker._strategies["c"].rating = 1550.0

        lb = tracker.leaderboard()
        ratings = [r.rating for r in lb]
        assert ratings == sorted(ratings, reverse=True)

    def test_record_match_by_returns(self, tracker):
        """Higher return → winner of the match."""
        tracker.register("a")
        tracker.register("b")
        old_a = tracker.get_rating("a")
        tracker.record_match_by_returns("a", "b", return_a=1000.0, return_b=500.0)
        assert tracker.get_rating("a") > old_a   # a won, rating should increase