"""Metric correctness against hand-computed values."""

import numpy as np
import pytest

from footyforecast.evaluation import metrics


def test_bits_coin_flip_scores_zero():
    # p = 0.5: correct → 1 + log2(0.5) = 0; wrong → 0; draw → 1 + 0.5·log2(0.25) = 0
    assert metrics.bits([0.5, 0.5, 0.5], [1, 0, 0.5]) == pytest.approx([0, 0, 0])


def test_bits_rewards_confident_correct_and_punishes_confident_wrong():
    assert metrics.bits([0.8], [1])[0] == pytest.approx(1 + np.log2(0.8))
    assert metrics.bits([0.8], [0])[0] == pytest.approx(1 + np.log2(0.2))
    assert metrics.bits([0.25], [0])[0] == pytest.approx(1 + np.log2(0.75))  # away tip right


def test_bits_draw_formula():
    assert metrics.bits([0.7], [0.5])[0] == pytest.approx(1 + 0.5 * np.log2(0.7 * 0.3))


def test_bits_is_finite_at_certainty():
    assert np.isfinite(metrics.bits([1.0, 0.0], [0, 1])).all()


def test_tips_correct_handles_draws_and_ties():
    p = [0.6, 0.6, 0.4, 0.5, 0.9]
    y = [1, 0, 0, 1, 0.5]
    assert metrics.tips_correct(p, y).tolist() == [1, 0, 1, 1, 1]
    assert metrics.tips_correct(p, y, draw_is_correct=False).tolist() == [1, 0, 1, 1, 0]


def test_brier_and_log_loss():
    assert metrics.brier([0.8, 0.3], [1, 0]) == pytest.approx((0.04 + 0.09) / 2)
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert metrics.log_loss([0.8, 0.3], [1, 0]) == pytest.approx(expected)


def test_margin_mae():
    assert metrics.margin_abs_error([10, -5], [20, 5]).tolist() == [10, 10]


def test_perfectly_calibrated_bins_have_zero_ece():
    # 10 games at p=0.2 with 2 home wins; 10 at p=0.8 with 8 home wins.
    p = [0.2] * 10 + [0.8] * 10
    y = [1] * 2 + [0] * 8 + [1] * 8 + [0] * 2
    assert metrics.expected_calibration_error(p, y) == pytest.approx(0.0)


def test_summarise_keys():
    out = metrics.summarise([0.6, 0.4], [1, 1], [5, -5], [10, 10])
    assert out["games"] == 2 and out["tips"] == 1
    assert out["mae"] == pytest.approx(10.0)
