import math

import pytest

from mplc_gradient import (
    capacity_factor,
    estimated_elution_column_volumes,
    pick_isocratic_point,
    recommend_mplc_gradient,
    suggest_gradient,
)


def test_capacity_factor_at_rf_one_is_zero():
    assert capacity_factor(1.0) == 0.0


def test_capacity_factor_matches_still_formula():
    # Still 1978: k' = (1 - Rf) / Rf
    assert math.isclose(capacity_factor(0.25), 3.0)
    assert math.isclose(capacity_factor(0.5), 1.0)


def test_capacity_factor_rejects_zero_and_out_of_range():
    with pytest.raises(ValueError):
        capacity_factor(0.0)
    with pytest.raises(ValueError):
        capacity_factor(1.5)
    with pytest.raises(ValueError):
        capacity_factor(-0.1)


def test_estimated_elution_column_volumes_is_one_plus_kprime():
    assert math.isclose(estimated_elution_column_volumes(0.5), 2.0)


def test_pick_isocratic_point_chooses_closest_to_target():
    candidates = [
        {"percent_b": 10, "rf": 0.05},
        {"percent_b": 30, "rf": 0.36},
        {"percent_b": 60, "rf": 0.8},
    ]
    chosen = pick_isocratic_point(candidates, target_rf=0.35)
    assert chosen["percent_b"] == 30


def test_pick_isocratic_point_rejects_empty_candidates():
    with pytest.raises(ValueError):
        pick_isocratic_point([])


def test_suggest_gradient_brackets_the_isocratic_point():
    plan = suggest_gradient(40.0, start_margin=10.0, end_margin=5.0)
    assert plan["start_percent_b"] == 30.0
    assert plan["end_percent_b"] == 45.0
    assert plan["isocratic_reference_percent_b"] == 40.0


def test_suggest_gradient_clamps_to_valid_percent_range():
    low = suggest_gradient(5.0, start_margin=10.0)
    assert low["start_percent_b"] == 0.0  # clamped, not -5

    high = suggest_gradient(98.0, end_margin=5.0)
    assert high["end_percent_b"] == 100.0  # clamped, not 103


def test_suggest_gradient_rejects_isocratic_point_outside_range():
    with pytest.raises(ValueError):
        suggest_gradient(150.0)


def test_recommend_mplc_gradient_combines_pick_and_suggest():
    candidates = [
        {"percent_b": 20, "rf": 0.1},
        {"percent_b": 35, "rf": 0.34},
        {"percent_b": 70, "rf": 0.9},
    ]
    plan = recommend_mplc_gradient(candidates, target_rf=0.35)
    assert plan["chosen_point"]["percent_b"] == 35
    assert plan["isocratic_reference_percent_b"] == 35.0
    assert plan["estimated_elution_column_volumes"] > 1.0


def test_recommend_mplc_gradient_rejects_empty_candidates():
    with pytest.raises(ValueError):
        recommend_mplc_gradient([])
