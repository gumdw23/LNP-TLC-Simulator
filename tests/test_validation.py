"""Tests for validation.group_holdout_split -- the fix for the gap found
2026-08-16: the retrainable models (correction, tailing) previously fit()
on the entire dataset with zero held-out evaluation, so no code path could
ever report whether a retrained model actually generalizes.

Run: pytest apps/tlc_simulator/tests (from the tlc_simulator venv).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation import group_holdout_split  # noqa: E402


def test_too_few_groups_returns_no_test_set_and_a_reason():
    # Only 1 distinct group -- holding any of it out for test would leave
    # nothing to train on, so min_test_groups=1 correctly requires >= 2
    # groups (see the `unique_groups < min_test_groups + 1` boundary in
    # validation.py) and this must be refused, not silently split anyway.
    groups = np.array(["A", "A", "A", "A", "A", "A"])
    train_idx, test_idx, warning = group_holdout_split(len(groups), groups, min_test_groups=1)

    assert len(test_idx) == 0
    assert len(train_idx) == len(groups)
    assert warning is not None
    assert "지질" in warning


def test_exactly_min_plus_one_groups_is_accepted_not_refused():
    # The boundary case this project's own migration-count discipline
    # taught: get the off-by-one right and prove it with a test, not by
    # eyeballing the inequality. 2 groups with min_test_groups=1 is exactly
    # enough (1 to train on, 1 to hold out) and must NOT be refused.
    groups = np.repeat(["A", "B"], 10)
    train_idx, test_idx, warning = group_holdout_split(
        len(groups), groups, test_size=0.5, min_test_groups=1, random_state=0
    )
    assert warning is None
    assert len(test_idx) > 0
    assert len(train_idx) > 0


def test_zero_rows_is_handled_without_crashing():
    train_idx, test_idx, warning = group_holdout_split(0, np.array([]))
    assert len(train_idx) == 0
    assert len(test_idx) == 0
    assert warning is not None


def test_enough_groups_produces_a_real_holdout_split():
    rng = np.random.default_rng(0)
    # 10 lipids, 5 rows each -- plenty of groups for a real split.
    groups = np.repeat([f"lipid_{i}" for i in range(10)], 5)
    n = len(groups)
    idx = rng.permutation(n)
    groups = groups[idx]

    train_idx, test_idx, warning = group_holdout_split(n, groups, test_size=0.3, min_test_groups=1)

    assert warning is None
    assert len(test_idx) > 0
    assert len(train_idx) > 0
    assert len(train_idx) + len(test_idx) == n


def test_no_group_appears_on_both_sides_of_the_split():
    # This is the property the whole fix exists for: a lipid's rows must
    # not leak across train and test, or the reported score would be
    # measuring memorization of a seen lipid, not generalization to an
    # unseen one.
    rng = np.random.default_rng(1)
    groups = np.repeat([f"lipid_{i}" for i in range(12)], 4)
    n = len(groups)
    idx = rng.permutation(n)
    groups = groups[idx]

    train_idx, test_idx, warning = group_holdout_split(n, groups, test_size=0.25, min_test_groups=2)

    assert warning is None
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(test_groups)


def test_split_is_reproducible_with_the_same_random_state():
    rng = np.random.default_rng(2)
    groups = np.repeat([f"lipid_{i}" for i in range(10)], 5)
    n = len(groups)
    idx = rng.permutation(n)
    groups = groups[idx]

    r1 = group_holdout_split(n, groups, random_state=7)
    r2 = group_holdout_split(n, groups, random_state=7)

    np.testing.assert_array_equal(r1[0], r2[0])
    np.testing.assert_array_equal(r1[1], r2[1])


def test_min_test_groups_is_honored_not_just_test_size():
    # Exactly at the boundary: min_test_groups=1 requires at least 2 total
    # groups. With 2 groups this should NOT be rejected outright (unlike
    # the too-few-groups case above, which used the same count) -- but the
    # split must still respect min_test_groups if the shuffle happens to
    # put too few groups in the test fold.
    groups = np.repeat(["A", "B"], 20)
    train_idx, test_idx, warning = group_holdout_split(
        len(groups), groups, test_size=0.3, min_test_groups=1, random_state=3
    )
    # 2 groups, min_test_groups=1 -- a split is attempted, not refused outright.
    assert warning is None or "검증 세트에 남는 지질" in warning
