"""Pure held-out validation logic for the retrainable models
(correction, tailing).

Split out of app.py so it can be unit-tested without a running Streamlit
session -- app.py's top-level code calls Streamlit UI functions on import,
which only work inside `streamlit run`, so anything worth testing on its
own has to live somewhere Streamlit-free.

Policy this module exists to enforce: a retrained model must never be
labeled trustworthy without an actual held-out score. Before this module
existed, train_correction_model_app()/train_tailing_model_app() called
model.fit(X, y) on the entire dataset and displayed a static "ACTIVE BUT
UNVALIDATED" warning every time, regardless of how much data existed or
how well the model actually performed -- there was no code path that
could ever produce a real number.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_holdout_split(n_rows, groups, test_size=0.25, min_test_groups=1, random_state=42):
    """Split by group (lipid) rather than by row.

    A plain random row split can leak: two spots from the same lipid can
    land on both sides, so the model gets tested partly on a lipid it
    already saw examples of, and the reported score looks better than the
    model's real ability to generalize to an unseen lipid -- which is
    exactly the claim "unseen lipid validation" is supposed to back up.

    Returns (train_idx, test_idx, warning). When there are too few groups
    for a meaningful test split, test_idx is an empty array and warning
    explains why -- callers must show that reason, not silently claim
    validation happened.
    """
    if n_rows == 0:
        return np.array([], dtype=int), np.array([], dtype=int), "데이터가 없습니다."

    groups = np.asarray(groups)
    unique_groups = pd.Series(groups).nunique()
    if unique_groups < (min_test_groups + 1):
        return (
            np.arange(n_rows),
            np.array([], dtype=int),
            f"서로 다른 지질이 {unique_groups}개뿐이라 held-out 검증을 위한 분리를 "
            "할 수 없습니다. 지질 종류가 더 쌓이면 자동으로 검증이 시작됩니다.",
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.arange(n_rows), groups=groups))

    if pd.Series(groups[test_idx]).nunique() < min_test_groups:
        return (
            np.arange(n_rows),
            np.array([], dtype=int),
            "검증 세트에 남는 지질 수가 너무 적어 held-out 검증을 신뢰할 수 없습니다.",
        )

    return train_idx, test_idx, None
