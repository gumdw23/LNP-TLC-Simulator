"""Pure mobile-phase math: no Streamlit, no session_state, no I/O.

Split out of app.py so it can be unit-tested directly -- app.py's
top-level code runs Streamlit UI calls at import time, which only works
inside `streamlit run`, so nothing that lives there can be imported and
tested on its own.
"""


def normalize_mobile_phase(components):
    """Take (solvent, ratio) pairs, drop empty/zero entries, and rescale
    the remaining ratios to percentages that sum to 100.

    Raises ValueError if the result isn't 1-3 distinct solvents -- the
    project's own stated mobile-phase constraint.
    """

    cleaned = []
    for solvent, ratio in components:
        if solvent is None or str(solvent) == "None":
            continue

        ratio = float(ratio)
        if ratio <= 0:
            continue

        cleaned.append((str(solvent), ratio))

    if not (1 <= len(cleaned) <= 3):
        raise ValueError("1~3개의 solvent를 사용하세요.")

    names = [name for name, _ in cleaned]
    if len(names) != len(set(names)):
        raise ValueError("같은 solvent를 두 번 선택할 수 없습니다.")

    total = sum(value for _, value in cleaned)

    return [(name, value / total * 100) for name, value in cleaned]


def mobile_phase_text(mobile_phase):
    """Render a normalized mobile phase as "Name X.X% / Name Y.Y%"."""

    return " / ".join(f"{name} {percent:.1f}%" for name, percent in mobile_phase)
