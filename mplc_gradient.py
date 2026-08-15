"""TLC Rf -> MPLC (flash) gradient suggestion, pure calculation only.

The owner asked for a way to turn a TLC result into a starting point for
medium-pressure/flash purification instead of guessing gradient conditions
from scratch. This module is the calculation half of that: it does not
predict Rf itself (app.py's loaded engine -- predict_rf_with_confidence,
recommend_tlc_conditions -- already does that from a SMILES and a mobile
phase). It only turns an Rf (or a handful of Rf-vs-%B points) into a
concrete flash gradient plan.

The math is W. C. Still, W. R. Kahn, A. Mitra, J. Org. Chem. 1978, 43,
2923 -- "still's rule" -- not a model fit to any data in this project:

    k' = (1 - Rf) / Rf                      (capacity factor)
    V_e / V0 ~= 1 + k'                       (elution volume, in column
                                              volumes, ignoring band
                                              broadening/loading/flow rate)

Still's own recommendation is to run the flash column isocratically at
whatever solvent ratio puts the target compound at Rf ~= 0.35 on TLC in
that same solvent system. This module extends that one step further to a
*gradient* plan (start weaker, ramp to at least the isocratic point) using
a fixed, documented margin -- not a second fitted model.

This has NOT been validated against a single real MPLC run in this lab.
Treat every number this module returns as a starting point to dial in on
the instrument, not a promise. See the module-level caveats in each
function's docstring for exactly what is and is not accounted for.
"""


def capacity_factor(rf):
    """k' = (1 - Rf) / Rf (Still 1978). Undefined at Rf = 0 (infinite
    retention) and negative for Rf > 1, both physically invalid inputs
    for a real spot, so both raise rather than returning a number that
    looks plausible but means nothing."""
    if not (0 < rf <= 1):
        raise ValueError(f"Rf must be in (0, 1], got {rf!r}")
    return (1 - rf) / rf


def estimated_elution_column_volumes(rf):
    """Rough single-compound elution volume, in column volumes (CV),
    from the capacity-factor approximation V_e/V0 ~= 1 + k'.

    What this ignores, all of which shift the real number: particle
    size and column packing quality, flow rate, sample loading amount
    relative to column capacity, and band broadening (this gives the
    peak's approximate center, not its width). Use it to plan roughly
    where to start watching for the compound, not to set a precise
    fraction cutoff.
    """
    return 1 + capacity_factor(rf)


def pick_isocratic_point(candidates, target_rf=0.35, rf_key="rf", percent_b_key="percent_b"):
    """Given several (percent_b, rf) points -- e.g. several rows from
    recommend_tlc_conditions's ranked results, or a manual TLC scan at
    a few solvent ratios -- return the one whose Rf is closest to
    target_rf. Still's rule targets Rf ~= 0.35 specifically (not just
    "somewhere in a broad band") because that value balances resolution
    against run time; this defaults to that value but accepts an
    override for cases with a different resolution/speed trade-off.

    candidates: a non-empty sequence of dict-like rows, each holding at
    least rf_key and percent_b_key.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    return min(candidates, key=lambda row: abs(row[rf_key] - target_rf))


def suggest_gradient(
    isocratic_percent_b,
    target_rf=0.35,
    start_margin=10.0,
    end_margin=5.0,
    min_percent=0.0,
    max_percent=100.0,
    ramp_column_volumes=15.0,
):
    """Turn a single isocratic %B (the ratio that puts the target
    compound at target_rf on TLC in this solvent system) into a linear
    flash gradient plan.

    start_percent_b is set start_margin points weaker than the
    isocratic ratio: early material that would co-elute with the
    solvent front under gradient elution gets more separation before
    the gradient catches up to the target compound's actual retention
    point. end_percent_b is set end_margin points stronger, so that
    anything running slightly behind the target compound's predicted
    Rf (due to loading, real-column effects, or plain prediction error)
    still comes off the column within the programmed gradient instead
    of needing a separate isocratic wash step afterward.

    The margins (10 / 5 percentage points) and the default 15 CV ramp
    length are common flash-chromatography practice, not a value fit
    to any dataset -- override them for a specific column/instrument
    once real runs establish what actually works.
    """
    if not (min_percent <= isocratic_percent_b <= max_percent):
        raise ValueError(
            f"isocratic_percent_b={isocratic_percent_b!r} outside "
            f"[{min_percent}, {max_percent}]"
        )
    start = max(min_percent, isocratic_percent_b - start_margin)
    end = min(max_percent, isocratic_percent_b + end_margin)
    return {
        "start_percent_b": round(start, 1),
        "end_percent_b": round(end, 1),
        "ramp_column_volumes": ramp_column_volumes,
        "isocratic_reference_percent_b": round(isocratic_percent_b, 1),
        "target_rf": target_rf,
    }


def recommend_mplc_gradient(candidates, target_rf=0.35, rf_key="rf", percent_b_key="percent_b", **gradient_kwargs):
    """One-call version: pick the best isocratic point from a set of
    Rf-vs-%B candidates, then propose a gradient around it. Also
    returns the estimated elution column-volume for the chosen point,
    since "roughly where in the run to expect the peak" is usually the
    next question after "what gradient to program."

    Raises ValueError if candidates is empty -- there is no gradient to
    suggest without at least one measured or predicted Rf point.
    """
    chosen = pick_isocratic_point(candidates, target_rf=target_rf, rf_key=rf_key, percent_b_key=percent_b_key)
    plan = suggest_gradient(chosen[percent_b_key], target_rf=target_rf, **gradient_kwargs)
    plan["chosen_point"] = chosen
    plan["estimated_elution_column_volumes"] = round(estimated_elution_column_volumes(chosen[rf_key]), 2)
    return plan
