# TLC -> MPLC gradient suggestion: design notes (2026-08-16)

## What the owner asked for

Recording a synthesis step's TLC result should be able to suggest a
starting MPLC (medium-pressure/flash) purification gradient, instead of
that being a separate manual guess every time. The owner specifically
asked for a recommended approach before any code was written.

## The approach: Still's rule, not a new model

This is not a machine-learning problem. Flash chromatography has had a
standard, well-cited heuristic since 1978:

> W. C. Still, W. R. Kahn, A. Mitra, *J. Org. Chem.* 1978, 43, 2923.

The two pieces of it this module implements:

1. **Isocratic point selection.** The solvent ratio that puts the target
   compound at **Rf ~= 0.35** on TLC, in the *same solvent system* used
   for the column, is Still's recommended isocratic flash condition.
2. **Capacity factor / elution volume.** `k' = (1 - Rf) / Rf`, and the
   elution volume is approximately `V0 * (1 + k')` column volumes. This
   gives a rough "where in the run to expect the peak" number from an Rf
   alone.

This module (`mplc_gradient.py`) extends point 1 from a single isocratic
ratio to a **linear gradient**: start a fixed margin weaker than the
isocratic point (so material that would otherwise co-elute with the
solvent front gets separated before the gradient catches up), end a
smaller margin stronger (so anything slightly more retained than
predicted still comes off within the programmed run), ramped over a
default 15 column volumes. The margins and ramp length are common flash
practice, not values fit to any dataset in this project -- see the
module docstring for the exact numbers and the reasoning behind each.

## What already existed vs. what's new

`apps/tlc_simulator/checkpoint/core_runtime_bundle.dill` already ships
`recommend_tlc_conditions(smiles, solvent_pool=None, target_rf=(0.2,
0.4), ...)` -- given a SMILES, it searches a solvent pool and ranks
candidate mobile phases by how close their predicted Rf lands to a
target band. That already solves "which solvent ratio gives a usable
Rf." What it does **not** do is turn that into a *gradient* elution plan
or estimate where in the run the compound will come off -- that's what
`mplc_gradient.py` adds, as a thin, pure-calculation layer on top.

`mplc_gradient.py` deliberately does not call into the dill bundle or
predict Rf itself. It takes Rf-vs-%B points as plain input (a list of
`{"percent_b": ..., "rf": ...}` dicts) so it can be fed either by
`recommend_tlc_conditions`'s output, by a manual multi-ratio TLC scan
the researcher already ran, or by test data -- whichever is available,
without this module needing to know which.

## Validation status

**Not validated against a single real MPLC run in this lab.** Still's
rule is decades-old, widely used, and its capacity-factor formula is
exact algebra given an Rf -- but the specific margins (10 points weaker
to start, 5 points stronger to end, 15 CV ramp) are defaults, not
measurements. Every function's docstring says this again at the point
where a number comes out, so it's not lost the first time someone reads
this doc and forgets it.

## What this does not do (yet)

- **Ternary/quaternary systems.** The module works on a single %B
  scalar (binary system: solvent A / solvent B). `optimize_ternary_mobile_phase`
  already exists in the dill bundle for 3-solvent search; wiring its
  output through this module's gradient math is a natural next step,
  not done here.
- **Data-quality filtering.** If this is ever driven by TLC results
  recorded in the main platform database (not this standalone app),
  those results carry a `quality_status` column (`accepted` /
  `provisional` / `rejected` / `reanalysis_required` / `not_reviewed`
  -- see `analysis_runs` in the main schema). A real integration must
  filter to `quality_status = 'accepted'` before treating a stored Rf
  as ground truth, exactly so a later-discovered bad result (e.g. one
  project's TLC turning out wrong, corrected by a different project's
  repeat of the same reaction) doesn't silently poison a future
  recommendation. This module has no opinion on that filtering itself
  -- it just consumes whatever points it's handed -- but any caller
  wiring real recorded data into it must do that filtering first.
- **Streamlit UI.** No screen calls this module yet. It's a tested,
  standalone calculation library (`tests/test_mplc_gradient.py`, 11
  tests, all pure -- no network, no Java, no trained model needed to
  run them) ready to be wired into a screen once the UI shape is
  decided.
