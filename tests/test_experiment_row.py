"""Tests for create_experiment_row's validation and assembly logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_row import create_experiment_row  # noqa: E402

VALID_SMILES = "CCO"  # ethanol -- a real, RDKit-parseable molecule


def _row(**overrides):
    base = dict(
        compound_name="Test Compound",
        smiles=VALID_SMILES,
        mobile_phase=[("Hexane", 75.0), ("EtOAc", 25.0)],
        additive="None",
        additive_percent=0,
        plate="Silica",
        concentration=1.0,
        spotting_volume=2.0,
        development_distance=5.0,
        chamber_saturated=True,
        detection_method="UV",
        experimental_rf=0.5,
        tailing="No",
        spot_quality="Good",
        notes="",
    )
    base.update(overrides)
    return base


def test_blank_compound_name_is_rejected():
    with pytest.raises(ValueError):
        create_experiment_row(**_row(compound_name="   "))


def test_unparseable_smiles_is_rejected():
    with pytest.raises(ValueError):
        create_experiment_row(**_row(smiles="not a real smiles!!"))


def test_rf_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        create_experiment_row(**_row(experimental_rf=1.5))
    with pytest.raises(ValueError):
        create_experiment_row(**_row(experimental_rf=-0.1))


def test_a_valid_row_is_assembled_with_canonical_smiles_and_solvent_slots():
    row = create_experiment_row(**_row())

    assert row["CompoundName"] == "Test Compound"
    assert row["Experimental_Rf"] == 0.5
    assert row["Solvent1"] == "Hexane"
    assert row["Solvent1Percent"] == 75.0
    assert row["Solvent2"] == "EtOAc"
    assert row["Solvent2Percent"] == 25.0
    # unfilled third slot defaults to the "no solvent" sentinel
    assert row["Solvent3"] == "None"
    assert row["DataOrigin"] == "RealExperiment"
    # SMILES is canonicalized by RDKit, not passed through verbatim
    assert row["SMILES"]


def test_experiment_id_is_unique_across_calls():
    row1 = create_experiment_row(**_row())
    row2 = create_experiment_row(**_row())
    assert row1["ExperimentID"] != row2["ExperimentID"]
