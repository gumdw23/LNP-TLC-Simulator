"""Tests for the pure mobile-phase math extracted from app.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mobile_phase import mobile_phase_text, normalize_mobile_phase  # noqa: E402


def test_normalizes_ratios_to_percentages_summing_to_100():
    result = normalize_mobile_phase([("Hexane", 3), ("EtOAc", 1)])
    assert result == [("Hexane", 75.0), ("EtOAc", 25.0)]


def test_single_solvent_normalizes_to_100_percent():
    result = normalize_mobile_phase([("MeOH", 5)])
    assert result == [("MeOH", 100.0)]


def test_drops_none_and_zero_or_negative_entries():
    result = normalize_mobile_phase([("Hexane", 3), (None, 1), ("EtOAc", 0), ("MeOH", 1)])
    assert result == [("Hexane", 75.0), ("MeOH", 25.0)]


def test_rejects_zero_solvents():
    with pytest.raises(ValueError):
        normalize_mobile_phase([(None, 1), ("EtOAc", 0)])


def test_rejects_more_than_three_solvents():
    with pytest.raises(ValueError):
        normalize_mobile_phase([("A", 1), ("B", 1), ("C", 1), ("D", 1)])


def test_rejects_duplicate_solvent_names():
    with pytest.raises(ValueError):
        normalize_mobile_phase([("Hexane", 1), ("Hexane", 2)])


def test_mobile_phase_text_formats_with_one_decimal_and_slash_separator():
    text = mobile_phase_text([("Hexane", 75.0), ("EtOAc", 25.0)])
    assert text == "Hexane 75.0% / EtOAc 25.0%"


def test_mobile_phase_text_handles_a_single_solvent():
    assert mobile_phase_text([("MeOH", 100.0)]) == "MeOH 100.0%"
