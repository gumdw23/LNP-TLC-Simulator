from cloud_record_mapping import (
    build_cloud_tlc_body,
    map_plate_type,
    map_spot_shape,
    map_visualization_method,
)


def test_map_plate_type_recognizes_silica():
    assert map_plate_type("Silica gel 60 F254") == "silica_gel"


def test_map_plate_type_recognizes_reverse_phase():
    assert map_plate_type("RP-18 TLC plate") == "reverse_phase"


def test_map_plate_type_falls_back_to_other_for_unrecognized_text():
    assert map_plate_type("some exotic plate nobody has heard of") == "other"


def test_map_visualization_method_recognizes_wavelengths():
    assert map_visualization_method("UV 254 nm") == "uv_254"
    assert map_visualization_method("UV 365 nm") == "uv_365"


def test_map_visualization_method_does_not_guess_a_wavelength_for_plain_uv():
    # "UV" alone doesn't say which wavelength -- must not fabricate one.
    assert map_visualization_method("UV") == "other"


def test_map_spot_shape_severe_tailing_overrides_spot_quality():
    assert map_spot_shape("Good", "Severe") == "tailing"
    assert map_spot_shape("Good", "Moderate") == "tailing"


def test_map_spot_shape_mild_or_no_tailing_uses_spot_quality():
    assert map_spot_shape("Diffuse", "No") == "diffuse"
    assert map_spot_shape("Streaking", "Mild") == "streak"
    assert map_spot_shape("Good", "No") == "compact"


def test_build_cloud_tlc_body_uses_placeholder_name_when_compound_name_blank():
    body = build_cloud_tlc_body(
        compound_name="  ",
        smiles="CCO",
        mobile_phase=[("Hexane", 70.0), ("EtOAc", 30.0)],
        plate="Silica gel 60 F254",
        development_distance_cm=7.0,
        chamber_saturated=True,
        detection_method="UV 254",
        experimental_rf=0.45,
        tailing_value="No",
        spot_quality_value="Good",
        spotting_volume_ul=1.0,
        notes_text="",
    )
    assert body["sample_name"] == "미등록 지질"
    assert body["title"] == "TLC: 미등록 지질"


def test_build_cloud_tlc_body_preserves_smiles_and_notes_in_analysis_notes():
    body = build_cloud_tlc_body(
        compound_name="Test Lipid",
        smiles="CCO",
        mobile_phase=[("Hexane", 70.0), ("EtOAc", 30.0)],
        plate="Silica gel 60 F254",
        development_distance_cm=7.0,
        chamber_saturated=True,
        detection_method="UV 254",
        experimental_rf=0.45,
        tailing_value="No",
        spot_quality_value="Good",
        spotting_volume_ul=1.0,
        notes_text="looked clean",
    )
    assert "CCO" in body["analysis_notes"]
    assert "looked clean" in body["analysis_notes"]


def test_build_cloud_tlc_body_converts_cm_to_mm():
    body = build_cloud_tlc_body(
        compound_name="Test Lipid",
        smiles="CCO",
        mobile_phase=[("Hexane", 100.0)],
        plate="Silica gel",
        development_distance_cm=7.0,
        chamber_saturated=False,
        detection_method="UV",
        experimental_rf=0.5,
        tailing_value="No",
        spot_quality_value="Good",
        spotting_volume_ul=1.0,
        notes_text="",
    )
    assert body["development_distance_mm"] == 70.0


def test_build_cloud_tlc_body_keeps_raw_plate_and_detection_text_even_when_mapped():
    body = build_cloud_tlc_body(
        compound_name="Test Lipid",
        smiles="CCO",
        mobile_phase=[("Hexane", 100.0)],
        plate="Silica gel 60 F254, Merck",
        development_distance_cm=7.0,
        chamber_saturated=False,
        detection_method="UV 254 handheld lamp",
        experimental_rf=0.5,
        tailing_value="No",
        spot_quality_value="Good",
        spotting_volume_ul=1.0,
        notes_text="",
    )
    assert "Silica gel 60 F254, Merck" in body["tlc_notes"]
    assert "UV 254 handheld lamp" in body["tlc_notes"]
    assert body["stationary_phase"] == "Silica gel 60 F254, Merck"


def test_build_cloud_tlc_body_single_spot_carries_both_raw_tailing_and_quality():
    body = build_cloud_tlc_body(
        compound_name="Test Lipid",
        smiles="CCO",
        mobile_phase=[("Hexane", 100.0)],
        plate="Silica gel",
        development_distance_cm=7.0,
        chamber_saturated=False,
        detection_method="UV",
        experimental_rf=0.33,
        tailing_value="Mild",
        spot_quality_value="Diffuse",
        spotting_volume_ul=1.0,
        notes_text="",
    )
    spot = body["spots"][0]
    assert spot["rf_value"] == 0.33
    assert "Mild" in spot["notes"]
    assert "Diffuse" in spot["notes"]
    assert spot["spot_shape"] == "diffuse"
