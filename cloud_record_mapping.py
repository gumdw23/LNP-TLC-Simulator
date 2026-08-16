"""Maps this app's own Record Experiment fields to tlc-create's request
shape (supabase/functions/tlc-create, OP-51/OP-53 in the main platform
repository). Pure logic, no Streamlit, no I/O -- split out the same way
mobile_phase.py/experiment_row.py/validation.py already are, so it can be
imported and tested on its own (app.py's top-level code runs Streamlit UI
calls at import time, which only works inside `streamlit run`).

This app has no CMP-######/BAT-###### compound codes of its own, so every
cloud-recorded run goes in as a free-text sample_name, never a code --
consistent with every other run this platform has recorded for an
unregistered lipid.

Design choice: never silently discard a value that has no clean home in
the DB's controlled vocabulary. Where this app's own free text/enum
doesn't map onto tlc_runs'/tlc_spots' fixed options (plate description,
detection method, spot quality + tailing severity), the exact original
text is preserved in tlc_notes/the spot's own notes alongside the
best-effort mapped value -- never fabricated, never lost.
"""

from mobile_phase import mobile_phase_text


def map_plate_type(plate_text):
    t = (plate_text or "").lower()
    if "reverse" in t or "c18" in t or " rp" in t or t.startswith("rp"):
        return "reverse_phase"
    if "alumina" in t:
        return "alumina"
    if "cellulose" in t:
        return "cellulose"
    if "amino" in t:
        return "amino"
    if "diol" in t:
        return "diol"
    if "silica" in t:
        return "silica_gel"
    return "other"


def map_visualization_method(detection_text):
    t = (detection_text or "").lower()
    if "254" in t:
        return "uv_254"
    if "365" in t:
        return "uv_365"
    if "iodine" in t or "요오드" in t:
        return "iodine"
    if "fluoresc" in t or "형광" in t:
        return "fluorescence"
    if "chemical" in t or "stain" in t:
        return "chemical_stain"
    if "visible" in t:
        return "visible_light"
    return "other"


def map_spot_shape(spot_quality_value, tailing_value):
    # Moderate/Severe tailing is the visually dominant feature at that
    # point -- overrides the spot_quality mapping, matching how a chemist
    # would actually describe the spot. Both raw values are kept in the
    # spot's own notes regardless of which one wins here.
    if tailing_value in ("Moderate", "Severe"):
        return "tailing"
    return {
        "Good": "compact",
        "Diffuse": "diffuse",
        "Streaking": "streak",
        "Overloaded": "other",
        "NotVisible": "not_assessed",
    }.get(spot_quality_value, "other")


def build_cloud_tlc_body(
    compound_name,
    smiles,
    mobile_phase,
    plate,
    development_distance_cm,
    chamber_saturated,
    detection_method,
    experimental_rf,
    tailing_value,
    spot_quality_value,
    spotting_volume_ul,
    notes_text,
):
    sample_label = (compound_name or "").strip() or "미등록 지질"

    analysis_notes_parts = []
    if smiles and smiles.strip():
        analysis_notes_parts.append(f"SMILES: {smiles.strip()}")
    if notes_text and notes_text.strip():
        analysis_notes_parts.append(notes_text.strip())

    return {
        "title": f"TLC: {sample_label}",
        "sample_name": sample_label,
        "analysis_notes": " / ".join(analysis_notes_parts) or None,
        "instrument_name": "TLC Simulator (local app)",
        "run_status": "completed",
        "plate_type": map_plate_type(plate),
        "stationary_phase": plate or None,
        "mobile_phase_description": mobile_phase_text(mobile_phase) or None,
        "mobile_phase_ratio_text": ":".join(f"{p:.1f}" for _n, p in mobile_phase) or None,
        "chamber_saturation": bool(chamber_saturated),
        "development_distance_mm": float(development_distance_cm) * 10 if development_distance_cm else None,
        "application_volume_ul": float(spotting_volume_ul) if spotting_volume_ul else None,
        "visualization_method": map_visualization_method(detection_method),
        "tlc_notes": f"Plate (원문): {plate} / Detection (원문): {detection_method}",
        "spots": [
            {
                "lane_number": 1,
                "spot_number": 1,
                "sample_label": sample_label,
                "rf_value": float(experimental_rf),
                "spot_shape": map_spot_shape(spot_quality_value, tailing_value),
                "assignment_status": "unassigned",
                "notes": f"Tailing: {tailing_value} / Spot quality: {spot_quality_value}",
            }
        ],
    }
