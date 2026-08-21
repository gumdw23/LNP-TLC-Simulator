"""Integration test for the cloud-recording path: build_cloud_tlc_body()'s
output actually gets accepted by the real tlc-create Edge Function on the
local Supabase stack, end to end -- not just that the mapping function
produces a plausible-looking dict.

Needs the local Supabase stack running (same requirement as the main
platform repository's own Deno/pytest suites). Skips cleanly if it isn't
reachable rather than failing the whole test run for an unrelated reason.

Run: pytest apps/tlc_simulator/tests (from the tlc_simulator venv), with
`supabase start` already running in the main platform repository.
"""

import sys
import uuid
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import db_client  # noqa: E402
from cloud_record_mapping import build_cloud_tlc_body  # noqa: E402

SUPABASE_URL = "http://127.0.0.1:54321"
ANON_KEY = db_client.SUPABASE_ANON_KEY


def _make_test_account():
    email = f"tlc-cloud-record-{uuid.uuid4().hex[:8]}@example.com"
    password = "tlc-cloud-record-test-pw-123"
    try:
        res = requests.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            headers={"apikey": ANON_KEY, "Content-Type": "application/json"},
            json={"email": email, "password": password},
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        pytest.skip("local Supabase stack is not reachable at 127.0.0.1:54321")
    body = res.json()
    if not body.get("access_token"):
        pytest.skip(f"local stack reachable but signup failed: {body}")
    return email, password


def test_a_mapped_tlc_body_is_accepted_by_the_real_tlc_create_function():
    email, password = _make_test_account()
    login = db_client.sign_in(email, password)
    token = login["access_token"]

    body = build_cloud_tlc_body(
        compound_name=f"cloud record test {uuid.uuid4().hex[:6]}",
        smiles="CCO",
        mobile_phase=[("Hexane", 70.0), ("EtOAc", 30.0)],
        plate="Silica gel 60 F254",
        development_distance_cm=7.0,
        chamber_saturated=True,
        detection_method="UV 254",
        experimental_rf=0.42,
        tailing_value="Mild",
        spot_quality_value="Diffuse",
        spotting_volume_ul=1.5,
        notes_text="integration test run",
    )

    result = db_client.call_function("tlc", token, {**body, "action": "create"})

    assert result["analysis"]["analysis_id"].startswith("ANL-")
    assert result["analysis"]["analysis_type"] == "tlc"
    assert result["analysis"]["run_status"] == "completed"
    assert result["analysis"]["sample_name"] == body["sample_name"]
    assert result["tlc_run"]["plate_type"] == "silica_gel"
    assert result["tlc_run"]["chamber_saturation"] is True
    assert len(result["spots"]) == 1
    assert result["spots"][0]["rf_value"] == 0.42
    assert result["spots"][0]["spot_shape"] == "diffuse"

    # Read it back through the read action too, the way the Lab Portal's own
    # TLC screen would -- confirms the round trip, not just the write.
    read_back = db_client.call_function(
        "tlc", token, {"action": "read", "analysis_id": result["analysis"]["analysis_id"]}
    )
    assert read_back["analysis"]["title"] == body["title"]
    assert "CCO" in read_back["analysis"]["notes"]


def test_a_blank_compound_name_still_records_under_the_unregistered_placeholder():
    email, password = _make_test_account()
    token = db_client.sign_in(email, password)["access_token"]

    body = build_cloud_tlc_body(
        compound_name="",
        smiles="",
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

    result = db_client.call_function("tlc", token, {**body, "action": "create"})
    assert result["analysis"]["sample_name"] == "미등록 지질"


def test_bad_login_raises_a_friendly_auth_error():
    try:
        with pytest.raises(db_client.ApiError) as exc_info:
            db_client.sign_in("nobody@example.invalid", "wrong-password")
    except requests.exceptions.ConnectionError:
        pytest.skip("local Supabase stack is not reachable at 127.0.0.1:54321")
    assert exc_info.value.code == "AUTH_FAILED"
