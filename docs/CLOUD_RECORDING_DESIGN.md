# Cloud recording — design note

## What this is

The owner's original architecture called for TLC (and HSP) experiment recordings to land
in the same shared database the Lab Portal manages, so a researcher's TLC results,
compound records, and formulation runs all trace back to one place with real provenance.
That connection was never built — every prior round explicitly deferred "modifying the
TLC Simulator's own code to call the database" as separate later work. This round is that
work, for TLC. HSP is next; it needs new backend tables/functions built first (none exist
yet for HSP screening data), so it did not fit in the same pass.

## What changed

- `db_client.py` — a small, self-contained HTTP client (`sign_in`, `call_function`,
  `friendly_error`), vendored from the Lab Portal's own `apps/lab_portal/lib/api_client.py`
  since this is a separate repository and cannot import the Portal's code directly. Holds
  only the signed-in researcher's own access token, never a service-role key — same
  boundary the Portal itself holds.
- `cloud_record_mapping.py` — pure functions mapping this app's own Record Experiment
  fields onto `tlc-create`'s request shape. Split out from `app.py` on purpose: `app.py`'s
  top-level code runs Streamlit UI calls at import time and can only run inside
  `streamlit run`, so nothing that lives there is importable by a plain test — the same
  reason `mobile_phase.py`/`experiment_row.py`/`validation.py` are already separate.
- `app.py` — a new, collapsed-by-default sidebar section ("☁️ 공유 데이터베이스 로그인
  (선택)") for optional login, and a hook in the Record Experiment tab: after the existing
  local `runtime_data/experiment_db.csv` save succeeds (unchanged), if the researcher is
  logged in, the same recorded experiment is also sent to `tlc-create`.

## Design decisions

**Entirely opt-in, local save untouched.** Not logging in changes nothing about existing
behavior — the local CSV save is unconditional and happens first. The cloud save is a
second, independent step that can fail without undoing or blocking the local one; a
failure surfaces as a warning naming what happened, not a crash.

**No compound/batch code linking in this pass.** This app has no concept of the Portal's
`CMP-######`/`BAT-######` codes. Every cloud-recorded run goes in as a free-text
`sample_name` (the entered compound name, or "미등록 지질" if left blank) — consistent
with how every other unregistered-material TLC run has been recorded on this platform
(`sample_name="미등록 지질 X"` precedent from the Portal's own TLC round). Adding a
code-search picker here is possible future work, not required for the database connection
itself to work.

**Never silently discard a value with no clean home in the DB's vocabulary.** This app's
own fields (free-text plate description, free-text detection method, a 4-level tailing
severity, a 5-value spot-quality enum) don't map cleanly onto `tlc_runs`/`tlc_spots`'
fixed option lists. Where a mapping is a judgment call rather than an exact match
(`plate_type`, `visualization_method`, `spot_shape`), the mapped value is a best-effort
classification and the exact original text is *also* kept verbatim in `tlc_notes` / the
spot's own `notes` — so a mapping decision, even a wrong one, never loses the source data.
`map_visualization_method("UV")` deliberately returns `"other"`, not a guessed wavelength:
"UV" alone doesn't say 254 or 365 nm, and fabricating one would misrepresent the record.

**Run status is `"completed"`, always.** Everything recorded through this form already
happened — this tab's own copy says "여기에는 실제로 수행한 TLC 결과만 저장하세요"
(only real, already-performed results). There's no "planned" or "in progress" TLC
experiment recordable from this screen.

## Verified

- 12 unit tests (`tests/test_cloud_record_mapping.py`) cover the pure mapping logic:
  vocabulary matches, the "don't guess a wavelength" case, tailing-severity overriding
  spot-quality, cm→mm conversion, and that raw text always survives alongside a mapped
  value.
- 3 integration tests (`tests/test_db_client_integration.py`) call the real local
  Supabase stack: sign up a throwaway account, sign in, submit a mapped body through the
  real `tlc-create` function, and read it back through `tlc-read` — confirming the whole
  path end to end, not just that the mapping function produces a plausible dict. Skip
  cleanly if the local stack isn't running.
- Full suite: 52/52 passing.
- A live browser click-through of the login form was attempted and could not complete in
  this sandbox — the same click-delivery limitation already root-caused (not a code
  defect) while building the Lab Portal's chromatography screen the same day: real
  OS-level clicks via the available browser-automation tool don't reliably reach
  Streamlit's React event handlers in this specific sandboxed session. The sidebar login
  code is a near-verbatim structural match to the Lab Portal's own already-working
  `login.py` (same `sign_in` → `session_state` → `st.rerun()` shape), and the backend path
  it calls into is proven end to end by the integration tests above. A human with working
  browser access should still click through it once.
