"""Optional cloud recording: connects this standalone app to the shared LNP
Lab Platform database (the same Supabase project the Lab Portal uses), via
its tlc-create/tlc-read Edge Functions.

Entirely opt-in. Logging in (sidebar) only adds a second save alongside the
existing local runtime_data/experiment_db.csv -- nothing about offline,
not-logged-in usage changes. Design reasoning and the field mapping this
enables live in docs/16_LNP_DATASET_STRATEGY.md's sibling note in the main
platform repository is not available here since this is a separate
repository; see this file's own docstrings and app.py's Record Experiment
tab for the mapping instead.

Deliberately plain `requests`, no Supabase SDK, no service-role key ever
held here -- only the signed-in researcher's own access token, mirroring
the boundary the Lab Portal's own lib/api_client.py holds
(apps/lab_portal/lib/api_client.py) and vendored from it, since this app is
a separate repository and cannot import the Portal's code directly.
"""

import os

import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "http://127.0.0.1:54321")
SUPABASE_ANON_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9."
    "CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0",
)


class ApiError(Exception):
    def __init__(self, code, message, status):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(f"{code}: {message}")


def sign_in(email, password):
    """Returns {access_token, refresh_token, user} on success. Raises
    ApiError on failure."""
    res = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
        json={"email": email, "password": password},
        timeout=15,
    )
    body = res.json()
    if res.status_code != 200:
        raise ApiError(
            "AUTH_FAILED",
            body.get("msg") or body.get("error_description") or "Sign-in failed.",
            res.status_code,
        )
    return body


def call_function(name, token, body=None):
    """POST to an Edge Function with the caller's own access token. Raises
    ApiError with the function's own {code, message} on any non-2xx
    response."""
    res = requests.post(
        f"{SUPABASE_URL}/functions/v1/{name}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body or {},
        timeout=30,
    )
    parsed = res.json() if res.content else {}
    if res.status_code >= 400:
        err = parsed.get("error", {})
        raise ApiError(err.get("code", "UNKNOWN_ERROR"), err.get("message", "Request failed."), res.status_code)
    return parsed


_MESSAGES = {
    "AUTH_FAILED": "이메일 또는 비밀번호가 올바르지 않습니다.",
    "VALIDATION_FAILED": "입력한 내용을 다시 확인해주세요.",
    "COMPOUND_NOT_FOUND": "해당 화합물 번호를 찾을 수 없습니다.",
    "BATCH_NOT_FOUND": "해당 배치 번호를 찾을 수 없습니다.",
    "NOT_AUTHENTICATED": "로그인이 만료되었습니다. 사이드바에서 다시 로그인해주세요.",
    "INTERNAL_ERROR": "예상치 못한 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
}
_DEFAULT_MESSAGE = "공유 데이터베이스 저장 중 문제가 발생했습니다."


def friendly_error(e):
    return _MESSAGES.get(e.code, _DEFAULT_MESSAGE)
