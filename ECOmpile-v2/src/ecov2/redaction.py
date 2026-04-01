from __future__ import annotations

import re
from pathlib import Path


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d")
SID_RE = re.compile(r"\bS-\d-\d+(?:-\d+){1,}\b", flags=re.IGNORECASE)
WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\(?:[^\\\n\r]+\\)*[^\\\n\r]*")


def redact_text(value: str) -> str:
    out = value
    out = EMAIL_RE.sub("[REDACTED_EMAIL]", out)
    out = PHONE_RE.sub("[REDACTED_PHONE]", out)
    out = SID_RE.sub("[REDACTED_ACCOUNT_ID]", out)
    out = WINDOWS_PATH_RE.sub(_replace_windows_path, out)
    return out


def redact_path(value: str | Path) -> str:
    return _replace_windows_path(str(value))


def _replace_windows_path(match_or_text: re.Match[str] | str) -> str:
    text = match_or_text.group(0) if hasattr(match_or_text, "group") else str(match_or_text)
    return "[REDACTED_LOCAL_PATH]"
