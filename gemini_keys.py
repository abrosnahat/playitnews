"""
Round-robin Gemini API key rotation.

Reads GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ... from the
environment (falls back to a single GEMINI_API_KEY / GOOGLE_API_KEY if no
numbered keys are set). The currently-active key index is persisted to a
small JSON file so that it survives restarts AND is shared between the
separate `main.py` / `webapp.py` processes.

Usage:
    from gemini_keys import get_current_key, rotate_key, is_quota_error

    api_key = get_current_key()
    try:
        ...call Gemini...
    except SomeError as exc:
        if is_quota_error(exc):
            rotate_key(reason=str(exc))
            # retry with the new key
"""
import json
import logging
import os
import threading

from config import BASE_DIR

logger = logging.getLogger(__name__)

_STATE_PATH = os.path.join(BASE_DIR, "gemini_key_state.json")
_LOCK = threading.Lock()


def _load_keys() -> list[str]:
    keys: list[str] = []
    i = 1
    while True:
        v = os.getenv(f"GEMINI_API_KEY_{i}")
        if not v:
            break
        v = v.strip()
        if v and v not in keys:
            keys.append(v)
        i += 1
    # Back-compat: a single unnumbered key.
    single = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if single and single not in keys:
        keys.append(single)
    return keys


_KEYS: list[str] = _load_keys()


def _read_index() -> int:
    if not _KEYS:
        return 0
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as fh:
            idx = int(json.load(fh).get("index", 0))
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        idx = 0
    return idx % len(_KEYS)


def _write_index(idx: int) -> None:
    tmp_path = _STATE_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump({"index": idx}, fh)
        os.replace(tmp_path, _STATE_PATH)
    except OSError as exc:
        logger.warning("Failed to persist Gemini key index: %s", exc)


def key_count() -> int:
    return len(_KEYS)


def get_current_key() -> str:
    """Return the currently-active Gemini API key ("" if none configured)."""
    if not _KEYS:
        return ""
    with _LOCK:
        return _KEYS[_read_index()]


def rotate_key(reason: str = "") -> str:
    """Advance to the next Gemini API key (wraps around), persist it, and
    return the newly-active key. No-op (returns the same key) if only one
    (or zero) keys are configured.
    """
    if len(_KEYS) <= 1:
        return _KEYS[0] if _KEYS else ""
    with _LOCK:
        idx = _read_index()
        new_idx = (idx + 1) % len(_KEYS)
        _write_index(new_idx)
        logger.warning(
            "Gemini API key rotated: #%d -> #%d/%d%s",
            idx + 1, new_idx + 1, len(_KEYS),
            f" ({reason})" if reason else "",
        )
        return _KEYS[new_idx]


def is_quota_error(exc: BaseException) -> bool:
    """Heuristic check whether an exception represents a Gemini quota /
    rate-limit error (HTTP 429 / RESOURCE_EXHAUSTED), as opposed to some
    other failure (network error, bad request, etc.)."""
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower()
