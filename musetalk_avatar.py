"""
musetalk_avatar.py — PlayItNews wrapper around MuseTalk.

Generates a realistic lip-synced talking-head video for a given narration
audio file. Runs the heavy model in an isolated environment
(``musetalk_repo/.venv-musetalk``, torch cu128) via a subprocess worker, while
this module itself is imported by the main app's ``.venv``.

Public API
----------
``render_talking_head(audio_path, out_path) -> bool`` (async)
    Produce a talking-head mp4 (avatar lip-synced to ``audio_path``) at
    ``out_path``. Returns True on success.

The talking-head video has the narration audio muxed in by MuseTalk; the
caller (video_generator) uses only its *frames* and supplies the final mixed
audio (TTS + music) itself.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import random
import subprocess
import threading

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
MUSETALK_REPO = os.environ.get("MUSETALK_REPO", os.path.join(_HERE, "musetalk_repo"))
_VENV_PY = os.path.join(MUSETALK_REPO, ".venv-musetalk", "Scripts", "python.exe")

# Source talking-person loop video the avatar is built from. Set
# MUSETALK_AVATAR_VIDEO to force one specific file; otherwise a random
# assets/avatar*.mp4 is picked per render so generations vary the presenter.
# The bundled MuseTalk sample is the last-resort fallback.
_FORCED_AVATAR = os.environ.get("MUSETALK_AVATAR_VIDEO", "").strip()
_AVATAR_DIR = os.path.join(_HERE, "assets")
_SAMPLE_AVATAR = os.path.join(MUSETALK_REPO, "data", "video", "yongen.mp4")

MUSETALK_FPS = int(os.environ.get("MUSETALK_FPS", "25"))
MUSETALK_BATCH = int(os.environ.get("MUSETALK_BATCH", "20"))
# Lip-sharpness / blending tunables (see README "improve lips" notes):
#   extra_margin   — how much jaw area is included in the inpainted region.
#   parsing_mode   — "jaw" (default, sharper teeth) or "raw".
#   *_cheek_width  — how far the blend mask extends sideways.
MUSETALK_EXTRA_MARGIN = int(os.environ.get("MUSETALK_EXTRA_MARGIN", "10"))
MUSETALK_PARSING_MODE = os.environ.get("MUSETALK_PARSING_MODE", "jaw")
# Generous: first call also prepares the avatar (one-time) + loads models.
MUSETALK_TIMEOUT = int(os.environ.get("MUSETALK_TIMEOUT", "1800"))


def _list_avatar_videos() -> list[str]:
    """All assets/avatar*.mp4 source videos, sorted for stable order."""
    return sorted(glob.glob(os.path.join(_AVATAR_DIR, "avatar*.mp4")))


def _resolve_avatar_video() -> str | None:
    # 1. Explicit override wins.
    if _FORCED_AVATAR:
        if os.path.isfile(_FORCED_AVATAR):
            return _FORCED_AVATAR
        logger.warning(
            "MUSETALK_AVATAR_VIDEO set but not found (%s) — falling back",
            _FORCED_AVATAR,
        )
    # 2. Random pick among assets/avatar*.mp4.
    candidates = _list_avatar_videos()
    if candidates:
        chosen = random.choice(candidates)
        logger.info(
            "Avatar source: %s (random of %d)", os.path.basename(chosen), len(candidates)
        )
        return chosen
    # 3. Bundled sample as last resort.
    if os.path.isfile(_SAMPLE_AVATAR):
        logger.warning(
            "No assets/avatar*.mp4 found — using bundled sample avatar %s",
            _SAMPLE_AVATAR,
        )
        return _SAMPLE_AVATAR
    return None


def is_available() -> bool:
    """True if the isolated env and at least one avatar source video exist."""
    if not os.path.isfile(_VENV_PY):
        return False
    if _FORCED_AVATAR and os.path.isfile(_FORCED_AVATAR):
        return True
    return bool(_list_avatar_videos()) or os.path.isfile(_SAMPLE_AVATAR)


def _drain(stream, sink: list[str]) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
        logger.debug("musetalk[stderr]: %s", line.rstrip())
    try:
        stream.close()
    except Exception:
        pass


def _render_sync(audio_path: str, out_path: str) -> bool:
    avatar_video = _resolve_avatar_video()
    if not avatar_video:
        logger.error("MuseTalk: no avatar video available (set MUSETALK_AVATAR_VIDEO)")
        return False
    if not os.path.isfile(_VENV_PY):
        logger.error("MuseTalk: isolated venv python not found at %s", _VENV_PY)
        return False

    cmd = [
        _VENV_PY, "-m", "scripts.playit_worker",
        "--avatar_video", os.path.abspath(avatar_video),
        "--fps", str(MUSETALK_FPS),
        "--batch_size", str(MUSETALK_BATCH),
        "--extra_margin", str(MUSETALK_EXTRA_MARGIN),
        "--parsing_mode", MUSETALK_PARSING_MODE,
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logger.info("MuseTalk: starting worker (avatar=%s)", os.path.basename(avatar_video))
    proc = subprocess.Popen(
        cmd, cwd=MUSETALK_REPO,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env,
    )
    err_lines: list[str] = []
    err_thread = threading.Thread(target=_drain, args=(proc.stderr, err_lines), daemon=True)
    err_thread.start()

    ok = False
    try:
        # Wait for READY.
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                logger.error("MuseTalk worker exited before READY (code %s)", proc.returncode)
                return False
            if line.strip() == "READY":
                break

        # Submit the single job.
        job = {"audio_path": os.path.abspath(audio_path), "out_path": os.path.abspath(out_path)}
        proc.stdin.write(json.dumps(job) + "\n")
        proc.stdin.flush()

        # Wait for RESULT / ERROR.
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                logger.error("MuseTalk worker exited before RESULT (code %s)", proc.returncode)
                break
            line = line.strip()
            if line.startswith("RESULT "):
                ok = os.path.exists(out_path)
                break
            if line.startswith("ERROR "):
                logger.error("MuseTalk render error: %s", line[6:])
                break
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        if not ok and err_lines:
            logger.error("MuseTalk stderr tail:\n%s", "".join(err_lines[-25:]))

    return ok


async def render_talking_head(audio_path: str, out_path: str) -> bool:
    """Async wrapper: render a lip-synced talking-head video for ``audio_path``."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_render_sync, audio_path, out_path),
            timeout=MUSETALK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("MuseTalk render timed out after %ds", MUSETALK_TIMEOUT)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("MuseTalk render failed: %s", exc, exc_info=True)
        return False
