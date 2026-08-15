"""
Video generation for TikTok / Reels / Shorts.

Pipeline:
  AI script → edge-tts voice + VTT subtitles → collect media (article images
  + YouTube gameplay clips + Pixabay stock fill) → ffmpeg slideshow + audio
  + burned-in subtitles → mp4

Output: 1080 × 1920 portrait video (~18–25 s target for max retention.
"""
from __future__ import annotations

import asyncio
import bisect
import concurrent.futures
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid
from typing import Optional

import ssl

import aiohttp
import certifi
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Windows: ensure ctranslate2 (used by faster-whisper) can find the CUDA
# runtime DLLs (cuBLAS, cuDNN, NVRTC) shipped via the nvidia-* pip wheels.
# Without this, faster-whisper on CUDA fails at first transcribe() with
# "Library cublas64_12.dll is not found or cannot be loaded".
# Harmless if the packages aren't installed.
# ---------------------------------------------------------------------------
if os.name == "nt":
    _site_packages = sysconfig.get_paths().get("purelib", "")
    _cuda_dll_dirs: list[str] = []
    for _sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
        _p = os.path.normpath(os.path.join(_site_packages, *_sub.split("/")))
        if os.path.isdir(_p):
            _cuda_dll_dirs.append(_p)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_p)
                except OSError:
                    pass
    if _cuda_dll_dirs:
        # Also prepend to PATH — ctranslate2's internal LoadLibrary calls in
        # some build configs ignore the per-thread DLL search dirs and only
        # consult PATH.
        os.environ["PATH"] = os.pathsep.join(_cuda_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

# Load .env BEFORE the module-level os.getenv() calls below.
# Required because this module is imported in webapp.py before `config`
# (which is the other place that calls load_dotenv()), so otherwise those
# defaults would be used instead of .env values.
load_dotenv()

import scraper as _scraper
import ai_adapter
from config import VIDEOS_DIR, YT_CLIP_SKIP, YT_MAX_FILESIZE
import gemini_keys

# Directory with royalty-free background music tracks (mp3/wav/flac/ogg/m4a)
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")

# Use certifi CA bundle — same fix as scraper.py prevents SSL errors on macOS
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

logger = logging.getLogger(__name__)

VID_W = 1080
VID_H = 1920
VID_FPS = 30

TTS_VOICE    = "en-US-AndrewMultilingualNeural"  # English — warm, authentic
TTS_VOICE_RU = "ru-RU-DmitryNeural"              # Russian — Microsoft Neural TTS
TTS_RATE     = "+15%"   # EN: 15% faster → targets 18–25s video length for max retention
TTS_RATE_RU  = "+15%"   # RU: Dmitry reads slower & inserts longer inter-sentence pauses;
                        # bump rate so audio length (and per-frame seg_dur) matches EN.
TTS_PITCH    = "-3Hz"   # Slightly lower pitch → warmer tone

# ---------------------------------------------------------------------------
# Gemini TTS backend (optional replacement for edge-tts).
#
# Set TTS_BACKEND=gemini in .env to route TTS through Gemini TTS models. The
# free tier at aistudio.google.com is sufficient for this pipeline (≤10 shorts
# / day, ~25 s each — well under 50 RPD and 10 RPM limits).
#
# Recommended model: gemini-2.5-flash-preview-tts (more stable). 3.1 Flash TTS
# is preview-quality with better prosody but ~3-5% "text-token-returned" 500
# errors — retry is built in.
#
# All 30 voices are language-agnostic; `languageCode` in the prompt steers
# pronunciation. Recommended picks for gaming-news shorts:
#   EN — Schedar (even/dj), Kore (firm), Puck (upbeat)
#   RU — Kore (neutral), Schedar (even), Sulafat (warm)
# ---------------------------------------------------------------------------
try:
    from google import genai as _genai
    from google.genai import types as _gtypes
    _GENAI_OK = True
except Exception:                                          # noqa: BLE001
    _GENAI_OK = False


async def _gemini_tts_generate(prompt: str, chosen_voice: str, language_code: str, max_retries: int):
    """Call Gemini TTS ``generate_content``, retrying transient errors and
    rotating to the next configured GEMINI_API_KEY_N on quota exhaustion
    (HTTP 429 / RESOURCE_EXHAUSTED). Returns the raw SDK response.

    Raises RuntimeError if no API key is configured or all keys/retries fail.
    """
    speech_cfg = _gtypes.SpeechConfig(
        voice_config=_gtypes.VoiceConfig(
            prebuilt_voice_config=_gtypes.PrebuiltVoiceConfig(voice_name=chosen_voice)
        ),
        language_code=language_code,
    )
    gen_config = _gtypes.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=speech_cfg,
    )

    max_key_attempts = max(gemini_keys.key_count(), 1)
    last_err: Exception | None = None
    for key_attempt in range(1, max_key_attempts + 1):
        api_key = gemini_keys.get_current_key() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env — "
                "get one at https://aistudio.google.com/apikey"
            )
        client = _genai.Client(api_key=api_key)
        quota_hit = False
        for attempt in range(1, max_retries + 1):
            try:
                return client.models.generate_content(
                    model=GEMINI_TTS_MODEL,
                    contents=prompt,
                    config=gen_config,
                )
            except Exception as exc:                      # noqa: BLE001
                last_err = exc
                if gemini_keys.is_quota_error(exc):
                    quota_hit = True
                    logger.warning(
                        "Gemini TTS quota exceeded on key #%d/%d: %s",
                        key_attempt, max_key_attempts, exc.__class__.__name__,
                    )
                    break   # don't burn retries on an exhausted key — rotate instead
                logger.warning(
                    "Gemini TTS attempt %d/%d failed: %s",
                    attempt, max_retries, exc.__class__.__name__,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
        if quota_hit and key_attempt < max_key_attempts:
            gemini_keys.rotate_key(reason="429 RESOURCE_EXHAUSTED (TTS)")
            continue
        break   # either quota exhausted on all keys, or a non-quota failure

    raise RuntimeError(f"Gemini TTS failed after retries: {last_err}")


# Active backend selector. "edge" = original edge-tts (default),
# "gemini" = route through Gemini TTS.
TTS_BACKEND = os.getenv("TTS_BACKEND", "edge").strip().lower()
GEMINI_TTS_MODEL     = os.getenv(
    "GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts",
)
# 3.1 Flash TTS is also valid: "gemini-3.1-flash-tts-preview"
GEMINI_TTS_VOICE_EN  = os.getenv("GEMINI_TTS_VOICE_EN",  "Schedar")
GEMINI_TTS_VOICE_RU  = os.getenv("GEMINI_TTS_VOICE_RU",  "Kore")
# Director's-note prelude — Gemini TTS is LLM-based; the prompt structure
# (instructions → transcript) materially changes prosody. Prepending this
# preamble gives reliable "gaming-news anchor" delivery and avoids the model
# reading its own style notes aloud (a known safety-classifier false positive).
# TUNED: energetic pace (≈165 wpm) without vocal emotion — gives a fast
# "professional newscast" feel, lands ~70-word scripts in ~22 s. Tested
# against the previous "flat" version: ~30% shorter audio, no detectable
# pitch variation increase.
_GEMINI_DIR_EN = (
    "Read at a brisk, newscast pace — about 115 words per minute. "
    "Professional, measured, and confident, but with no vocal smile, "
    "no exclamations, and no pitch variation. Keep your energy high and "
    "your delivery quick; The result should sound like an experienced blogger "
    "on a tight deadline, not a storyteller. Speak ONLY the transcript below:\n\n"
)
_GEMINI_DIR_RU = (
    "Читай в быстром темпе диктора новостей — около 95 слов в минуту. "
    "Профессионально, собранно, уверенно, но без улыбки в голосе, без "
    "восклицаний и без перепадов тона. Держи высокую энергию и быструю "
    "подачу; Звучать должно как опытный блоггер на сжатых "
    "сроках, а не как рассказчик. Произнеси ТОЛЬКО текст ниже:\n\n"
)

# Optional call-to-action appended to the narration (opt-in via UI checkbox).
# Drives viewers from Shorts/Reels to the Telegram channels. The hook is the
# "full game trailer" — something the short clip doesn't fully show.
CTA_PHRASE    = "Full trailer is in our Telegram, link in bio!"
CTA_PHRASE_RU = "Полный трейлер в нашем Telegram канале, ссылка в профиле!"


# yt-dlp cookie arguments — passes browser cookies to bypass YouTube bot check.
#
# Chrome on Windows locks its cookie DB while the browser is running, which makes
# `--cookies-from-browser chrome` fail with "Could not copy Chrome cookie database"
# (yt-dlp/yt-dlp#7271). To work around that, allow overriding via env:
#   YT_COOKIES_FILE     — path to an exported cookies.txt (preferred on Windows)
#   YT_COOKIES_BROWSER  — browser name for --cookies-from-browser (default: chrome)
_YT_COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "").strip()
_YT_COOKIES_BROWSER = os.getenv("YT_COOKIES_BROWSER", "chrome").strip()
if _YT_COOKIES_FILE:
    _YT_COOKIE_ARGS = ["--cookies", _YT_COOKIES_FILE]
elif _YT_COOKIES_BROWSER.lower() in ("", "none", "off", "0"):
    _YT_COOKIE_ARGS = []
else:
    _YT_COOKIE_ARGS = ["--cookies-from-browser", _YT_COOKIES_BROWSER]

# YouTube extractor args: tv_downgraded client works best with authenticated cookies
# (per yt-dlp docs: used for free accounts when logged-in cookies are passed)
_YT_EXTRACTOR_ARGS = ["--extractor-args", "youtube:player_client=tv_downgraded,web_safari"]

# Sleep between requests to avoid YouTube rate-limiting / bot detection
_YT_SLEEP_ARGS = ["--sleep-requests", "1", "--sleep-interval", "2"]

# ---------------------------------------------------------------------------
# Generic subprocess helper
# ---------------------------------------------------------------------------

def _run(args: list[str], cwd: str | None = None, timeout: int = 120) -> bool:
    """Run a command, return True on success."""
    try:
        # Extend PATH so yt-dlp can find JS runtimes (deno, node) for n-challenge solving.
        _env = os.environ.copy()
        if os.name != "nt":
            _extra_paths = [
                "/opt/homebrew/bin",                                      # deno (macOS Homebrew)
                "/usr/local/bin",
                os.path.expanduser("~/.nvm/versions/node/v20.19.5/bin"), # node (nvm)
            ]
            _env["PATH"] = os.pathsep.join(_extra_paths) + os.pathsep + _env.get("PATH", "")
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=_env,
        )
        # yt-dlp exits with 101 when --max-downloads limit is reached — treat as success
        if result.returncode not in (0, 101):
            logger.error("Command failed [%s] (rc=%d): %s", args[0], result.returncode, result.stderr[-600:])
            return False
        if result.stderr:
            logger.debug("Command stderr [%s]: %s", args[0], result.stderr[-400:])
        return True
    except subprocess.TimeoutExpired:
        logger.error("Command timed out: %s", args[:3])
        return False
    except FileNotFoundError:
        logger.error("Command not found: %s — please install it", args[0])
        return False
    except Exception as exc:
        logger.error("Command error (%s): %s", args[0], exc)
        return False


async def _run_async(args: list[str], cwd: str | None = None, timeout: int = 160) -> bool:
    """Async wrapper for _run (runs in thread pool to not block the event loop)."""
    return await asyncio.to_thread(_run, args, cwd, timeout)


def _run_capture(args: list[str], timeout: int = 120) -> str:
    """Run a command and return its captured stdout (empty string on failure).
    Used where we need the JSON output of a command (e.g. yt-dlp --dump-json),
    as opposed to `_run`, which only reports success/failure."""
    try:
        _env = os.environ.copy()
        if os.name != "nt":
            _extra_paths = [
                "/opt/homebrew/bin",
                "/usr/local/bin",
                os.path.expanduser("~/.nvm/versions/node/v20.19.5/bin"),
            ]
            _env["PATH"] = os.pathsep.join(_extra_paths) + os.pathsep + _env.get("PATH", "")
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, env=_env,
        )
        if result.returncode not in (0, 101) and not result.stdout:
            logger.warning("Command failed [%s] (rc=%d): %s", args[0], result.returncode, result.stderr[-400:])
        return result.stdout or ""
    except Exception as exc:
        logger.warning("Command capture error (%s): %s", args[0] if args else "?", exc)
        return ""


async def _run_capture_async(args: list[str], timeout: int = 120) -> str:
    """Async wrapper for _run_capture (runs in thread pool)."""
    return await asyncio.to_thread(_run_capture, args, timeout)


# ---------------------------------------------------------------------------
# Hardware H.264 encoder auto-detection
#
# Falls back to libx264. On Windows 11 with a modern GPU this typically gives
# a 4-10× speed-up over CPU encoding for our slideshow / composite passes.
# Override with VIDEO_ENCODER env var:
#   VIDEO_ENCODER=libx264|h264_nvenc|h264_qsv|h264_amf
# ---------------------------------------------------------------------------

def _probe_encoder(name: str) -> bool:
    """Try a 1-frame encode to confirm the encoder actually works on this host."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                # 320×240 — NVENC requires at least 145×49, AMF/QSV have similar
                # minimums, so 64×64 is too small for a generic probe.
                "-f", "lavfi", "-i", "color=size=320x240:rate=1:duration=1",
                "-c:v", name, "-frames:v", "1", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _detect_h264_encoder() -> str:
    override = (os.getenv("VIDEO_ENCODER") or "").strip()
    if override:
        return override
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "libx264"
    # NVIDIA → Intel → AMD priority; NVENC is usually fastest and highest quality.
    for name in ("h264_nvenc", "h264_qsv", "h264_amf"):
        if name in out and _probe_encoder(name):
            return name
    return "libx264"


_SELECTED_ENCODER = _detect_h264_encoder()
logger.info("Video encoder: %s", _SELECTED_ENCODER)


def _video_encoder_args(crf: int = 23, preset: str = "fast") -> list[str]:
    """Return ffmpeg args for the H.264 video encoder.

    Maps libx264's `-crf` to the equivalent quality knob on each HW encoder.
    """
    enc = _SELECTED_ENCODER
    if enc == "h264_nvenc":
        # p1=fastest..p7=slowest. p4 ≈ libx264 'medium'; use VBR HQ + CQ.
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(crf),
            "-b:v", "0",
        ]
    if enc == "h264_qsv":
        return [
            "-c:v", "h264_qsv",
            "-preset", "faster",
            "-global_quality", str(crf),
            "-look_ahead", "0",
        ]
    if enc == "h264_amf":
        return [
            "-c:v", "h264_amf",
            "-quality", "speed",
            "-rc", "cqp",
            "-qp_i", str(crf),
            "-qp_p", str(crf),
        ]
    # CPU fallback. -threads 0 = auto.
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-threads", "0"]


# ---------------------------------------------------------------------------
# faster-whisper device selection + model cache
#
# RTX-class NVIDIA GPUs give a 5-10× speed-up over CPU for the 'small' model.
# Override device with WHISPER_DEVICE env var:
#   WHISPER_DEVICE=cuda|cpu|auto
# Override compute type with WHISPER_COMPUTE_TYPE (e.g. float16/int8_float16/int8).
# ---------------------------------------------------------------------------

_WHISPER_DEVICE: str | None = None
_WHISPER_COMPUTE_TYPE: str | None = None
_WHISPER_MODELS: dict[str, object] = {}


def _detect_whisper_device() -> tuple[str, str]:
    """Return (device, compute_type). Probes CUDA via a tiny WhisperModel load."""
    override = (os.getenv("WHISPER_DEVICE") or "").strip().lower()
    ct_override = (os.getenv("WHISPER_COMPUTE_TYPE") or "").strip()
    if override and override != "auto":
        return override, (ct_override or ("float16" if override == "cuda" else "int8"))
    # Probe CUDA by trying to load the smallest model on cuda; cheap (~1-2 s).
    try:
        from faster_whisper import WhisperModel
        _probe = WhisperModel("tiny", device="cuda", compute_type="float16")
        del _probe
        return "cuda", (ct_override or "float16")
    except Exception as exc:
        logger.info("CUDA not available for faster-whisper (%s) — falling back to CPU.", exc.__class__.__name__)
        return "cpu", (ct_override or "int8")


def _get_whisper_model(model_name: str):
    """Return a cached WhisperModel for the given size."""
    global _WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE
    if _WHISPER_DEVICE is None:
        _WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE = _detect_whisper_device()
        logger.info("Whisper device: %s (%s)", _WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE)
    cached = _WHISPER_MODELS.get(model_name)
    if cached is not None:
        return cached
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device=_WHISPER_DEVICE, compute_type=_WHISPER_COMPUTE_TYPE)
    except Exception as exc:
        # Last-ditch fallback to CPU if a previously-OK CUDA setup just broke
        # (driver reload, OOM, etc.).
        logger.warning("Whisper load %s on %s failed (%s); retrying on CPU.", model_name, _WHISPER_DEVICE, exc)
        try:
            from faster_whisper import WhisperModel
            _WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE = "cpu", "int8"
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as exc2:
            logger.error("Whisper load %s on CPU also failed: %s", model_name, exc2)
            return None
    _WHISPER_MODELS[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _get_audio_duration(path: str) -> float:
    """Return audio duration in seconds via ffprobe, default 40 s on error."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        val = result.stdout.strip()
        return float(val) if val else 40.0
    except Exception:
        return 40.0


# ---------------------------------------------------------------------------
# VTT → SRT conversion (edge-tts produces word-level VTT)
# ---------------------------------------------------------------------------

def _parse_vtt_entries(vtt_path: str) -> list[dict]:
    """Return list of {start, end, text} from a VTT file."""
    with open(vtt_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    entries: list[dict] = []
    for block in re.split(r"\n{2,}", content.strip()):
        block = block.strip()
        if not block or block.startswith("WEBVTT") or block.startswith("NOTE"):
            continue
        lines = block.splitlines()
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line:
            continue
        parts = ts_line.split("-->")
        start = parts[0].strip()
        # drop any positional cue settings after the end timestamp
        end = parts[1].strip().split()[0]
        text_parts = [
            l.strip()
            for l in lines
            if "-->" not in l and not re.match(r"^\d+$", l.strip()) and l.strip()
        ]
        text = " ".join(text_parts).strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})
    return entries


def _vtt_ts_to_sec(ts: str) -> float:
    """Convert VTT/SRT timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    try:
        ts = ts.strip().replace(",", ".")
        if ts.count(":") == 1:
            ts = "00:" + ts
        h, m, rest = ts.split(":")
        s_parts = rest.split(".")
        s  = int(s_parts[0])
        ms = int(s_parts[1][:3].ljust(3, "0")) if len(s_parts) > 1 else 0
        return int(h) * 3600 + int(m) * 60 + s + ms / 1000
    except Exception:
        return 0.0


def _parse_vtt_cues(vtt_path: str) -> list[tuple[float, float, str]]:
    """
    Parse a VTT/SRT file and return [(start_sec, end_sec, text), ...].
    Each cue keeps its exact timing from the file.
    """
    entries = _parse_vtt_entries(vtt_path)
    cues = []
    for e in entries:
        t_start = _vtt_ts_to_sec(e["start"])
        t_end   = _vtt_ts_to_sec(e["end"])
        text    = e["text"].strip()
        if text and t_end > t_start:
            cues.append((t_start, t_end, text))
    return cues


def _find_system_font(size: int):
    """Return a PIL ImageFont, preferring Impact on macOS."""
    from PIL import ImageFont
    _win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts")
    candidates = [
        # Windows
        os.path.join(_win_fonts, "impact.ttf"),
        os.path.join(_win_fonts, "arialbd.ttf"),
        os.path.join(_win_fonts, "arial.ttf"),
        os.path.join(_win_fonts, "seguiemj.ttf"),
        # macOS
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


async def _burn_subtitles_pillow(
    mixed_mp4: str,
    cues: list[tuple[float, float, str]],
    output_mp4: str,
    workdir: str,
) -> bool:
    """
    Burn subtitles using frame extraction + Pillow composite + single re-encode.
    Works with any ffmpeg build — no libass/libfreetype/drawtext needed.

    cues: word-level [(start_sec, end_sec, word)] from edge-tts WordBoundary
    events — exact TTS timing, no estimation or splitting needed.

    Pipeline:
      1. Extract video frames at VID_FPS
      2. Composite the current word onto each frame with Pillow
      3. Re-encode frames → mp4, mux original audio back in one pass
    """
    from PIL import Image

    if not cues:
        logger.warning("No subtitle cues — skipping subtitle burn")
        return False
    logger.info("Burning %d word-level subtitle cues", len(cues))

    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    EXTRACT_FPS = VID_FPS  # 30 fps — matches target output framerate

    # 1. Extract frames — force portrait 1080x1920 and correct SAR
    ok = await _run_async(
        [
            "ffmpeg", "-y",
            "-i", mixed_mp4,
            "-vf", (
                f"scale={VID_W}:{VID_H}:force_original_aspect_ratio=increase,"
                f"crop={VID_W}:{VID_H},"
                f"setsar=1,"
                f"fps={EXTRACT_FPS}"
            ),
            "-f", "image2",
            os.path.join(frames_dir, "frame_%06d.png"),
        ],
        timeout=180,
    )
    if not ok:
        logger.warning("Frame extraction failed")
        return False

    # 2. Build a lookup: for each frame index → subtitle text (or None)
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not frame_files:
        logger.warning("No frames extracted")
        return False

    def _composite_frame(fname: str) -> None:
        """Composite subtitle text onto a single frame."""
        idx = int(fname.replace("frame_", "").replace(".png", ""))
        t   = (idx - 1) / EXTRACT_FPS

        text: str | None = None
        cue_start: float = 0.0
        for t_start, t_end, cue_text in cues:
            if t_start <= t < t_end:
                text = cue_text
                cue_start = t_start
                break

        if not text:
            return

        scale = _bounce_scale(t - cue_start)
        fpath = os.path.join(frames_dir, fname)
        try:
            base    = Image.open(fpath).convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            _render_subtitle_onto(text, overlay, scale=scale)
            combined = Image.alpha_composite(base, overlay).convert("RGB")
            combined.save(fpath, "PNG")
        except Exception as exc:
            logger.debug("Frame composite error %s: %s", fname, exc)

    # Run compositing in a thread pool (Pillow is CPU-bound) — parallel across all cores
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(
            pool,
            lambda: list(pool.map(_composite_frame, frame_files)),
        )

    # 3. Re-encode frames + original audio, force portrait dimensions
    ok = await _run_async(
        [
            "ffmpeg", "-y",
            "-framerate", str(EXTRACT_FPS),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-i", mixed_mp4,
            "-map", "0:v",
            "-map", "1:a",
            "-vf", f"scale={VID_W}:{VID_H},setsar=1",
            *_video_encoder_args(crf=20),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ar", "44100",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            output_mp4,
        ],
        timeout=300,
    )
    return ok


_SUB_GRAD_TOP    = (255, 255, 255)       # white
_SUB_GRAD_BOTTOM = (0xC7, 0xF8, 0xFD)   # #C7F8FD light cyan

# Duration of the bounce-in animation per word (seconds)
_BOUNCE_DUR = 0.18


def _bounce_scale(elapsed: float) -> float:
    """easeOutBack: scale goes 0 → ~1.30 → 1.0 over _BOUNCE_DUR seconds."""
    p = min(1.0, elapsed / _BOUNCE_DUR)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (p - 1.0) ** 3 + c1 * (p - 1.0) ** 2


def _render_subtitle_onto(text: str, img, scale: float = 1.0) -> None:
    """Composite subtitle text onto an RGBA PIL image in-place.
    Uses Impact font, white→cyan gradient fill and italic shear (top leans right).
    scale: bounce-in scale factor (easeOutBack, 0→1.3→1.0).
    """
    from PIL import Image, ImageDraw

    font_size = 112
    font      = _find_system_font(font_size)

    # Measure lines using a temporary draw on a throwaway image
    _tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    max_w   = VID_W - 80
    words   = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        bbox = _tmp.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) > max_w and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    line_h  = font_size + 18
    total_h = len(lines) * line_h
    y_start = (VID_H - 330) - total_h // 2

    # --- Render text to a separate RGBA layer ---
    text_layer = Image.new("RGBA", (VID_W, VID_H), (0, 0, 0, 0))
    draw_t     = ImageDraw.Draw(text_layer)

    cur_y = y_start
    for line in lines:
        bbox_measure = draw_t.textbbox((0, 0), line, font=font)
        line_w = bbox_measure[2] - bbox_measure[0]
        x = (VID_W - line_w) // 2 - bbox_measure[0]

        # Thin black outline (2 px offset in 4 directions)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            draw_t.text((x + dx, cur_y + dy), line, font=font, fill=(0, 0, 0, 255))

        # Gradient fill (white top → cyan bottom)
        bbox = draw_t.textbbox((x, cur_y), line, font=font)
        ly0, ly1 = bbox[1], bbox[3]
        lh = max(1, ly1 - ly0)

        grad = Image.new("RGBA", (VID_W, lh), (0, 0, 0, 0))
        for gy in range(lh):
            t = gy / (lh - 1) if lh > 1 else 0
            r = int(_SUB_GRAD_TOP[0] + (_SUB_GRAD_BOTTOM[0] - _SUB_GRAD_TOP[0]) * t)
            g = int(_SUB_GRAD_TOP[1] + (_SUB_GRAD_BOTTOM[1] - _SUB_GRAD_TOP[1]) * t)
            b = int(_SUB_GRAD_TOP[2] + (_SUB_GRAD_BOTTOM[2] - _SUB_GRAD_TOP[2]) * t)
            ImageDraw.Draw(grad).line([(0, gy), (VID_W, gy)], fill=(r, g, b, 255))

        mask_full = Image.new("L", (VID_W, VID_H), 0)
        ImageDraw.Draw(mask_full).text((x, cur_y), line, font=font, fill=255)
        line_mask = mask_full.crop((0, ly0, VID_W, ly1))
        grad.putalpha(line_mask)
        text_layer.paste(grad, (0, ly0), mask=line_mask)

        cur_y += line_h

    # --- Italic shear: top of text leans right ---
    shear  = 0.18
    affine = (1, shear, -shear * y_start, 0, 1, 0)
    text_layer = text_layer.transform(
        (VID_W, VID_H), Image.AFFINE, affine, resample=Image.BICUBIC,
    )

    # --- Bounce-in scale: resize layer around text-block centre ---
    if abs(scale - 1.0) > 0.005:
        from PIL import Image as _PILImage
        cx = VID_W // 2
        cy = y_start + total_h // 2
        new_w = max(1, int(VID_W * scale))
        new_h = max(1, int(VID_H * scale))
        scaled = text_layer.resize((new_w, new_h), _PILImage.BICUBIC)
        dest = _PILImage.new("RGBA", (VID_W, VID_H), (0, 0, 0, 0))
        ox = cx - round(cx * scale)
        oy = cy - round(cy * scale)
        dest.paste(scaled, (ox, oy))
        text_layer = dest

    # --- Composite onto caller's image ---
    img.alpha_composite(text_layer)


# ---------------------------------------------------------------------------
# Text-to-Speech  (edge-tts)
# ---------------------------------------------------------------------------

def _detect_word_boundaries_from_audio(
    audio_path: str,
    sentence_cues: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """
    Use ffmpeg silencedetect to find word boundaries in TTS audio.

    Strategy:
    1. Try progressively more sensitive silencedetect configurations.
    2. If we get exactly N segments for N words → perfect, use them directly.
    3. If we get K < N segments → use them as anchors: distribute words
       across those K chunks proportionally by char length (bounded drift).
    4. If no silence detected → char-proportional across the sentence.

    This bounds sync error to within a single detected speech chunk (typically
    1-2 words) rather than drifting across the whole sentence.
    """
    # Try progressively more sensitive configs (most sensitive first)
    CONFIGS = [
        "noise=-28dB:d=0.012",   # very sensitive — catches short TTS pauses
        "noise=-30dB:d=0.018",
        "noise=-33dB:d=0.025",
        "noise=-35dB:d=0.035",   # original-ish
    ]

    def _run_silencedetect(config: str) -> list[tuple[float, float]]:
        res = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af", f"silencedetect={config}",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        silences: list[tuple[float, float]] = []
        s_start: float | None = None
        for line in res.stderr.split("\n"):
            if "silence_start" in line:
                m = re.search(r"silence_start:\s*([\d.]+)", line)
                if m:
                    s_start = float(m.group(1))
            elif "silence_end" in line and s_start is not None:
                m = re.search(r"silence_end:\s*([\d.]+)", line)
                if m:
                    silences.append((s_start, float(m.group(1))))
                    s_start = None
        return silences

    def _speech_segs(
        silences: list[tuple[float, float]], t_from: float, t_to: float
    ) -> list[tuple[float, float]]:
        """Non-silent regions within [t_from, t_to]."""
        segs: list[tuple[float, float]] = []
        prev = t_from
        for ss, se in silences:
            if se <= t_from or ss >= t_to:
                continue
            gap_start = max(ss, t_from)
            if gap_start > prev + 0.01:       # at least 10ms of speech
                segs.append((prev, gap_start))
            prev = max(se, prev)
        if prev < t_to - 0.01:
            segs.append((prev, t_to))
        return segs

    def _clean(word: str) -> str:
        return word.strip(".,!?;:\"'()-\u2013\u2014")

    def _char_weight(word: str) -> float:
        return max(1.0, float(len(word)))

    def _distribute_words_in_region(
        words: list[str], t_from: float, t_to: float
    ) -> list[tuple[float, float, str]]:
        """Distribute words inside [t_from, t_to] by char-length proportion."""
        total = sum(_char_weight(w) for w in words) or 1.0
        dur   = t_to - t_from
        result = []
        t = t_from
        for word in words:
            w_dur = dur * _char_weight(word) / total
            c = _clean(word)
            if c:
                result.append((t, t + w_dur, c))
            t += w_dur
        return result

    # Pick the silencedetect config that gives the closest match to total words
    best_silences: list[tuple[float, float]] = []
    best_score = -1
    total_words = sum(len(s.split()) for _, _, s in sentence_cues)

    for cfg in CONFIGS:
        silences = _run_silencedetect(cfg)
        # Count speech segments across all sentence windows
        n_segs = sum(
            len(_speech_segs(silences, ts, te))
            for ts, te, txt in sentence_cues
            for _ in [None]          # dummy loop to use ts/te
        )
        # Recount properly
        n_segs = 0
        for ts, te, txt in sentence_cues:
            n_segs += len(_speech_segs(silences, ts, te))

        score = -(abs(n_segs - total_words))   # closer to total_words = better
        if score > best_score:
            best_score = score
            best_silences = silences

    # Build word cues using best silences
    word_cues: list[tuple[float, float, str]] = []

    for t_start, t_end, sentence in sentence_cues:
        words = [w for w in sentence.split() if _clean(w)]
        if not words:
            continue

        segs = _speech_segs(best_silences, t_start, t_end)

        if len(segs) == len(words):
            # Perfect — each word → its speech segment
            for word, (ws, we) in zip(words, segs):
                c = _clean(word)
                if c:
                    word_cues.append((ws, we, c))

        elif len(segs) > 1:
            # Use detected segments as anchors; distribute words between them
            # proportionally by character length within each chunk.
            avg_word_dur = (t_end - t_start) / len(words)
            chunk_words: list[list[str]] = [[] for _ in segs]
            remaining = list(words)
            for i, (ws, we) in enumerate(segs):
                n = max(1, round((we - ws) / avg_word_dur))
                if i == len(segs) - 1:
                    chunk_words[i] = remaining
                else:
                    n = min(n, len(remaining) - (len(segs) - i - 1))
                    chunk_words[i] = remaining[:n]
                    remaining = remaining[n:]
            for (ws, we), chunk in zip(segs, chunk_words):
                word_cues.extend(_distribute_words_in_region(chunk, ws, we))

        else:
            # No usable silence detected — char-proportional across sentence
            word_cues.extend(_distribute_words_in_region(words, t_start, t_end))

    return word_cues


def _run_whisper_sync(
    audio_path: str,
    sentence_cues: list[tuple[float, float, str]],
    language: str = "en",
) -> list[tuple[float, float, str]]:
    """
    Use Whisper to get acoustic word timing, then map our known script words
    to that timing.

    Mapping strategy (per-sentence):
      For each VTT sentence boundary [t_start, t_end], collect Whisper word
      timings that fall inside that window, then map script words → Whisper
      timings by position ratio **within the window only**.  This bounds any
      drift to a single sentence (~1-2 s) instead of accumulating globally.

    Model: 'small' for Russian (better Cyrillic recognition), 'tiny' for EN.
    Returns [] if Whisper is unavailable or produces no word segments.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []

    def _clean(word: str) -> str:
        return word.strip(".,!?;:\"'()-\u2013\u2014")

    if not sentence_cues:
        return []

    model_name = "small"
    model = _get_whisper_model(model_name)
    if model is None:
        return []

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            segments, _info = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            # faster-whisper returns a generator — materialise it.
            seg_list = list(segments)
        except Exception as exc:
            logger.warning("Whisper transcribe failed: %s", exc)
            return []

    # Collect all Whisper word timings globally
    wh_timings: list[tuple[float, float]] = [
        (float(w.start), float(w.end))
        for seg in seg_list
        for w in (seg.words or [])
    ]

    if not wh_timings:
        return []

    cues: list[tuple[float, float, str]] = []

    if language == "ru":
        # Per-sentence mapping: Whisper timings filtered to each VTT sentence window.
        # Bounds drift to one sentence — needed because RU VTT has long phrase-level cues.
        for t_start, t_end, sentence in sentence_cues:
            script_words = [_clean(w) for w in sentence.split() if _clean(w)]
            if not script_words:
                continue
            window = [
                (s, e) for s, e in wh_timings
                if s >= t_start - 0.2 and e <= t_end + 0.2
            ]
            if window:
                n_sc = len(script_words)
                n_wh = len(window)
                for i, word in enumerate(script_words):
                    j = round(i * (n_wh - 1) / max(n_sc - 1, 1)) if n_sc > 1 else 0
                    j = min(j, n_wh - 1)
                    start, end = window[j]
                    cues.append((start, end, word))
            else:
                dur = t_end - t_start
                w_dur = dur / len(script_words)
                for i, word in enumerate(script_words):
                    cues.append((t_start + i * w_dur, t_start + (i + 1) * w_dur, word))
    else:
        # Global position-ratio mapping for EN: works well because EN VTT cues
        # are granular and Whisper EN accuracy is high.
        script_words = [
            _clean(w)
            for _, _, text in sentence_cues
            for w in text.split()
            if _clean(w)
        ]
        n_sc = len(script_words)
        n_wh = len(wh_timings)
        for i, word in enumerate(script_words):
            j = round(i * (n_wh - 1) / max(n_sc - 1, 1)) if n_sc > 1 else 0
            j = min(j, n_wh - 1)
            start, end = wh_timings[j]
            cues.append((start, end, word))

    return cues


# ---------------------------------------------------------------------------
# Russian TTS text pre-processing — transliterate English brands/terms
# ---------------------------------------------------------------------------

# Common gaming brands and terms → Russian phonetic spelling.
# Order inside the list doesn't matter — at compile time keys are sorted by
# length DESC so longer phrases match before their substrings
# (e.g. "Epic Games Store" before "Epic Games" before "Epic").
_RU_BRAND_MAP: list[tuple[str, str]] = [
    # ── Companies / studios ─────────────────────────────────────────────
    ("Activision Blizzard", "Активижн Близзард"),
    ("Activision", "Активижн"),
    ("Blizzard", "Близзард"),
    ("Bethesda", "Бесезда"),
    ("Ubisoft", "Юбисофт"),
    ("Capcom", "Капком"),
    ("Konami", "Конами"),
    ("Bandai Namco", "Бандай Намко"),
    ("Bandai", "Бандай"),
    ("Namco", "Намко"),
    ("FromSoftware", "Фром Софтвэр"),
    ("From Software", "Фром Софтвэр"),
    ("CD Projekt Red", "Си Ди Прожект Ред"),
    ("CD Projekt", "Си Ди Прожект"),
    ("Obsidian", "Обсидиан"),
    ("Insomniac", "Инсомниак"),
    ("Naughty Dog", "Нотти Дог"),
    ("Rockstar Games", "Рокстар Геймс"),
    ("Rockstar", "Рокстар"),
    ("Take-Two", "Тейк Ту"),
    ("Take Two", "Тейк Ту"),
    ("Valve", "Вэлв"),
    ("Epic Games Store", "Эпик Геймс Стор"),
    ("Epic Games", "Эпик Геймс"),
    ("Epic", "Эпик"),
    ("Nintendo", "Нинтендо"),
    ("Square Enix", "Скуэр Эникс"),
    ("Bungie", "Банджи"),
    ("Respawn", "Респаун"),
    ("DICE", "Дайс"),
    ("Crytek", "Крайтек"),
    ("505 Games", "Файв О Файв Геймс"),
    ("2K Games", "Ту Кей Геймс"),
    ("THQ Nordic", "Ти Эйч Кью Нордик"),
    ("Sega", "Сега"),
    ("Atari", "Атари"),
    ("id Software", "Ай Ди Софтвэр"),
    ("Larian Studios", "Лариан Студиос"),
    ("Larian", "Лариан"),
    ("Paradox Interactive", "Парадокс Интерактив"),
    ("Paradox", "Парадокс"),
    ("Warhorse Studios", "Уорхорс Студиос"),
    ("Warhorse", "Уорхорс"),
    ("Nacon", "Након"),
    ("Focus Entertainment", "Фокус Энтертейнмент"),
    ("Piranha Bytes", "Пиранья Байтс"),
    ("Deep Silver", "Дип Силвер"),
    ("Annapurna Interactive", "Аннапурна Интерактив"),
    ("Annapurna", "Аннапурна"),
    ("Devolver Digital", "Девольвер Диджитал"),
    ("Devolver", "Девольвер"),
    ("Remedy", "Ремеди"),
    ("Arkane", "Аркейн"),
    ("MachineGames", "Машин Геймс"),
    ("Microsoft", "Майкрософт"),
    ("Sony", "Сони"),
    ("Sony Interactive Entertainment", "Сони Интерактив Энтертейнмент"),
    ("Electronic Arts", "Электроник Артс"),
    ("EA Sports", "И Эй Спортс"),
    # ── Platforms / stores ──────────────────────────────────────────────
    ("PlayStation 5", "ПлейСтейшн Пять"),
    ("PlayStation 4", "ПлейСтейшн Четыре"),
    ("PlayStation", "ПлейСтейшн"),
    ("PS5", "Пи Эс Пять"),
    ("PS4", "Пи Эс Четыре"),
    ("Xbox Series X", "Иксбокс Сириес Икс"),
    ("Xbox Series S", "Иксбокс Сириес Эс"),
    ("Xbox One", "Иксбокс Уан"),
    ("Xbox", "Иксбокс"),
    ("Steam Deck", "Стим Дек"),
    ("Steam", "Стим"),
    ("Nintendo Switch 2", "Нинтендо Свитч Два"),
    ("Nintendo Switch", "Нинтендо Свитч"),
    ("Switch", "Свитч"),
    ("Game Pass Ultimate", "Гейм Пасс Алтимит"),
    ("Game Pass", "Гейм Пасс"),
    ("PS Plus", "Пи Эс Плюс"),
    ("PlayStation Plus", "ПлейСтейшн Плюс"),
    ("GOG", "Гог"),
    # ── Common gaming terms / acronyms ──────────────────────────────────
    ("MMORPG", "ММО Эр Пи Джи"),
    ("RPG", "Эр Пи Джи"),
    ("FPS", "Эф Пи Эс"),
    ("MMO", "ММО"),
    ("DLC", "Ди Эл Си"),
    ("AAA", "Трипл Эй"),
    ("NPC", "Эн Пи Си"),
    ("PvP", "Пи Ви Пи"),
    ("PvE", "Пи Ви И"),
    ("Early Access", "Эрли Эксесс"),
    ("Open World", "Опен Ворлд"),
    ("Battle Royale", "Батл Рояль"),
    ("Soulslike", "Соулслайк"),
    ("Roguelike", "Роглайк"),
    ("Roguelite", "Роглайт"),
    ("Co-op", "Ко-оп"),
    ("Crossplay", "Кросс-плей"),
    ("Cross-play", "Кросс-плей"),
    ("The Game Awards", "Зе Гейм Авордс"),
    ("Game Awards", "Гейм Авордс"),
    ("State of Play", "Стейт оф Плей"),
    ("Xbox Showcase", "Иксбокс Шоукейс"),
    ("Nintendo Direct", "Нинтендо Директ"),
    ("Summer Game Fest", "Саммер Гейм Фест"),
    # ── Specific game titles ────────────────────────────────────────────
    ("Resident Evil", "Резидент Ивл"),
    ("Devil May Cry", "Девил Мэй Край"),
    ("Monster Hunter", "Монстер Хантер"),
    ("Street Fighter", "Стрит Файтер"),
    ("Tekken", "Теккен"),
    ("Dark Souls", "Дарк Соулс"),
    ("Demon's Souls", "Демонс Соулс"),
    ("Elden Ring", "Элден Ринг"),
    ("Bloodborne", "Бладборн"),
    ("Sekiro", "Секиро"),
    ("Hollow Knight", "Холлоу Найт"),
    ("Silksong", "Силксонг"),
    ("Ghost of Tsushima", "Гост оф Цусима"),
    ("Ghost of Yotei", "Гост оф Йотей"),
    ("Death Stranding", "Дэт Стрэндинг"),
    ("Red Dead Redemption", "Ред Дэд Редемпшн"),
    ("Red Dead", "Ред Дэд"),
    # NOTE: standalone "Red" → "Ред" removed intentionally.
    # Silero TTS озвучивает одиночное "Ред" как "редакция" (трактует
    # как типографское сокращение "ред."). Длинные фразы выше уже
    # покрывают "CD Projekt Red" и "Red Dead*".
    ("GTA", "Джи Ти Эй"),
    ("Grand Theft Auto", "Гранд Тефт Авто"),
    ("The Witcher", "Ведьмак"),
    ("Witcher", "Ведьмак"),
    ("Cyberpunk 2077", "Киберпанк две тысячи семьдесят семь"),
    ("Cyberpunk", "Киберпанк"),
    ("Baldur's Gate", "Балдурс Гейт"),
    ("Dragon Age", "Драгон Эйдж"),
    ("Mass Effect", "Масс Эффект"),
    ("Starfield", "Старфилд"),
    ("Fallout", "Фоллаут"),
    ("Skyrim", "Скайрим"),
    ("Oblivion", "Обливион"),
    ("Morrowind", "Морровинд"),
    ("S.T.A.L.K.E.R.", "Сталкер"),
    ("STALKER", "Сталкер"),
    ("Metro Exodus", "Метро Исход"),
    ("Metro", "Метро"),
    ("Diablo", "Диабло"),
    ("Path of Exile", "Пас оф Экзайл"),
    ("Overwatch", "Овервотч"),
    ("World of Warcraft", "Ворлд оф Варкрафт"),
    ("Warcraft", "Варкрафт"),
    ("StarCraft", "Старкрафт"),
    ("Hearthstone", "Хартстоун"),
    ("League of Legends", "Лига Легенд"),
    ("Valorant", "Валорант"),
    ("Counter-Strike 2", "Каунтер-Страйк Ту"),
    ("Counter-Strike", "Каунтер-Страйк"),
    ("CS2", "Си Эс Ту"),
    ("CS:GO", "Си Эс Гоу"),
    ("Dota 2", "Дота Два"),
    ("Dota", "Дота"),
    ("Minecraft", "Майнкрафт"),
    ("Fortnite", "Фортнайт"),
    ("Apex Legends", "Эйпекс Лэджендс"),
    ("Call of Duty", "Кол оф Дьюти"),
    ("Modern Warfare", "Модерн Уорфэр"),
    ("Black Ops", "Блэк Опс"),
    ("Warzone", "Уорзон"),
    ("Battlefield", "Баттлфилд"),
    ("Assassin's Creed Shadows", "Ассасинс Крид Шэдоус"),
    ("Assassin's Creed", "Ассасинс Крид"),
    ("Far Cry", "Фар Край"),
    ("Watch Dogs", "Вотч Догс"),
    ("Rainbow Six Siege", "Рэйнбоу Сикс Сидж"),
    ("Rainbow Six", "Рэйнбоу Сикс"),
    ("God of War Ragnarok", "Год оф Вор Рагнарёк"),
    ("God of War", "Год оф Вор"),
    ("Spider-Man 2", "Спайдер-Мэн Два"),
    ("Spider-Man", "Спайдер-Мэн"),
    ("Horizon Forbidden West", "Хорайзон Форбидден Уэст"),
    ("Horizon Zero Dawn", "Хорайзон Зеро Дон"),
    ("Horizon", "Хорайзон"),
    ("The Last of Us Part II", "Зе Ласт оф Ас Парт Ту"),
    ("The Last of Us", "Зе Ласт оф Ас"),
    ("Uncharted", "Анчартед"),
    ("Gran Turismo", "Гран Туризмо"),
    ("Forza Horizon", "Форца Хорайзон"),
    ("Forza Motorsport", "Форца Моторспорт"),
    ("Forza", "Форца"),
    ("Halo Infinite", "Хейло Инфинит"),
    ("Halo", "Хейло"),
    ("Gears of War", "Гирс оф Вор"),
    ("Final Fantasy", "Файнал Фэнтези"),
    ("Persona", "Персона"),
    ("Yakuza", "Якудза"),
    ("Like a Dragon", "Лайк э Драгон"),
    ("Silent Hill", "Сайлент Хилл"),
    ("Alan Wake", "Алан Уэйк"),
    ("Control", "Контрол"),
    ("Hitman", "Хитмэн"),
    ("Doom Eternal", "Дум Этернал"),
    ("Doom", "Дум"),
    ("Half-Life: Alyx", "Халф-Лайф Аликс"),
    ("Half-Life 2", "Халф-Лайф Два"),
    ("Half-Life", "Халф-Лайф"),
    ("Quake", "Квейк"),
    ("Wolfenstein", "Вольфенштайн"),
    ("Dying Light", "Даинг Лайт"),
    ("Dead Space", "Дэд Спейс"),
    ("Borderlands", "Бордерлендс"),
    ("BioShock", "Биошок"),
    ("Mortal Kombat", "Мортал Комбат"),
    ("Stellar Blade", "Стеллар Блейд"),
    ("Black Myth: Wukong", "Блэк Миф Вуконг"),
    ("Black Myth Wukong", "Блэк Миф Вуконг"),
    ("Wukong", "Вуконг"),
    ("Helldivers", "Хеллдайверс"),
    ("Palworld", "Палворлд"),
    ("Hogwarts Legacy", "Хогвартс Легаси"),
    ("Marvel Rivals", "Марвел Райвалс"),
    ("Marvel", "Марвел"),
    ("Kingdom Come: Deliverance", "Кингдом Кам Деливеренс"),
    ("Kingdom Come Deliverance", "Кингдом Кам Деливеренс"),
    ("Perceptum", "Перцептум"),
    ("Meccha Chameleon", "Мекча Хамелеон"),
    ("Genshin Impact", "Геншин Импакт"),
]

# Sort keys by length DESC so the regex tries the longest phrases first.
# (Python's `re` alternation is left-to-right, not longest-match.)
_RU_BRAND_KEYS_SORTED = sorted(
    {k for k, _ in _RU_BRAND_MAP}, key=lambda s: (-len(s), s)
)

# Build compiled pattern once at module load.
# Use lookarounds instead of \b — \b doesn't fire correctly next to punctuation
# inside keys like "S.T.A.L.K.E.R." or "Counter-Strike".
# Граница включает и кириллицу: иначе короткий латинский ключ (напр. "Red")
# с IGNORECASE может зацепить кириллический фрагмент рядом и наоборот.
_RU_BRAND_RE = re.compile(
    r"(?<![A-Za-z0-9\u0400-\u04FF])("
    + "|".join(re.escape(k) for k in _RU_BRAND_KEYS_SORTED)
    + r")(?![A-Za-z0-9\u0400-\u04FF])",
    re.IGNORECASE,
)
_RU_BRAND_LOOKUP = {k.lower(): v for k, v in _RU_BRAND_MAP}


def _transliterate_for_ru_tts(text: str) -> str:
    """Replace English gaming brands/terms with Russian phonetic spelling."""
    def _replace(m: re.Match) -> str:
        return _RU_BRAND_LOOKUP.get(m.group(0).lower(), m.group(0))
    return _RU_BRAND_RE.sub(_replace, text)


async def _synthesize_voice_gemini(
    text: str,
    workdir: str,
    voice: str | None = None,
    max_retries: int = 3,
) -> tuple[str, list[tuple[float, float, str]]]:
    """
    Synthesize narration via Gemini TTS and recover word-level timing via the
    existing Whisper pipeline (`_run_whisper_sync`).

    The Gemini API returns inline audio (WAV L16 24 kHz mono) but no
    word-level timestamps, so we feed the WAV through faster-whisper — already
    a dependency — the same way the edge-tts path does. Output files:
      voice.wav  — PCM audio, 24 kHz mono, 16-bit
      subs.vtt   — single cue [0..audio_dur] used as a Whisper anchor

    Returns (audio_path, [(start_sec, end_sec, word), ...]).
    Raises RuntimeError on API key missing, sdk missing, or all retries failing.
    """
    if not _GENAI_OK:
        raise RuntimeError(
            "google-genai SDK not installed. Run: pip install google-genai"
        )

    # Determine language from the voice name passed by the caller (edge-tts
    # uses locale-prefixed names like "ru-RU-DmitryNeural"). For Gemini TTS
    # callers can pass either a plain voice name ("Kore") or a locale-prefixed
    # hint ("ru-RU") — we extract the language either way.
    whisper_lang = "en"
    if voice:
        v_lower = voice.lower()
        if v_lower.startswith("ru") or "-ru-" in v_lower or v_lower == "ru-ru":
            whisper_lang = "ru"

    chosen_voice = voice or (
        GEMINI_TTS_VOICE_RU if whisper_lang == "ru" else GEMINI_TTS_VOICE_EN
    )
    # Voice name from edge-tts style is unusable for Gemini (no locale strip);
    # fall back to the configured RU/EN pick if a non-Gemini voice leaks in.
    if "-" in chosen_voice and not _is_gemini_voice_name(chosen_voice):
        chosen_voice = (
            GEMINI_TTS_VOICE_RU if whisper_lang == "ru" else GEMINI_TTS_VOICE_EN
        )

    director = _GEMINI_DIR_RU if whisper_lang == "ru" else _GEMINI_DIR_EN
    prompt = f"{director}{text.strip()}\n"

    response = await _gemini_tts_generate(
        prompt, chosen_voice,
        "ru-RU" if whisper_lang == "ru" else "en-US",
        max_retries,
    )

    # ---- Decode inline audio (WAV L16 24 kHz mono per Google docs) ----
    try:
        part = response.candidates[0].content.parts[0]
        pcm_bytes = part.inline_data.data
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(f"Gemini TTS returned no audio: {exc}") from exc

    audio_path = os.path.join(workdir, "voice.wav")
    # Gemini returns raw PCM (L16, 24 kHz, mono, little-endian). Wrap in a
    # real WAV container so downstream ffmpeg and Whisper read it cleanly.
    import base64 as _b64
    import wave as _wave
    try:
        pcm = _b64.b64decode(pcm_bytes) if isinstance(pcm_bytes, str) else pcm_bytes
    except Exception as exc:
        raise RuntimeError(f"Gemini TTS audio decode failed: {exc}") from exc
    with _wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)         # 16-bit
        wf.setframerate(24_000)
        wf.writeframes(pcm)

    # ---- Log token/audio usage from response.usage_metadata ----
    # Tracks daily Gemini TTS quota in the bot's own logs alongside the
    # dashboard at https://aistudio.google.com/usage.
    try:
        um = getattr(response, "usage_metadata", None)
        if um:
            prompt_tok     = int(getattr(um, "prompt_token_count",     0) or 0)
            audio_tok      = int(getattr(um, "candidates_token_count", 0) or 0)  # audio
            thoughts_tok   = int(getattr(um, "thoughts_token_count",   0) or 0)
            audio_seconds  = audio_tok / 25.0        # 25 tokens / sec per docs
            logger.info(
                "Gemini TTS usage: model=%s prompt=%d tok, audio=%d tok (%.1f s), "
                "thoughts=%d tok, voice=%s",
                GEMINI_TTS_MODEL, prompt_tok, audio_tok, audio_seconds,
                thoughts_tok, chosen_voice,
            )
    except Exception as exc:                                # noqa: BLE001
        logger.debug("Gemini usage_metadata read failed: %s", exc)

    # ---- Recover word timing via existing Whisper pipeline ----
    audio_dur = _get_audio_duration(audio_path)

    # Synthesise a single-cue VTT covering [0, audio_dur] — same trick the
    # docs recommend so _run_whisper_sync has a sentence anchor to bind to.
    vtt_path = os.path.join(workdir, "subs.vtt")
    h, rem = divmod(audio_dur, 3600)
    m, s   = divmod(rem, 60)
    end_ts = f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    with open(vtt_path, "w", encoding="utf-8") as fh:
        fh.write(f"WEBVTT\n\n00:00:00.000 --> {end_ts}\n{text.strip()}\n")

    sentence_cues = _parse_vtt_cues(vtt_path)
    word_cues = await asyncio.to_thread(
        _run_whisper_sync, audio_path, sentence_cues, whisper_lang,
    )
    if not word_cues:                                     # Whisper missing/failed
        word_cues = await asyncio.to_thread(
            _detect_word_boundaries_from_audio, audio_path, sentence_cues,
        )
        logger.info(
            "TTS (gemini): %d word cues (silencedetect fallback), audio %.1f s",
            len(word_cues), audio_dur,
        )
    else:
        logger.info(
            "TTS (gemini): %d word cues (whisper), audio %.1f s, voice=%s",
            len(word_cues), audio_dur, chosen_voice,
        )
    return audio_path, word_cues


_GEMINI_VOICE_NAMES = {
    # 30 voices from the Gemini TTS docs (June 2026).
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
}


def _is_gemini_voice_name(name: str) -> bool:
    """True if `name` looks like a Gemini prebuilt voice (e.g. 'Kore'),
    False if it's an edge-tts style locale+name (e.g. 'ru-RU-DmitryNeural').
    """
    return name in _GEMINI_VOICE_NAMES


GEMINI_VOICE_NAMES = sorted(_GEMINI_VOICE_NAMES)


def _gemini_speed_prompt(speed: float, lang: str) -> str:
    """
    Build a pace instruction for the Gemini TTS prompt.

    Gemini TTS has NO numeric speaking-rate parameter in the API (verified —
    `SpeechConfig`/`VoiceConfig`/`PrebuiltVoiceConfig` only expose voice name
    + language code). Speed is steered purely through natural-language
    instructions in the prompt text, so `speed` (0.5–2.0, 1.0 = natural) is
    translated into a target words-per-minute figure the model reads back.
    """
    speed = max(0.5, min(2.0, speed or 1.0))
    base_wpm = 130 if lang == "ru" else 150   # natural conversational pace
    wpm = round(base_wpm * speed)
    if lang == "ru":
        return (
            f"Читай текст в естественном темпе, примерно {wpm} слов в минуту. "
            "Произнеси ТОЛЬКО текст ниже:\n\n"
        )
    return (
        f"Read the text at a natural pace, approximately {wpm} words per minute. "
        "Speak ONLY the transcript below:\n\n"
    )


async def synthesize_gemini_tts_standalone(
    text: str,
    voice: str | None = None,
    lang: str = "en",
    speed: float = 1.0,
    max_retries: int = 3,
) -> str:
    """
    Ad-hoc Gemini TTS synthesis for the webapp's "paste text → voice" tool.

    Unlike `_synthesize_voice_gemini` (used by the video pipeline), this skips
    the Whisper word-timing recovery step — callers just want a playable /
    downloadable audio file, not subtitle cues. `speed` (0.5–2.0, default 1.0)
    controls pace via a words-per-minute instruction — see `_gemini_speed_prompt`.

    Returns the absolute path to the generated WAV file (24 kHz mono PCM).
    Raises RuntimeError on missing SDK/API key, empty text, or API failure.
    """
    if not _GENAI_OK:
        raise RuntimeError(
            "google-genai SDK not installed. Run: pip install google-genai"
        )

    text = (text or "").strip()
    if not text:
        raise RuntimeError("Text is empty — nothing to synthesize")

    lang = (lang or "en").strip().lower()
    if lang not in ("en", "ru"):
        lang = "en"

    chosen_voice = voice or (GEMINI_TTS_VOICE_RU if lang == "ru" else GEMINI_TTS_VOICE_EN)
    if not _is_gemini_voice_name(chosen_voice):
        chosen_voice = GEMINI_TTS_VOICE_RU if lang == "ru" else GEMINI_TTS_VOICE_EN

    director = _gemini_speed_prompt(speed, lang)
    prompt = f"{director}{text}\n"

    response = await _gemini_tts_generate(
        prompt, chosen_voice, "ru-RU" if lang == "ru" else "en-US", max_retries,
    )

    try:
        part = response.candidates[0].content.parts[0]
        pcm_bytes = part.inline_data.data
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(f"Gemini TTS returned no audio: {exc}") from exc

    import base64 as _b64
    import wave as _wave
    try:
        pcm = _b64.b64decode(pcm_bytes) if isinstance(pcm_bytes, str) else pcm_bytes
    except Exception as exc:
        raise RuntimeError(f"Gemini TTS audio decode failed: {exc}") from exc

    out_dir = os.path.join(VIDEOS_DIR, "tts_manual")
    os.makedirs(out_dir, exist_ok=True)
    audio_path = os.path.join(out_dir, f"tts_{uuid.uuid4().hex[:10]}.wav")
    with _wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)         # 16-bit
        wf.setframerate(24_000)
        wf.writeframes(pcm)

    try:
        um = getattr(response, "usage_metadata", None)
        if um:
            audio_tok = int(getattr(um, "candidates_token_count", 0) or 0)
            logger.info(
                "Gemini TTS (standalone) usage: model=%s voice=%s audio=%d tok (%.1f s)",
                GEMINI_TTS_MODEL, chosen_voice, audio_tok, audio_tok / 25.0,
            )
    except Exception as exc:                                # noqa: BLE001
        logger.debug("Gemini usage_metadata read failed: %s", exc)

    return audio_path


async def _synthesize_voice(
    text: str, workdir: str, voice: str | None = None
) -> tuple[str, list[tuple[float, float, str]]]:
    """
    Generate audio via edge-tts (EN) or Silero TTS (Russian), then use Whisper
    to get accurate word-level timestamps from the actual audio signal.
    Falls back to ffmpeg silencedetect if Whisper is unavailable.
    Returns (audio_path, [(start_sec, end_sec, word), ...]).
    """
    # Backend selector. "edge" = original edge-tts; "gemini" = route through
    # Gemini TTS (Flash TTS Preview models). Switch is .env-controlled.
    if TTS_BACKEND == "gemini":
        return await _synthesize_voice_gemini(text, workdir, voice=voice)

    chosen_voice = voice or TTS_VOICE

    audio_path = os.path.join(workdir, "voice.mp3")
    vtt_path   = os.path.join(workdir, "subs.vtt")

    # Derive Whisper language from voice locale prefix (e.g. "ru-RU-..." → "ru")
    whisper_lang = chosen_voice.split("-")[0].lower() if chosen_voice else "en"
    # Apply Russian brand transliteration for Russian-locale voices
    tts_text = _transliterate_for_ru_tts(text) if whisper_lang == "ru" else text

    if not tts_text or not tts_text.strip():
        raise RuntimeError("TTS text is empty — cannot synthesize voice")

    # Russian voice (Dmitry) is naturally slower & has longer inter-sentence
    # pauses than the EN voice — use a higher rate to keep audio length comparable.
    tts_rate = TTS_RATE_RU if whisper_lang == "ru" else TTS_RATE

    async def _run_edge_tts(tts_input: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "edge-tts",
            "--voice", chosen_voice,
            f"--rate={tts_rate}",
            f"--pitch={TTS_PITCH}",
            "--text",  tts_input,
            "--write-media",     audio_path,
            "--write-subtitles", vtt_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"edge-tts error: {stderr.decode()[:400]}")

    try:
        await _run_edge_tts(tts_text)
    except FileNotFoundError:
        raise RuntimeError("edge-tts not found. Install with: pip install edge-tts")

    # Validate generated audio — retry with original text if transliteration caused issues
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
        logger.warning("edge-tts produced invalid audio (size=%d), retrying with original text",
                       os.path.getsize(audio_path) if os.path.exists(audio_path) else 0)
        if tts_text != text:
            await _run_edge_tts(text)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            raise RuntimeError("edge-tts failed to produce valid audio")

    if not os.path.exists(vtt_path):
        open(vtt_path, "w").close()

    sentence_cues = _parse_vtt_cues(vtt_path)

    if whisper_lang == "ru":
        # Use Whisper 'small' model for Russian — better Cyrillic accuracy.
        # Mapping is done per-sentence using VTT boundaries as anchors,
        # so drift is bounded to one sentence even if word counts differ.
        word_cues = await asyncio.to_thread(
            _run_whisper_sync, audio_path, sentence_cues, whisper_lang
        )
        if word_cues:
            logger.info("TTS: %d word cues (whisper/small RU), audio %.1f s",
                        len(word_cues), _get_audio_duration(audio_path))
        else:
            word_cues = await asyncio.to_thread(
                _detect_word_boundaries_from_audio, audio_path, sentence_cues
            )
            logger.info("TTS: %d word cues (silencedetect RU fallback), audio %.1f s",
                        len(word_cues), _get_audio_duration(audio_path))
    else:
        # Try Whisper first (most accurate for EN — extracts timing from audio).
        word_cues = await asyncio.to_thread(
            _run_whisper_sync, audio_path, sentence_cues, whisper_lang
        )
        if word_cues:
            logger.info("TTS: %d word cues (whisper), audio %.1f s",
                        len(word_cues), _get_audio_duration(audio_path))
        else:
            # Fallback: ffmpeg silencedetect (no ML model required)
            word_cues = await asyncio.to_thread(
                _detect_word_boundaries_from_audio, audio_path, sentence_cues
            )
            logger.info("TTS: %d word cues (silencedetect fallback), audio %.1f s",
                        len(word_cues), _get_audio_duration(audio_path))

    return audio_path, word_cues


# ---------------------------------------------------------------------------
# YouTube gameplay footage via yt-dlp
# ---------------------------------------------------------------------------

async def _download_multiple_yt_videos(
    search_query: str,
    count: int,
    workdir: str,
    skip: int = 0,
) -> list[str]:
    """
    Download *count* full YouTube videos (no time-slicing) for *search_query*.
    Uses a single yt-dlp call for efficiency.
    Returns a list of local .mp4/.webm/.mkv paths (may be fewer than *count*
    if some downloads fail or exceed the file-size limit).
    """
    clips_dir = os.path.join(workdir, "yt_full")
    os.makedirs(clips_dir, exist_ok=True)

    _FMT = (
        "bestvideo[height=720][ext=mp4]/bestvideo[height<=720][ext=mp4]"
        "/bestvideo[height<=720]/bestvideo[ext=mp4]/bestvideo/best"
    )

    fetch_count = count + skip + 2
    ydl_args = [
        "yt-dlp",
        "--no-playlist", "--no-warnings",
        "--ignore-errors",
        "--match-filters", "live_status=not_live",  # skip premieres (is_upcoming) and live streams
        *_YT_COOKIE_ARGS,
        *_YT_EXTRACTOR_ARGS,
        *_YT_SLEEP_ARGS,
        "--format", _FMT,
        "--max-filesize", f"{YT_MAX_FILESIZE}M",
        "--max-downloads", str(count + skip),
        "--output", os.path.join(clips_dir, "%(autonumber)s.%(ext)s"),
        f"ytsearch{fetch_count}:{search_query}",
    ]

    logger.info(
        "Downloading %d full YT videos for '%s' (skip=%d)", count, search_query, skip
    )
    ok = await _run_async(ydl_args, timeout=360 * (count + 1))
    if not ok:
        logger.warning("yt-dlp finished with error — checking partial downloads")

    all_paths = sorted(
        os.path.join(clips_dir, f)
        for f in os.listdir(clips_dir)
        if f.endswith((".mp4", ".webm", ".mkv")) and not f.endswith(".part")
    )
    result = [p for p in all_paths[skip:][:count] if os.path.getsize(p) > 10 * 1024]
    logger.info("Downloaded %d full YT videos for '%s'", len(result), search_query)
    return result


async def _download_full_yt_video(
    url_or_search: str,
    workdir: str,
    skip: int = 0,
    is_url: bool = False,
) -> str | None:
    """
    Download a single full YouTube video (no time slicing).
    For search queries, picks the (skip+1)-th result for regeneration diversity.
    Returns local .mp4/.mkv path or None on failure.

    Uses a FRESH subdirectory per call (never a shared one) — this function
    may now be called several times against the same ``workdir`` (once per
    Shorts candidate, once per fallback attempt, etc.), and a shared output
    folder previously let a later call silently pick up a stale/unrelated
    file left behind by an earlier call (or by `_download_yt_segment`).
    """
    clips_dir = tempfile.mkdtemp(dir=workdir, prefix="src_")

    # Prefer 720p mp4 to avoid YouTube n-challenge throttling on higher formats;
    # fall back to merged 'best' (always available without n-challenge). Used
    # for bulk ytsearch downloads, where 720p is plenty for cut-up gameplay clips.
    _FMT = ("bestvideo[height=720][ext=mp4]/bestvideo[height<=720][ext=mp4]"
            "/bestvideo[height<=720]/bestvideo[ext=mp4]/bestvideo/best")

    # A direct link (is_url=True) is a single, deliberately-chosen video —
    # e.g. a pasted YouTube Short/Reel/TikTok URL — so fetch it at the highest
    # resolution available instead of the 720p cap used for bulk search results.
    _FMT_URL = "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best"

    if is_url:
        ydl_args = [
            "yt-dlp",
            "--no-playlist", "--no-warnings", "--force-overwrites",
            *_YT_COOKIE_ARGS,
            *_YT_EXTRACTOR_ARGS,
            "--format", _FMT_URL,
            "--max-filesize", f"{YT_MAX_FILESIZE}M",
            "--output", os.path.join(clips_dir, "source.%(ext)s"),
            url_or_search,
        ]
    else:
        fetch_count = skip + 3
        ydl_args = [
            "yt-dlp",
            "--no-playlist", "--no-warnings", "--force-overwrites",
            "--ignore-errors",
            "--match-filters", "live_status=not_live",  # skip premieres (is_upcoming) and live streams
            *_YT_COOKIE_ARGS,
            *_YT_EXTRACTOR_ARGS,
            *_YT_SLEEP_ARGS,
            "--format", _FMT,
            "--max-filesize", f"{YT_MAX_FILESIZE}M",
            "--max-downloads", str(skip + 1),
            "--output", os.path.join(clips_dir, "%(autonumber)s.%(ext)s"),
            f"ytsearch{fetch_count}:{url_or_search}",
        ]

    logger.info(
        "Downloading full YT video: '%s' (is_url=%s, skip=%d)",
        url_or_search[:80], is_url, skip,
    )
    ok = await _run_async(ydl_args, timeout=360)
    if not ok:
        logger.warning("yt-dlp finished with error — checking partial downloads")

    dir_contents = os.listdir(clips_dir)
    logger.info("clips_dir contents after yt-dlp: %s", dir_contents)
    all_paths = sorted(
        os.path.join(clips_dir, f)
        for f in dir_contents
        if f.endswith((".mp4", ".webm", ".mkv")) and not f.endswith(".part")
    )
    if not all_paths:
        logger.warning("No video file found after yt-dlp for '%s'", url_or_search[:60])
        return None

    # For search with skip, the last downloaded file = the skip-th search result
    chosen = all_paths[-1]
    if os.path.getsize(chosen) < 10 * 1024:  # < 10 KB → probably broken
        logger.warning("Downloaded video too small: %s", chosen)
        return None

    logger.info(
        "Full YT video ready: %s (%.1f MB)",
        os.path.basename(chosen), os.path.getsize(chosen) / 1024 / 1024,
    )
    return chosen


# ---------------------------------------------------------------------------
# Metadata-only YouTube search + AI-picked segment download
#
# Instead of always downloading a whole matched video and randomly cutting a
# few seconds out of it, we can (a) search YouTube Shorts too — a short is
# already a complete, relevant moment, no cutting needed — and (b) for
# longer videos, fetch just the auto-generated transcript first (tiny,
# skip-download), ask the LLM which timestamp range best matches what we're
# looking for, and download ONLY that section via yt-dlp --download-sections.
# Falls back to the old "download the whole video" behaviour whenever a step
# fails (no transcript available, AI pick fails, etc.).
# ---------------------------------------------------------------------------

# A duration at/under this is treated like a YouTube Short/Reel/TikTok —
# used whole, no segment-picking needed (mirrors _SHORT_VIDEO_THRESHOLD below).
_SHORTS_MAX_DURATION = 90.0
# Padding (s) added around an AI-picked transcript segment before download,
# so the cut doesn't start/end mid-sentence.
_SEGMENT_PAD = 1.5


async def _search_yt_candidates(query: str, count: int = 6) -> list[dict]:
    """
    Search YouTube for *query* and return lightweight metadata WITHOUT
    downloading anything: [{"id","url","title","duration"}, ...] in search-rank
    order. ``duration`` may be None if yt-dlp couldn't determine it from the
    flat search listing (callers should treat that as "long-form, unknown").
    """
    args = [
        "yt-dlp",
        "--flat-playlist", "--dump-json", "--no-warnings",
        "--ignore-errors",
        "--match-filters", "live_status=not_live",
        *_YT_COOKIE_ARGS, *_YT_EXTRACTOR_ARGS, *_YT_SLEEP_ARGS,
        f"ytsearch{count}:{query}",
    ]
    out = await _run_capture_async(args, timeout=90)
    candidates: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        vid = e.get("id")
        if not vid:
            continue
        url = e.get("url") or f"https://www.youtube.com/watch?v={vid}"
        duration = e.get("duration")
        try:
            duration = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        candidates.append({
            "id": vid, "url": url,
            "title": e.get("title") or "",
            "duration": duration,
        })
    return candidates


async def _fetch_yt_transcript_cues(
    video_url: str, workdir: str, langs: tuple[str, ...] = ("ru", "en"),
) -> list[tuple[float, float, str]] | None:
    """
    Download only the auto-generated captions (no video) for *video_url* and
    parse them into [(start_sec, end_sec, text), ...]. Tries each language in
    *langs* until one yields captions. Returns None if no captions exist.
    """
    sub_dir = tempfile.mkdtemp(dir=workdir, prefix="subs_")
    for lang in langs:
        out_tmpl = os.path.join(sub_dir, f"sub.%(ext)s")
        args = [
            "yt-dlp", "--no-playlist", "--no-warnings", "--skip-download",
            "--write-auto-sub", "--sub-lang", lang, "--sub-format", "vtt",
            *_YT_COOKIE_ARGS, *_YT_EXTRACTOR_ARGS,
            "-o", out_tmpl,
            video_url,
        ]
        ok = await _run_async(args, timeout=90)
        vtt_files = [f for f in os.listdir(sub_dir) if f.endswith(".vtt")]
        if vtt_files:
            cues = _parse_vtt_cues(os.path.join(sub_dir, vtt_files[0]))
            if cues:
                logger.info("Fetched %d transcript cues (%s) for %s", len(cues), lang, video_url)
                return cues
        # Clean up before trying the next language.
        for f in vtt_files:
            try:
                os.remove(os.path.join(sub_dir, f))
            except OSError:
                pass
    logger.info("No usable auto-captions found for %s", video_url)
    return None


async def _download_yt_segment(
    video_url: str, start: float, end: float, workdir: str, out_name: str = "segment",
) -> str | None:
    """
    Download ONLY the [start, end] section (seconds) of *video_url* via
    yt-dlp --download-sections, instead of the whole video. Returns the local
    file path, or None on failure.

    Uses a fresh subdirectory per call — this can be invoked repeatedly with
    the same ``out_name`` (one per entity query, tried against several
    candidate videos), and yt-dlp skips re-downloading when the destination
    file already exists, which previously caused every retry to silently
    reuse the FIRST candidate's (stale) file instead of the new one.
    """
    clips_dir = tempfile.mkdtemp(dir=workdir, prefix="seg_")
    _FMT = ("bestvideo[height=720][ext=mp4]/bestvideo[height<=720][ext=mp4]"
            "/bestvideo[height<=720]/bestvideo[ext=mp4]/bestvideo/best")
    section = f"*{max(0.0, start):.2f}-{end:.2f}"
    out_tmpl = os.path.join(clips_dir, f"{out_name}.%(ext)s")
    args = [
        "yt-dlp", "--no-playlist", "--no-warnings", "--force-overwrites",
        *_YT_COOKIE_ARGS, *_YT_EXTRACTOR_ARGS,
        "--format", _FMT,
        "--max-filesize", f"{YT_MAX_FILESIZE}M",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "-o", out_tmpl,
        video_url,
    ]
    logger.info("Downloading YT segment %s from %s", section, video_url)
    ok = await _run_async(args, timeout=150)
    if not ok:
        logger.warning("yt-dlp segment download reported an error for %s", video_url)
    matches = [
        os.path.join(clips_dir, f) for f in os.listdir(clips_dir)
        if f.startswith(out_name) and f.endswith((".mp4", ".webm", ".mkv")) and not f.endswith(".part")
    ]
    if not matches:
        logger.warning("No segment file produced for %s (%s)", video_url, section)
        return None
    path = matches[0]
    if os.path.getsize(path) < 10 * 1024:
        logger.warning("Downloaded segment too small: %s", path)
        return None
    logger.info("YT segment ready: %s (%.1f MB)", os.path.basename(path), os.path.getsize(path) / 1024 / 1024)
    return path


async def _fetch_clips_for_query(
    query: str,
    n_clips: int,
    workdir: str,
    yt_skip: int = 0,
    clip_prefix: str = "yt_clip",
    used_ids: set[str] | None = None,
) -> list[str]:
    """
    Find footage matching *query* and return cut clips, preferring (in order):
      1. A YouTube Short/Reel-length result (already a complete relevant
         moment) — used whole, no download-then-cut needed.
      2. A long-form video: fetch its transcript only, ask the LLM for the
         best-matching timestamp range, and download ONLY that section.
      3. Fallback: download the whole top search result and randomly cut
         *n_clips* segments from it (the pre-existing behaviour).
    ``used_ids`` (optional, shared across calls) avoids re-picking the same
    source video for multiple different queries in one render.
    """
    if used_ids is None:
        used_ids = set()

    candidates = await _search_yt_candidates(query, count=5)
    candidates += await _search_yt_candidates(f"{query} shorts", count=4)
    seen: set[str] = set()
    uniq: list[dict] = []
    for c in candidates:
        if c["id"] in seen or c["id"] in used_ids:
            continue
        seen.add(c["id"])
        uniq.append(c)
    if not uniq:
        logger.warning("No YT search results for '%s'", query)
        return []

    # Rotate the candidate order by yt_skip so clicking "regenerate" (which
    # bumps yt_skip) picks different source videos instead of the same ones
    # every time — mirrors the diversity the old skip-based search gave.
    if yt_skip:
        offset = yt_skip % len(uniq)
        uniq = uniq[offset:] + uniq[:offset]

    shorts = [c for c in uniq if c["duration"] and c["duration"] <= _SHORTS_MAX_DURATION]
    long_form = [c for c in uniq if not c["duration"] or c["duration"] > _SHORTS_MAX_DURATION]

    clips: list[str] = []

    # 1. Shorts first — each is a whole, ready-to-use clip.
    for si, c in enumerate(shorts):
        if len(clips) >= n_clips:
            break
        video = await _download_full_yt_video(c["url"], workdir, is_url=True)
        if not video:
            continue
        used_ids.add(c["id"])
        clips.extend(await _clips_from_source(video, n_clips - len(clips), workdir, 0.0, f"{clip_prefix}_short{si}"))

    # 2. AI-picked segment from a long-form video. Bounded to a few attempts —
    #    each one costs a transcript fetch + LLM call + a slow yt-dlp download,
    #    so we don't keep trying candidates forever if segments aren't panning out.
    _MAX_LONGFORM_ATTEMPTS = 3
    for li, c in enumerate(long_form[:_MAX_LONGFORM_ATTEMPTS]):
        if len(clips) >= n_clips:
            break
        try:
            cues = await _fetch_yt_transcript_cues(c["url"], workdir)
            segment = None
            if cues:
                transcript_lines = [
                    f"{int(s // 60):02d}:{int(s % 60):02d} {t}" for s, _e, t in cues
                ][:400]
                segment = await ai_adapter.pick_video_segment(query, transcript_lines, c["duration"] or 0.0)
        except Exception as exc:
            logger.warning("Segment-pick failed for '%s' (%s): %s", query, c["url"], exc)
            segment = None

        if segment:
            start, end = segment
            seg_path = await _download_yt_segment(
                c["url"], start - _SEGMENT_PAD, end + _SEGMENT_PAD, workdir, f"{clip_prefix}_seg{li}",
            )
            if seg_path:
                used_ids.add(c["id"])
                clips.extend(await _clips_from_source(
                    seg_path, n_clips - len(clips), workdir, 0.0, f"{clip_prefix}_seg{li}",
                ))
                continue

        # 3. Fallback: no transcript / no AI match / segment download failed
        #    → download the whole video and cut randomly (old behaviour).
        video = await _download_full_yt_video(c["url"], workdir, is_url=True)
        if video:
            used_ids.add(c["id"])
            clips.extend(await _clips_from_source(
                video, n_clips - len(clips), workdir, float(YT_CLIP_SKIP), f"{clip_prefix}_fb{li}",
            ))
        break  # only fall back on one candidate, not all of them

    return clips


# Detects a direct short-form video link (YouTube incl. Shorts, Instagram Reel,
# TikTok, VK clip) pasted into the per-post "YouTube search query" field — as
# opposed to a plain-text search query, which is the field's normal use.
_DIRECT_VIDEO_URL_RE = re.compile(
    r"^https?://([\w-]+\.)?(youtube\.com|youtu\.be|instagram\.com|tiktok\.com|vk\.com|vkvideo\.ru)/",
    re.I,
)


_OUTRO_SKIP = 5.0  # don't take material from the last 5 s of a source video

# Below this duration, a source video is used whole (no cutting into multiple
# clips, no intro skip) — typical for YouTube Shorts / Reels / TikTok clips,
# which are already a single short, complete moment.
_SHORT_VIDEO_THRESHOLD = 60.0


async def _clips_from_source(
    video_path: str,
    n_clips: int,
    workdir: str,
    intro_skip: float,
    clip_name_prefix: str = "clip",
) -> list[str]:
    """Cut n_clips segments from video_path, unless it's shorter than
    _SHORT_VIDEO_THRESHOLD — then use it whole, un-cut and without the
    intro skip (nothing to trim on an already-short clip)."""
    duration = _get_audio_duration(video_path)
    if duration < _SHORT_VIDEO_THRESHOLD:
        logger.info(
            "Source video is short (%.1fs < %.0fs) — using it whole, no cutting/intro-skip: %s",
            duration, _SHORT_VIDEO_THRESHOLD, os.path.basename(video_path),
        )
        return [video_path]
    return await asyncio.to_thread(
        _cut_clips_from_video, video_path, n_clips, workdir, intro_skip, clip_name_prefix,
    )



# Smart clip selection: pick the most visually active windows (scene cuts /
# fast motion / action) instead of random positions. Disable with
# SMART_CLIP_SELECTION=0 to fall back to the old random-bucket behaviour.
_SMART_CLIPS = (os.getenv("SMART_CLIP_SELECTION", "1").strip().lower()
                not in ("0", "false", "no", "off", ""))

# Tuning for smart clip scoring. A static text/title card produces near-zero
# scene_scores; an editing cut produces a single big spike. To avoid picking
# windows that merely *start* on a cut into a static screen we:
#   * cap each sample so one cut spike can't "carry" an otherwise static window
#   * penalise windows that are mostly static (many near-zero samples)
_STATIC_SCORE   = float(os.getenv("SMART_STATIC_SCORE", "0.012"))   # below this = "static" frame
_SPIKE_CAP      = float(os.getenv("SMART_SPIKE_CAP", "0.35"))       # clamp per-sample motion
_STATIC_PENALTY = float(os.getenv("SMART_STATIC_PENALTY", "0.6"))   # weight of static-ratio penalty

# Text-density penalty. Even action-packed windows sometimes contain big
# title/lower-third/text-card frames. We densely sample frames across each
# candidate window and run the EAST deep-learning text detector on them; the
# fraction of frame area covered by detected text boxes is the "text ratio".
# Windows that show text at ANY sampled frame get their motion score scaled
# down (trailer text fades in/out, so sparse sampling misses it — we sample at
# _TEXT_PROBE_FPS and take the worst frame). Disable with SMART_TEXT_PENALTY=0.
# Needs opencv (cv2.dnn) + the EAST model file; if either is missing or a frame
# can't be analysed it's treated as text-free (no penalty), so the pipeline
# degrades gracefully.
_TEXT_PENALTY   = float(os.getenv("SMART_TEXT_PENALTY", "0.7"))     # weight of text-ratio penalty
_TEXT_TOPN      = int(os.getenv("SMART_TEXT_TOPN", "4"))            # candidate pool = n_clips * this
_TEXT_PROBE_FPS = float(os.getenv("SMART_TEXT_PROBE_FPS", "2.0"))  # frames/sec sampled per window
_TEXT_FRAME_W   = int(os.getenv("SMART_TEXT_FRAME_W", "320"))      # downscale width for frame extraction
_EAST_MODEL     = os.getenv(
    "SMART_EAST_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "assets", "frozen_east_text_detection.pb"),
)
_EAST_CONF      = float(os.getenv("SMART_EAST_CONF", "0.5"))       # min text-box confidence
_EAST_SIZE      = int(os.getenv("SMART_EAST_SIZE", "320"))         # EAST input size (multiple of 32)
# Text coverage (fraction of frame area inside text boxes) that we treat as
# "definitely a text/title card" → normalised text signal of 1.0. Real title
# cards cover ~0.10–0.45; gameplay is ~0.00. Tune via SMART_TEXT_FULL_COVER.
_TEXT_FULL_COVER = float(os.getenv("SMART_TEXT_FULL_COVER", "0.10"))
# Normalised text signal at/above which a frame counts as "has text" when
# building the busy-times map (0.5 ⇒ coverage ≥ 0.05).
_TEXT_CLEAN_THR  = float(os.getenv("SMART_TEXT_CLEAN_THR", "0.5"))
# Safety margin (s) added around a clip window when checking it against the
# text map, so a clip never starts/ends right on a text flash.
_TEXT_MARGIN     = float(os.getenv("SMART_TEXT_MARGIN", "0.4"))
# EAST output layer names (frozen graph).
_EAST_LAYERS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
# Lazily-loaded singleton: None = not tried, False = unavailable, else cv2 net.
_east_net = None


def _scene_scores(
    video_path: str,
    sample_fps: float = 4.0,
    timeout: int = 180,
) -> list[tuple[float, float]]:
    """
    Analyse a video and return [(timestamp, scene_score), ...].

    Uses ffmpeg's built-in scene-change detector. A high score means a big
    visual change between consecutive sampled frames — i.e. a cut, fast camera
    movement, explosion or other "action", which is a cheap proxy for the most
    interesting moments. The video is first downsampled to `sample_fps` so the
    analysis pass stays fast even on long sources.

    Returns an empty list if ffmpeg fails or nothing could be parsed (callers
    then fall back to random selection).
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "info",
                "-i", video_path,
                "-vf", f"fps={sample_fps},select='gte(scene,0)',"
                       "metadata=print:file=-",
                "-an", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Scene analysis failed for %s: %s",
                       os.path.basename(video_path), exc)
        return []

    scores: list[tuple[float, float]] = []
    cur_t: float | None = None
    # metadata=print emits pairs of lines:
    #   frame:0    pts:7616    pts_time:7.616
    #   lavfi.scene_score=0.045339
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([0-9.]+)", line)
            cur_t = float(m.group(1)) if m else None
        elif "lavfi.scene_score=" in line and cur_t is not None:
            try:
                scores.append((cur_t, float(line.split("=", 1)[1])))
            except ValueError:
                pass
    return scores


def _best_window_start(
    scores: list[tuple[float, float]],
    lo: float,
    hi: float,
    clip_dur: float,
    top_n: int = 1,
) -> list[tuple[float, float]]:
    """
    Within [lo, hi], rank clip starts by how much *sustained* motion their
    [start, start+clip_dur] window contains and return the best `top_n` as
    [(start, motion_score), ...] (highest first).

    We don't just sum scene-scores — that rewards a single editing cut spike
    that often jumps straight to a static title/text card. Instead, for each
    candidate window we:

      * clamp every sample to _SPIKE_CAP so one big cut can't dominate;
      * measure the fraction of near-static samples (< _STATIC_SCORE) and
        subtract a penalty proportional to it.

    Candidate starts are the scored timestamps inside the bucket. Returns an
    empty list if no scores fall in range.
    """
    cands = [t for t, _ in scores if lo <= t <= max(lo, hi)]
    if not cands:
        return []
    ranked: list[tuple[float, float]] = []
    for start in cands:
        window_end = start + clip_dur
        window = [s for t, s in scores if start <= t < window_end]
        if not window:
            continue
        motion = sum(min(s, _SPIKE_CAP) for s in window)
        static_ratio = sum(1 for s in window if s < _STATIC_SCORE) / len(window)
        score = motion * (1.0 - _STATIC_PENALTY * static_ratio)
        ranked.append((start, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:max(1, top_n)]


def _get_east_net():
    """
    Lazily load the EAST text-detector once and cache it. Returns the cv2 DNN
    net, or None if OpenCV/the model file is unavailable (callers then skip the
    text penalty gracefully).
    """
    global _east_net
    if _east_net is not None:
        return _east_net or None
    try:
        import cv2  # type: ignore
    except Exception:
        logger.info("Text penalty disabled: OpenCV not available")
        _east_net = False
        return None
    if not os.path.exists(_EAST_MODEL):
        logger.info("Text penalty disabled: EAST model not found at %s", _EAST_MODEL)
        _east_net = False
        return None
    try:
        net = cv2.dnn.readNet(_EAST_MODEL)
    except Exception as exc:
        logger.warning("Text penalty disabled: failed to load EAST model: %s", exc)
        _east_net = False
        return None
    _east_net = net
    logger.info("EAST text detector loaded (%s)", os.path.basename(_EAST_MODEL))
    return net


def _frame_text_ratio(img_path: str) -> float:
    """
    Estimate how "text-heavy" a single frame is, in [0.0, 1.0], using the EAST
    deep-learning text detector (cv2.dnn). EAST was trained to find real text
    regions, so unlike edge/MSER heuristics it does NOT fire on building
    windows, fences or HUD textures — it reliably separates gameplay from
    title/lower-third/text cards.

    The frame is run through EAST, detected text boxes are NMS-merged and
    rasterised into a mask, and the returned ratio is the fraction of frame
    area covered by text.

    Returns 0.0 on any failure or if no text is found (i.e. no penalty).
    """
    net = _get_east_net()
    if net is None:
        return 0.0
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return 0.0
    try:
        img = cv2.imread(img_path)
    except Exception:
        return 0.0
    if img is None or img.size == 0:
        return 0.0

    inp = max(32, (_EAST_SIZE // 32) * 32)
    try:
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (inp, inp),
            (123.68, 116.78, 103.94), swapRB=True, crop=False,
        )
        net.setInput(blob)
        scores, geometry = net.forward(_EAST_LAYERS)
    except Exception:
        return 0.0

    n_rows, n_cols = scores.shape[2], scores.shape[3]
    rects: list[tuple[int, int, int, int]] = []
    confs: list[float] = []
    for y in range(n_rows):
        s = scores[0, 0, y]
        d0, d1, d2, d3 = (geometry[0, 0, y], geometry[0, 1, y],
                          geometry[0, 2, y], geometry[0, 3, y])
        angles = geometry[0, 4, y]
        for x in range(n_cols):
            sc = float(s[x])
            if sc < _EAST_CONF:
                continue
            off_x, off_y = x * 4.0, y * 4.0
            a = float(angles[x])
            cos_a, sin_a = math.cos(a), math.sin(a)
            h = float(d0[x] + d2[x])
            w = float(d1[x] + d3[x])
            end_x = off_x + cos_a * d1[x] + sin_a * d2[x]
            end_y = off_y - sin_a * d1[x] + cos_a * d2[x]
            start_x = end_x - w
            start_y = end_y - h
            rects.append((int(start_x), int(start_y), int(w), int(h)))
            confs.append(sc)

    if not rects:
        return 0.0
    try:
        idxs = cv2.dnn.NMSBoxes(rects, confs, _EAST_CONF, 0.4)
    except Exception:
        return 0.0
    if idxs is None or len(idxs) == 0:
        return 0.0

    mask = np.zeros((inp, inp), dtype=np.uint8)
    for i in np.array(idxs).flatten():
        x, y, w, h = rects[int(i)]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(inp, x + w); y1 = min(inp, y + h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return float((mask > 0).sum()) / float(inp * inp)


def _text_busy_times(
    video_path: str,
    workdir: str,
    fps: float = _TEXT_PROBE_FPS,
) -> list[float]:
    """
    Scan the whole video once (one ffmpeg pass at `fps` frames/sec) and return
    the sorted list of timestamps (seconds) whose frame shows on-screen text
    (normalised EAST signal ≥ _TEXT_CLEAN_THR).

    Doing a single full-video pass is both cheaper and more accurate than
    re-probing every candidate window: trailer text fades in/out, so we need
    dense temporal coverage to know exactly which moments are clean. Callers
    use the returned "busy" times to reject any clip window that overlaps text.
    Returns an empty list if text detection is unavailable (→ no windows are
    considered texty, i.e. the penalty is effectively off).
    """
    if _get_east_net() is None:
        return []
    fps = max(0.5, fps)
    tmp = os.path.join(workdir, "_textscan")
    os.makedirs(tmp, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={fps:.3f},scale={_TEXT_FRAME_W}:-1",
            "-q:v", "5",
            os.path.join(tmp, "f_%05d.jpg"),
        ],
        timeout=240,
    )
    busy: list[float] = []
    try:
        frames = sorted(f for f in os.listdir(tmp) if f.endswith(".jpg"))
        for idx, fname in enumerate(frames):
            fp = os.path.join(tmp, fname)
            if os.path.getsize(fp) <= 256:
                continue
            cov = _frame_text_ratio(fp)
            sig = min(1.0, cov / max(1e-6, _TEXT_FULL_COVER))
            if sig >= _TEXT_CLEAN_THR:
                busy.append(idx / fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return busy


def _window_has_text(
    busy_times: list[float],
    start: float,
    clip_dur: float,
    margin: float = _TEXT_MARGIN,
) -> bool:
    """
    True if any known text timestamp falls within [start-margin, start+clip_dur+margin].
    `busy_times` must be sorted. Empty list ⇒ always False (no text known).
    """
    if not busy_times:
        return False
    lo = start - margin
    hi = start + clip_dur + margin
    i = bisect.bisect_left(busy_times, lo)
    return i < len(busy_times) and busy_times[i] <= hi


def _cut_clips_from_video(
    video_path: str,
    n_clips: int,
    workdir: str,
    intro_skip: float = 15.0,
    clip_name_prefix: str = "clip",
) -> list[str]:
    """
    Cut n_clips 5–7 second segments from video_path, skipping the opening and
    the final _OUTRO_SKIP seconds (typical outro/end-card territory).

    With scene analysis enabled, start positions are chosen GLOBALLY: candidate
    windows are ranked by sustained motion, the strongest are scored for
    on-screen text (EAST), and the least-texty / most-active windows are picked
    greedily while kept spaced out for coverage. This lets the picker skip
    whole text/title regions instead of being forced to take one clip per
    fixed time bucket. Without scene scores it falls back to even random
    buckets. Each clip is stream-copied (no re-encode) for speed.

    Why 5–7 s (not 3–4 s):
      `_make_video_segment` uses `-stream_loop -1 -t seg_dur`. If seg_dur is
      bigger than the source clip, ffmpeg loops the clip *inside* the segment
      — the loop boundary looks like a visible freeze/jump on screen.
      RU TTS is longer than EN, so seg_dur (≈ audio_dur / n_media) on RU can
      exceed 3–4 s. Cutting clips with headroom means `-t` just trims and we
      never hit a loop seam.
    """
    clips_dir = os.path.join(workdir, "yt_clips")
    os.makedirs(clips_dir, exist_ok=True)

    duration = _get_audio_duration(video_path)  # ffprobe works on video too
    usable_end = max(0.0, duration - _OUTRO_SKIP)
    available = max(0.0, usable_end - intro_skip)
    if available < 7.0:
        # Video too short after intro skip — start from the very beginning
        intro_skip = 0.0
        available = max(0.0, usable_end)

    if available < 1.0:
        logger.warning("Video too short to cut clips: %.1f s", duration)
        return []

    # Optionally analyse the video and prefer the most action-packed windows.
    scores: list[tuple[float, float]] = []
    if _SMART_CLIPS:
        scores = _scene_scores(video_path)
        if scores:
            logger.info(
                "Scene analysis: %d samples for %s — selecting interesting moments",
                len(scores), os.path.basename(video_path),
            )

    result: list[str] = []
    bucket_size = available / n_clips
    text_penalty_on = _SMART_CLIPS and _TEXT_PENALTY > 0.0
    nominal_dur = 6.0                       # window length used for ranking
    cut_max = 7.0                           # longest clip we may cut
    max_start = max(intro_skip, usable_end - cut_max)  # leave room for a cut

    # ------------------------------------------------------------------
    # Pick n_clips start positions.
    #
    # Old behaviour cut exactly one clip per equal-size bucket, so a bucket
    # that was entirely a title/text card produced a text clip no matter what.
    # Instead we now (1) scan the whole video ONCE for on-screen text, then
    # (2) rank candidate windows by motion and keep only those that fall
    # entirely inside text-free stretches. Several clips may come from one long
    # clean run, and text regions are skipped outright. If the source is mostly
    # text we relax gracefully (see passes below).
    # ------------------------------------------------------------------
    starts: list[float] = []
    if scores:
        busy_times = _text_busy_times(video_path, workdir) if text_penalty_on else []
        ranked = _best_window_start(
            scores, intro_skip, max_start, nominal_dur, top_n=10_000,
        )
        ranked = [(s, m) for s, m in ranked if m > 0]

        def _clean(start: float) -> bool:
            return not _window_has_text(busy_times, start, cut_max)

        # Pass 1: highest-motion windows that are fully text-free, spaced by one
        # clip length so we don't cut overlapping footage (but several clips may
        # share a long clean run).
        for s, m in ranked:
            if len(starts) >= n_clips:
                break
            if any(abs(s - u) < cut_max for u in starts):
                continue
            if not _clean(s):
                continue
            starts.append(s)
        # Pass 2: relax spacing to allow tighter packing of clean runs.
        if len(starts) < n_clips:
            for s, m in ranked:
                if len(starts) >= n_clips:
                    break
                if s in starts or any(abs(s - u) < nominal_dur * 0.5 for u in starts):
                    continue
                if not _clean(s):
                    continue
                starts.append(s)
        # Pass 3: source is mostly text — accept highest-motion windows even if
        # they contain some text, still avoiding duplicates.
        if len(starts) < n_clips:
            for s, m in ranked:
                if len(starts) >= n_clips:
                    break
                if s in starts or any(abs(s - u) < nominal_dur * 0.5 for u in starts):
                    continue
                starts.append(s)
        # Last resort: still short → random fill.
        guard = 0
        while len(starts) < n_clips and guard < n_clips * 4:
            starts.append(random.uniform(intro_skip, max_start))
            guard += 1
        starts = starts[:n_clips]
        starts.sort()

        if text_penalty_on:
            texty = sum(1 for s in starts if _window_has_text(busy_times, s, cut_max))
            logger.info(
                "Clip selection: %d windows, %d busy-text frames in source, "
                "%d clips still overlap text",
                len(starts), len(busy_times), texty,
            )

    for i in range(n_clips):
        clip_dur = random.uniform(5.0, 7.0)
        if i < len(starts):
            start = min(starts[i], max_start)
        else:
            # Fallback (no scene scores): even-coverage random buckets.
            bucket_start = intro_skip + i * bucket_size
            bucket_end = bucket_start + max(0.0, bucket_size - clip_dur)
            start = random.uniform(bucket_start, max(bucket_start, bucket_end))

        out_path = os.path.join(clips_dir, f"{clip_name_prefix}_{i:02d}.mp4")
        ok = _run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{clip_dur:.3f}",
                "-c", "copy",
                out_path,
            ],
            timeout=60,
        )
        if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            result.append(out_path)

    logger.info(
        "Cut %d %s 5–7 s clips from %s (intro_skip=%.1fs)",
        len(result), "smart" if scores else "random",
        os.path.basename(video_path), intro_skip,
    )
    return result


# ---------------------------------------------------------------------------
# ffmpeg segment builders
# ---------------------------------------------------------------------------

def _probe_image_dims(img_path: str) -> tuple[int, int]:
    """Return (width, height) of an image file via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "default=noprint_wrappers=1", img_path],
            capture_output=True, text=True, timeout=10,
        )
        w, h = VID_W, VID_H
        for line in r.stdout.splitlines():
            if line.startswith("width="):
                w = int(line.split("=")[1])
            elif line.startswith("height="):
                h = int(line.split("=")[1])
        return w, h
    except Exception:
        return VID_W, VID_H


def _make_image_segment(
    img_path: str,
    duration: float,
    out_path: str,
    target_w: int = VID_W,
    target_h: int = VID_H,
) -> bool:
    """Create a fixed-duration silent video segment from a still image.

    Landscape images (AR ≥ target):
      – Scaled to fill height; L→R pan reveals the full image width.

    Portrait images (AR < target):
      – Scaled to fit width, centre-cropped vertically.
    """
    fps      = VID_FPS
    n_frames = max(1, round(duration * fps))

    orig_w, orig_h = _probe_image_dims(img_path)
    img_ar = orig_w / max(orig_h, 1)
    vid_ar = target_w / target_h

    if img_ar >= vid_ar:
        # Landscape: scale to full frame height, pan L→R to reveal full picture.
        pan_h = target_h
        pan_w = max(target_w + 2, (int(pan_h * img_ar + 0.5) // 2) * 2)
        pan_range  = pan_w - target_w
        pan_speed  = pan_range / max(duration, 0.001)  # px/s
        vf = (
            f"scale={pan_w}:{pan_h},"
            f"crop={target_w}:{target_h}:'min(t*{pan_speed:.4f},{pan_range})':0,"
            f"setsar=1"
        )
    else:
        # Portrait: fit width, centre-crop height (static)
        vf = (
            f"scale={target_w}:-2,"
            f"crop={target_w}:{target_h}:0:'(ih-{target_h})/2',"
            f"setsar=1"
        )
    # Feed image as a looped stream at exactly VID_FPS so each output frame comes
    # from a distinct input tick — this makes zoompan's `on` counter advance
    # frame-by-frame, enabling the L→R pan expression to work correctly.
    # d=1 (one output per input frame) avoids the d×input_frames duration explosion.
    return _run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-r", str(fps),          # input rate for the looped still image
            "-t", f"{duration:.3f}",
            "-i", img_path,
            "-vf", vf,
            "-r", str(fps),
            "-frames:v", str(n_frames),
            *_video_encoder_args(crf=23),
            "-pix_fmt", "yuv420p",
            "-an",
            out_path,
        ],
        timeout=120,
    )


def _make_video_segment(
    vid_path: str,
    duration: float,
    out_path: str,
    skip_secs: float = 0.0,
    target_w: int = VID_W,
    target_h: int = VID_H,
) -> bool:
    """Trim and scale a video clip to the required output format.
    Uses -stream_loop so source clips shorter than duration are looped.
    skip_secs: seek into the source before cutting (e.g. to skip intros)."""
    ss_args = ["-ss", f"{skip_secs:.3f}"] if skip_secs > 0 else []
    return _run(
        [
            "ffmpeg", "-y",
            "-stream_loop", "-1",   # loop input if shorter than -t
        ] + ss_args + [
            "-i", vid_path,
            "-t", f"{duration:.3f}",
            "-vf", (
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h},"
                f"setsar=1,"
                f"fps={VID_FPS}"
            ),
            *_video_encoder_args(crf=23),
            "-pix_fmt", "yuv420p",
            "-an",
            out_path,
        ],
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Full video assembly
# ---------------------------------------------------------------------------

def _find_music_track() -> str | None:
    """Return a random background music track from the local music/ cache, or None."""
    if not os.path.isdir(MUSIC_DIR):
        return None
    tracks = [
        os.path.join(MUSIC_DIR, f)
        for f in os.listdir(MUSIC_DIR)
        if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a"))
    ]
    return random.choice(tracks) if tracks else None


async def _assemble_video(
    image_paths: list[str],
    video_clip_paths: list[str],
    audio_path: str,
    cues: list[tuple[float, float, str]],
    output_path: str,
    workdir: str,
    n_article_clips: int = 0,
) -> bool:
    """
    Assemble portrait video:
      1. Build per-item silent segments (equal screen time)
      2. Concat segments
      3. Overlay TTS audio
      4. Burn subtitles
    Final file written to output_path.
    """
    audio_dur = _get_audio_duration(audio_path)
    logger.info("Audio duration: %.1f s", audio_dur)

    # Order: first clip → images (pan animation) → remaining clips.
    # This keeps the video opening with motion footage and returns to gameplay after images.
    clips  = list(video_clip_paths)
    images = list(image_paths)
    if clips and images:
        all_media = [clips[0]] + images + clips[1:]
        is_image  = [False] + [True] * len(images) + [False] * len(clips[1:])
    elif clips:
        all_media = clips
        is_image  = [False] * len(clips)
    else:
        all_media = images
        is_image  = [True] * len(images)

    if not all_media:
        logger.error("No media available for video assembly")
        return False

    seg_dur = audio_dur / len(all_media)       # equal time per media item

    seg_w, seg_h = VID_W, VID_H

    # ── Step 1: build segments (parallel) ────────────────────────────────
    # Article clips are always the first n_article_clips entries in `clips`.
    # Track which media indices correspond to article clips for skip logic.
    article_clip_set = set(clips[:n_article_clips])

    async def _build_segment(i, media, img_flag):
        seg_path = os.path.join(workdir, f"seg_{i:03d}.mp4")
        if img_flag:
            ok = await asyncio.to_thread(
                _make_image_segment, media, seg_dur, seg_path, seg_w, seg_h,
            )
        else:
            skip = YT_CLIP_SKIP if media in article_clip_set else 0.0
            ok = await asyncio.to_thread(
                _make_video_segment, media, seg_dur, seg_path, skip, seg_w, seg_h,
            )
        if not ok:
            return None
        return seg_path

    seg_results = await asyncio.gather(
        *[_build_segment(i, media, img_flag) for i, (media, img_flag) in enumerate(zip(all_media, is_image))]
    )
    segments = []
    for i, p in enumerate(seg_results):
        if p is None:
            logger.warning("Skipping failed segment %d", i)
            continue
        segments.append(p)

    if not segments:
        logger.error("All segments failed — cannot build video")
        return False

    # ── Step 2: concat ────────────────────────────────────────────────────
    concat_txt = os.path.join(workdir, "concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as fh:
        for s in segments:
            fh.write(f"file '{s}'\n")

    raw_mp4 = os.path.join(workdir, "raw.mp4")
    # Re-encode (not -c copy) so all segments share an identical timebase /
    # GOP structure. With -c copy, transitions between image segments and
    # video segments can leave a brief PTS gap that desyncs downstream
    # filters, causing visible glitches for one GOP.
    ok = await _run_async(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
         "-vf", f"fps={VID_FPS},setsar=1,format=yuv420p",
         *_video_encoder_args(crf=20),
         "-pix_fmt", "yuv420p",
         "-video_track_timescale", "15360",
         "-an",
         raw_mp4],
        timeout=300,
    )
    if not ok:
        return False

    # ── Step 3: add audio ─────────────────────────────────────────────────
    # Use -stream_loop so the video loops if somehow still shorter than audio.
    # -t audio_dur ensures we never cut the audio short.
    mixed_mp4 = os.path.join(workdir, "mixed.mp4")
    ok = await _run_async(
        [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", raw_mp4,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-t", f"{audio_dur:.3f}",
            mixed_mp4,
        ],
        timeout=120,
    )
    if not ok:
        return False

    # ── Step 3.5: optional background music ──────────────────────────────────
    music_track = _find_music_track()  # local music/ folder only
    if music_track:
        music_mp4 = os.path.join(workdir, "music_mixed.mp4")
        music_dur = _get_audio_duration(music_track)
        music_start = random.uniform(0, max(0.0, music_dur - audio_dur - 1))
        ok_music = await _run_async(
            [
                "ffmpeg", "-y",
                "-i", mixed_mp4,
                "-ss", f"{music_start:.3f}",
                "-stream_loop", "-1",
                "-i", music_track,
                "-filter_complex",
                "[1:a]volume=0.12[bg];[0:a][bg]amix=inputs=2:normalize=0[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac",
                "-t", f"{audio_dur:.3f}",
                music_mp4,
            ],
            timeout=120,
        )
        if ok_music:
            mixed_mp4 = music_mp4
            logger.info("Background music added: %s", os.path.basename(music_track))
        else:
            logger.warning("Music mixing failed — continuing without background music")

    # ── Step 4: burn subtitles (Pillow PNG → ffmpeg overlay) ───────────────
    # Uses only the always-available `overlay` filter — no libass/libfreetype.
    final_mp4 = os.path.join(workdir, "final.mp4")
    subs_ok = await _burn_subtitles_pillow(
        mixed_mp4, cues, final_mp4, workdir,
    )
    if not subs_ok:
        logger.warning("Subtitle burn failed — sending video without subtitles")
        shutil.copy2(mixed_mp4, final_mp4)

    shutil.copy2(final_mp4, output_path)
    return os.path.exists(output_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def fetch_gameplay_clips(
    post: dict,
    search_query: str,
    yt_skip: int = 0,
    user_query: bool = False,
    search_queries: list[str] | None = None,
) -> tuple[list[str], list[str], str]:
    """
    Find source video(s), cut random 3–4 s segments to fill the full runtime.

    If the article has a downloaded video:
      Cut N_CLIPS_ARTICLE random segments from it (skipping the intro).
      Assembly order: first clip → article images → remaining clips.

    If the article has NO downloaded video:
      Download N_YT_VIDEOS YouTube search results and cut CLIPS_PER_VIDEO
      random segments from each → N_YT_VIDEOS × CLIPS_PER_VIDEO clips total.

    ``search_queries`` (optional): a list of distinct search queries — one per
    entity/subject mentioned in the video's *script* (see
    ``ai_adapter.extract_entity_queries``). When given (and the user hasn't
    pasted a custom direct video link), one YT video is downloaded PER query
    and cut into clips, pooling the results — so the resulting clip pool
    actually contains footage of each entity mentioned in the narration,
    instead of just whatever the single headline-based query returned.
    Ignored when empty/None (falls back to the single ``search_query`` path).

    Returns ([], pre_cut_clips, shared_workdir).
    article_videos is always [] — clips are pre-cut, so no additional
    intro-skip is applied during assembly (n_article_clips=0 in callers).
    Caller is responsible for cleaning up shared_workdir after rendering.
    """
    workdir = tempfile.mkdtemp(dir=VIDEOS_DIR, prefix="clips_")
    N_CLIPS_ARTICLE = 8    # ~7 × 3–4 s ≈ 24–32 s — fits the target short duration

    source_videos: list[str] = []

    # ── 1. Article video already on disk — skip if user provided a custom query ──
    if not user_query:
        article_hls = [p for p in post.get("video_paths", []) if os.path.exists(p)]
        if article_hls:
            source_videos = article_hls
            logger.info("Found %d article video(s) on disk", len(source_videos))

    # ── 2. Re-fetch article embeds if nothing on disk ─────────────────────
    if not user_query and not source_videos and post.get("article_url"):
        try:
            async with aiohttp.ClientSession() as _sess:
                article = await _scraper.scrape_article(_sess, post["article_url"])
                if article and article.pg_embeds:
                    # Try Playground HLS download first — collect ALL HLS videos
                    _, hls_paths = await _scraper.download_videos(_sess, article.pg_embeds)
                    if hls_paths:
                        source_videos = hls_paths
                        logger.info("Downloaded %d article Playground video(s)", len(source_videos))
                    else:
                        # Fall back to first YouTube embed in the article
                        yt_embeds = [e for e in article.pg_embeds if e["type"] == "youtube"]
                        if yt_embeds:
                            yt_url = f"https://www.youtube.com/watch?v={yt_embeds[0]['id']}"
                            logger.info("Downloading article YouTube embed: %s", yt_url)
                            src = await _download_full_yt_video(
                                yt_url, workdir, is_url=True
                            )
                            if src:
                                source_videos = [src]
        except Exception as exc:
            logger.warning("Could not fetch article embeds: %s", exc)

    if source_videos:
        # ── Article video(s) found: cut N_CLIPS_ARTICLE random 3–4 s clips from ALL videos ──
        all_clips: list[str] = []
        for vi, src_vid in enumerate(source_videos):
            logger.info(
                "Cutting clips from article video %d/%d: %s",
                vi + 1, len(source_videos), src_vid,
            )
            prefix = f"vid{vi}_clip" if len(source_videos) > 1 else "clip"
            vid_clips = await _clips_from_source(
                src_vid, N_CLIPS_ARTICLE, workdir, float(YT_CLIP_SKIP), prefix,
            )
            all_clips.extend(vid_clips)
            logger.info(
                "Prepared %d clips from article video %d for '%s'",
                len(vid_clips), vi + 1, search_query,
            )
        clips = all_clips
        logger.info(
            "Total: prepared %d clips from %d article video(s) for '%s'",
            len(clips), len(source_videos), search_query,
        )
    else:
        # ── No article video: download ONE YT video and cut clips from it ──
        # If the user pasted a direct video link (YouTube/Shorts, Instagram Reel,
        # TikTok, VK) into the search-query field, fetch that exact video instead
        # of treating the link text as a search query.
        direct_url = user_query and bool(_DIRECT_VIDEO_URL_RE.match(search_query.strip()))
        clips: list[str] = []
        entity_queries = [q for q in (search_queries or []) if q and q.strip()]
        if direct_url:
            logger.info("Custom query is a direct video link — downloading it: %s", search_query[:100])
            yt_video = await _download_full_yt_video(
                search_query.strip(), workdir, is_url=True,
            )
            if yt_video:
                clips = await _clips_from_source(
                    yt_video, N_CLIPS_ARTICLE, workdir, float(YT_CLIP_SKIP),
                    "yt_clip",
                )
        elif entity_queries and not user_query:
            # ── Multi-entity fetch: footage PER mentioned subject ───────────
            # Prefers Shorts (used whole) or an AI-picked transcript segment
            # of a long video, only falling back to a full-video download.
            CLIPS_PER_ENTITY = 4
            logger.info(
                "Fetching footage for %d entity queries: %s",
                len(entity_queries), entity_queries,
            )
            used_ids: set[str] = set()
            for qi, q in enumerate(entity_queries):
                q_clips = await _fetch_clips_for_query(
                    q, CLIPS_PER_ENTITY, workdir, yt_skip=yt_skip,
                    clip_prefix=f"yt_clip_e{qi}", used_ids=used_ids,
                )
                logger.info("Prepared %d clips for entity query '%s'", len(q_clips), q)
                clips.extend(q_clips)
            logger.info(
                "Total: prepared %d clips from %d entity queries",
                len(clips), len(entity_queries),
            )
        else:
            logger.info(
                "No article video — fetching footage for '%s' (skip=%d)",
                search_query, yt_skip,
            )
            clips = await _fetch_clips_for_query(
                search_query, N_CLIPS_ARTICLE, workdir, yt_skip=yt_skip,
                clip_prefix="yt_clip",
            )
            logger.info(
                "Prepared %d clips for '%s'",
                len(clips), search_query,
            )

    return [], clips, workdir


async def create_short_video(
    post: dict,
    script: str,
    search_query: str,
    yt_skip: int = 0,
    lang: str = "en",
    prefetched_clips: list[str] | None = None,
    n_article_clips: int = 0,
    include_article_images: bool = False,
    add_cta: bool = False,
) -> Optional[str]:
    """
    Generate a TikTok/Reels/Shorts video for an approved post.

    Args:
        post:              DB post dict (needs id, image_paths, article_title).
        script:            Pre-generated narration (~70–90 words, plain text).
        search_query:      Keywords for YouTube gameplay clip search.
        lang:              'en' for English TTS, 'ru' for Russian TTS.
        prefetched_clips:  Already-downloaded clip paths to reuse (skip download).
        include_article_images: If True, append article still images after the
            video clips. Defaults to False (clips/footage only).

    Returns:
        Absolute path to the generated .mp4 file, or None on failure.
    """
    workdir = tempfile.mkdtemp(dir=VIDEOS_DIR, prefix="gen_")
    logger.info("Video workdir: %s", workdir)

    tts_voice = TTS_VOICE_RU if lang == "ru" else TTS_VOICE
    if TTS_BACKEND == "gemini":
        # Translate edge-tts style "ru-RU"/"en-US" hint for the Gemini backend
        # so _synthesize_voice_gemini picks the right voice config.
        tts_voice = "ru-RU" if lang == "ru" else "en-US"

    # Optionally append a Telegram call-to-action to the end of the narration.
    if add_cta:
        cta = CTA_PHRASE_RU if lang == "ru" else CTA_PHRASE
        script = f"{script.rstrip()} {cta}"
        logger.info("Appended Telegram CTA to narration (%s)", lang)

    try:
        # 1. TTS voice + word-level subtitle cues (from WordBoundary events)
        logger.info("Synthesizing voice with edge-tts (%s)...", tts_voice)
        audio_path, cues = await _synthesize_voice(script, workdir, voice=tts_voice)
        audio_dur = _get_audio_duration(audio_path)

        # 2. Article images — may have been cleaned up after publish; re-fetch from source if needed
        # Article images are opt-in via `include_article_images` (default off).
        if include_article_images:
            article_images = [p for p in post.get("image_paths", []) if os.path.exists(p)]
        else:
            article_images = []

        # Article videos (Playground HLS downloaded at scrape time) — same check
        article_videos = [p for p in post.get("video_paths", []) if os.path.exists(p)]

        # Re-fetch from source if any stored files are missing on disk
        images_missing = include_article_images and not article_images and bool(post.get("image_paths"))
        videos_missing = not article_videos and bool(post.get("video_paths"))
        if images_missing or videos_missing or (not article_videos and include_article_images and not article_images):
            article_url = post.get("article_url", "")
            if article_url:
                logger.info("Re-downloading article media from %s", article_url)
                try:
                    async with aiohttp.ClientSession() as _sess:
                        article = await _scraper.scrape_article(_sess, article_url)
                        if article:
                            if include_article_images and article.image_urls and not article_images:
                                article_images = await _scraper.download_images(_sess, article.image_urls)
                                logger.info("Re-fetched %d article images", len(article_images))
                            if article.pg_embeds and not article_videos:
                                _, article_videos = await _scraper.download_videos(_sess, article.pg_embeds)
                                logger.info("Re-fetched %d article videos", len(article_videos))
                        else:
                            logger.info("Article returned nothing at source URL")
                except Exception as exc:
                    logger.warning("Could not re-fetch article media: %s", exc)
            else:
                logger.info("No article_url in post — cannot re-fetch media")

        # 3. Media collection: article videos first, then YouTube gameplay clips

        # ── Primary: article videos (Playground HLS) ─────────────────────────
        # ── Secondary: YouTube gameplay footage ──────────────────────────────
        if prefetched_clips is not None:
            # Reuse already-downloaded clips (shared between EN and RU renders).
            # prefetched_clips already contains article_videos + yt_clips —
            # do NOT add article_videos again from post.get("video_paths").
            all_clips = prefetched_clips
        else:
            # Fallback: download 4 YT videos and cut random 3–4 s clips from each.
            _N_YT  = 4
            _N_CUT = 3
            _yt_videos = await _download_multiple_yt_videos(
                search_query, _N_YT, workdir, skip=yt_skip,
            )
            all_clips = []
            for _vi, _vid in enumerate(_yt_videos):
                _segs = await asyncio.to_thread(
                    _cut_clips_from_video,
                    _vid, _N_CUT, workdir, float(YT_CLIP_SKIP), f"yt{_vi}_clip",
                )
                all_clips.extend(_segs)

        if not all_clips:
            logger.error("No video media available for post #%s", post.get("id"))
            return None

        # Images appended after clips so video always starts with motion footage.
        # article_images is populated above by re-fetch logic if needed.
        all_images: list[str] = article_images

        # 4. Assemble
        out_name    = f"short_{post['id']}_{uuid.uuid4().hex[:6]}.mp4"
        output_path = os.path.join(VIDEOS_DIR, out_name)
        logger.info(
            "Assembling video: %d images, %d clips, %.1f s audio",
            len(all_images), len(all_clips), audio_dur,
        )
        ok = await _assemble_video(
            all_images, all_clips, audio_path, cues, output_path, workdir,
            n_article_clips=n_article_clips,
        )
        return output_path if ok else None

    except RuntimeError as exc:
        logger.error("Video creation error (post #%s): %s", post.get("id"), exc)
        return None
    except Exception as exc:
        logger.error(
            "Unexpected error in video creation (post #%s): %s",
            post.get("id"), exc, exc_info=True,
        )
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
