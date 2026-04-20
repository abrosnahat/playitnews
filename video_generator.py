"""
Video generation for TikTok / Reels / Shorts.

Pipeline:
  AI script → edge-tts voice + VTT subtitles → collect media (article images
  + YouTube gameplay clips + Pixabay stock fill) → ffmpeg slideshow + audio
  + burned-in subtitles → mp4

Output: 1080 × 1920 portrait video (~30–45 s).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional

import ssl

import aiofiles
import aiohttp
import certifi

import scraper as _scraper
from config import PIXABAY_API_KEY, VIDEOS_DIR, YT_CLIP_DURATION, YT_CLIP_SKIP, YT_MAX_CLIPS, YT_MAX_FILESIZE

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
TTS_RATE  = "+10%"   # Slightly faster than default → more energetic, TikTok-style
TTS_PITCH = "-3Hz"   # Slightly lower pitch → warmer tone


# ---------------------------------------------------------------------------
# Generic subprocess helper
# ---------------------------------------------------------------------------

def _run(args: list[str], cwd: str | None = None, timeout: int = 120) -> bool:
    """Run a command, return True on success."""
    try:
        # Extend PATH so yt-dlp can find JS runtimes (deno, node) for n-challenge solving.
        _env = os.environ.copy()
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


def _ts_vtt_to_ass(ts: str) -> str:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to ASS format (H:MM:SS.cc).
    Handles both '.' and ',' as the decimal separator."""
    ts = ts.strip().replace(",", ".")   # SRT-style comma → dot
    # Normalise to HH:MM:SS.mmm
    if ts.count(":") == 1:
        ts = "00:" + ts
    h, m, rest = ts.split(":")
    secs, ms = (rest.split(".") + ["0"])[:2]
    cs = int(ms[:3].ljust(3, "0")) // 10   # centiseconds
    return f"{int(h)}:{int(m):02d}:{int(secs):02d}.{cs:02d}"


def _ass_ts_to_sec(ts: str) -> float:
    """Convert ASS timestamp H:MM:SS.cc to seconds."""
    try:
        h, m, rest = ts.strip().split(":")
        s_parts = rest.split(".")
        s  = int(s_parts[0])
        cs = int(s_parts[1]) if len(s_parts) > 1 else 0
        return int(h) * 3600 + int(m) * 60 + s + cs / 100
    except Exception:
        return 0.0


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
    candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
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


def _render_subtitle_png(text: str, out_path: str) -> bool:
    """Render subtitle text as RGBA PNG (transparent background)."""
    try:
        from PIL import Image
        img = Image.new("RGBA", (VID_W, VID_H), (0, 0, 0, 0))
        _render_subtitle_onto(text, img)
        img.save(out_path, "PNG")
        return True
    except Exception as exc:
        logger.warning("subtitle PNG render error: %s", exc)
        return False


def _parse_ass_cues(ass_path: str) -> list[tuple[float, float, str]]:
    """Return [(start_sec, end_sec, text), ...] from an ASS Dialogue file."""
    cues: list[tuple[float, float, str]] = []
    try:
        with open(ass_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("Dialogue:"):
                    continue
                parts = line.split(",", 9)
                if len(parts) < 10:
                    continue
                t_start = _ass_ts_to_sec(parts[1].strip())
                t_end   = _ass_ts_to_sec(parts[2].strip())
                text    = re.sub(r"\{[^}]*\}", "", parts[9].strip()).strip()
                if text:
                    cues.append((t_start, t_end, text))
    except Exception as exc:
        logger.warning("ASS parse error: %s", exc)
    return cues


def _expand_cues_to_words(
    cues: list[tuple[float, float, str]]
) -> list[tuple[float, float, str]]:
    """
    Split each sentence-level cue into individual words,
    distributing the cue's time window evenly across its words.
    Punctuation is stripped so only clean words appear.
    """
    result: list[tuple[float, float, str]] = []
    for t_start, t_end, text in cues:
        words = text.split()
        if not words:
            continue
        word_dur = (t_end - t_start) / len(words)
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:\"'()-–—")  # remove surrounding punctuation
            if not clean:
                continue
            result.append((
                t_start + i * word_dur,
                t_start + (i + 1) * word_dur,
                clean,
            ))
    return result


def _script_to_timed_cues(
    script: str,
    audio_dur: float,
    words_per_cue: int = 1,
) -> list[tuple[float, float, str]]:
    """
    Split the script into fixed-size word groups and assign each an equal
    time slice proportional to the audio duration.
    This is reliable regardless of VTT/ASS parsing issues.
    """
    words  = script.split()
    chunks = [" ".join(words[i: i + words_per_cue])
              for i in range(0, len(words), words_per_cue)]
    if not chunks:
        return []
    dur = audio_dur / len(chunks)
    return [(i * dur, (i + 1) * dur, text) for i, text in enumerate(chunks)]


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
        for t_start, t_end, cue_text in cues:
            if t_start <= t < t_end:
                text = cue_text
                break

        if not text:
            return

        fpath = os.path.join(frames_dir, fname)
        try:
            base    = Image.open(fpath).convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            _render_subtitle_onto(text, overlay)
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
            "-c:v", "libx264",
            "-crf", "20",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ar", "44100",
            "-b:a", "192k",
            "-shortest",
            output_mp4,
        ],
        timeout=300,
    )
    return ok


_SUB_GRAD_TOP    = (255, 255, 255)       # white
_SUB_GRAD_BOTTOM = (0xC7, 0xF8, 0xFD)   # #C7F8FD light cyan


def _render_subtitle_onto(text: str, img) -> None:
    """Composite subtitle text onto an RGBA PIL image in-place.
    Uses Impact font, white→cyan gradient fill and italic shear (top leans right).
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

    # --- Composite onto caller's image ---
    img.alpha_composite(text_layer)


def _vtt_to_ass(
    vtt_path: str,
    ass_path: str,
    res_x: int = VID_W,
    res_y: int = VID_H,
    words_per_cue: int = 6,
) -> None:
    """
    Convert word-level VTT → styled ASS subtitle file.
    Styles are baked into the file so the ffmpeg `ass=` filter needs no
    extra escaping — no more filtergraph parsing errors.
    """
    entries = _parse_vtt_entries(vtt_path)

    ass_header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {res_x}\n"
        f"PlayResY: {res_y}\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # White text, black outline, no shadow, bottom-centre, large font
        "Style: Default,Arial,56,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,3,0,2,10,10,120,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    with open(ass_path, "w", encoding="utf-8") as fh:
        fh.write(ass_header)
        for i in range(0, len(entries), words_per_cue):
            chunk = entries[i: i + words_per_cue]
            start = _ts_vtt_to_ass(chunk[0]["start"])
            end   = _ts_vtt_to_ass(chunk[-1]["end"])
            text  = " ".join(e["text"] for e in chunk)
            fh.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


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
        import whisper as _whisper  # openai-whisper
    except ImportError:
        return []

    def _clean(word: str) -> str:
        return word.strip(".,!?;:\"'()-\u2013\u2014")

    if not sentence_cues:
        return []

    model_name = "small"

    import ssl, certifi
    import warnings
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _whisper.load_model(model_name)
            result = model.transcribe(
                audio_path,
                word_timestamps=True,
                language=language,
                condition_on_previous_text=False,
            )
    finally:
        ssl._create_default_https_context = _orig_ctx

    # Collect all Whisper word timings globally
    wh_timings: list[tuple[float, float]] = [
        (float(w["start"]), float(w["end"]))
        for seg in result.get("segments", [])
        for w in seg.get("words", [])
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

# Common gaming brands and terms → Russian phonetic spelling
_RU_BRAND_MAP: list[tuple[str, str]] = [
    # Companies
    ("Activision Blizzard", "Активижн Близзард"),
    ("Activision", "Активижн"),
    ("Blizzard", "Близзард"),
    ("Bethesda", "Бетесда"),
    ("Ubisoft", "Юбисофт"),
    ("Capcom", "Капком"),
    ("Konami", "Конами"),
    ("Bandai Namco", "Бандай Намко"),
    ("Bandai", "Бандай"),
    ("FromSoftware", "Фром Софтвэр"),
    ("From Software", "Фром Софтвэр"),
    ("CD Projekt Red", "Си Ди Проджект Рэд"),
    ("CD Projekt", "Си Ди Проджект"),
    ("Obsidian", "Обсидиан"),
    ("Insomniac", "Инсомниак"),
    ("Naughty Dog", "Наути Дог"),
    ("Rockstar", "Рокстар"),
    ("Valve", "Вэлв"),
    ("Epic Games", "Эпик Геймс"),
    ("Epic", "Эпик"),
    ("Nintendo", "Нинтендо"),
    ("Square Enix", "Сквэр Эникс"),
    ("Square", "Сквэр"),
    ("Enix", "Эникс"),
    ("Bungie", "Банджи"),
    ("Respawn", "Риспон"),
    ("DICE", "Дайс"),
    ("Crytek", "Крайтек"),
    ("505 Games", "505 Геймс"),
    ("2K Games", "Ту Кей Геймс"),
    ("2K", "Ту Кей"),
    ("THQ Nordic", "ТХК Нордик"),
    ("Sega", "Сега"),
    ("Atari", "Атари"),
    ("id Software", "Ай Ди Софтвэр"),
    ("Larian Studios", "Ляриан Студиос"),
    ("Larian", "Ляриан"),
    ("Paradox", "Парадокс"),
    ("Warhorse", "Уорхорс"),
    ("Nacon", "Након"),
    ("Focus Entertainment", "Фокус Энтертейнмент"),
    ("Piranha Bytes", "Пираньа Байтс"),
    ("Deep Silver", "Дип Сильвер"),
    ("505", "505"),
    # Platforms / stores
    ("PlayStation", "ПлейСтэйшн"),
    ("Xbox", "Иксбокс"),
    ("Steam", "Стим"),
    ("Nintendo Switch", "Нинтендо Свитч"),
    ("Switch", "Свитч"),
    ("PC", "Пи Си"),
    ("Game Pass", "Гейм Пасс"),
    ("Epic Games Store", "Эпик Геймс Стор"),
    # Common gaming terms
    ("DLC", "ДЛЦ"),
    ("RPG", "РПГ"),
    ("FPS", "ФПС"),
    ("MMO", "ММО"),
    ("MMORPG", "ММОRPG"),
    ("Early Access", "Ёрли Эксесс"),
    ("Open World", "Опэн Ворлд"),
    ("Battle Royale", "Батл Рояль"),
    ("Game Awards", "Гейм Эворс"),
    ("The Game Awards", "Зе Гейм Эворс"),
    ("State of Play", "Стейт оф Плей"),
    ("Xbox Showcase", "Иксбокс Шоукейс"),
    ("Nintendo Direct", "Нинтендо Директ"),
    ("Summer Game Fest", "Саммер Гейм Фест"),
    # Specific game titles often left in English in RU scripts
    ("Resident Evil", "Резидент Ивл"),
    ("Devil May Cry", "Дэвил Мэй Край"),
    ("Dark Souls", "Дарк Соулс"),
    ("Elden Ring", "Элден Ринг"),
    ("Hollow Knight", "Холлоу Найт"),
    ("Ghost of Tsushima", "Гост оф Цусима"),
    ("Death Stranding", "Дэт Стрэндинг"),
    ("Red Dead Redemption", "Рэд Дэд Редэмпшн"),
    ("The Witcher", "Зе Витчер"),
    ("Cyberpunk", "Сайберпанк"),
    ("Baldur's Gate", "Балдурс Гейт"),
    ("Dragon Age", "Дрэгон Эйдж"),
    ("Mass Effect", "Масс Эффект"),
    ("Starfield", "Старфилд"),
    ("Fallout", "Фоллаут"),
    ("Skyrim", "Скайрим"),
    ("Oblivion", "Обливион"),
    ("Morrowind", "Морровинд"),
    ("S.T.A.L.K.E.R.", "СТАЛКЕР"),
    ("STALKER", "СТАЛКЕР"),
    ("Metro", "Метро"),
    ("Diablo", "Диабло"),
    ("Overwatch", "Овервотч"),
    ("World of Warcraft", "Ворлд оф Варкрафт"),
    ("Warcraft", "Варкрафт"),
    ("Hearthstone", "Хартстоун"),
    ("League of Legends", "Лига Легенд"),
    ("Counter-Strike", "Контер Страйк"),
    ("Dota", "Дота"),
    ("Minecraft", "Майнкрафт"),
    ("Fortnite", "Фортнайт"),
    ("Apex Legends", "Эпекс Лэджэндс"),
    ("Call of Duty", "Кол оф Дьюти"),
    ("Battlefield", "Баттлфилд"),
    ("Assassin's Creed", "Ассасинс Крид"),
    ("Far Cry", "Фар Край"),
    ("Watch Dogs", "Вотч Догс"),
    ("Rainbow Six", "Рэйнбоу Сикс"),
    ("God of War", "Год оф Вор"),
    ("Spider-Man", "Спайдермэн"),
    ("Horizon", "Хорайзон"),
    ("The Last of Us", "Зе Ласт оф Ас"),
    ("Uncharted", "Unchartd"),
    ("Gran Turismo", "Гран Туризмо"),
]

# Build compiled pattern once at module load
_RU_BRAND_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k, _ in _RU_BRAND_MAP) + r")\b",
    re.IGNORECASE,
)
_RU_BRAND_LOOKUP = {k.lower(): v for k, v in _RU_BRAND_MAP}


def _transliterate_for_ru_tts(text: str) -> str:
    """Replace English gaming brands/terms with Russian phonetic spelling."""
    def _replace(m: re.Match) -> str:
        return _RU_BRAND_LOOKUP.get(m.group(0).lower(), m.group(0))
    return _RU_BRAND_RE.sub(_replace, text)


async def _synthesize_voice(
    text: str, workdir: str, voice: str | None = None
) -> tuple[str, list[tuple[float, float, str]]]:
    """
    Generate audio via edge-tts (EN) or Silero TTS (Russian), then use Whisper
    to get accurate word-level timestamps from the actual audio signal.
    Falls back to ffmpeg silencedetect if Whisper is unavailable.
    Returns (audio_path, [(start_sec, end_sec, word), ...]).
    """
    chosen_voice = voice or TTS_VOICE

    audio_path = os.path.join(workdir, "voice.mp3")
    vtt_path   = os.path.join(workdir, "subs.vtt")

    # Derive Whisper language from voice locale prefix (e.g. "ru-RU-..." → "ru")
    whisper_lang = chosen_voice.split("-")[0].lower() if chosen_voice else "en"
    # Apply Russian brand transliteration for Russian-locale voices
    tts_text = _transliterate_for_ru_tts(text) if whisper_lang == "ru" else text

    if not tts_text or not tts_text.strip():
        raise RuntimeError("TTS text is empty — cannot synthesize voice")

    async def _run_edge_tts(tts_input: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "edge-tts",
            "--voice", chosen_voice,
            f"--rate={TTS_RATE}",
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

async def _fetch_youtube_clips(
    game_query: str,
    count: int,
    workdir: str,
    skip: int = 0,
) -> list[str]:
    """
    Search YouTube for gameplay footage of *game_query*, download short clips.
    Uses yt-dlp (already in requirements) — no API key needed.
    *skip*: skip the first N search results (for regenerate diversity).
    Returns list of local .mp4 paths.
    """
    clips_dir = os.path.join(workdir, "yt_clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Request more results than needed so we can skip the first *skip* entries.
    fetch_count = count + skip + 2
    yt_search = f"ytsearch{fetch_count}:{game_query}"

    # yt-dlp: download best video<=720p, no audio, max 50 MB, no playlist
    # Randomise the clip start position for visual diversity across videos.
    # Picks anywhere from YT_CLIP_SKIP to ~90 s into the video to vary the footage.
    clip_start = random.randint(YT_CLIP_SKIP, YT_CLIP_SKIP + 90)

    ydl_args = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        "--format", "bestvideo[height=1080][ext=mp4]/bestvideo[height=1080]/bestvideo[height<=1080][ext=mp4]/bestvideo[height<=720]",
        "--max-filesize", f"{YT_MAX_FILESIZE}M",
        "--max-downloads", str(count + skip),   # download skip+count, discard first skip
        "--output", os.path.join(clips_dir, "%(autonumber)s.%(ext)s"),
        # Random start within the video — varies on every generation
        "--download-sections", f"*{clip_start}-{clip_start + YT_CLIP_DURATION}",
        "--force-keyframes-at-cuts",
        yt_search,
    ]

    logger.info("Searching YouTube: '%s' (%d clips, skip=%d)", game_query, count, skip)
    ok = await _run_async(ydl_args, timeout=180)
    if not ok:
        logger.warning("yt-dlp exited with error — checking partial downloads")

    all_paths = [
        os.path.join(clips_dir, f)
        for f in sorted(os.listdir(clips_dir))
        if f.endswith((".mp4", ".webm", ".mkv"))
    ]
    # Discard the first *skip* results, take up to *count* of the rest
    result = all_paths[skip:][:count]
    logger.info("Downloaded %d YouTube clips for '%s' (skipped %d)", len(result), game_query, skip)
    return result


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
    """
    clips_dir = os.path.join(workdir, "yt_clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Prefer 720p mp4 to avoid YouTube n-challenge throttling on higher formats;
    # fall back to merged 'best' (always available without n-challenge).
    _FMT = ("bestvideo[height=720][ext=mp4]/bestvideo[height<=720][ext=mp4]"
            "/bestvideo[height<=720]/bestvideo[ext=mp4]/bestvideo/best")

    if is_url:
        ydl_args = [
            "yt-dlp",
            "--no-playlist", "--no-warnings",
            "--format", _FMT,
            "--max-filesize", f"{YT_MAX_FILESIZE}M",
            "--output", os.path.join(clips_dir, "source.%(ext)s"),
            url_or_search,
        ]
    else:
        fetch_count = skip + 3
        ydl_args = [
            "yt-dlp",
            "--no-playlist", "--no-warnings",
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


def _cut_clips_from_video(
    video_path: str,
    n_clips: int,
    workdir: str,
    intro_skip: float = 15.0,
    clip_name_prefix: str = "clip",
) -> list[str]:
    """
    Cut n_clips random 3–4 second segments from video_path, skipping the opening.
    The available range is divided into n_clips equal buckets; a random start
    position is chosen within each bucket to ensure even coverage with variety.
    Each clip is stream-copied (no re-encode) for speed.
    """
    clips_dir = os.path.join(workdir, "yt_clips")
    os.makedirs(clips_dir, exist_ok=True)

    duration = _get_audio_duration(video_path)  # ffprobe works on video too
    available = max(0.0, duration - intro_skip)
    if available < 4.0:
        # Video too short after intro skip — start from the very beginning
        intro_skip = 0.0
        available = duration

    if available < 1.0:
        logger.warning("Video too short to cut clips: %.1f s", duration)
        return []

    result: list[str] = []
    bucket_size = available / n_clips

    for i in range(n_clips):
        clip_dur = random.uniform(3.0, 4.0)
        bucket_start = intro_skip + i * bucket_size
        bucket_end   = bucket_start + max(0.0, bucket_size - clip_dur)
        start        = random.uniform(bucket_start, max(bucket_start, bucket_end))

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
        "Cut %d random 3–4 s clips from %s (intro_skip=%.1fs)",
        len(result), os.path.basename(video_path), intro_skip,
    )
    return result


# ---------------------------------------------------------------------------
# Pixabay stock media (images only — used as fallback fill)
# ---------------------------------------------------------------------------

async def _download_file(
    session: aiohttp.ClientSession, url: str, dest: str
) -> bool:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=90), ssl=SSL_CONTEXT) as resp:
            if resp.status == 200:
                data = await resp.read()
                async with aiofiles.open(dest, "wb") as fh:
                    await fh.write(data)
                return True
        logger.debug("Download HTTP %s: %s", resp.status, url)
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
    return False


async def _fetch_pixabay_images(
    query: str, count: int, workdir: str
) -> list[str]:
    if not PIXABAY_API_KEY:
        return []
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "image_type": "photo",
        "per_page": min(count + 5, 20),
        "safesearch": "true",
        "order": "popular",
    }
    paths: list[str] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://pixabay.com/api/",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
                ssl=SSL_CONTEXT,
            ) as resp:
                if resp.status != 200:
                    logger.warning("Pixabay images HTTP %s", resp.status)
                    return []
                data = await resp.json()
            for i, hit in enumerate(data.get("hits", [])[:count]):
                img_url = hit.get("largeImageURL") or hit.get("webformatURL")
                if not img_url:
                    continue
                ext = (img_url.split("?")[0].rsplit(".", 1)[-1] or "jpg")[:4]
                dest = os.path.join(workdir, f"pb_img_{i}.{ext}")
                if await _download_file(session, img_url, dest):
                    paths.append(dest)
    except Exception as exc:
        logger.warning("Pixabay images error: %s", exc)
    logger.info("Fetched %d Pixabay images for '%s'", len(paths), query)
    return paths


async def _fetch_pixabay_videos(
    query: str, count: int, workdir: str
) -> list[str]:
    if not PIXABAY_API_KEY:
        return []
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "video_type": "film",       # realistic footage, not animation
        "per_page": min(count + 5, 20),
        "safesearch": "true",
        "order": "relevant",
    }
    paths: list[str] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://pixabay.com/api/videos/",
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
                ssl=SSL_CONTEXT,
            ) as resp:
                if resp.status != 200:
                    logger.warning("Pixabay videos HTTP %s", resp.status)
                    return []
                data = await resp.json()
            downloads = []
            for i, hit in enumerate(data.get("hits", [])[:count]):
                videos = hit.get("videos", {})
                video_url = (
                    (videos.get("medium") or {}).get("url")
                    or (videos.get("small") or {}).get("url")
                    or (videos.get("tiny") or {}).get("url")
                )
                if not video_url:
                    continue
                dest = os.path.join(workdir, f"pb_vid_{i}.mp4")
                downloads.append((video_url, dest))
            # Download in parallel
            results = await asyncio.gather(
                *[_download_file(session, url, dest) for url, dest in downloads],
                return_exceptions=True,
            )
            for (_, dest), ok in zip(downloads, results):
                if ok is True:
                    paths.append(dest)
    except Exception as exc:
        logger.warning("Pixabay videos error: %s", exc)
    logger.info("Fetched %d Pixabay videos for '%s'", len(paths), query)
    return paths


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


def _make_image_segment(img_path: str, duration: float, out_path: str) -> bool:
    """Create a fixed-duration silent video segment from a still image.

    Landscape images (AR ≥ 9:16):
      – Displayed letterboxed in their native 16:9 ratio (black bars top/bottom).
      – Scaled so image height ≈ 60% of frame height, then padded to VID_H.
      – L→R pan reveals the full image width without any cropping or squishing.

    Portrait images (AR < 9:16):
      – Scaled to fit width=VID_W, centre-cropped vertically.
    """
    fps      = VID_FPS
    n_frames = max(1, round(duration * fps))

    orig_w, orig_h = _probe_image_dims(img_path)
    img_ar = orig_w / max(orig_h, 1)
    vid_ar = VID_W / VID_H  # 9/16

    if img_ar >= vid_ar:
        # Landscape: scale to full frame height, pan L→R to reveal full picture.
        # Image fills 1920px tall; width grows proportionally (wider than 1080) → pan.
        pan_h = VID_H
        pan_w = max(VID_W + 2, (int(pan_h * img_ar + 0.5) // 2) * 2)  # e.g. 3413 for 16:9
        pan_range  = pan_w - VID_W
        pan_speed  = pan_range / max(duration, 0.001)  # px/s
        vf = (
            f"scale={pan_w}:{pan_h},"
            # crop window slides from x=0 to x=pan_range using PTS time 't'
            f"crop={VID_W}:{VID_H}:'min(t*{pan_speed:.4f},{pan_range})':0,"
            f"setsar=1"
        )
    else:
        # Portrait: fit width, centre-crop height (static)
        vf = (
            f"scale={VID_W}:-2,"
            f"crop={VID_W}:{VID_H}:0:'(ih-{VID_H})/2',"
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
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            out_path,
        ],
        timeout=120,
    )


def _make_video_segment(vid_path: str, duration: float, out_path: str, skip_secs: float = 0.0) -> bool:
    """Trim and scale a video clip to the required portrait format.
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
                f"scale={VID_W}:{VID_H}:force_original_aspect_ratio=increase,"
                f"crop={VID_W}:{VID_H},"
                f"setsar=1,"
                f"fps={VID_FPS}"
            ),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
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


async def _fetch_pixabay_music(search_query: str) -> str | None:
    """
    Download a royalty-free background music track via yt-dlp.
    Searches YouTube for "royalty free gaming music no copyright" tracks.
    Caches in music/ for reuse. Returns local mp3 path or None on failure.
    """
    os.makedirs(MUSIC_DIR, exist_ok=True)

    # Reuse cached track if available
    cached = _find_music_track()
    if cached:
        return cached

    search_terms = [
        f"{search_query} background music no copyright",
        "gaming background music no copyright free use",
        "electronic background music no copyright",
        "lofi gaming music no copyright",
    ]

    for term in search_terms:
        track_id = uuid.uuid4().hex[:8]
        dest = os.path.join(MUSIC_DIR, f"yt_{track_id}.mp3")
        logger.info("Searching background music: '%s'", term)
        ok = await _run_async(
            [
                "yt-dlp",
                "--no-warnings", "--quiet",
                "--no-playlist",
                "-x", "--audio-format", "mp3", "--audio-quality", "5",
                "--match-filter", "duration > 30",    # skip very short clips
                "--max-downloads", "1",
                "-o", dest,
                f"ytsearch3:{term}",
            ],
            timeout=60,
        )
        if ok and os.path.exists(dest):
            logger.info("Background music cached: %s", os.path.basename(dest))
            return dest
        # Clean up partial file if any
        try:
            os.remove(dest)
        except OSError:
            pass

    logger.warning("Could not download background music")
    return None


async def _assemble_video(
    image_paths: list[str],
    video_clip_paths: list[str],
    audio_path: str,
    cues: list[tuple[float, float, str]],
    output_path: str,
    workdir: str,
    search_query: str = "",
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

    # ── Step 1: build segments (parallel) ────────────────────────────────
    # Article clips are always the first n_article_clips entries in `clips`.
    # Track which media indices correspond to article clips for skip logic.
    article_clip_set = set(clips[:n_article_clips])

    async def _build_segment(i, media, img_flag):
        seg_path = os.path.join(workdir, f"seg_{i:03d}.mp4")
        if img_flag:
            ok = await asyncio.to_thread(_make_image_segment, media, seg_dur, seg_path)
        else:
            skip = YT_CLIP_SKIP if media in article_clip_set else 0.0
            ok = await asyncio.to_thread(_make_video_segment, media, seg_dur, seg_path, skip)
        return seg_path if ok else None

    seg_results = await asyncio.gather(
        *[_build_segment(i, media, img_flag) for i, (media, img_flag) in enumerate(zip(all_media, is_image))]
    )
    segments = [p for p in seg_results if p is not None]
    for i, p in enumerate(seg_results):
        if p is None:
            logger.warning("Skipping failed segment %d", i)

    if not segments:
        logger.error("All segments failed — cannot build video")
        return False

    # ── Step 2: concat ────────────────────────────────────────────────────
    concat_txt = os.path.join(workdir, "concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as fh:
        for s in segments:
            fh.write(f"file '{s}'\n")

    raw_mp4 = os.path.join(workdir, "raw.mp4")
    ok = await _run_async(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
         "-c", "copy", raw_mp4],
        timeout=180,
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
) -> tuple[list[str], list[str], str]:
    """
    Find source video(s), cut random 3–4 s segments to fill the full runtime.

    If the article has a downloaded video:
      Cut N_CLIPS_ARTICLE random segments from it (skipping the intro).
      Assembly order: first clip → article images → remaining clips.

    If the article has NO downloaded video:
      Download N_YT_VIDEOS YouTube search results and cut CLIPS_PER_VIDEO
      random segments from each → N_YT_VIDEOS × CLIPS_PER_VIDEO clips total.

    Returns ([], pre_cut_clips, shared_workdir).
    article_videos is always [] — clips are pre-cut, so no additional
    intro-skip is applied during assembly (n_article_clips=0 in callers).
    Caller is responsible for cleaning up shared_workdir after rendering.
    """
    workdir = tempfile.mkdtemp(dir=VIDEOS_DIR, prefix="clips_")
    N_CLIPS_ARTICLE = 12   # ~12 × 3–4 s ≈ 36–48 s — fills a typical short
    N_YT_VIDEOS     = 4    # source videos when no article video available
    CLIPS_PER_VIDEO = 3    # cuts per YT video → 4 × 3 = 12 clips total

    source_video: str | None = None

    # ── 1. Playground HLS video already on disk ───────────────────────────
    article_hls = [p for p in post.get("video_paths", []) if os.path.exists(p)]
    if article_hls:
        source_video = article_hls[0]
        logger.info("Using article Playground video as source: %s", source_video)

    # ── 2. Re-fetch article embeds if nothing on disk ─────────────────────
    if not source_video and post.get("article_url"):
        try:
            async with aiohttp.ClientSession() as _sess:
                article = await _scraper.scrape_article(_sess, post["article_url"])
                if article and article.pg_embeds:
                    # Try Playground HLS download first
                    _, hls_paths = await _scraper.download_videos(_sess, article.pg_embeds)
                    if hls_paths:
                        source_video = hls_paths[0]
                        logger.info("Downloaded article Playground video: %s", source_video)
                    else:
                        # Fall back to first YouTube embed in the article
                        yt_embeds = [e for e in article.pg_embeds if e["type"] == "youtube"]
                        if yt_embeds:
                            yt_url = f"https://www.youtube.com/watch?v={yt_embeds[0]['id']}"
                            logger.info("Downloading article YouTube embed: %s", yt_url)
                            source_video = await _download_full_yt_video(
                                yt_url, workdir, is_url=True
                            )
        except Exception as exc:
            logger.warning("Could not fetch article embeds: %s", exc)

    if source_video:
        # ── Article video found: cut N_CLIPS_ARTICLE random 3–4 s clips ──
        clips = await asyncio.to_thread(
            _cut_clips_from_video,
            source_video, N_CLIPS_ARTICLE, workdir, float(YT_CLIP_SKIP),
        )
        logger.info(
            "Prepared %d random clips from article video for '%s'",
            len(clips), search_query,
        )
    else:
        # ── No article video: download N_YT_VIDEOS and cut from each ──────
        logger.info(
            "No article video — downloading %d YT videos for '%s' (skip=%d)",
            N_YT_VIDEOS, search_query, yt_skip,
        )
        yt_videos = await _download_multiple_yt_videos(
            search_query, N_YT_VIDEOS, workdir, skip=yt_skip,
        )
        clips: list[str] = []
        for vi, vid_path in enumerate(yt_videos):
            segs = await asyncio.to_thread(
                _cut_clips_from_video,
                vid_path, CLIPS_PER_VIDEO, workdir, float(YT_CLIP_SKIP),
                f"yt{vi}_clip",
            )
            clips.extend(segs)
        logger.info(
            "Prepared %d clips from %d YT videos for '%s'",
            len(clips), len(yt_videos), search_query,
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
) -> Optional[str]:
    """
    Generate a TikTok/Reels/Shorts video for an approved post.

    Args:
        post:              DB post dict (needs id, image_paths, article_title).
        script:            Pre-generated narration (~70–90 words, plain text).
        search_query:      Keywords for YouTube gameplay clip search.
        lang:              'en' for English TTS, 'ru' for Russian TTS.
        prefetched_clips:  Already-downloaded clip paths to reuse (skip download).

    Returns:
        Absolute path to the generated .mp4 file, or None on failure.
    """
    workdir = tempfile.mkdtemp(dir=VIDEOS_DIR, prefix="gen_")
    logger.info("Video workdir: %s", workdir)

    tts_voice = TTS_VOICE_RU if lang == "ru" else TTS_VOICE

    try:
        # 1. TTS voice + word-level subtitle cues (from WordBoundary events)
        logger.info("Synthesizing voice with edge-tts (%s)...", tts_voice)
        audio_path, cues = await _synthesize_voice(script, workdir, voice=tts_voice)
        audio_dur = _get_audio_duration(audio_path)

        # 2. Article images — may have been cleaned up after publish; re-fetch from source if needed
        article_images = [p for p in post.get("image_paths", []) if os.path.exists(p)]

        # Article videos (Playground HLS downloaded at scrape time) — same check
        article_videos = [p for p in post.get("video_paths", []) if os.path.exists(p)]

        # Re-fetch from source if any stored files are missing on disk
        images_missing = not article_images and bool(post.get("image_paths"))
        videos_missing = not article_videos and bool(post.get("video_paths"))
        if images_missing or videos_missing or (not article_images and not article_videos):
            article_url = post.get("article_url", "")
            if article_url:
                logger.info("Re-downloading article media from %s", article_url)
                try:
                    async with aiohttp.ClientSession() as _sess:
                        article = await _scraper.scrape_article(_sess, article_url)
                        if article:
                            if article.image_urls and not article_images:
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
            search_query=search_query,
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
