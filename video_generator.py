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
import logging
import os
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
from config import PIXABAY_API_KEY, VIDEOS_DIR

# Use certifi CA bundle — same fix as scraper.py prevents SSL errors on macOS
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Clip settings for YouTube footage
YT_CLIP_DURATION = 8       # seconds to cut from each YouTube video
YT_CLIP_SKIP     = 15      # skip first N seconds to avoid intros/title cards
YT_MAX_CLIPS     = 5       # max clips to download
YT_MAX_FILESIZE  = 50      # MB per clip (yt-dlp limit)

logger = logging.getLogger(__name__)

VID_W = 1080
VID_H = 1920
VID_FPS = 30
TTS_VOICE = "en-US-AndrewMultilingualNeural"  # Warm, authentic, most human-sounding
TTS_RATE  = "+10%"  # Slightly faster than default → more energetic, TikTok-style
TTS_PITCH = "-3Hz"  # Slightly lower pitch → warmer tone


# ---------------------------------------------------------------------------
# Generic subprocess helper
# ---------------------------------------------------------------------------

def _run(args: list[str], cwd: str | None = None, timeout: int = 120) -> bool:
    """Run a command, return True on success."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        if result.returncode != 0:
            logger.error("Command failed [%s]: %s", args[0], result.stderr[-600:])
            return False
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


async def _run_async(args: list[str], cwd: str | None = None, timeout: int = 120) -> bool:
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
    """Return a PIL ImageFont, preferring Arial Bold on macOS."""
    from PIL import ImageFont
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
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
        """Composite subtitle text onto a single frame in-place."""
        # frame index is 1-based in ffmpeg's image2 muxer
        idx = int(fname.replace("frame_", "").replace(".png", ""))
        t = (idx - 1) / EXTRACT_FPS

        text: str | None = None
        for t_start, t_end, cue_text in cues:
            if t_start <= t < t_end:
                text = cue_text
                break
        if not text:
            return  # nothing to draw

        fpath = os.path.join(frames_dir, fname)
        try:
            base = Image.open(fpath).convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            _render_subtitle_onto(text, overlay)
            combined = Image.alpha_composite(base, overlay).convert("RGB")
            combined.save(fpath, "PNG")
        except Exception as exc:
            logger.debug("Frame composite error %s: %s", fname, exc)

    # Run compositing in a thread pool (Pillow is CPU-bound)
    await asyncio.to_thread(
        lambda: [_composite_frame(f) for f in frame_files]
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
            "-crf", "28",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            output_mp4,
        ],
        timeout=300,
    )
    return ok


def _render_subtitle_onto(text: str, img) -> None:
    """Composite subtitle text onto an RGBA PIL image in-place."""
    from PIL import ImageDraw
    font_size = 112         # large karaoke-style single word
    draw      = ImageDraw.Draw(img)
    font      = _find_system_font(font_size)
    max_w     = VID_W - 80
    words     = text.split()
    lines: list[str] = []
    current   = ""
    for word in words:
        candidate = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) > max_w and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    line_h  = font_size + 18
    total_h = len(lines) * line_h
    # Position at ~55% from top (lower-third upper edge) — well above the bottom
    y       = (VID_H - 330) - total_h // 2
    # Thicker outline for legibility at larger size
    outline = [
        (dx, dy)
        for dx in range(-4, 5)
        for dy in range(-4, 5)
        if abs(dx) + abs(dy) <= 5
    ]
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (VID_W - (bbox[2] - bbox[0])) / 2
        for dx, dy in outline:
            draw.text((x+dx, y+dy), line, font=font, fill=(0, 0, 0, 220))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h


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
    Use ffmpeg silencedetect to find acoustic word boundaries in clean TTS audio.
    For each sentence cue the silence gaps between words are detected and each
    word in the sentence is mapped to the corresponding non-silent audio chunk.
    Falls back to syllable-proportional split if not enough chunks are found.
    """
    # Run silencedetect on the full audio (-35dB threshold, 40ms min silence)
    result = subprocess.run(
        [
            "ffmpeg", "-i", audio_path,
            "-af", "silencedetect=noise=-35dB:d=0.04",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=30,
    )
    # Parse silence_start / silence_end from stderr
    silences: list[tuple[float, float]] = []
    s_start: float | None = None
    for line in result.stderr.split("\n"):
        if "silence_start" in line:
            m = re.search(r"silence_start:\s*([\d.]+)", line)
            if m:
                s_start = float(m.group(1))
        elif "silence_end" in line and s_start is not None:
            m = re.search(r"silence_end:\s*([\d.]+)", line)
            if m:
                silences.append((s_start, float(m.group(1))))
                s_start = None

    audio_dur = _get_audio_duration(audio_path)

    # Build list of non-silent regions from the silence gaps
    def _speech_segments(t_from: float, t_to: float) -> list[tuple[float, float]]:
        """Non-silent segments within [t_from, t_to]."""
        segs: list[tuple[float, float]] = []
        prev = t_from
        for ss, se in silences:
            if se <= t_from or ss >= t_to:
                continue
            gap_start = max(ss, t_from)
            if gap_start > prev + 0.02:
                segs.append((prev, gap_start))
            prev = max(se, t_from)
        if prev < t_to - 0.02:
            segs.append((prev, t_to))
        return segs

    def _syllables(word: str) -> int:
        """Rough syllable count for proportional fallback."""
        word = re.sub(r"[^a-z]", "", word.lower())
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and len(word) > 2:
            count -= 1
        return max(1, count)

    word_cues: list[tuple[float, float, str]] = []

    for t_start, t_end, sentence in sentence_cues:
        words = sentence.split()
        if not words:
            continue
        segs = _speech_segments(t_start, t_end)

        if len(segs) == len(words):
            # Perfect match — assign each word to its detected speech segment
            for word, (ws, we) in zip(words, segs):
                clean = word.strip(".,!?;:\"'()-\u2013\u2014")
                if clean:
                    word_cues.append((ws, we, clean))
        else:
            # Fallback: distribute proportionally by syllable count
            total_syl = sum(_syllables(w) for w in words) or 1
            t = t_start
            for word in words:
                frac = _syllables(word) / total_syl
                dur  = (t_end - t_start) * frac
                clean = word.strip(".,!?;:\"'()-\u2013\u2014")
                if clean:
                    word_cues.append((t, t + dur, clean))
                t += dur

    return word_cues


async def _synthesize_voice(
    text: str, workdir: str
) -> tuple[str, list[tuple[float, float, str]]]:
    """
    Generate MP3 audio via edge-tts CLI, then use ffmpeg silencedetect
    to find accurate word-level boundaries from the actual audio signal.
    Sentence-level VTT anchors the start/end of each sentence, then
    silencedetect finds the gap between words within each sentence.
    Returns (audio_path, [(start_sec, end_sec, word), ...]).
    """
    audio_path = os.path.join(workdir, "voice.mp3")
    vtt_path   = os.path.join(workdir, "subs.vtt")

    try:
        proc = await asyncio.create_subprocess_exec(
            "edge-tts",
            "--voice", TTS_VOICE,
            f"--rate={TTS_RATE}",
            f"--pitch={TTS_PITCH}",
            "--text",  text,
            "--write-media",     audio_path,
            "--write-subtitles", vtt_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"edge-tts error: {stderr.decode()[:400]}")
    except FileNotFoundError:
        raise RuntimeError("edge-tts not found. Install with: pip install edge-tts")

    if not os.path.exists(vtt_path):
        open(vtt_path, "w").close()

    # Get sentence-level anchors from VTT, then detect exact word boundaries
    # from the audio signal using ffmpeg silencedetect (no ML model needed).
    sentence_cues = _parse_vtt_cues(vtt_path)
    word_cues = await asyncio.to_thread(
        _detect_word_boundaries_from_audio, audio_path, sentence_cues
    )
    logger.info("TTS: %d word cues (silencedetect), audio %.1f s",
                len(word_cues), _get_audio_duration(audio_path))
    return audio_path, word_cues


# ---------------------------------------------------------------------------
# YouTube gameplay footage via yt-dlp
# ---------------------------------------------------------------------------

async def _fetch_youtube_clips(
    game_query: str,
    count: int,
    workdir: str,
) -> list[str]:
    """
    Search YouTube for gameplay footage of *game_query*, download short clips.
    Uses yt-dlp (already in requirements) — no API key needed.
    Returns list of local .mp4 paths.
    """
    clips_dir = os.path.join(workdir, "yt_clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Search query: add 'gameplay footage' to narrow to in-game content
    yt_search = f"ytsearch{count + 2}:{game_query} gameplay footage"

    # yt-dlp: download best video<=720p, no audio, max 50 MB, no playlist
    ydl_args = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        "--format", "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
        "--max-filesize", f"{YT_MAX_FILESIZE}M",
        "--max-downloads", str(count),
        "--output", os.path.join(clips_dir, "%(autonumber)s.%(ext)s"),
        # Trim: skip first YT_CLIP_SKIP seconds (intros/title cards),
        # then take YT_CLIP_DURATION seconds of actual gameplay
        "--download-sections", f"*{YT_CLIP_SKIP}-{YT_CLIP_SKIP + YT_CLIP_DURATION}",
        "--force-keyframes-at-cuts",
        yt_search,
    ]

    logger.info("Searching YouTube: '%s gameplay footage' (%d clips)", game_query, count)
    ok = await _run_async(ydl_args, timeout=180)
    if not ok:
        logger.warning("yt-dlp exited with error — checking partial downloads")

    paths = [
        os.path.join(clips_dir, f)
        for f in sorted(os.listdir(clips_dir))
        if f.endswith((".mp4", ".webm", ".mkv"))
    ]
    logger.info("Downloaded %d YouTube clips for '%s'", len(paths), game_query)
    return paths[:count]


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

def _make_image_segment(img_path: str, duration: float, out_path: str) -> bool:
    """Create a fixed-duration silent video segment from a still image."""
    return _run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-t", f"{duration:.3f}",
            "-i", img_path,
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
        timeout=90,
    )


def _make_video_segment(vid_path: str, duration: float, out_path: str) -> bool:
    """Trim and scale a video clip to the required portrait format.
    Uses -stream_loop so source clips shorter than duration are looped."""
    return _run(
        [
            "ffmpeg", "-y",
            "-stream_loop", "-1",   # loop input if shorter than -t
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

async def _assemble_video(
    image_paths: list[str],
    video_clip_paths: list[str],
    audio_path: str,
    cues: list[tuple[float, float, str]],
    output_path: str,
    workdir: str,
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

    all_media = list(image_paths) + list(video_clip_paths)
    is_image  = [True] * len(image_paths) + [False] * len(video_clip_paths)

    if not all_media:
        logger.error("No media available for video assembly")
        return False

    seg_dur = audio_dur / len(all_media)       # equal time per media item

    # ── Step 1: build segments ─────────────────────────────────────────────
    segments: list[str] = []
    for i, (media, img_flag) in enumerate(zip(all_media, is_image)):
        seg_path = os.path.join(workdir, f"seg_{i:03d}.mp4")
        fn = _make_image_segment if img_flag else _make_video_segment
        ok = await asyncio.to_thread(fn, media, seg_dur, seg_path)
        if ok:
            segments.append(seg_path)
        else:
            logger.warning("Skipping failed segment %d: %s", i, media)

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

async def create_short_video(
    post: dict,
    script: str,
    search_query: str,
) -> Optional[str]:
    """
    Generate a TikTok/Reels/Shorts video for an approved post.

    Args:
        post:         DB post dict (needs id, image_paths, article_title).
        script:       Pre-generated English narration (~70–90 words, plain text).
        search_query: English keywords for Pixabay stock media search.

    Returns:
        Absolute path to the generated .mp4 file, or None on failure.
    """
    workdir = tempfile.mkdtemp(dir=VIDEOS_DIR, prefix="gen_")
    logger.info("Video workdir: %s", workdir)

    try:
        # 1. TTS voice + word-level subtitle cues (from WordBoundary events)
        logger.info("Synthesizing voice with edge-tts...")
        audio_path, cues = await _synthesize_voice(script, workdir)
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
        target_n = max(4, int(audio_dur / 5))

        # ── Primary: article videos (Playground HLS) ─────────────────────────
        # ── Secondary: YouTube gameplay footage ──────────────────────────────
        yt_needed = max(0, target_n - len(article_videos))
        yt_clips = await _fetch_youtube_clips(search_query, min(yt_needed, YT_MAX_CLIPS), workdir)

        # Article videos first, then YouTube clips — no images
        all_images: list[str] = []
        all_clips  = article_videos + yt_clips

        if not all_clips:
            logger.error("No video media available for post #%s", post.get("id"))
            return None

        # 4. Assemble
        out_name    = f"short_{post['id']}_{uuid.uuid4().hex[:6]}.mp4"
        output_path = os.path.join(VIDEOS_DIR, out_name)
        logger.info(
            "Assembling video: %d images, %d clips, %.1f s audio",
            len(all_images), len(all_clips), audio_dur,
        )
        ok = await _assemble_video(
            all_images, all_clips, audio_path, cues, output_path, workdir,
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
