#!/usr/bin/env python3
"""
Interactive monitor-screen corner picker.

Opens the background (image OR first frame of a video) using the SAME ffmpeg
scale+crop chain as the production pipeline so the coordinates you click are
in the exact pixel space ffmpeg sees at render time.

Workflow:
  1. Click ANY 4 points around the monitor screen — order doesn't matter,
     the tool auto-sorts them into TL/TR/BL/BR by geometry.
  2. (Optional) Press "Test render" — bakes a colourful test pattern into
     your quad using the EXACT compose filter from the pipeline. Inspect
     preview.mp4 to see what the pipeline will actually produce.
     The test pattern has labels TL / TR / BL / BR — if any label appears
     mirrored or in the wrong corner, the quad is wrong.
  3. (Video bg only) Use the time slider to confirm monitor is static.
  4. Press Enter (or "Done") — prints the env values, with --write also
     updates .env in place.

Usage:
    python3 calibrate_monitor.py                       # uses MONITOR_BG_PATH from .env
    python3 calibrate_monitor.py path/to/bg.mp4
    python3 calibrate_monitor.py path/to/bg.jpg --write
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH     = os.path.join(PROJECT_ROOT, ".env")

VID_W, VID_H = 1080, 1920

PREVIEW_H = 900
PREVIEW_W = int(VID_W * PREVIEW_H / VID_H)   # 506

# Inner margin around the image inside the canvas, so points right at the
# image edge are still comfortably clickable (you can overshoot into the
# margin and the coord is clamped to the image edge).
CANVAS_MARGIN = 40

CANVAS_W = PREVIEW_W + 2 * CANVAS_MARGIN
CANVAS_H = PREVIEW_H + 2 * CANVAS_MARGIN

# Magnifier loupe settings
LOUPE_SIZE = 220   # px (square)
LOUPE_ZOOM = 8     # full-res pixels visible across the loupe = LOUPE_SIZE/ZOOM

LABELS = ["TL", "TR", "BL", "BR"]


# Same chain the pipeline uses for the bg input
BG_FILTER = (
    f"scale={VID_W}:{VID_H}:force_original_aspect_ratio=increase,"
    f"crop={VID_W}:{VID_H},setsar=1"
)


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------

def _read_env_value(key: str) -> str | None:
    if not os.path.exists(ENV_PATH):
        return None
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.*)\s*$")
    with open(ENV_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return None


def _update_env(quad: str, rect: str) -> None:
    if not os.path.exists(ENV_PATH):
        with open(ENV_PATH, "w", encoding="utf-8") as fh:
            fh.write("")

    with open(ENV_PATH, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    def _replace_or_append(key: str, value: str, lines: list[str]) -> list[str]:
        pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
        new_line = f"{key}={value}\n"
        for i, line in enumerate(lines):
            if pat.match(line):
                lines[i] = new_line
                return lines
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)
        return lines

    lines = _replace_or_append("MONITOR_SCREEN_RECT", rect, lines)
    lines = _replace_or_append("MONITOR_SCREEN_QUAD", quad, lines)

    with open(ENV_PATH, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


# ---------------------------------------------------------------------------
# Geometry: auto-sort 4 arbitrary points into TL, TR, BL, BR
# ---------------------------------------------------------------------------

def _sort_quad(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Return points reordered as [TL, TR, BL, BR].

    Robust for tilted/perspective quads (where a "top-right" point can sit
    lower in Y than a "bottom-left" point):

      1. Compute centroid.
      2. Sort all 4 points clockwise by angle around the centroid.
      3. Pick the starting point of the cycle as the one closest to the
         top-left of the quad's own bounding box (smallest x+y).
      4. Walking clockwise from there gives TL → TR → BR → BL,
         which we then re-index to [TL, TR, BL, BR].
    """
    if len(points) != 4:
        return list(points)

    import math
    cx = sum(p[0] for p in points) / 4.0
    cy = sum(p[1] for p in points) / 4.0
    # In screen space (y grows down) sorting atan2 ascending walks CCW,
    # so we sort DESCENDING to walk clockwise around the centroid.
    cw = sorted(points,
                key=lambda p: math.atan2(-(p[1] - cy), p[0] - cx),
                reverse=True)

    # Find the index of the corner closest to the top-left direction
    # (minimal x+y). On any reasonable monitor quad this is unambiguous.
    start = min(range(4), key=lambda i: cw[i][0] + cw[i][1])
    cycle = cw[start:] + cw[:start]   # TL, TR, BR, BL (clockwise)

    tl, tr, br, bl = cycle
    return [tl, tr, bl, br]


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v"))


def _ffprobe_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _extract_calibration_frame(bg_path: str, timestamp: float = 1.0) -> Image.Image:
    """Render the bg through the SAME ffmpeg scale+crop chain the pipeline uses."""
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    try:
        if _is_video(bg_path):
            cmd = ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}",
                   "-i", bg_path, "-frames:v", "1", "-vf", BG_FILTER, out]
        else:
            cmd = ["ffmpeg", "-y", "-i", bg_path,
                   "-frames:v", "1", "-vf", BG_FILTER, out]
        subprocess.run(cmd, check=True, capture_output=True)
        return Image.open(out).convert("RGB").copy()
    finally:
        if os.path.exists(out):
            os.unlink(out)


def _make_test_placeholder(workdir: str, dur: float = 4.0) -> str:
    """1280×720 mp4 with bold corner labels (TL/TR/BL/BR) + frame counter."""
    out = os.path.join(workdir, "placeholder.mp4")
    centre = (
        "drawtext=text='%{e\\:n}':x=(w-tw)/2:y=(h-th)/2:"
        "fontsize=160:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10"
    )
    corners = (
        "drawtext=text='TL':x=30:y=30:fontsize=110:fontcolor=#00ff66:"
        "box=1:boxcolor=black@0.7:boxborderw=8,"
        "drawtext=text='TR':x=w-tw-30:y=30:fontsize=110:fontcolor=#ff3366:"
        "box=1:boxcolor=black@0.7:boxborderw=8,"
        "drawtext=text='BL':x=30:y=h-th-30:fontsize=110:fontcolor=#00aaff:"
        "box=1:boxcolor=black@0.7:boxborderw=8,"
        "drawtext=text='BR':x=w-tw-30:y=h-th-30:fontsize=110:fontcolor=#ffcc00:"
        "box=1:boxcolor=black@0.7:boxborderw=8"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"smptebars=size=1280x720:duration={dur}:rate=30",
        "-vf", f"{centre},{corners}",
        "-pix_fmt", "yuv420p", out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _build_test_render(
    bg_path: str,
    quad: list[tuple[int, int]],   # already TL, TR, BL, BR
    out_path: str,
) -> tuple[bool, str]:
    """Run the EXACT pipeline filter chain with a labelled placeholder."""
    workdir = tempfile.mkdtemp(prefix="calib_")
    try:
        placeholder = _make_test_placeholder(workdir)

        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        bbox_x, bbox_y = min(xs), min(ys)
        bbox_w = max(xs) - bbox_x
        bbox_h = max(ys) - bbox_y
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = [
            (p[0] - bbox_x, p[1] - bbox_y) for p in quad
        ]

        # Build alpha mask matching the quad shape so perspective output is
        # clipped exactly to the on-monitor region (no leakage onto bezel).
        from PIL import Image, ImageDraw
        mask_path = os.path.join(workdir, "_quad_mask.png")
        # 4× supersampling + LANCZOS downscale for smooth edges.
        ss = 4
        big = Image.new("L", (bbox_w * ss, bbox_h * ss), 0)
        ImageDraw.Draw(big).polygon(
            [(x0 * ss, y0 * ss), (x1 * ss, y1 * ss),
             (x3 * ss, y3 * ss), (x2 * ss, y2 * ss)],   # TL→TR→BR→BL
            fill=255,
        )
        mask_img = big.resize((bbox_w, bbox_h), Image.LANCZOS)
        mask_img.save(mask_path)

        bg_chain = f"[1:v]{BG_FILTER}[bg];"
        screen_chain = (
            f"[0:v]scale={bbox_w}:{bbox_h},setsar=1,"
            f"perspective="
            f"x0={x0}:y0={y0}:"
            f"x1={x1}:y1={y1}:"
            f"x2={x2}:y2={y2}:"
            f"x3={x3}:y3={y3}:"
            f"sense=destination:interpolation=linear[warp];"
            f"[2:v]format=gray,scale={bbox_w}:{bbox_h},setsar=1[mask];"
            f"[warp][mask]alphamerge[screen];"
        )
        overlay_xy = f"{bbox_x}:{bbox_y}"
        filter_complex = bg_chain + screen_chain + (
            f"[bg][screen]overlay={overlay_xy}:format=auto[final]"
        )

        bg_input = (
            ["-stream_loop", "-1", "-i", bg_path]
            if _is_video(bg_path)
            else ["-loop", "1", "-i", bg_path]
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", placeholder,
            *bg_input,
            "-loop", "1", "-i", mask_path,
            "-filter_complex", filter_complex,
            "-map", "[final]",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-t", "4",
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (r.returncode == 0 and os.path.exists(out_path)), r.stderr[-1200:]
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def _format_quad(points: list[tuple[int, int]]) -> str:
    return ";".join(f"{x},{y}" for x, y in points)


def _format_rect(points: list[tuple[int, int]]) -> str:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x = min(xs)
    y = min(ys)
    return f"{x},{y},{max(xs) - x},{max(ys) - y}"


# ---------------------------------------------------------------------------
# Tk UI
# ---------------------------------------------------------------------------

class Picker:
    def __init__(self, root: tk.Tk, bg_path: str, write_env: bool):
        self.root      = root
        self.bg_path   = bg_path
        self.write_env = write_env
        self.is_video  = _is_video(bg_path)
        self.duration  = _ffprobe_duration(bg_path) if self.is_video else 0.0
        self.timestamp = 1.0 if self.is_video else 0.0
        self.points: list[tuple[int, int]] = []   # raw clicks, any order

        root.title("Monitor screen calibration")

        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H,
                                cursor="cross", highlightthickness=0,
                                bg="#222222")
        self.canvas.pack(side=tk.LEFT, padx=(24, 0), pady=12)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self._hide_loupe())
        self._image_id = None
        self.tk_img    = None
        self.full_img: Image.Image | None = None   # 1080×1920 source

        sidebar = tk.Frame(root, padx=12, pady=12)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(sidebar, text="Click 4 corners of the screen.\n"
                               "Order does NOT matter — the tool sorts them\n"
                               "automatically into TL / TR / BL / BR.",
                 font=("Helvetica", 11), justify="left").pack(anchor="w")

        tk.Frame(sidebar, height=10).pack()
        self.status = tk.StringVar(value="0 / 4 points")
        tk.Label(sidebar, textvariable=self.status,
                 font=("Helvetica", 13, "bold")).pack(anchor="w")

        tk.Frame(sidebar, height=8).pack()
        tk.Label(sidebar, text="Sorted corners:",
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.order_lbls: list[tk.Label] = []
        for name in LABELS:
            lbl = tk.Label(sidebar, text=f"  {name}: —",
                           font=("Menlo", 11))
            lbl.pack(anchor="w")
            self.order_lbls.append(lbl)

        if self.is_video and self.duration > 0:
            tk.Frame(sidebar, height=10).pack()
            tk.Label(sidebar, text=f"Frame time (0–{self.duration:.1f}s):",
                     font=("Helvetica", 11, "bold")).pack(anchor="w")
            self.time_var = tk.DoubleVar(value=self.timestamp)
            ttk.Scale(sidebar, from_=0.0,
                      to=max(0.1, self.duration - 0.05),
                      orient=tk.HORIZONTAL, variable=self.time_var,
                      command=self._on_time_change).pack(fill=tk.X)
            self.time_lbl = tk.Label(sidebar,
                                     text=f"t = {self.timestamp:.2f} s",
                                     font=("Menlo", 10))
            self.time_lbl.pack(anchor="w")

        tk.Frame(sidebar, height=12).pack()
        tk.Label(sidebar, text="Magnifier (hover canvas):",
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.loupe = tk.Canvas(sidebar, width=LOUPE_SIZE, height=LOUPE_SIZE,
                               highlightthickness=1, highlightbackground="#888",
                               bg="#111")
        self.loupe.pack(anchor="w", pady=(2, 4))
        self._loupe_img_id = None
        self._loupe_tk = None
        self.loupe_coord = tk.StringVar(value="—")
        tk.Label(sidebar, textvariable=self.loupe_coord,
                 font=("Menlo", 10)).pack(anchor="w")

        tk.Frame(sidebar, height=12).pack()
        tk.Button(sidebar, text="Undo last  (U)",
                  command=self._undo).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Reset  (R)",
                  command=self._reset).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Test render → preview.mp4  (T)",
                  command=self._test_render,
                  bg="#1c6", fg="white",
                  activebackground="#1a5",
                  font=("Helvetica", 11, "bold")).pack(fill=tk.X, pady=8)
        tk.Button(sidebar, text="Done  (Enter)",
                  command=self._done).pack(fill=tk.X, pady=2)

        tk.Frame(sidebar, height=12).pack()
        self.coord_text = tk.Text(sidebar, width=46, height=10,
                                  font=("Menlo", 10),
                                  takefocus=0)
        self.coord_text.pack()

        for ch in ("u", "U"): root.bind_all(ch, lambda e: self._undo())
        for ch in ("r", "R"): root.bind_all(ch, lambda e: self._reset())
        for ch in ("t", "T"): root.bind_all(ch, lambda e: self._test_render())
        root.bind_all("<Return>", lambda e: self._done())

        self._reload_bg()
        self._redraw()

    # ---- bg frame ---------------------------------------------------------

    def _reload_bg(self):
        try:
            img = _extract_calibration_frame(self.bg_path, self.timestamp)
        except subprocess.CalledProcessError as exc:
            messagebox.showerror(
                "ffmpeg error",
                (exc.stderr or b"")[-800:].decode("utf-8", "ignore"),
            )
            return
        self.preview = img.resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        self.tk_img  = ImageTk.PhotoImage(self.preview)
        self.full_img = img
        if self._image_id is None:
            self._image_id = self.canvas.create_image(
                CANVAS_MARGIN, CANVAS_MARGIN,
                anchor=tk.NW, image=self.tk_img,
            )
            # Subtle frame around the image so the margin is visible.
            self.canvas.create_rectangle(
                CANVAS_MARGIN - 1, CANVAS_MARGIN - 1,
                CANVAS_MARGIN + PREVIEW_W, CANVAS_MARGIN + PREVIEW_H,
                outline="#444", width=1, tags="frame",
            )
        else:
            self.canvas.itemconfig(self._image_id, image=self.tk_img)

    def _on_time_change(self, _val):
        self.timestamp = float(self.time_var.get())
        self.time_lbl.config(text=f"t = {self.timestamp:.2f} s")
        if hasattr(self, "_time_after"):
            self.root.after_cancel(self._time_after)
        self._time_after = self.root.after(120, self._reload_and_redraw)

    def _reload_and_redraw(self):
        self._reload_bg()
        self._redraw()

    # ---- magnifier --------------------------------------------------------

    def _on_motion(self, event):
        if self.full_img is None:
            return
        # Map preview-canvas coords → full-res 1080×1920 coords.
        # Image is drawn at (CANVAS_MARGIN, CANVAS_MARGIN); subtract that
        # offset and clamp so cursor in the margin still tracks edge pixels.
        ex = max(0, min(PREVIEW_W - 1, event.x - CANVAS_MARGIN))
        ey = max(0, min(PREVIEW_H - 1, event.y - CANVAS_MARGIN))
        fx = ex * VID_W / PREVIEW_W
        fy = ey * VID_H / PREVIEW_H
        crop_size = LOUPE_SIZE / LOUPE_ZOOM   # full-res pixels visible
        half = crop_size / 2.0
        left   = int(round(fx - half))
        top    = int(round(fy - half))
        right  = left + int(round(crop_size))
        bottom = top  + int(round(crop_size))
        # Clamp into image bounds, keep crop square
        if left   < 0:           right -= left;            left = 0
        if top    < 0:           bottom -= top;            top = 0
        if right  > VID_W:       left -= (right - VID_W);  right = VID_W
        if bottom > VID_H:       top  -= (bottom - VID_H); bottom = VID_H
        left = max(0, left); top = max(0, top)
        crop = self.full_img.crop((left, top, right, bottom))
        # NEAREST keeps pixel grid sharp for precise picking
        zoomed = crop.resize((LOUPE_SIZE, LOUPE_SIZE), Image.NEAREST)
        self._loupe_tk = ImageTk.PhotoImage(zoomed)
        if self._loupe_img_id is None:
            self._loupe_img_id = self.loupe.create_image(
                0, 0, anchor=tk.NW, image=self._loupe_tk,
            )
            # Crosshair
            cx = cy = LOUPE_SIZE / 2
            self.loupe.create_line(0, cy, LOUPE_SIZE, cy,
                                   fill="#00FF88", tags="xhair")
            self.loupe.create_line(cx, 0, cx, LOUPE_SIZE,
                                   fill="#00FF88", tags="xhair")
            self.loupe.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                   outline="#00FF88", tags="xhair")
        else:
            self.loupe.itemconfig(self._loupe_img_id, image=self._loupe_tk)
            self.loupe.tag_raise("xhair")
        self.loupe_coord.set(f"x={int(round(fx))}  y={int(round(fy))}")

    def _hide_loupe(self):
        self.loupe_coord.set("—")

    # ---- click / control --------------------------------------------------

    def _on_click(self, event):
        if len(self.points) >= 4:
            return
        # Translate by image offset; allow clicking into the margin to land
        # exactly on the image edge.
        ex = max(0, min(PREVIEW_W - 1, event.x - CANVAS_MARGIN))
        ey = max(0, min(PREVIEW_H - 1, event.y - CANVAS_MARGIN))
        x = round(ex * VID_W / PREVIEW_W)
        y = round(ey * VID_H / PREVIEW_H)
        x = max(0, min(VID_W - 1, x))
        y = max(0, min(VID_H - 1, y))
        self.points.append((x, y))
        self._redraw()

    def _undo(self):
        if self.points:
            self.points.pop()
            self._redraw()

    def _reset(self):
        self.points.clear()
        self._redraw()

    def _sorted(self) -> list[tuple[int, int]]:
        return _sort_quad(self.points)

    def _test_render(self):
        if len(self.points) != 4:
            messagebox.showwarning(
                "Need 4 points",
                f"Pick all 4 corners first ({len(self.points)}/4).",
            )
            return
        out_path = os.path.join(PROJECT_ROOT, "preview.mp4")
        self.status.set("Rendering preview…")
        self.root.update_idletasks()
        ok, stderr_tail = _build_test_render(self.bg_path,
                                             self._sorted(), out_path)
        if not ok:
            messagebox.showerror("ffmpeg failed", stderr_tail or "Unknown error")
            self.status.set("Preview failed")
            return
        self.status.set(f"preview.mp4 ready — opening…")
        try:
            subprocess.Popen(["open", out_path])
        except Exception:
            pass

    def _done(self):
        if len(self.points) != 4:
            messagebox.showwarning(
                "Need 4 points",
                f"Click all 4 corners ({len(self.points)}/4 done).",
            )
            return
        sorted_pts = self._sorted()
        quad = _format_quad(sorted_pts)
        rect = _format_rect(sorted_pts)
        print()
        print("# Paste into .env:")
        print(f"MONITOR_SCREEN_QUAD={quad}")
        print(f"MONITOR_SCREEN_RECT={rect}")
        if self.write_env:
            _update_env(quad, rect)
            print(f"\n✓ .env updated at {ENV_PATH}")
        self.root.destroy()

    # ---- drawing ----------------------------------------------------------

    def _redraw(self):
        self.canvas.delete("marker")
        sorted_pts = self._sorted() if len(self.points) == 4 else None

        # Raw click markers
        for i, (x, y) in enumerate(self.points):
            px = CANVAS_MARGIN + x * PREVIEW_W / VID_W
            py = CANVAS_MARGIN + y * PREVIEW_H / VID_H
            r = 6
            self.canvas.create_oval(
                px - r, py - r, px + r, py + r,
                outline="#00FF88", width=2, fill="", tags="marker",
            )

        # Sorted-corner labels (only when all 4 picked)
        if sorted_pts:
            colours = {"TL": "#00FF88", "TR": "#FF4477",
                       "BL": "#33AAFF", "BR": "#FFCC00"}
            for name, (x, y) in zip(LABELS, sorted_pts):
                px = CANVAS_MARGIN + x * PREVIEW_W / VID_W
                py = CANVAS_MARGIN + y * PREVIEW_H / VID_H
                self.canvas.create_text(
                    px + 12, py - 12, text=name,
                    fill=colours[name],
                    font=("Helvetica", 13, "bold"), tags="marker",
                )

            # Polygon outline TL → TR → BR → BL
            tl, tr, bl, br = sorted_pts
            poly = []
            for x, y in (tl, tr, br, bl):
                poly.extend([
                    CANVAS_MARGIN + x * PREVIEW_W / VID_W,
                    CANVAS_MARGIN + y * PREVIEW_H / VID_H,
                ])
            self.canvas.create_polygon(
                poly, outline="#00FF88", fill="", width=2,
                tags="marker", dash=(4, 3),
            )

        # Sidebar
        self.status.set(f"{len(self.points)} / 4 points")
        for i, lbl in enumerate(self.order_lbls):
            if sorted_pts:
                x, y = sorted_pts[i]
                lbl.config(text=f"  {LABELS[i]}: ({x:>4}, {y:>4})")
            else:
                lbl.config(text=f"  {LABELS[i]}: —")

        self.coord_text.delete("1.0", tk.END)
        if sorted_pts:
            self.coord_text.insert(tk.END,
                "MONITOR_SCREEN_QUAD=" + _format_quad(sorted_pts) + "\n")
            self.coord_text.insert(tk.END,
                "MONITOR_SCREEN_RECT=" + _format_rect(sorted_pts) + "\n")
        else:
            self.coord_text.insert(tk.END, "Pick 4 corners…\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bg", nargs="?",
                        help="Path to background image or video. "
                             "Defaults to MONITOR_BG_PATH from .env.")
    parser.add_argument("--write", action="store_true",
                        help="Write MONITOR_SCREEN_QUAD/RECT into .env on finish.")
    args = parser.parse_args()

    bg_path = args.bg or _read_env_value("MONITOR_BG_PATH")
    if not bg_path:
        print("ERROR: provide a background path or set MONITOR_BG_PATH in .env",
              file=sys.stderr)
        return 1
    if not os.path.isabs(bg_path):
        bg_path = os.path.join(PROJECT_ROOT, bg_path)
    if not os.path.exists(bg_path):
        print(f"ERROR: background not found: {bg_path}", file=sys.stderr)
        return 1

    print(f"Loading background: {bg_path}")
    root = tk.Tk()
    Picker(root, bg_path, write_env=args.write)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
