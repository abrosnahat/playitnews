"""
Thumbnail generator: overlays article title text on the first article image.

Creates a single 1080×1920 JPEG used for both YouTube Shorts and Instagram Reels covers.
Uses Pillow — same dependency already used by video_generator.
"""
import os
import logging
import textwrap

logger = logging.getLogger(__name__)

# Thumbnail size: 1080×1920 (9:16) — used for Instagram Reels
IG_W, IG_H = 1080, 1920

_WIN_FONTS_DIR = os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts")
_FONT_CANDIDATES = [
    # Windows
    os.path.join(_WIN_FONTS_DIR, "impact.ttf"),
    os.path.join(_WIN_FONTS_DIR, "arialbd.ttf"),
    os.path.join(_WIN_FONTS_DIR, "arial.ttf"),
    os.path.join(_WIN_FONTS_DIR, "seguiemj.ttf"),
    # macOS
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Linux
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Outline / stroke colour (black)
_OUTLINE_COLOR = (0x00, 0x00, 0x00)  # #000000
# Gradient fill: top colour → bottom colour
_GRAD_TOP    = (255, 255, 255)        # white
_GRAD_BOTTOM = (0xC7, 0xF8, 0xFD)    # #C7F8FD (light cyan)


def _find_font(size: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_thumbnail(
    source_image_path: str,
    title: str,
    out_path: str,
    width: int,
    height: int,
) -> bool:
    """
    Core render: gaming-style thumbnail.
    - scale-to-fill crop of source image
    - dark gradient overlay on bottom half
    - Impact font, italic shear
    - thick #31B3BD outline
    - white→pink gradient fill
    Returns True on success.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageChops

        # --- 1. Load + fill frame (scale-to-fill, centre-crop) ---
        img = Image.open(source_image_path).convert("RGB")
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - width) // 2
        top  = (new_h - height) // 2
        img = img.crop((left, top, left + width, top + height))

        # --- 2. Dark gradient overlay on the bottom 60% ---
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)
        grad_start = int(height * 0.40)
        for y_g in range(grad_start, height):
            alpha = int(230 * (y_g - grad_start) / (height - grad_start))
            draw_ov.line([(0, y_g), (width, y_g)], fill=(0, 0, 0, alpha))
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)

        # --- 3. Truncate title to ~50 chars at word boundary ---
        if len(title) > 50:
            short = title[:50].rsplit(" ", 1)[0]
            title = (short if short else title[:50]) + "…"

        # --- 4. Font + layout ---
        font_size = max(90, width // 8)   # larger than before
        font = _find_font(font_size)

        padding = int(width * 0.06)
        # Impact is wider per char — use a tighter estimate
        max_chars_per_line = max(6, int((width - 2 * padding) / (font_size * 0.62)))
        lines = textwrap.wrap(title, width=max_chars_per_line)
        if not lines:
            lines = [title]
        lines = lines[:3]

        line_spacing = int(font_size * 1.05)
        total_text_h = len(lines) * line_spacing

        # Text block ends ~24% from bottom
        y_start = height - int(height * 0.24) - total_text_h

        # --- 5. Render text to a separate RGBA layer ---
        text_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_t = ImageDraw.Draw(text_layer)

        outline_r = max(10, font_size // 4)  # thick outline

        cur_y = y_start
        for line in lines:
            # Compute line width to centre horizontally
            bbox_measure = draw_t.textbbox((0, 0), line, font=font)
            line_w = bbox_measure[2] - bbox_measure[0]
            x = (width - line_w) // 2 - bbox_measure[0]

            # 5a. Sharp outline via square morphological dilation (no rounded corners)
            # Render text as a binary mask, expand with MaxFilter (square kernel),
            # then subtract original to get the border ring only.
            text_mask_line = Image.new("L", (width, height), 0)
            ImageDraw.Draw(text_mask_line).text((x, cur_y), line, font=font, fill=255)
            kernel_size = outline_r * 2 + 1
            dilated = text_mask_line.filter(ImageFilter.MaxFilter(kernel_size))
            outline_mask = ImageChops.subtract(dilated, text_mask_line)
            outline_fill = Image.new("RGBA", (width, height), (*_OUTLINE_COLOR, 255))
            text_layer.paste(outline_fill, mask=outline_mask)

            # 5b. Gradient fill (white top → cyan bottom)
            bbox = draw_t.textbbox((x, cur_y), line, font=font)
            ly0, ly1 = bbox[1], bbox[3]
            lh = max(1, ly1 - ly0)

            # Build gradient strip (full width, height = lh)
            grad = Image.new("RGBA", (width, lh), (0, 0, 0, 0))
            for gy in range(lh):
                t = gy / (lh - 1) if lh > 1 else 0
                r = int(_GRAD_TOP[0] + (_GRAD_BOTTOM[0] - _GRAD_TOP[0]) * t)
                g = int(_GRAD_TOP[1] + (_GRAD_BOTTOM[1] - _GRAD_TOP[1]) * t)
                b = int(_GRAD_TOP[2] + (_GRAD_BOTTOM[2] - _GRAD_TOP[2]) * t)
                ImageDraw.Draw(grad).line([(0, gy), (width, gy)], fill=(r, g, b, 255))

            # Text alpha mask (full-size L image, then crop to line bounds)
            mask_full = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask_full).text((x, cur_y), line, font=font, fill=255)
            line_mask = mask_full.crop((0, ly0, width, ly1))
            grad.putalpha(line_mask)
            text_layer.paste(grad, (0, ly0), mask=line_mask)

            cur_y += line_spacing

        # --- 6. Apply italic shear to the whole text layer ---
        # PIL AFFINE inverse-maps: src_x = dst_x + shear*(dst_y - y_start)
        # shear > 0: at top (dst_y < y_start) → src_x < dst_x → pixel pulled from left → top appears RIGHT ✓
        shear = 0.18
        affine = (1, shear, -shear * y_start, 0, 1, 0)
        text_layer = text_layer.transform(
            (width, height), Image.AFFINE, affine, resample=Image.BICUBIC,
        )

        # --- 7. Composite text layer onto image ---
        img = Image.alpha_composite(img, text_layer).convert("RGB")

        # --- 8. Save ---
        img.save(out_path, "JPEG", quality=92)
        logger.info("Thumbnail saved: %s (%dx%d)", out_path, width, height)
        return True

    except Exception as exc:
        logger.warning("Thumbnail render failed: %s", exc)
        return False


def generate_instagram_thumbnail(
    source_image_path: str,
    title: str,
    out_path: str,
) -> bool:
    """Generate a 1080×1920 thumbnail for Instagram Reels (9:16). Returns True on success."""
    return _render_thumbnail(source_image_path, title, out_path, IG_W, IG_H)


def generate_thumbnail(
    source_image_path: str,
    title: str,
    out_path: str,
) -> bool:
    """Alias for generate_instagram_thumbnail (1080×1920). Returns True on success."""
    return _render_thumbnail(source_image_path, title, out_path, IG_W, IG_H)


# ---------------------------------------------------------------------------
# Instagram CAROUSEL slides (1080×1350, 4:5 portrait)
# ---------------------------------------------------------------------------

CAR_W, CAR_H = 1080, 1350

# Brand colours for text slides
_CAR_GRAD_BG_TOP    = (0x12, 0x1B, 0x2E)   # deep navy
_CAR_GRAD_BG_BOTTOM = (0x2A, 0x5A, 0x8A)   # cool steel blue
_CAR_ACCENT         = (0x31, 0xB3, 0xBD)   # teal accent line


def _fit_cover(img, w: int, h: int):
    """Scale-to-fill crop preserving aspect ratio."""
    from PIL import Image
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top  = (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    """Word-wrap by measuring actual font width."""
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _draw_text_block(
    img, text: str, *, max_lines: int = 6, font_size: int | None = None,
    align: str = "center", v_align: str = "middle",
    color=(255, 255, 255), shadow=True, padding_pct: float = 0.08,
) -> None:
    """Draw word-wrapped, vertically-centred text on *img* in place."""
    from PIL import ImageDraw, ImageFilter
    w, h = img.size
    pad = int(min(w, h) * padding_pct)
    max_w = w - 2 * pad

    # Auto font size: shrink until text fits in (max_w, max_lines)
    fs = font_size or int(min(w, h) * 0.07)
    while fs > 28:
        font = _find_font(fs)
        d = ImageDraw.Draw(img)
        lines = _wrap_text(d, text, font, max_w)
        if len(lines) <= max_lines:
            break
        fs -= 4
    else:
        font = _find_font(fs)
        lines = _wrap_text(ImageDraw.Draw(img), text, font, max_w)[:max_lines]

    line_h = int(fs * 1.25)
    block_h = line_h * len(lines)
    if v_align == "top":
        y = pad
    elif v_align == "bottom":
        y = h - pad - block_h
    else:  # middle
        y = (h - block_h) // 2

    # Render shadow + text via separate layer for nicer blur
    if shadow:
        from PIL import Image as _Im
        shadow_layer = _Im.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow_layer)
        cy = y
        for ln in lines:
            tw = sd.textlength(ln, font=font)
            x = (w - tw) // 2 if align == "center" else pad
            sd.text((x + 4, cy + 4), ln, font=font, fill=(0, 0, 0, 220))
            cy += line_h
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(6))
        img.alpha_composite(shadow_layer) if img.mode == "RGBA" else img.paste(
            shadow_layer.convert("RGB"), (0, 0), shadow_layer
        )

    d = ImageDraw.Draw(img)
    cy = y
    for ln in lines:
        tw = d.textlength(ln, font=font)
        x = (w - tw) // 2 if align == "center" else pad
        d.text((x, cy), ln, font=font, fill=color)
        cy += line_h


def _gradient_bg(w: int, h: int, top, bottom):
    """Vertical gradient background as RGB Image."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), top)
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        d.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _blurred_photo_bg(source_image_path: str, w: int, h: int):
    """Use article photo as background — scaled-to-fill, lightly blurred + slightly darkened."""
    from PIL import Image, ImageFilter, ImageEnhance
    img = Image.open(source_image_path).convert("RGB")
    img = _fit_cover(img, w, h)
    # Soft blur so the photo is still recognisable behind the text
    img = img.filter(ImageFilter.GaussianBlur(7))
    img = ImageEnhance.Brightness(img).enhance(0.78)
    return img


# Subtitle-style text colours (match video_generator._render_subtitle_onto)
_SUB_GRAD_TOP    = (255, 255, 255)      # white
_SUB_GRAD_BOTTOM = (0xC7, 0xF8, 0xFD)   # #C7F8FD light cyan
_SUB_OUTLINE     = (0, 0, 0)


def _draw_subtitle_style_text(
    img,
    text: str,
    *,
    max_lines: int = 6,
    font_size: int | None = None,
    v_align: str = "middle",
    padding_pct: float = 0.08,
    shear: float = 0.18,
) -> None:
    """Render *text* on *img* in place using the same Impact + white→cyan
    gradient + italic-shear style as the burned-in video subtitles
    (see video_generator._render_subtitle_onto). The text layer is composited
    via alpha so it sits cleanly on any background.
    """
    from PIL import Image, ImageDraw
    w, h = img.size
    pad = int(min(w, h) * padding_pct)
    max_w = w - 2 * pad

    # Auto-shrink until the text fits within max_lines
    fs = font_size or int(min(w, h) * 0.075)
    while fs > 28:
        font = _find_font(fs)
        d = ImageDraw.Draw(img)
        lines = _wrap_text(d, text, font, max_w)
        if len(lines) <= max_lines:
            break
        fs -= 4
    else:
        font = _find_font(fs)
        lines = _wrap_text(ImageDraw.Draw(img), text, font, max_w)[:max_lines]

    line_h = int(fs * 1.18)
    block_h = line_h * len(lines)
    if v_align == "top":
        y_start = pad
    elif v_align == "bottom":
        y_start = h - pad - block_h
    else:
        y_start = (h - block_h) // 2

    outline_r = max(3, fs // 22)
    text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_t = ImageDraw.Draw(text_layer)

    cy = y_start
    for line in lines:
        bbox_m = draw_t.textbbox((0, 0), line, font=font)
        line_w = bbox_m[2] - bbox_m[0]
        x = (w - line_w) // 2 - bbox_m[0]

        # Thin black outline (4 offsets, like the video subtitles)
        for dx in range(-outline_r, outline_r + 1):
            for dy in range(-outline_r, outline_r + 1):
                if dx == 0 and dy == 0:
                    continue
                draw_t.text((x + dx, cy + dy), line, font=font, fill=(*_SUB_OUTLINE, 255))

        # White→cyan vertical gradient fill
        bbox = draw_t.textbbox((x, cy), line, font=font)
        ly0, ly1 = bbox[1], bbox[3]
        lh = max(1, ly1 - ly0)
        grad = Image.new("RGBA", (w, lh), (0, 0, 0, 0))
        for gy in range(lh):
            t = gy / (lh - 1) if lh > 1 else 0
            r = int(_SUB_GRAD_TOP[0] + (_SUB_GRAD_BOTTOM[0] - _SUB_GRAD_TOP[0]) * t)
            g = int(_SUB_GRAD_TOP[1] + (_SUB_GRAD_BOTTOM[1] - _SUB_GRAD_TOP[1]) * t)
            b = int(_SUB_GRAD_TOP[2] + (_SUB_GRAD_BOTTOM[2] - _SUB_GRAD_TOP[2]) * t)
            ImageDraw.Draw(grad).line([(0, gy), (w, gy)], fill=(r, g, b, 255))
        mask_full = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask_full).text((x, cy), line, font=font, fill=255)
        line_mask = mask_full.crop((0, ly0, w, ly1))
        grad.putalpha(line_mask)
        text_layer.paste(grad, (0, ly0), mask=line_mask)
        cy += line_h

    # Italic shear: top of text leans right
    if shear:
        affine = (1, shear, -shear * y_start, 0, 1, 0)
        from PIL import Image as _Im
        text_layer = text_layer.transform(
            (w, h), _Im.AFFINE, affine, resample=_Im.BICUBIC,
        )

    if img.mode != "RGBA":
        img2 = img.convert("RGBA")
        img2.alpha_composite(text_layer)
        img.paste(img2.convert(img.mode))
    else:
        img.alpha_composite(text_layer)


def render_carousel_cover(
    source_image_path: str,
    hook: str,
    out_path: str,
) -> bool:
    """
    Render the first ("cover") slide of an Instagram carousel: 1080×1350,
    big punchy ALL-CAPS hook over the article image (uses the same gradient
    text style as the Reels thumbnail, but in 4:5).
    """
    return _render_thumbnail(source_image_path, hook, out_path, CAR_W, CAR_H)


def render_carousel_image_slide(
    source_image_path: str | None,
    text: str,
    out_path: str,
) -> bool:
    """
    Render a middle slide: article photo as background (or gradient fallback),
    light dark overlay for legibility, subtitle-style text rendered on top.
    """
    try:
        from PIL import Image, ImageDraw
        if source_image_path and os.path.exists(source_image_path):
            bg = _blurred_photo_bg(source_image_path, CAR_W, CAR_H)
        else:
            bg = _gradient_bg(CAR_W, CAR_H, _CAR_GRAD_BG_TOP, _CAR_GRAD_BG_BOTTOM)
        bg = bg.convert("RGBA")

        # Gentle bottom-up overlay (keeps the photo visible, helps the text)
        overlay = Image.new("RGBA", (CAR_W, CAR_H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        band_top = int(CAR_H * 0.45)
        for y in range(band_top, CAR_H):
            a = int(140 * (y - band_top) / (CAR_H - band_top))
            od.line([(0, y), (CAR_W, y)], fill=(0, 0, 0, a))
        bg.alpha_composite(overlay)

        # Accent bar above text
        bar_y = int(CAR_H * 0.62)
        ImageDraw.Draw(bg).rectangle(
            [int(CAR_W * 0.08), bar_y, int(CAR_W * 0.08) + 80, bar_y + 8],
            fill=(*_CAR_ACCENT, 255),
        )

        _draw_subtitle_style_text(
            bg, text,
            max_lines=6,
            font_size=int(CAR_W * 0.072),
            v_align="bottom",
            padding_pct=0.09,
        )
        bg.convert("RGB").save(out_path, "JPEG", quality=92)
        return True
    except Exception as exc:
        logger.warning("Carousel image slide render failed: %s", exc)
        return False


def render_carousel_text_slide(
    text: str,
    out_path: str,
    *,
    bg_image: str | None = None,
) -> bool:
    """Render a centred-text slide. Uses an article photo as background when
    *bg_image* is provided (lightly blurred), otherwise a brand gradient.
    Text is drawn in the same subtitle style used in generated videos.
    """

    try:
        from PIL import Image, ImageDraw
        if bg_image and os.path.exists(bg_image):
            bg = _blurred_photo_bg(bg_image, CAR_W, CAR_H).convert("RGBA")
            # Mild dark wash so the centred text stays legible
            wash = Image.new("RGBA", (CAR_W, CAR_H), (0, 0, 0, 90))
            bg.alpha_composite(wash)
        else:
            bg = _gradient_bg(CAR_W, CAR_H, _CAR_GRAD_BG_TOP, _CAR_GRAD_BG_BOTTOM).convert("RGBA")
        _draw_subtitle_style_text(
            bg, text,
            max_lines=8,
            font_size=int(CAR_W * 0.082),
            v_align="middle",
            padding_pct=0.10,
        )
        bg.convert("RGB").save(out_path, "JPEG", quality=92)
        return True
    except Exception as exc:
        logger.warning("Carousel text slide render failed: %s", exc)
        return False


def render_carousel_video_overlay(
    text: str,
    out_path: str,
) -> bool:
    """
    Render a TRANSPARENT 1080×1350 PNG that contains only the bottom dark
    gradient + accent bar + subtitle-styled text. Used as an ffmpeg overlay
    on top of a video clip so the rest of the frame stays visible.
    """
    try:
        from PIL import Image, ImageDraw
        overlay = Image.new("RGBA", (CAR_W, CAR_H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        band_top = int(CAR_H * 0.45)
        for y in range(band_top, CAR_H):
            a = int(155 * (y - band_top) / (CAR_H - band_top))
            od.line([(0, y), (CAR_W, y)], fill=(0, 0, 0, a))
        bar_y = int(CAR_H * 0.62)
        ImageDraw.Draw(overlay).rectangle(
            [int(CAR_W * 0.08), bar_y, int(CAR_W * 0.08) + 80, bar_y + 8],
            fill=(*_CAR_ACCENT, 255),
        )
        _draw_subtitle_style_text(
            overlay, text,
            max_lines=6,
            font_size=int(CAR_W * 0.072),
            v_align="bottom",
            padding_pct=0.09,
        )
        overlay.save(out_path, "PNG")
        return True
    except Exception as exc:
        logger.warning("Carousel video overlay render failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI test — run as:  python thumbnail_generator.py <post_id>
#                 or: python thumbnail_generator.py <image_path> "Title text"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) == 3 and os.path.exists(sys.argv[1]):
        # Direct mode: python thumbnail_generator.py path/to/image.jpg "My Title"
        source    = sys.argv[1]
        title_ru  = sys.argv[2]
        title_en  = None  # no translation in direct mode
    elif len(sys.argv) == 2 and sys.argv[1].isdigit():
        # Database mode: python thumbnail_generator.py <post_id>
        sys.path.insert(0, os.path.dirname(__file__))
        import database as db
        import ai_adapter
        post = db.get_scheduled_post(int(sys.argv[1]))
        if not post:
            print(f"Post #{sys.argv[1]} not found in database.")
            sys.exit(1)
        images = [p for p in post.get("image_paths", []) if os.path.exists(p)]
        if not images:
            print(f"Post #{sys.argv[1]} has no local images.")
            sys.exit(1)
        source   = images[0]
        title_ru = post.get("article_title", "No title")
        print(f"Post #{sys.argv[1]}: {title_ru}")
        print(f"Source image: {source}")
        print("Translating title to English via Ollama...")
        title_en = asyncio.run(ai_adapter.translate_title_to_english(title_ru))
        print(f"EN title: {title_en}")
    else:
        print("Usage:")
        print("  python thumbnail_generator.py <post_id>")
        print("  python thumbnail_generator.py <image_path> \"Title text\"")
        sys.exit(1)

    import subprocess, tempfile

    def _open_preview(path: str) -> None:
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception:
            pass

    _tmp = tempfile.gettempdir()

    # Generate AI hooks
    sys.path.insert(0, os.path.dirname(__file__))
    import ai_adapter as _ai  # noqa: F811 (may already be imported)
    hook_ru = asyncio.run(_ai.generate_thumbnail_hook(title_ru, lang="ru"))
    print(f"RU hook: {hook_ru}")

    # --- RU thumbnail (IG 1080×1920) ---
    out_ru_ig = os.path.join(_tmp, "thumb_test_ru_ig.jpg")
    if generate_instagram_thumbnail(source, hook_ru, out_ru_ig):
        print(f"RU Instagram thumbnail (1080×1920): {out_ru_ig}")
        _open_preview(out_ru_ig)
    else:
        print("RU Instagram thumbnail FAILED")

    # --- EN thumbnail (only when title_en is available) ---
    if title_en:
        hook_en = asyncio.run(_ai.generate_thumbnail_hook(title_en, lang="en"))
        print(f"EN hook: {hook_en}")
        out_en_ig = os.path.join(_tmp, "thumb_test_en_ig.jpg")
        if generate_instagram_thumbnail(source, hook_en, out_en_ig):
            print(f"EN Instagram thumbnail (1080×1920): {out_en_ig}")
            _open_preview(out_en_ig)
        else:
            print("EN Instagram thumbnail FAILED")
