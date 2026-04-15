"""
Thumbnail generator: overlays article title text on the first article image.

Creates a single 1080×1920 JPEG used for both YouTube Shorts and Instagram Reels covers.
Uses Pillow — same dependency already used by video_generator.
"""
import os
import logging
import textwrap

logger = logging.getLogger(__name__)

# Thumbnail size: 1080×1920 (9:16) — used for both YouTube Shorts and Instagram Reels
THUMB_W, THUMB_H = 1080, 1920

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
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


def generate_thumbnail(
    source_image_path: str,
    title: str,
    out_path: str,
) -> bool:
    """Generate a 1080×1920 thumbnail for YouTube Shorts / Instagram Reels. Returns True on success."""
    return _render_thumbnail(source_image_path, title, out_path, THUMB_W, THUMB_H)


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

    import subprocess

    # Generate AI hooks
    sys.path.insert(0, os.path.dirname(__file__))
    import ai_adapter as _ai  # noqa: F811 (may already be imported)
    hook_ru = asyncio.run(_ai.generate_thumbnail_hook(title_ru, lang="ru"))
    print(f"RU hook: {hook_ru}")

    # --- RU thumbnail ---
    out_ru = "/tmp/thumb_test_ru.jpg"
    if generate_thumbnail(source, hook_ru, out_ru):
        print(f"RU thumbnail (1080×1920): {out_ru}")
        subprocess.run(["open", out_ru])
    else:
        print("RU thumbnail FAILED")

    # --- EN thumbnail (only when title_en is available) ---
    if title_en:
        hook_en = asyncio.run(_ai.generate_thumbnail_hook(title_en, lang="en"))
        print(f"EN hook: {hook_en}")
        out_en = "/tmp/thumb_test_en.jpg"
        if generate_thumbnail(source, hook_en, out_en):
            print(f"EN thumbnail (1080×1920): {out_en}")
            subprocess.run(["open", out_en])
        else:
            print("EN thumbnail FAILED")
