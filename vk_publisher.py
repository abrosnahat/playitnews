"""
VK video publisher via VK API (поддержка вертикальных коротких видео / VK Клипы).

Только канал.

Поток публикации (2 шага, см. https://dev.vk.com/ru/api/upload/video-in-profile):
  1. video.save  -> возвращает upload_url (+ owner_id, video_id, access_key).
  2. POST на upload_url с полем `video_file` (multipart/form-data) — заливаем сам файл.

После загрузки видео проходит обработку на стороне VK и появляется в списке
видео сообщества/профиля. Вертикальные короткие ролики VK автоматически
относит к разделу «Клипы» по формату — отдельного публичного метода
«создать клип» в API нет.

Важные ограничения VK API:
  - Право доступа `video` выдаётся в исключительных случаях по запросу в
    поддержку (devsupport@corp.vk.com). Без него video.save вернёт ошибку.
  - Не более 5 000 вызовов video.save в сутки на приложение.
  - Допустимые форматы: AVI, MP4, 3GP, MPEG, MOV, FLV, WMV.

Требования:
  - pip install aiohttp certifi
  - VK access token (пользователя или сообщества) со scope `video`.

Переменные окружения (см. config.py):
  VK_ACCESS_TOKEN — токен для сообщества/профиля
  VK_GROUP_ID     — числовой ID сообщества (без минуса); если пусто — грузим в профиль токена
  VK_API_VERSION      — версия API (по умолчанию 5.199)
"""
import asyncio
import logging
import os
import ssl

import aiohttp
import certifi

from config import (
    VK_ACCESS_TOKEN,
    VK_API_VERSION,
    VK_GROUP_ID,
    VK_WALLPOST,
)

logger = logging.getLogger(__name__)

VK_API_BASE = "https://api.vk.com/method"
_UPLOAD_TIMEOUT = 600  # секунд на саму загрузку файла (большие видео)


def is_configured() -> bool:
    """True, если задан токен VK."""
    return bool(VK_ACCESS_TOKEN)


def _make_session() -> aiohttp.ClientSession:
    """aiohttp-сессия с certifi SSL-контекстом."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ctx)
    return aiohttp.ClientSession(connector=connector)


async def _vk_call(session: aiohttp.ClientSession, method: str, params: dict) -> dict:
    """
    Вызвать метод VK API. Возвращает содержимое поля `response`.
    Бросает RuntimeError при ошибке VK (error) или HTTP-ошибке.
    """
    payload = {**params, "v": VK_API_VERSION}
    url = f"{VK_API_BASE}/{method}"
    async with session.post(url, data=payload) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"VK {method} HTTP {resp.status}: {text[:300]}")
        try:
            data = await resp.json(content_type=None)
        except Exception as exc:
            raise RuntimeError(f"VK {method}: invalid JSON: {text[:300]}") from exc

    if "error" in data:
        err = data["error"]
        code = err.get("error_code")
        msg = err.get("error_msg", "unknown error")
        hint = ""
        if code == 5:
            hint = (" — токен недействителен/просрочен. Обновите VK_ACCESS_TOKEN "
                    "(нужен scope video).")
        elif code == 1051:
            hint = (" — video.save недоступен для этого профиля. Используйте токен "
                    "СООБЩЕСТВА и задайте VK_GROUP_ID (грузим видео в группу, а не в профиль).")
        raise RuntimeError(f"VK {method} error {code}: {msg}{hint}")
    return data.get("response", {})


def _build_video_url(owner_id: int, video_id: int) -> str:
    """Публичная ссылка на загруженное видео VK."""
    return f"https://vk.com/video{owner_id}_{video_id}"


async def upload_video(
    video_path: str,
    title: str,
    description: str = "",
    *,
    wallpost: bool | None = None,
) -> str:
    """
    Загрузить *video_path* в сообщество VK (если задан VK_GROUP_ID),
    иначе в профиль владельца токена.

    Если *wallpost* (или VK_WALLPOST по умолчанию) и задан VK_GROUP_ID — после
    загрузки публикует запись с клипом на стене сообщества.

    Возвращает публичную ссылку на видео (https://vk.com/video{owner}_{id}).
    Бросает RuntimeError при ошибке.
    """
    tok = VK_ACCESS_TOKEN
    grp = VK_GROUP_ID
    do_wallpost = VK_WALLPOST if wallpost is None else wallpost
    if not tok:
        raise RuntimeError(
            "VK credentials not configured. Set VK_ACCESS_TOKEN in .env"
        )
    if not os.path.exists(video_path):
        raise RuntimeError(f"Video file not found: {video_path}")

    save_params: dict = {
        "access_token": tok,
        "name": (title or "")[:128],
        "description": (description or "")[:5000],
        "wallpost": 0,
    }
    if grp:
        # group_id передаётся положительным числом (без минуса).
        save_params["group_id"] = str(grp).lstrip("-")

    async with _make_session() as session:
        # Шаг 1: получить upload_url.
        logger.info("VK video.save: запрашиваю upload_url для «%s»…", title[:60])
        save_resp = await _vk_call(session, "video.save", save_params)
        upload_url = save_resp.get("upload_url")
        owner_id = save_resp.get("owner_id")
        video_id = save_resp.get("video_id")
        if not upload_url:
            raise RuntimeError(f"VK video.save: нет upload_url в ответе: {save_resp}")

        # Шаг 2: залить файл на upload_url полем video_file.
        logger.info("VK: загружаю видеофайл (%.2f MB)…",
                    os.path.getsize(video_path) / (1024 * 1024))
        with open(video_path, "rb") as fh:
            form = aiohttp.FormData()
            form.add_field(
                "video_file",
                fh,
                filename=os.path.basename(video_path),
                content_type="video/mp4",
            )
            timeout = aiohttp.ClientTimeout(total=_UPLOAD_TIMEOUT)
            async with session.post(upload_url, data=form, timeout=timeout) as up:
                up_text = await up.text()
                if up.status != 200:
                    raise RuntimeError(f"VK upload HTTP {up.status}: {up_text[:300]}")
                try:
                    up_data = await up.json(content_type=None)
                except Exception:
                    up_data = {}

        # Сервер загрузки возвращает video_id/owner_id; страхуемся значениями из video.save.
        final_owner = up_data.get("owner_id", owner_id)
        final_video = up_data.get("video_id", video_id)
        url = _build_video_url(int(final_owner), int(final_video))
        logger.info("VK: видео загружено и обрабатывается — %s", url)

        # Шаг 3 (опц.): опубликовать запись с клипом на стене сообщества.
        if do_wallpost and grp:
            await _post_to_wall(
                session,
                token=tok,
                group_id=str(grp).lstrip("-"),
                message=(description or title or ""),
                owner_id=int(final_owner),
                video_id=int(final_video),
            )

        return url


async def _post_to_wall(
    session: aiohttp.ClientSession,
    *,
    token: str,
    group_id: str,
    message: str,
    owner_id: int,
    video_id: int,
) -> None:
    """Опубликовать запись с видео-вложением на стене сообщества."""
    wall_params = {
        "access_token": token,
        "owner_id": f"-{group_id}",          # стена сообщества — отрицательный id
        "from_group": 1,                      # пост от имени сообщества
        "message": (message or "")[:4000],
        "attachments": f"video{owner_id}_{video_id}",
    }
    try:
        resp = await _vk_call(session, "wall.post", wall_params)
        post_id = resp.get("post_id")
        logger.info("VK: запись опубликована на стене сообщества (post_id=%s)", post_id)
    except Exception as exc:
        # Не рвём публикацию из-за неудавшегося поста — видео уже загружено.
        logger.warning("VK: не удалось опубликовать запись на стене: %s", exc)
