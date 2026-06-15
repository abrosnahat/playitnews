#!/usr/bin/env python3
"""
One-shot script to obtain a VK user access token with the `video` scope.

Зачем: метод video.save в публичном VK API недоступен для обычных
приложений/сообществ (ошибки 1051 / 27). Рабочий путь — пользовательский
токен, полученный через client_id официального приложения VK (Kate Mobile),
у которого право `video` уже выдано.

Как пользоваться:
  1. Запусти:  python get_vk_token.py
  2. Откроется браузер на oauth.vk.com — залогинься под нужным аккаунтом
     (тем, что админит сообщество) и нажми «Разрешить».
  3. Тебя перебросит на страницу вида
        https://oauth.vk.com/blank.html#access_token=vk1.a.XXXX&expires_in=0&user_id=12345
     Скопируй ВЕСЬ адрес из адресной строки и вставь его в терминал.
  4. Скрипт распарсит токен и подскажет, что положить в .env:
        VK_ACCESS_TOKEN=vk1.a.XXXX
        VK_GROUP_ID=<id_сообщества_без_минуса>   (если публикуешь в группу)

Примечание: scope=offline делает токен бессрочным (expires_in=0).
"""
import os
import urllib.parse
import webbrowser

# client_id официального приложения Kate Mobile (право `video` уже есть).
# Можно переопределить своим через переменную окружения VK_APP_ID.
VK_APP_ID = os.getenv("VK_APP_ID", "2685278")

# Запрашиваемые права: video — загрузка видео, offline — бессрочный токен,
# groups/wall — чтобы видеть сообщества и (опц.) постить на стену.
SCOPE = "video,offline,groups,wall"
API_VERSION = os.getenv("VK_API_VERSION", "5.199")
REDIRECT_URI = "https://oauth.vk.com/blank.html"

AUTH_URL = (
    "https://oauth.vk.com/authorize"
    f"?client_id={VK_APP_ID}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT_URI, safe='')}"
    "&display=page"
    f"&scope={SCOPE}"
    "&response_type=token"
    f"&v={API_VERSION}"
)


def _parse_token(raw: str) -> tuple[str, str]:
    """
    Достать access_token и user_id из вставленного адреса или фрагмента.
    Принимает как полный URL (...#access_token=...), так и просто строку
    'access_token=...&user_id=...' или сам токен.
    """
    raw = raw.strip()
    # Если вставлен полный URL — берём часть после '#'.
    if "#" in raw:
        raw = raw.split("#", 1)[1]
    elif raw.startswith("http"):
        # На случай, если параметры оказались в query (?...).
        parsed = urllib.parse.urlparse(raw)
        raw = parsed.fragment or parsed.query

    if "access_token=" in raw:
        params = urllib.parse.parse_qs(raw)
        token = params.get("access_token", [""])[0]
        user_id = params.get("user_id", [""])[0]
        return token, user_id

    # Иначе считаем, что вставили сам токен.
    return raw, ""


def main() -> None:
    print("=" * 70)
    print("  Получение VK-токена (scope: video, offline)")
    print("=" * 70)
    print(f"\nclient_id (Kate Mobile): {VK_APP_ID}")
    print("Открываю браузер. Залогинься под нужным аккаунтом и нажми «Разрешить».\n")
    print("Если браузер не открылся — перейди вручную по ссылке:\n")
    print(AUTH_URL + "\n")

    try:
        webbrowser.open(AUTH_URL)
    except Exception:
        pass

    print("После авторизации тебя перебросит на пустую страницу")
    print("  https://oauth.vk.com/blank.html#access_token=...")
    print("Скопируй ВЕСЬ адрес из адресной строки и вставь сюда.\n")

    raw = input("Вставь адрес (или сам токен): ").strip()
    if not raw:
        print("ERROR: ничего не введено.")
        raise SystemExit(1)

    token, user_id = _parse_token(raw)
    if not token or not token.startswith(("vk1.", "vk2.")) and len(token) < 40:
        print("ERROR: не удалось распознать токен. Проверь, что вставил полный адрес.")
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("  Токен получен. Добавь в .env:")
    print("=" * 70)
    print(f"\nVK_ACCESS_TOKEN={token}")
    if user_id:
        print(f"# user_id={user_id}")
    print("VK_GROUP_ID=<числовой ID сообщества без минуса, если публикуешь в группу>\n")
    print("Затем перезапусти приложение (.\\start.ps1).")


if __name__ == "__main__":
    main()
