#!/usr/bin/env python3
"""
One-shot script to obtain a long-lived Instagram access token.

Steps:
  1. Add  http://localhost:8080/callback  to your Meta App's
     Valid OAuth Redirect URIs (App Dashboard → Instagram → API Setup)
  2. Run:  python get_instagram_token.py
  3. A browser window opens — log in and approve.
  4. The script prints INSTAGRAM_USER_ID and INSTAGRAM_ACCESS_TOKEN for .env

Required env vars (or edit the constants below):
  APP_ID, APP_SECRET
"""
import http.server
import os
import threading
import urllib.parse
import urllib.request
import webbrowser
import json

# ── Edit these ──────────────────────────────────────────────────────────────
APP_ID     = os.getenv("INSTAGRAM_APP_ID", "")
APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET", "")
# ────────────────────────────────────────────────────────────────────────────

REDIRECT_URI = "http://localhost:8080/callback"
SCOPE        = "instagram_business_basic,instagram_business_content_publish"
AUTH_URL     = (
    f"https://api.instagram.com/oauth/authorize"
    f"?client_id={APP_ID}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT_URI, safe='')}"
    f"&scope={SCOPE}"
    f"&response_type=code"
)

_code_holder: list[str] = []
_server_done = threading.Event()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            _code_holder.append(params["code"][0].rstrip("#_"))
            body = b"<h2>Authorization successful! You can close this tab.</h2>"
        else:
            error = params.get("error_description", ["Unknown error"])[0]
            body = f"<h2>Error: {error}</h2>".encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)
        _server_done.set()

    def log_message(self, *args):
        pass  # silence request logs


def _exchange_code(code: str) -> tuple[str, str]:
    """Exchange authorization code for short-lived token."""
    data = urllib.parse.urlencode({
        "client_id":     APP_ID,
        "client_secret": APP_SECRET,
        "grant_type":    "authorization_code",
        "redirect_uri":  REDIRECT_URI,
        "code":          code,
    }).encode()
    req = urllib.request.Request("https://api.instagram.com/oauth/access_token", data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["access_token"], str(result["user_id"])


def _exchange_long_lived(short_token: str) -> str:
    """Exchange short-lived token for a long-lived one (60 days)."""
    url = (
        f"https://graph.instagram.com/access_token"
        f"?grant_type=ig_exchange_token"
        f"&client_secret={APP_SECRET}"
        f"&access_token={short_token}"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["access_token"]


def main():
    if not APP_ID or not APP_SECRET:
        print("ERROR: Set INSTAGRAM_APP_ID and INSTAGRAM_APP_SECRET env vars, or edit this script.")
        raise SystemExit(1)

    # Start local callback server
    server = http.server.HTTPServer(("localhost", 8080), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Opening browser for Instagram authorization...")
    print(f"(Make sure  {REDIRECT_URI}  is added to Valid OAuth Redirect URIs in Meta App Dashboard)\n")
    webbrowser.open(AUTH_URL)

    _server_done.wait(timeout=120)
    server.shutdown()

    if not _code_holder:
        print("ERROR: No authorization code received (timed out or user denied).")
        raise SystemExit(1)

    code = _code_holder[0]
    print("Authorization code received. Exchanging for token...")

    short_token, user_id = _exchange_code(code)
    print("Short-lived token obtained. Getting long-lived token...")

    long_token = _exchange_long_lived(short_token)

    print("\n" + "="*60)
    print("Add these to your .env file:")
    print("="*60)
    print(f"INSTAGRAM_USER_ID={user_id}")
    print(f"INSTAGRAM_ACCESS_TOKEN={long_token}")
    print("="*60)
    print("\nToken expires in ~60 days. Refresh with:")
    print(f"  curl 'https://graph.instagram.com/refresh_access_token"
          f"?grant_type=ig_refresh_token&access_token=YOUR_TOKEN'")


if __name__ == "__main__":
    main()
