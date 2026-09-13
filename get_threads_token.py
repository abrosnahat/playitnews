#!/usr/bin/env python3
"""
One-shot script to obtain a long-lived Threads access token.

Steps:
  1. Create a Meta app with the "Threads" use case:
     https://developers.facebook.com/documentation/development/create-an-app/threads-use-case
     (App Dashboard shows a separate Threads App ID / App Secret — use those,
     not the main Facebook app ID/secret.)
  2. Add  http://localhost:8081/callback  to the app's Threads product
     "Redirect Callback URLs".
  3. Invite yourself as a Threads Tester (App Dashboard → App roles → Roles
     → Add People → Threads Tester) and accept the invite at
     https://www.threads.net/settings/account → Website permissions.
  4. Run:  python get_threads_token.py
  5. A browser window opens — log in and approve.
  6. The script prints THREADS_USER_ID and THREADS_ACCESS_TOKEN for .env.

Required env vars (or edit the constants below):
  THREADS_APP_ID, THREADS_APP_SECRET
"""
import http.server
import json
import os
import threading
import urllib.parse
import urllib.request
import webbrowser

# ── Edit these ──────────────────────────────────────────────────────────────
APP_ID     = os.getenv("THREADS_APP_ID", "")
APP_SECRET = os.getenv("THREADS_APP_SECRET", "")
# ────────────────────────────────────────────────────────────────────────────

REDIRECT_URI = "http://localhost:8081/callback"
SCOPE        = "threads_basic,threads_content_publish"
AUTH_URL     = (
    f"https://threads.net/oauth/authorize"
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
    """Exchange authorization code for a short-lived Threads user access token."""
    data = urllib.parse.urlencode({
        "client_id":     APP_ID,
        "client_secret": APP_SECRET,
        "grant_type":    "authorization_code",
        "redirect_uri":  REDIRECT_URI,
        "code":          code,
    }).encode()
    req = urllib.request.Request("https://graph.threads.net/oauth/access_token", data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["access_token"], str(result["user_id"])


def _exchange_long_lived(short_token: str) -> str:
    """Exchange short-lived token for a long-lived one (60 days)."""
    url = (
        f"https://graph.threads.net/access_token"
        f"?grant_type=th_exchange_token"
        f"&client_secret={APP_SECRET}"
        f"&access_token={short_token}"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["access_token"]


def main():
    if not APP_ID or not APP_SECRET:
        print("ERROR: Set THREADS_APP_ID and THREADS_APP_SECRET env vars, or edit this script.")
        raise SystemExit(1)

    server = http.server.HTTPServer(("localhost", 8081), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("Opening browser for Threads authorization...")
    print(f"(Make sure  {REDIRECT_URI}  is added to the app's Threads Redirect Callback URLs)\n")
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
    print(f"THREADS_USER_ID={user_id}")
    print(f"THREADS_ACCESS_TOKEN={long_token}")
    print("="*60)
    print("\nToken expires in ~60 days. Refresh (before expiry) with:")
    print("  curl 'https://graph.threads.net/refresh_access_token"
          "?grant_type=th_refresh_token&access_token=YOUR_TOKEN'")


if __name__ == "__main__":
    main()
