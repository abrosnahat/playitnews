#!/usr/bin/env python3
"""
One-shot script to obtain YouTube OAuth2 token.

Usage:
  python get_youtube_token.py        # EN account → youtube_token.json
  python get_youtube_token.py --ru   # RU account → youtube_token_ru.json

Steps:
  1. Go to console.cloud.google.com → New Project
  2. APIs & Services → Enable APIs → YouTube Data API v3
  3. OAuth consent screen → External → Publish App (no expiry) or add Test Users
  4. Credentials → Create OAuth 2.0 Client ID → Desktop App
  5. Download JSON → save as client_secrets.json in this folder
  6. Run the script for each account (see Usage above)
  7. Browser opens → log in with the correct Google account → approve
"""
import os
import sys

BASE_DIR     = os.path.dirname(__file__)
SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS", os.path.join(BASE_DIR, "client_secrets.json"))
SCOPES       = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def main():
    ru = "--ru" in sys.argv

    token_env  = "YOUTUBE_TOKEN_FILE_RU" if ru else "YOUTUBE_TOKEN_FILE"
    token_default = "youtube_token_ru.json" if ru else "youtube_token.json"
    TOKEN_FILE = os.getenv(token_env, os.path.join(BASE_DIR, token_default))

    label = "RU" if ru else "EN"
    port  = 8082 if ru else 8081

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: Install required packages first:")
        print("  .venv/bin/pip install google-auth-oauthlib google-api-python-client")
        sys.exit(1)

    if not os.path.exists(SECRETS_FILE):
        print(f"ERROR: {SECRETS_FILE} not found.")
        print("Download it from Google Cloud Console:")
        print("  APIs & Services → Credentials → OAuth 2.0 Client ID → Download JSON")
        print(f"  Save as: {SECRETS_FILE}")
        sys.exit(1)

    print(f"[{label}] Opening browser for YouTube authorization...")
    print(f"       Make sure to log in with your {label} Google account!\n")

    flow = InstalledAppFlow.from_client_secrets_file(SECRETS_FILE, SCOPES)
    creds = flow.run_local_server(port=port, prompt="consent")

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"\n[{label}] Token saved to: {TOKEN_FILE}")
    print(f"\nToken auto-refreshes — no need to re-run unless you revoke access.")


if __name__ == "__main__":
    main()
