#!/usr/bin/env python3
"""
One-shot script to obtain YouTube OAuth2 token and save it to youtube_token.json.

Steps:
  1. Go to console.cloud.google.com → New Project
  2. APIs & Services → Enable APIs → YouTube Data API v3
  3. OAuth consent screen → External → add yourself as Test User
  4. Credentials → Create OAuth 2.0 Client ID → Desktop App
  5. Download JSON → save as client_secrets.json in this folder
  6. Run: python get_youtube_token.py
  7. Browser opens → log in → approve → token saved to youtube_token.json
"""
import os
import sys

SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS", os.path.join(os.path.dirname(__file__), "client_secrets.json"))
TOKEN_FILE   = os.getenv("YOUTUBE_TOKEN_FILE",    os.path.join(os.path.dirname(__file__), "youtube_token.json"))
SCOPES       = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.oauth2.credentials import Credentials
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

    print("Opening browser for YouTube authorization...")
    flow = InstalledAppFlow.from_client_secrets_file(SECRETS_FILE, SCOPES)
    creds = flow.run_local_server(port=8081, prompt="consent")

    # Save token
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"\nToken saved to: {TOKEN_FILE}")
    print("Add to .env (optional, defaults already set):")
    print(f"  YOUTUBE_CLIENT_SECRETS={SECRETS_FILE}")
    print(f"  YOUTUBE_TOKEN_FILE={TOKEN_FILE}")
    print("\nToken auto-refreshes — no need to re-run unless you revoke access.")


if __name__ == "__main__":
    main()
