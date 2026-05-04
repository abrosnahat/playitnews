"""
Upload local media files to a GitHub repo via the Contents API and return a
raw.githubusercontent.com URL that Instagram (Meta) can fetch directly.

Why not Release assets? GitHub release downloads redirect to
`release-assets.githubusercontent.com`, whose /robots.txt returns an HTML
404 — Meta's crawler interprets that as "restricted by robots.txt" and
refuses to download. raw.githubusercontent.com avoids that path.

Auth : GITHUB_MEDIA_TOKEN (classic PAT with `repo` scope)
Repo : GITHUB_MEDIA_REPO  (e.g. user/playitnews-media — must be public)
Files are written to a `media/` directory at the repo root with unique
filenames so concurrent uploads don't clash. Files are deleted after a
successful publish if GITHUB_MEDIA_DELETE_AFTER_PUBLISH=1.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import secrets
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

import certifi
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

_API = "https://api.github.com"
# We serve files via jsDelivr instead of raw.githubusercontent.com because
# raw.github always returns `Content-Type: application/octet-stream`, which
# triggers Meta's small "non-media payload" limit (~10 MB → HTTP 413 when
# Instagram tries to ingest the video). jsDelivr proxies the same content
# but sets `Content-Type: video/mp4` + `Content-Length`, so Meta accepts it.
_RAW = "https://cdn.jsdelivr.net/gh"
_MEDIA_DIR = "media"  # path inside the repo where files are stored


def _cfg() -> tuple[str, str]:
    token = os.getenv("GITHUB_MEDIA_TOKEN", "").strip()
    repo  = os.getenv("GITHUB_MEDIA_REPO", "").strip()
    if not token or not repo:
        raise RuntimeError(
            "GitHub uploader not configured. Set GITHUB_MEDIA_TOKEN and "
            "GITHUB_MEDIA_REPO in .env (repo must be public)."
        )
    return token, repo


def _request(method: str, url: str, *, token: str, data: bytes | None = None,
             headers: dict | None = None, timeout: int = 120,
             retries: int = 6) -> tuple[int, bytes]:
    h = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "playitnews-media-uploader",
    }
    if headers:
        h.update(headers)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            # Retry on transient errors and on 409 (concurrent commit race
            # on the branch tip — happens when EN and RU publish in parallel
            # and both PUT to /contents/ at the same time).
            if exc.code in (409, 403, 429, 500, 502, 503, 504) and attempt < retries:
                wait = min(2 ** attempt, 30) + random.uniform(0, 2)
                logger.warning("GitHub %s %s: HTTP %s — retry %d/%d in %.1fs",
                               method, url, exc.code, attempt, retries, wait)
                time.sleep(wait)
                continue
            return exc.code, exc.read()
        except (urllib.error.URLError, ssl.SSLError, ConnectionError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries:
                wait = min(2 ** attempt, 30)
                logger.warning("GitHub %s %s: %s — retry %d/%d in %ds",
                               method, url, exc, attempt, retries, wait)
                time.sleep(wait)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("GitHub request failed without exception")


def _default_branch(token: str, repo: str) -> str:
    """Return the repo's default branch (creating it via README if empty)."""
    code, body = _request("GET", f"{_API}/repos/{repo}", token=token)
    if code != 200:
        raise RuntimeError(f"GitHub repo lookup failed ({code}): {body[:300]!r}")
    info = json.loads(body)
    branch = info.get("default_branch")
    if branch:
        return branch

    # Empty repo — create README.md so a default branch exists.
    payload = json.dumps({
        "message": "Initial commit",
        "content": base64.b64encode(
            b"# playitnews-media\n\nAuto-uploaded media used as a CDN.\n"
        ).decode(),
    }).encode()
    code, body = _request(
        "PUT", f"{_API}/repos/{repo}/contents/README.md",
        token=token, data=payload,
        headers={"Content-Type": "application/json"},
    )
    if code not in (200, 201):
        raise RuntimeError(f"GitHub bootstrap repo failed ({code}): {body[:300]!r}")
    # Refresh repo info to learn the (now-existing) default branch.
    code, body = _request("GET", f"{_API}/repos/{repo}", token=token)
    info = json.loads(body) if code == 200 else {}
    return info.get("default_branch", "main")


def upload(local_path: str) -> tuple[str, str]:
    """
    Upload *local_path* to the repo. Returns (raw_public_url, repo_path).

    raw_public_url is served by raw.githubusercontent.com — Meta accepts it.
    repo_path is the path inside the repo (used by `delete()`).
    """
    if not os.path.isfile(local_path):
        raise RuntimeError(f"File not found: {local_path}")
    token, repo = _cfg()
    branch = _default_branch(token, repo)

    base = os.path.basename(local_path)
    stem, ext = os.path.splitext(base)
    repo_path = f"{_MEDIA_DIR}/{stem}_{secrets.token_hex(4)}{ext}"

    with open(local_path, "rb") as f:
        body = f.read()
    payload = json.dumps({
        "message": f"Upload {repo_path}",
        "content": base64.b64encode(body).decode(),
        "branch": branch,
    }).encode()
    api_path = urllib.parse.quote(repo_path)
    code, resp = _request(
        "PUT", f"{_API}/repos/{repo}/contents/{api_path}",
        token=token, data=payload,
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    if code not in (200, 201):
        raise RuntimeError(f"GitHub upload failed ({code}): {resp[:300]!r}")

    # jsDelivr URL format: https://cdn.jsdelivr.net/gh/{repo}@{branch}/{path}
    raw_url = f"{_RAW}/{repo}@{branch}/{repo_path}"
    logger.info("Uploaded %s → %s", base, raw_url)
    return raw_url, repo_path


def delete(repo_path: str) -> None:
    """Best-effort delete of an uploaded file (errors are swallowed)."""
    try:
        token, repo = _cfg()
        branch = _default_branch(token, repo)
        api_path = urllib.parse.quote(repo_path)

        # Need the file SHA to delete via Contents API.
        code, body = _request(
            "GET", f"{_API}/repos/{repo}/contents/{api_path}?ref={branch}",
            token=token,
        )
        if code != 200:
            logger.warning("GitHub delete: SHA lookup %s failed (%s)", repo_path, code)
            return
        sha = json.loads(body)["sha"]

        payload = json.dumps({
            "message": f"Delete {repo_path}",
            "sha": sha,
            "branch": branch,
        }).encode()
        code, body = _request(
            "DELETE", f"{_API}/repos/{repo}/contents/{api_path}",
            token=token, data=payload,
            headers={"Content-Type": "application/json"},
        )
        if code not in (200, 204):
            logger.warning("GitHub delete %s failed (%s): %s", repo_path, code, body[:200])
    except Exception as exc:
        logger.warning("GitHub delete %s exception: %s", repo_path, exc)
