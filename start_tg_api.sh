#!/bin/bash
# Launch a local telegram-bot-api server so the bot can upload files >50 MB.
#
# One-time setup:
#   1) brew install telegram-bot-api      (macOS)
#   2) Get api_id / api_hash on https://my.telegram.org/apps
#      and put them into .env as TELEGRAM_API_ID / TELEGRAM_API_HASH.
#   3) Switch the bot from cloud to local (run ONCE):
#        curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/logOut"
#   4) Set TELEGRAM_LOCAL_API_URL=http://127.0.0.1:8088 in .env
#
# After that, ./start.sh starts this server automatically.

set -e
cd "$(dirname "$0")"

# Read a single key from .env without sourcing it
# (some values contain ';' / spaces that break `source`).
read_env() {
  local key="$1"
  [ -f .env ] || return 0
  awk -F= -v k="$key" '
    /^[[:space:]]*#/ || NF<2 { next }
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      if ($1 != k) next
      val = $0
      sub(/^[^=]*=/, "", val)
      sub(/^"/, "", val); sub(/"$/, "", val)
      sub(/^\x27/, "", val); sub(/\x27$/, "", val)
      print val
      exit
    }
  ' .env
}

TELEGRAM_API_ID="${TELEGRAM_API_ID:-$(read_env TELEGRAM_API_ID)}"
TELEGRAM_API_HASH="${TELEGRAM_API_HASH:-$(read_env TELEGRAM_API_HASH)}"
TELEGRAM_LOCAL_API_PORT="${TELEGRAM_LOCAL_API_PORT:-$(read_env TELEGRAM_LOCAL_API_PORT)}"
TELEGRAM_LOCAL_API_DIR="${TELEGRAM_LOCAL_API_DIR:-$(read_env TELEGRAM_LOCAL_API_DIR)}"

if ! command -v telegram-bot-api >/dev/null 2>&1; then
  echo "❌ telegram-bot-api not found. Install: brew install telegram-bot-api"
  exit 1
fi

if [ -z "$TELEGRAM_API_ID" ] || [ -z "$TELEGRAM_API_HASH" ]; then
  echo "❌ TELEGRAM_API_ID / TELEGRAM_API_HASH are not set in .env"
  echo "   Create them at https://my.telegram.org/apps"
  exit 1
fi

PORT="${TELEGRAM_LOCAL_API_PORT:-8088}"
WORKDIR="${TELEGRAM_LOCAL_API_DIR:-/tmp/tgbotapi}"
mkdir -p "$WORKDIR"

echo "Starting telegram-bot-api on http://127.0.0.1:${PORT} (data: ${WORKDIR})"
exec telegram-bot-api \
  --local \
  --api-id="$TELEGRAM_API_ID" \
  --api-hash="$TELEGRAM_API_HASH" \
  --http-port="$PORT" \
  --dir="$WORKDIR" \
  --temp-dir="$WORKDIR/tmp" \
  --verbosity=1
