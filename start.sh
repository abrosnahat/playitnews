#!/bin/bash
# Start Telegram bot, web dashboard, and optionally ngrok
VENV=".venv/bin/python"
cd "$(dirname "$0")"

# Read a single key from .env without sourcing it
# (the file may contain values with ';' / spaces that break `source`).
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
      # strip optional surrounding quotes
      sub(/^"/, "", val); sub(/"$/, "", val)
      sub(/^\x27/, "", val); sub(/\x27$/, "", val)
      print val
      exit
    }
  ' .env
}

TELEGRAM_LOCAL_API_URL="$(read_env TELEGRAM_LOCAL_API_URL)"
TELEGRAM_LOCAL_API_PORT="$(read_env TELEGRAM_LOCAL_API_PORT)"

# Kill any leftover instances from previous runs to avoid port/getUpdates conflicts
pkill -f "$PWD/webapp.py" 2>/dev/null
pkill -f "$PWD/main.py" 2>/dev/null
pkill -f "[Pp]ython.*[/ ]webapp\.py( |$)" 2>/dev/null
pkill -f "[Pp]ython.*[/ ]main\.py( |$)" 2>/dev/null
PORT_PID=$(lsof -tnP -i :5003 2>/dev/null)
[ -n "$PORT_PID" ] && kill "$PORT_PID" 2>/dev/null
pkill -f "cloudflared tunnel" 2>/dev/null
pkill -f "telegram-bot-api" 2>/dev/null
sleep 1

# --- Optional: local Bot API server (raises 50 MB upload cap to 2 GB) ---
TG_API_PID=""
if [ -n "$TELEGRAM_LOCAL_API_URL" ]; then
  PORT="${TELEGRAM_LOCAL_API_PORT:-8088}"
  # If the port is already answering (e.g. Docker container is running) — just use it.
  if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    echo "  ✅ telegram-bot-api already running on :$PORT (external/Docker)"
  elif command -v telegram-bot-api >/dev/null 2>&1; then
    echo "Starting local telegram-bot-api server..."
    ./start_tg_api.sh > /tmp/tgbotapi_playitnews.log 2>&1 &
    TG_API_PID=$!
    READY=0
    for i in $(seq 1 30); do
      if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
        echo "  ✅ telegram-bot-api ready on :$PORT"
        READY=1
        break
      fi
      sleep 0.5
    done
    if [ "$READY" != "1" ]; then
      echo "  ❌ telegram-bot-api did not start. Last log lines:"
      tail -n 20 /tmp/tgbotapi_playitnews.log 2>/dev/null
      echo "Aborting — bot would fail to connect."
      kill $TG_API_PID 2>/dev/null
      exit 1
    fi
  else
    echo "❌ TELEGRAM_LOCAL_API_URL is set but nothing is listening on :$PORT"
    echo "   and the telegram-bot-api binary is missing."
    echo "   Either:"
    echo "     • run the Docker image (recommended):"
    echo "         docker run -d --name tg-bot-api --restart unless-stopped \\"
    echo "           -p 127.0.0.1:${PORT}:8081 \\"
    echo "           -e TELEGRAM_API_ID=\$TELEGRAM_API_ID \\"
    echo "           -e TELEGRAM_API_HASH=\$TELEGRAM_API_HASH \\"
    echo "           -e TELEGRAM_LOCAL=1 \\"
    echo "           -v \$HOME/.tgbotapi:/var/lib/telegram-bot-api \\"
    echo "           aiogram/telegram-bot-api:latest"
    echo "     • or build telegram-bot-api from source: https://github.com/tdlib/telegram-bot-api"
    echo "     • or remove TELEGRAM_LOCAL_API_URL from .env to use the cloud API (50 MB cap)."
    exit 1
  fi
fi

echo "Starting Telegram bot (main.py)..."
$VENV main.py &
BOT_PID=$!

echo "Starting web dashboard (webapp.py)..."
$VENV webapp.py &
WEB_PID=$!

TUNNEL_PID=""
if command -v cloudflared &>/dev/null; then
  # Kill any leftover cloudflared processes from previous runs
  pkill -f "cloudflared tunnel" 2>/dev/null; sleep 0.5
  echo "Starting Cloudflare tunnel..."
  cloudflared --config /dev/null tunnel --url http://localhost:5003 --no-autoupdate > /tmp/cloudflared_playitnews.log 2>&1 &
  TUNNEL_PID=$!
  # Wait for URL to appear in log
  for i in $(seq 1 15); do
    TUNNEL_URL=$(grep -ao 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cloudflared_playitnews.log 2>/dev/null | head -1)
    [ -n "$TUNNEL_URL" ] && break
    sleep 1
  done
  if [ -n "$TUNNEL_URL" ]; then
    echo ""
    echo "  ✅ Cloudflare public URL: $TUNNEL_URL"
  else
    echo "  ⚠️  cloudflared started but URL not ready — check /tmp/cloudflared_playitnews.log"
  fi
else
  echo "  ℹ️  cloudflared not found. Install: brew install cloudflared"
fi

echo ""
echo "  Local:   http://localhost:5003"
echo "  Bot PID: $BOT_PID  |  Web PID: $WEB_PID"
echo ""
echo "Press Ctrl+C to stop all."

trap "echo 'Stopping...'; kill $BOT_PID $WEB_PID $TUNNEL_PID $TG_API_PID 2>/dev/null; pkill -f 'telegram-bot-api' 2>/dev/null; exit 0" INT TERM

wait
