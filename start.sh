#!/bin/bash
# Start Telegram bot, web dashboard, and optionally ngrok
VENV=".venv/bin/python"
cd "$(dirname "$0")"

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
  cloudflared tunnel --url http://localhost:5001 --no-autoupdate > /tmp/cloudflared_playitnews.log 2>&1 &
  TUNNEL_PID=$!
  # Wait for URL to appear in log
  for i in $(seq 1 15); do
    TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cloudflared_playitnews.log 2>/dev/null | head -1)
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
  echo "  Local:   http://localhost:5001"
echo "  Bot PID: $BOT_PID  |  Web PID: $WEB_PID"
echo ""
echo "Press Ctrl+C to stop all."

trap "echo 'Stopping...'; kill $BOT_PID $WEB_PID $TUNNEL_PID 2>/dev/null; exit 0" INT TERM

wait
