#!/bin/bash
# Start both the Telegram bot and the web dashboard
VENV=".venv/bin/python"
cd "$(dirname "$0")"

echo "Starting Telegram bot (main.py)..."
$VENV main.py &
BOT_PID=$!

echo "Starting web dashboard (webapp.py)..."
$VENV webapp.py &
WEB_PID=$!

echo ""
echo "Both processes started."
echo "  Bot PID:  $BOT_PID"
echo "  Web PID:  $WEB_PID"
echo "  Dashboard → http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both."

# On Ctrl+C kill both children
trap "echo 'Stopping...'; kill $BOT_PID $WEB_PID 2>/dev/null; exit 0" INT TERM

wait
