# PlayItNews — Telegram News Bot

Monitors [playground.ru/news](https://www.playground.ru/news), automatically translates and adapts new articles into English Telegram posts, and schedules them to the **@playitnews** channel.

---

## Features

| Feature           | Details                                                                      |
| ----------------- | ---------------------------------------------------------------------------- |
| Source monitoring | Checks playground.ru/news every 30 min (configurable)                        |
| AI adaptation     | Translates RU→EN, rephrases for gaming audience, adds hashtags via Claude    |
| Image handling    | Downloads article images and attaches them to the post                       |
| Scheduled posting | Posts are delayed 1 hour (configurable) before publishing                    |
| Admin review      | You receive a Telegram notification with **Approve / Edit / Cancel** buttons |
| Edit support      | Reply with new text to update the draft before it publishes                  |
| Persistent state  | SQLite tracks seen articles and scheduled posts across restarts              |

---

## Setup

### 1. Create a Telegram Bot

1. Open Telegram → `@BotFather` → `/newbot`
2. Copy the **token** it gives you.
3. Add the bot as **Administrator** in your channel `@playitnews` with _Post Messages_ permission.

### 2. Get your personal Chat ID

Send a message to `@userinfobot` — it will reply with your numeric user ID.

### 3. Get an Anthropic API key

Sign up at [console.anthropic.com](https://console.anthropic.com) and create an API key.

### 4. Install dependencies

```bash
cd /Users/abrosnahat/Desktop/playitnews

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in all values
nano .env
```

### 6. Run

```bash
source .venv/bin/activate
python3 main.py
```

---

## Admin Controls

When a new article is detected you receive a Telegram message like this:

```
New article detected

Source: <article title>
https://playground.ru/news/xxx

Scheduled: 2026-02-25 15:30 UTC

--- DRAFT POST ---
<first 400 chars of the adapted post>

[Approve]  [Edit text]  [Cancel]
```

| Button        | Action                                                                |
| ------------- | --------------------------------------------------------------------- |
| **Approve**   | Marks the post as approved; it publishes at the scheduled time        |
| **Edit text** | Bot asks you to send the new text; updates and approves automatically |
| **Cancel**    | Cancels the post — it will not be published                           |

> If you do nothing, the post publishes at the scheduled time regardless (default behaviour: **auto-publish**).

---

## Project Structure

```
playitnews/
├── main.py          — Entry point, job scheduling, pipeline orchestration
├── bot.py           — Telegram handlers (approve/edit/cancel)
├── scraper.py       — playground.ru scraping and image downloading
├── ai_adapter.py    — Claude translation & adaptation
├── database.py      — SQLite helpers
├── config.py        — Environment variable loading
├── requirements.txt
├── .env.example
├── images/          — Downloaded article images (auto-created)
├── data.db          — SQLite database (auto-created)
└── playitnews.log   — Log file (auto-created)
```

---

## Running as a Service (macOS)

To keep the bot running after closing the terminal, use a launch agent:

```bash
# Create plist
cat > ~/Library/LaunchAgents/com.playitnews.bot.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.playitnews.bot</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/abrosnahat/Desktop/playitnews/.venv/bin/python</string>
    <string>/Users/abrosnahat/Desktop/playitnews/main.py</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/abrosnahat/Desktop/playitnews</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/abrosnahat/Desktop/playitnews/playitnews.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/abrosnahat/Desktop/playitnews/playitnews.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.playitnews.bot.plist
```

Stop it:

```bash
launchctl unload ~/Library/LaunchAgents/com.playitnews.bot.plist
```
