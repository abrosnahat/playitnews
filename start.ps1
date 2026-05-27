# Windows 11 launcher for PlayItNews (analog of start.sh).
# Starts: main.py (Telegram bot) + webapp.py (dashboard) + optional cloudflared tunnel
#         + optional local telegram-bot-api server.
#
# Run from project root:
#   .\start.ps1
# If PowerShell blocks the script:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$VenvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Error ".venv not found. Create it: py -3.12 -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt"
}

# Force UTF-8 in child Python processes so logs/cyrillic don't end up as mojibake.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"

# --- Read a key from .env (no `source`, handles ; and spaces) ---
function Read-EnvValue([string]$Key) {
    if (-not (Test-Path ".env")) { return $null }
    foreach ($line in Get-Content .env -Encoding UTF8) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith("#")) { continue }
        $eq = $t.IndexOf("=")
        if ($eq -lt 1) { continue }
        $k = $t.Substring(0, $eq).Trim()
        if ($k -ne $Key) { continue }
        $v = $t.Substring($eq + 1).Trim()
        if ($v.Length -ge 2) {
            if (($v.StartsWith('"') -and $v.EndsWith('"')) -or
                ($v.StartsWith("'") -and $v.EndsWith("'"))) {
                $v = $v.Substring(1, $v.Length - 2)
            }
        }
        return $v
    }
    return $null
}

$TelegramLocalApiUrl  = Read-EnvValue "TELEGRAM_LOCAL_API_URL"
$TelegramLocalApiPort = Read-EnvValue "TELEGRAM_LOCAL_API_PORT"

# --- Kill stale instances to avoid port / getUpdates conflicts ---
function Stop-PyByScript([string]$Script) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -match [regex]::Escape($Script) } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Stop-ByName([string]$Name) {
    Get-Process -Name $Name -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

Stop-PyByScript "webapp.py"
Stop-PyByScript "main.py"
Stop-ByName "cloudflared"

# Free :5003 if something is squatting on it
try {
    Get-NetTCPConnection -LocalPort 5003 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
} catch { }

Start-Sleep -Milliseconds 700

$TmpDir = $env:TEMP
$CloudflaredLog = Join-Path $TmpDir "cloudflared_playitnews.log"
$BotLog         = Join-Path $TmpDir "playitnews_bot.log"
$BotErrLog      = Join-Path $TmpDir "playitnews_bot.err.log"
$WebLog         = Join-Path $TmpDir "playitnews_web.log"
$WebErrLog      = Join-Path $TmpDir "playitnews_web.err.log"

# --- Optional: local Bot API server (lifts upload cap from 50 MB to 2 GB) ---
$TgApiProc = $null
if ($TelegramLocalApiUrl) {
    $port = if ($TelegramLocalApiPort) { [int]$TelegramLocalApiPort } else { 8088 }

    $listening = $false
    try {
        $listening = [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    } catch { }

    if ($listening) {
        Write-Host "  [ok] telegram-bot-api already running on :$port (external/Docker)"
    }
    elseif (Get-Command telegram-bot-api -ErrorAction SilentlyContinue) {
        Write-Host "Starting local telegram-bot-api server..."
        $TgApiProc = Start-Process -FilePath (Join-Path $PSScriptRoot "start_tg_api.ps1") `
            -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden
        $ready = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
                    $ready = $true; break
                }
            } catch { }
        }
        if (-not $ready) {
            Write-Host "  [fail] telegram-bot-api did not start." -ForegroundColor Red
            if ($TgApiProc) { Stop-Process -Id $TgApiProc.Id -Force -ErrorAction SilentlyContinue }
            exit 1
        }
        Write-Host "  [ok] telegram-bot-api ready on :$port"
    }
    else {
        Write-Host "[fail] TELEGRAM_LOCAL_API_URL is set but nothing is listening on :$port" -ForegroundColor Red
        Write-Host "       and telegram-bot-api binary is missing."
        Write-Host "       Options:"
        Write-Host "         * run the Docker image (recommended on Windows):"
        Write-Host "             docker run -d --name tg-bot-api --restart unless-stopped \"
        Write-Host "               -p 127.0.0.1:${port}:8081 \"
        Write-Host "               -e TELEGRAM_API_ID=`$env:TELEGRAM_API_ID \"
        Write-Host "               -e TELEGRAM_API_HASH=`$env:TELEGRAM_API_HASH \"
        Write-Host "               -e TELEGRAM_LOCAL=1 \"
        Write-Host "               -v `$env:USERPROFILE\.tgbotapi:/var/lib/telegram-bot-api \"
        Write-Host "               aiogram/telegram-bot-api:latest"
        Write-Host "         * or remove TELEGRAM_LOCAL_API_URL from .env to use the cloud API (50 MB cap)."
        exit 1
    }
}

Write-Host "Starting Telegram bot (main.py)..."
$BotProc = Start-Process -FilePath $VenvPy -ArgumentList "-u", "main.py" `
    -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $BotLog -RedirectStandardError $BotErrLog

Write-Host "Starting web dashboard (webapp.py)..."
$WebProc = Start-Process -FilePath $VenvPy -ArgumentList "-u", "webapp.py" `
    -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $WebLog -RedirectStandardError $WebErrLog

$TunnelProc = $null
if (Get-Command cloudflared -ErrorAction SilentlyContinue) {
    Write-Host "Starting Cloudflare tunnel..."
    if (Test-Path $CloudflaredLog) { Remove-Item $CloudflaredLog -Force -ErrorAction SilentlyContinue }
    # cloudflared prints the tunnel URL on stderr. Start-Process refuses to
    # share one file between stdout and stderr, so wrap the call in cmd /c
    # and merge streams with 2>&1.
    $cfCmd = 'cloudflared --config NUL tunnel --url http://localhost:5003 --no-autoupdate > "{0}" 2>&1' -f $CloudflaredLog
    $TunnelProc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", $cfCmd `
        -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden

    $TunnelUrl = $null
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $CloudflaredLog) {
            $m = (Select-String -Path $CloudflaredLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches -ErrorAction SilentlyContinue |
                  Select-Object -First 1).Matches
            if ($m -and $m.Count -gt 0) { $TunnelUrl = $m[0].Value; break }
        }
    }
    if ($TunnelUrl) {
        Write-Host ""
        Write-Host "  [ok] Cloudflare public URL: $TunnelUrl" -ForegroundColor Green
    } else {
        Write-Host "  [warn] cloudflared started but URL not ready - check $CloudflaredLog" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [info] cloudflared not found. Install: winget install -e --id Cloudflare.cloudflared"
}

Write-Host ""
Write-Host "  Local:   http://localhost:5003"
Write-Host ("  Bot PID: {0}  |  Web PID: {1}" -f $BotProc.Id, $WebProc.Id)
Write-Host ("  Bot log: {0}" -f $BotLog)
Write-Host ("  Web log: {0}" -f $WebLog)
Write-Host ""
Write-Host "Press Ctrl+C to stop all."

# Wait for any child to exit; on Ctrl+C clean everything up.
try {
    while ($true) {
        if ($BotProc.HasExited) {
            Write-Host ("[fail] main.py exited (code {0}). See {1}" -f $BotProc.ExitCode, $BotErrLog) -ForegroundColor Red
            break
        }
        if ($WebProc.HasExited) {
            Write-Host ("[fail] webapp.py exited (code {0}). See {1}" -f $WebProc.ExitCode, $WebErrLog) -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "Stopping..."
    foreach ($p in @($BotProc, $WebProc, $TunnelProc, $TgApiProc)) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Stop-ByName "cloudflared"
}
