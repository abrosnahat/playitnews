# Launch a local telegram-bot-api server on Windows so the bot can upload files >50 MB.
#
# Native Windows build of telegram-bot-api is not officially distributed.
# On Windows the easiest path is Docker Desktop (aiogram/telegram-bot-api image).
# This script is provided for parity with start_tg_api.sh and only works if you
# already have a telegram-bot-api.exe on PATH (e.g. built from source via vcpkg).

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

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
        if ($v.Length -ge 2 -and (
            ($v.StartsWith('"') -and $v.EndsWith('"')) -or
            ($v.StartsWith("'") -and $v.EndsWith("'")))) {
            $v = $v.Substring(1, $v.Length - 2)
        }
        return $v
    }
    return $null
}

$ApiId    = if ($env:TELEGRAM_API_ID)           { $env:TELEGRAM_API_ID }           else { Read-EnvValue "TELEGRAM_API_ID" }
$ApiHash  = if ($env:TELEGRAM_API_HASH)         { $env:TELEGRAM_API_HASH }         else { Read-EnvValue "TELEGRAM_API_HASH" }
$ApiPort  = if ($env:TELEGRAM_LOCAL_API_PORT)   { $env:TELEGRAM_LOCAL_API_PORT }   else { Read-EnvValue "TELEGRAM_LOCAL_API_PORT" }
$WorkDir  = if ($env:TELEGRAM_LOCAL_API_DIR)    { $env:TELEGRAM_LOCAL_API_DIR }    else { Read-EnvValue "TELEGRAM_LOCAL_API_DIR" }

if (-not (Get-Command telegram-bot-api -ErrorAction SilentlyContinue)) {
    Write-Host "[fail] telegram-bot-api.exe not found on PATH." -ForegroundColor Red
    Write-Host "       On Windows the recommended path is Docker Desktop:"
    Write-Host "         docker run -d --name tg-bot-api --restart unless-stopped \"
    Write-Host "           -p 127.0.0.1:8088:8081 \"
    Write-Host "           -e TELEGRAM_API_ID=<id> -e TELEGRAM_API_HASH=<hash> -e TELEGRAM_LOCAL=1 \"
    Write-Host "           -v $env:USERPROFILE\.tgbotapi:/var/lib/telegram-bot-api \"
    Write-Host "           aiogram/telegram-bot-api:latest"
    exit 1
}

if (-not $ApiId -or -not $ApiHash) {
    Write-Host "[fail] TELEGRAM_API_ID / TELEGRAM_API_HASH are not set in .env" -ForegroundColor Red
    Write-Host "       Create them at https://my.telegram.org/apps"
    exit 1
}

if (-not $ApiPort) { $ApiPort = "8088" }
if (-not $WorkDir) { $WorkDir = Join-Path $env:TEMP "tgbotapi" }

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "tmp") | Out-Null

Write-Host "Starting telegram-bot-api on http://127.0.0.1:$ApiPort (data: $WorkDir)"
& telegram-bot-api `
    --local `
    --api-id=$ApiId `
    --api-hash=$ApiHash `
    --http-port=$ApiPort `
    --dir=$WorkDir `
    --temp-dir=(Join-Path $WorkDir "tmp") `
    --verbosity=1
