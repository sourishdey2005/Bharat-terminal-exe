# scripts/download-qwen.ps1
# Author: Sourish Dey
#
# Fetches the optional Qwen2.5-1.5B-Instruct GGUF used by the chat window.
#
# The app does not need this: SmolLM2-135M already ships in the repository and
# answers chat. Qwen is the better, slower model, and its weights are ~990 MB, so
# they are not committed. Install them only if you want the stronger answers and
# have the disk space.
#
# Needs nothing but curl.exe, which ships with Windows. No Python, no pip, no
# huggingface_hub: those would add a dependency the app does not otherwise have.
#
# Usage, from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/download-qwen.ps1

$ErrorActionPreference = "Stop"

# Derived from this script's location rather than hardcoded, so the script works
# in any checkout. Every other script in this repo does the same.
$root = Split-Path -Parent $PSScriptRoot
$models = Join-Path $root "bharat-terminal\models"

# Must match chat_qwen::GGUF_FILENAME and TOKENIZER_FILENAME exactly; the badge
# only turns green when the loader resolves these two names.
$ggufName = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
$tokenizerName = "tokenizer.json"

$ggufUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$ggufName"
$tokenizerUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json"

# The GGUF's real size, used to prove the download completed. A truncated file
# would otherwise load as a corrupt model much later, with a far worse error.
$expectedGgufBytes = 1120000000

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " BHARAT TERMINAL - QWEN2.5-1.5B DOWNLOAD (optional)" -ForegroundColor Cyan
Write-Host " Made by Sourish Dey" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " This model is optional. The chat window already answers with the" -ForegroundColor Gray
Write-Host " bundled SmolLM2-135M. Qwen gives better, slower replies and" -ForegroundColor Gray
Write-Host " needs roughly 1 GB on disk." -ForegroundColor Gray
Write-Host ""

New-Item -ItemType Directory -Force -Path $models | Out-Null

# The tokenizer is shared with SmolLM2 and already committed by the v4.2.0 work,
# so it is usually a no-op here. Fetched anyway so this script works standalone.
$tokenizerPath = Join-Path $models $tokenizerName
if (-not (Test-Path $tokenizerPath)) {
    Write-Host "Downloading $tokenizerName ..." -ForegroundColor Yellow
    curl.exe -sL --max-time 300 -o $tokenizerPath $tokenizerUrl
    if (-not (Test-Path $tokenizerPath)) {
        throw "tokenizer.json was not created"
    }
    # Guard against a captive portal or proxy returning an HTML error page,
    # which curl reports as success because the HTTP status was 200.
    $head = Get-Content $tokenizerPath -TotalCount 1
    if ($head -notmatch '^\s*\{') {
        Remove-Item $tokenizerPath -Force -ErrorAction SilentlyContinue
        throw "tokenizer.json is not JSON (proxy or captive portal?). File removed."
    }
}
Write-Host "tokenizer.json present." -ForegroundColor DarkGray

$ggufPath = Join-Path $models $ggufName
if (Test-Path $ggufPath) {
    $have = (Get-Item $ggufPath).Length
    if ($have -ge ($expectedGgufBytes * 0.9)) {
        $mb = [math]::Round($have / 1MB, 1)
        Write-Host "$ggufName already present ($mb MB) - nothing to do." -ForegroundColor Green
        Write-Host ""
        Write-Host "Rebuild to pick it up: cargo build --release -p bt-app" -ForegroundColor Yellow
        exit 0
    }
    # A partial file from an interrupted download would be silently loaded as a
    # corrupt model later, so it is removed and refetched rather than resumed.
    Write-Host "Found a partial download ($have bytes); refetching." -ForegroundColor Yellow
    Remove-Item $ggufPath -Force
}

Write-Host "Downloading $ggufName (about 1 GB, this takes a while)..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
# -C - resumes, so a dropped connection does not restart the whole gigabyte.
curl.exe -L --fail --retry 3 --retry-delay 5 -C - --max-time 7200 `
    -o $ggufPath $ggufUrl
$sw.Stop()

if (-not (Test-Path $ggufPath)) {
    throw "download failed and no file was created"
}
$size = (Get-Item $ggufPath).Length
if ($size -lt ($expectedGgufBytes * 0.9)) {
    throw "download looks truncated: $size bytes. Delete $ggufPath and re-run."
}

$mb = [math]::Round($size / 1MB, 1)
Write-Host ""
Write-Host "SUCCESS: $ggufName ($mb MB in $([math]::Round($sw.Elapsed.TotalMinutes, 1)) min)" -ForegroundColor Green
Write-Host ""
Write-Host "Next: rebuild so the app finds it." -ForegroundColor Yellow
Write-Host "  cd $root\bharat-terminal" -ForegroundColor White
Write-Host "  cargo build --release -p bt-app" -ForegroundColor White
Write-Host ""
Write-Host "Open the chat window; the badge should read Qwen2.5-1.5B." -ForegroundColor Yellow
Write-Host "Expect roughly 60-120s per reply on an i3 - it is a 1.5B model." -ForegroundColor DarkGray