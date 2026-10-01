# scripts/download-tokenizer.ps1
# Author: Sourish Dey
# Downloads tokenizer.json for SmolLM2-135M-Instruct from HuggingFace.
# Run once after cloning. Needs nothing but curl.exe (ships with Windows):
# the file is fetched straight from the CDN, no Python required.

$ErrorActionPreference = "Stop"
$PROJECT = "E:\Projects\bharat-terminal\bharat-terminal"
$MODELS  = "$PROJECT\models"

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " BHARAT TERMINAL - SMOLLM2 TOKENIZER DOWNLOAD" -ForegroundColor Cyan
Write-Host " Made by Sourish Dey" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Force -Path $MODELS | Out-Null

$tok = "$MODELS\tokenizer.json"
if (Test-Path $tok) {
    $size = [math]::Round((Get-Item $tok).Length / 1KB, 1)
    Write-Host "Already present: $tok ($size KB) — nothing to do." -ForegroundColor Green
    exit 0
}

Write-Host "Downloading tokenizer.json (~2 MB)..." -ForegroundColor Yellow
curl.exe -sL --max-time 300 -o $tok `
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json"

if (-not (Test-Path $tok)) {
    Write-Host ""
    Write-Host "FAILED: tokenizer.json was not created" -ForegroundColor Red
    exit 1
}

# Sanity: must be JSON with a version field, not an HTML error page.
$head = Get-Content $tok -TotalCount 1
if ($head -notmatch '^\s*\{') {
    Remove-Item $tok -Force -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "FAILED: download is not JSON (proxy/captive portal?)." -ForegroundColor Red
    exit 1
}

$size = [math]::Round((Get-Item $tok).Length / 1KB, 1)
Write-Host ""
Write-Host "SUCCESS: $tok ($size KB)" -ForegroundColor Green
Write-Host ""
Write-Host "Next step: rebuild bt-app" -ForegroundColor Yellow
Write-Host "  cd $PROJECT" -ForegroundColor White
Write-Host "  cargo build --release -p bt-app" -ForegroundColor White
