# Bharat Terminal — Windows Installer Builder
# Author: Sourish Dey
#
# Builds the release binaries (with embedded icons) and packages them into a
# per-user MSI with branded dialogs using the WiX Toolset v5.
#
# Prerequisites (one time):
#   dotnet tool install --global wix --version 5.0.2
#   wix extension add --global WixToolset.UI.wixext/5.0.2
#
# Usage: run from the repository root.
#   powershell -ExecutionPolicy Bypass -File scripts/build-installer.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root "bharat-terminal"
$installerDir = Join-Path $project "installer"
$nativeDir = Join-Path $project "native"
$msiOut = Join-Path $project "releases\BharatTerminal-v3.0.0.msi"

Write-Host "Building Bharat Terminal v3 (release)..." -ForegroundColor Cyan
cargo build --release --workspace --manifest-path (Join-Path $project "Cargo.toml")

# Pinned ONNX Runtime 1.28.0 (matches ort-sys; never committed, ~16 MB).
$ortVersion = "1.28.0"
$ortDll = Join-Path $nativeDir "onnxruntime.dll"

# Portable copies. These are advertised in the summary, so they must actually be
# refreshed; a stale binary here means the "download the portable build" path
# silently ships the previous release.
$releasesDir = Join-Path $project "releases"
$portableExe = Join-Path $releasesDir "BharatTerminal-v3.0.0.exe"
$portableCli = Join-Path $releasesDir "BharatTerminal-v3.0.0-cli.exe"
if (-not (Test-Path $ortDll)) {
    Write-Host "Fetching ONNX Runtime $ortVersion..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $nativeDir -Force | Out-Null
    $zip = Join-Path ([System.IO.Path]::GetTempPath()) "onnxruntime-win-x64.zip"
    curl.exe -sL --max-time 300 -o $zip `
        "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/onnxruntime-win-x64-$ortVersion.zip"
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "ortx"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    Copy-Item "$tmp\onnxruntime-win-x64-$ortVersion\lib\*.dll" $nativeDir -Force
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "ONNX Runtime already present." -ForegroundColor DarkGray
}

Write-Host "Packaging per-user MSI..." -ForegroundColor Yellow
Push-Location $installerDir
try {
    # `wix build` writes errors to stdout and does not set a non-zero exit code
    # for every failure, so capture the output and fail loudly. Otherwise a
    # broken .wxs still prints "BUILD COMPLETE" and ships a stale MSI.
    $wixLog = & wix build BharatTerminal.wxs `
        -o $msiOut `
        -ext WixToolset.UI.wixext `
        -arch x64 `
        -d AppVersion=3.0.0 2>&1
    $wixLog | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "wix build failed with exit code $LASTEXITCODE"
    }
    if ($wixLog -match "error WIX") {
        throw "wix reported errors; refusing to report success"
    }
    if (-not (Test-Path $msiOut)) {
        throw "wix build reported success but $msiOut does not exist"
    }
} finally {
    Pop-Location
}

Write-Host "Copying portable binaries..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $releasesDir -Force | Out-Null
Copy-Item (Join-Path $project "target\release\bt-app.exe") $portableExe -Force
Copy-Item (Join-Path $project "target\release\bt-cli.exe") $portableCli -Force

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "BUILD COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " Installer: $msiOut"
Write-Host " Portable:  $portableExe"
Write-Host " Made by Sourish Dey" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
