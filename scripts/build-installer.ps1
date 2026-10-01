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
$msiOut = Join-Path $project "releases\BharatTerminal-v4.0.0.msi"

Write-Host "Building Bharat Terminal v4 (release)..." -ForegroundColor Cyan
cargo build --release --workspace --manifest-path (Join-Path $project "Cargo.toml")
# PowerShell does not throw when a native command fails, so a compile error - or
# a running app holding bt-app.exe - would sail through and the MSI would be built
# from the previous binary while still printing BUILD COMPLETE.
if ($LASTEXITCODE -ne 0) {
    throw "cargo build failed (exit $LASTEXITCODE); refusing to package a stale binary"
}

# Pinned ONNX Runtime 1.28.0 (matches ort-sys; never committed, ~16 MB).
$ortVersion = "1.28.0"
$ortDll = Join-Path $nativeDir "onnxruntime.dll"

# Portable copies. These are advertised in the summary, so they must actually be
# refreshed; a stale binary here means the "download the portable build" path
# silently ships the previous release.
$releasesDir = Join-Path $project "releases"
$portableExe = Join-Path $releasesDir "BharatTerminal-v4.0.0.exe"
$portableCli = Join-Path $releasesDir "BharatTerminal-v4.0.0-cli.exe"
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

Write-Host "Staging install payload..." -ForegroundColor Yellow

# The embedded interpreter is staged, not installed in place, so trimming the
# shipped payload never touches the repository copy.
#
# `python_runtime/` carries pandas (63 MB), lightgbm (5 MB) and a __pycache__
# tree, none of which `scripts/predictor.py` imports - it needs json, sys and
# numpy. numpy IS vendored into the runtime, and the bridge runs the interpreter
# with `-E -s`, so the shipped app never falls back to whatever numpy the user's
# machine happens to have. Dropping the unused packages takes the payload from
# 107 MB to roughly 40 MB, and keeps pip/setuptools so anything added later can be
# installed without re-downloading the interpreter.
$payloadDir = Join-Path $project "build\payload"
if (Test-Path $payloadDir) { Remove-Item $payloadDir -Recurse -Force }
New-Item -ItemType Directory -Path $payloadDir -Force | Out-Null

$pyRuntimeSrc = Join-Path $project "python_runtime"
$payloadRuntime = Join-Path $payloadDir "python_runtime"
if (-not (Test-Path (Join-Path $pyRuntimeSrc "python.exe"))) {
    throw "python_runtime\python.exe is missing; the Python bridge cannot ship without it"
}
Copy-Item $pyRuntimeSrc -Destination $payloadRuntime -Recurse -Force

# Trim only the staged copy.
$stagedSite = Join-Path $payloadDir "python_runtime\Lib\site-packages"
foreach ($dropped in @("pandas", "pandas.libs", "pandas-*.dist-info", "lightgbm", "lightgbm-*.dist-info")) {
    Get-ChildItem $stagedSite -Filter $dropped -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem $payloadDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# The prediction engine is useless without numpy, so fail the build rather than
# ship an installer whose Python route always 503s.
if (-not (Test-Path (Join-Path $stagedSite "numpy\__init__.py"))) {
    throw "numpy is not vendored in python_runtime\Lib\site-packages; the staged payload would not be self-contained"
}

# The staged interpreter must actually run isolated, before an MSI is built
# around it. `wix build` does not know what is inside a 3000-file glob.
$probe = Join-Path ([System.IO.Path]::GetTempPath()) "bt_payload_probe.json"
try {
    $probeOut = & (Join-Path $payloadDir "python_runtime\python.exe") -E -s -c `
        "import json,numpy;print(json.dumps({'ok':True,'v':numpy.__version__}))" 2>&1
    if ($LASTEXITCODE -ne 0 -or ($probeOut -notmatch '"ok":\s*true')) {
        throw "staged runtime cannot import numpy in isolation: $probeOut"
    }
    Write-Host "  staged runtime imports numpy: $($probeOut.Trim())" -ForegroundColor DarkGray
} finally {
    Remove-Item $probe -Force -ErrorAction SilentlyContinue
}

$payloadMb = [math]::Round((Get-ChildItem $payloadDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
$runtimeMb = [math]::Round((Get-ChildItem $pyRuntimeSrc -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "  payload $payloadMb MB (from $runtimeMb MB runtime, unused packages dropped)" -ForegroundColor DarkGray

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
        -d AppVersion=4.0.0 `
        -d "PayloadDir=$payloadRuntime" 2>&1
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

# A `<Files>` glob that matches nothing fails silently: wix reports success and
# ships an MSI with no interpreter in it, so the /api/ai/py_forecast route 503s on
# every install. The file table is the only honest check.
function Get-MsiFileCount {
    param([string]$MsiPath)
    $wi = New-Object -ComObject WindowsInstaller.Installer
    $db = $wi.GetType().InvokeMember("OpenDatabase", "InvokeMethod", $null, $wi, @($MsiPath, 0))
    $view = $db.GetType().InvokeMember("OpenView", "InvokeMethod", $null, $db, @("SELECT FileName FROM File"))
    $view.GetType().InvokeMember("Execute", "InvokeMethod", $null, $view, $null)
    $names = @()
    while ($true) {
        $rec = $view.GetType().InvokeMember("Fetch", "InvokeMethod", $null, $view, $null)
        if (-not $rec) { break }
        $names += $rec.GetType().InvokeMember("StringData", "GetProperty", $null, $rec, @(1))
    }
    return $names
}

$msiFiles = Get-MsiFileCount -MsiPath $msiOut
$stagedCount = (Get-ChildItem $payloadRuntime -Recurse -File).Count
# The MSI File table stores "longname|shortname" for names over 8.3 chars, and a
# short name for the rest, so counting on a prefix would only see python.exe and
# friends. Sentinels are matched anywhere in the name instead.
$pyInMsi = @($msiFiles | Where-Object { $_ -like "*numpy*" }).Count
$interpInMsi = @($msiFiles | Where-Object { $_ -like "python*.e*" -or $_ -like "python*.e|*" }).Count
if ($interpInMsi -lt 1) {
    throw "MSI contains no python interpreter; the glob matched nothing. Refusing to ship."
}
if ($pyInMsi -lt 1) {
    throw "MSI contains no numpy files; the Python route would 503 on every install. Refusing to ship."
}
# Total staged files must all be present, with a little slack for the pre-existing
# components (executables, runtime, models) that are not part of the payload.
$otherFiles = @($msiFiles | Where-Object { $_ -notlike "*numpy*" }).Count
if ($msiFiles.Count -lt $stagedCount) {
    throw "MSI holds $($msiFiles.Count) files but the payload staged $stagedCount; the glob is incomplete. Refusing to ship."
}
Write-Host "  verified $($msiFiles.Count) files in the MSI, $pyInMsi numpy, $interpInMsi interpreter (payload staged $stagedCount, $otherFiles other)" -ForegroundColor DarkGray

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
