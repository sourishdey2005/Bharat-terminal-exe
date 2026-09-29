# Native runtime (not committed)

This folder holds the pinned ONNX Runtime 1.28.0 CPU build
(`onnxruntime.dll` + `onnxruntime_providers_shared.dll`) that `ort`
dlopens at runtime. It is fetched, not vendored — see
`scripts/build-installer.ps1` (or run that snippet manually):

```powershell
$ortVersion = "1.28.0"
$zip = "$env:TEMP\onnxruntime-win-x64.zip"
curl.exe -sL --max-time 300 -o $zip `
  "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/onnxruntime-win-x64-$ortVersion.zip"
Expand-Archive -Path $zip -DestinationPath "$env:TEMP\ortx" -Force
Copy-Item "$env:TEMP\ortx\onnxruntime-win-x64-$ortVersion\lib\*.dll" .
```

Why pinned: the OS-resolved `onnxruntime.dll` (e.g. an older inbox build in
System32) can be ABI-incompatible and crash natively instead of erroring,
so the app points `ort` at exactly this build.
