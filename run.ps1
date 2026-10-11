$env:PYTHONPATH = "$PSScriptRoot\.deps;$PSScriptRoot"
$env:TEMP = "$PSScriptRoot\.piptemp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& 'C:\Users\黄金海\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' @args
exit $LASTEXITCODE
