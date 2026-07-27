[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = Join-Path ([IO.Path]::GetTempPath()) ('beellama-harness-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $root
try {
    $fake = Join-Path $PSScriptRoot 'fake-validation-command.ps1'
    $config = Join-Path $root 'config.json'
    $pwsh = (Get-Process -Id $PID).Path
    $model = Join-Path $root 'model.gguf'
    [IO.File]::WriteAllBytes($model, [byte[]](1, 2, 3, 4))
    @{
        executable = $pwsh
        model = $model
        context = 16384
        coordinates = @(
            @{ id = 'pass'; task = 'smoke'; timeout_seconds = 15; arguments = @('-NoProfile', '-File', $fake, 'pass', '-c', '16384') },
            @{ id = 'semantic'; task = 'smoke'; timeout_seconds = 15; arguments = @('-NoProfile', '-File', $fake, 'semantic', '-c', '16384') },
            @{ id = 'timeout'; task = 'smoke'; timeout_seconds = 1; arguments = @('-NoProfile', '-File', $fake, 'timeout', '-c', '16384') },
            @{ id = 'stale'; task = 'smoke'; timeout_seconds = 15; arguments = @('-NoProfile', '-File', $fake, 'pass', '-c', '16384') }
        )
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $config -Encoding utf8

    $dry = & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $config -OutputRoot $root -DryRun
    if (@($dry).Count -ne 4 -or $dry[0] -notmatch '^pass:' -or $dry[1] -notmatch '^semantic:' -or
            $dry[2] -notmatch '^timeout:' -or $dry[3] -notmatch '^stale:') {
        throw 'Dry-run matrix construction failed.'
    }
    & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $config -OutputRoot $root -Filter pass
    if ($LASTEXITCODE -ne 0) { throw 'Passing fake coordinate failed.' }
    $run = Get-ChildItem -LiteralPath $root -Directory | Where-Object Name -Match '^\d{8}T' | Select-Object -First 1
    if (-not $run) { throw 'Runner did not create an isolated run directory.' }
    $passPath = Join-Path $run.FullName 'results\pass.attempt-1.json'
    $passHash = (Get-FileHash -LiteralPath $passPath -Algorithm SHA256).Hash
    & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $config -RunPath $run.FullName -Filter pass
    if ($LASTEXITCODE -ne 0) { throw 'Resuming a completed coordinate failed.' }
    if (@(Get-ChildItem -LiteralPath (Join-Path $run.FullName 'results') -Filter 'pass.attempt-*.json').Count -ne 1 -or
            (Get-FileHash -LiteralPath $passPath -Algorithm SHA256).Hash -ne $passHash) {
        throw 'Resume overwrote or retried a completed coordinate.'
    }

    & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $config -RunPath $run.FullName -Filter semantic
    if ($LASTEXITCODE -eq 0) { throw 'Semantic failure was not rejected.' }
    $semantic = Get-ChildItem -LiteralPath (Join-Path $run.FullName 'results') -Filter 'semantic.attempt-1.json' |
        ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json }
    if ($semantic.classification -ne 'semantic-failure') { throw 'Semantic failure was misclassified.' }
    $semanticPath = Join-Path $run.FullName 'results\semantic.attempt-1.json'
    $semanticHash = (Get-FileHash -LiteralPath $semanticPath -Algorithm SHA256).Hash
    & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $config -RunPath $run.FullName -Filter semantic
    if ($LASTEXITCODE -eq 0) { throw 'Retried semantic failure unexpectedly passed.' }
    if (-not (Test-Path -LiteralPath (Join-Path $run.FullName 'results\semantic.attempt-2.json')) -or
            (Get-FileHash -LiteralPath $semanticPath -Algorithm SHA256).Hash -ne $semanticHash) {
        throw 'Retry did not create a new attempt or overwrote the first attempt.'
    }

    & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $config -RunPath $run.FullName -Filter timeout
    if ($LASTEXITCODE -eq 0) { throw 'Timeout was not rejected.' }
    $timeout = Get-Content -LiteralPath (Join-Path $run.FullName 'results\timeout.attempt-1.json') -Raw | ConvertFrom-Json
    if ($timeout.classification -ne 'timeout') { throw 'Timeout was misclassified.' }

    $stalePath = Join-Path $run.FullName 'results\stale.attempt-1.json'
    @{ id = 'stale'; task = 'smoke'; attempt = 1; classification = 'running' } |
        ConvertTo-Json | Set-Content -LiteralPath $stalePath -Encoding utf8
    $staleHash = (Get-FileHash -LiteralPath $stalePath -Algorithm SHA256).Hash
    & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $config -RunPath $run.FullName -Filter stale
    if (-not (Test-Path -LiteralPath (Join-Path $run.FullName 'results\stale.attempt-2.json')) -or
            (Get-FileHash -LiteralPath $stalePath -Algorithm SHA256).Hash -ne $staleHash) {
        throw 'A stale running record was not preserved before retry.'
    }

    $changedConfig = Join-Path $root 'changed-config.json'
    ((Get-Content -LiteralPath $config -Raw) + "`n ") |
        Set-Content -LiteralPath $changedConfig -Encoding utf8
    $identityRejected = $false
    try {
        & (Join-Path $PSScriptRoot 'Start-MultiGpuKvValidation.ps1') -ConfigPath $changedConfig -RunPath $run.FullName -Filter pass
    } catch {
        $identityRejected = $_.Exception.Message -match 'identity mismatch'
    }
    if (-not $identityRejected) { throw 'Resume did not reject an identity mismatch.' }

    'multi-GPU KV validation harness self-test: OK'
    $global:LASTEXITCODE = 0
} finally {
    if (-not $env:BEELLAMA_KEEP_HARNESS_TEST) {
        Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        "retained harness self-test root: $root"
    }
}
