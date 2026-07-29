[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $ConfigPath,
    [string] $OutputRoot = (Join-Path $PSScriptRoot 'runs'),
    [string] $RunPath,
    [string] $Filter = '*',
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-FileIdentity([string] $Path) {
    $item = Get-Item -LiteralPath $Path
    [ordered]@{
        path = $item.FullName
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Get-ExecutableIdentity([string] $Path) {
    $exe = Get-Item -LiteralPath $Path
    $files = @($exe) + @(Get-ChildItem -LiteralPath $exe.DirectoryName -Filter '*.dll' -File)
    $parts = foreach ($file in ($files | Sort-Object FullName)) {
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        [ordered]@{ path = $file.FullName; bytes = $file.Length; sha256 = $hash }
    }
    $text = ($parts | ForEach-Object { '{0}|{1}|{2}' -f $_.path, $_.bytes, $_.sha256 }) -join "`n"
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $sha = $hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))
    } finally {
        $hasher.Dispose()
    }
    [ordered]@{
        composite_sha256 = (($sha | ForEach-Object { $_.ToString('x2') }) -join '')
        files = @($parts)
    }
}

function Write-JsonAtomic([string] $Path, $Value) {
    $temp = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temp -Encoding utf8
    Move-Item -LiteralPath $temp -Destination $Path -Force
}

function Quote-Command([string] $Executable, [string[]] $Arguments) {
    $quoted = foreach ($arg in $Arguments) {
        if ($arg -match '[\s"]') { '"' + ($arg -replace '"', '\"') + '"' } else { $arg }
    }
    '"{0}" {1}' -f $Executable, ($quoted -join ' ')
}

function ConvertTo-WindowsArgumentString([string[]] $Arguments) {
    (@($Arguments | ForEach-Object {
        if ($_ -notmatch '[\s"]') { return $_ }
        # CommandLineToArgvW-compatible quoting for the argument shapes used by
        # llama.cpp. Backslashes before a closing quote must be doubled.
        $escaped = $_ -replace '(\\*)"', '$1$1\"'
        $escaped = $escaped -replace '(\\+)$', '$1$1'
        '"' + $escaped + '"'
    }) -join ' ')
}

function Assert-NoComputeProcess {
    $busyNames = @('llama-cli', 'llama-bench', 'llama-perplexity', 'llama-server', 'nsys', 'ncu')
    $busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in $busyNames })
    if ($busy.Count -gt 0) {
        throw 'A llama or Nsight process is already active: ' + (($busy | ForEach-Object ProcessName) -join ', ')
    }
}

$configItem = Get-Item -LiteralPath $ConfigPath
$config = Get-Content -LiteralPath $configItem.FullName -Raw | ConvertFrom-Json
$exe = (Get-Item -LiteralPath $config.executable).FullName
$model = (Get-Item -LiteralPath $config.model).FullName
if ($config.context -ne 16384) {
    throw 'The runtime harness requires context=16384; only task=kld64k may use 65536.'
}

$selected = @($config.coordinates | Where-Object { $_.id -like $Filter })
if ($selected.Count -eq 0) { throw "Filter '$Filter' selected no coordinates." }
foreach ($coordinate in $selected) {
    $args = @($coordinate.arguments | ForEach-Object {
        $_.Replace('{model}', $model).Replace('{context}', [string]$config.context)
    })
    $ctxIndex = [Array]::IndexOf($args, '-c')
    if ($ctxIndex -lt 0) { $ctxIndex = [Array]::IndexOf($args, '--ctx-size') }
    $ctx = if ($ctxIndex -ge 0) { [int]$args[$ctxIndex + 1] } else { [int]$config.context }
    if ($coordinate.task -eq 'kld64k') {
        if ($ctx -ne 65536) { throw "$($coordinate.id): kld64k must use context 65536." }
    } elseif ($ctx -ne 16384) {
        throw "$($coordinate.id): runtime coordinate must use context 16384."
    }
    $coordinate | Add-Member -NotePropertyName resolved_arguments -NotePropertyValue $args -Force
}

if ($DryRun) {
    foreach ($coordinate in $selected) {
        '{0}: {1}' -f $coordinate.id, (Quote-Command $exe $coordinate.resolved_arguments)
    }
    return
}

try {
    $runnerLock = [IO.File]::Open(
        (Join-Path $PSScriptRoot '.runner.lock'),
        [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite,
        [IO.FileShare]::None)
} catch {
    throw 'Another multi-GPU KV validation runner owns the launch lock.'
}
Assert-NoComputeProcess

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$gitCommit = (& git -C $repo rev-parse HEAD).Trim()
$timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
if (-not $RunPath) {
    $RunPath = Join-Path $OutputRoot ("$timestamp-" + $gitCommit.Substring(0, 9))
    if (Test-Path -LiteralPath $RunPath) { throw "Run path already exists: $RunPath" }
    $null = New-Item -ItemType Directory -Path $RunPath
} else {
    $RunPath = (Get-Item -LiteralPath $RunPath).FullName
}
foreach ($name in @('results', 'logs', 'profiles', 'summaries', 'identity-cache')) {
    $null = New-Item -ItemType Directory -Path (Join-Path $RunPath $name) -Force
}

$manifestPath = Join-Path $RunPath 'manifest.json'
$configIdentity = Get-FileIdentity $configItem.FullName
$executableIdentity = Get-ExecutableIdentity $exe
$modelIdentity = Get-FileIdentity $model
$corpusIdentity = if ($config.PSObject.Properties.Name -contains 'corpus' -and $config.corpus) {
    Get-FileIdentity $config.corpus
} else { $null }
$baselineIdentity = if ($config.PSObject.Properties.Name -contains 'baseline' -and $config.baseline) {
    Get-FileIdentity $config.baseline
} else { $null }
if (-not (Test-Path -LiteralPath $manifestPath)) {
    $device = (& nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>$null) -join "`n"
    $manifest = [ordered]@{
        schema = 1
        created_utc = [DateTime]::UtcNow.ToString('o')
        git_commit = $gitCommit
        config = $configIdentity
        executable = $executableIdentity
        model = $modelIdentity
        corpus = $corpusIdentity
        baseline = $baselineIdentity
        device = $device
        commands = @($selected | ForEach-Object {
            [ordered]@{ id = $_.id; task = $_.task; command = Quote-Command $exe $_.resolved_arguments }
        })
    }
    Write-JsonAtomic $manifestPath $manifest
} else {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $identityChecks = @(
        @{ name = 'config'; expected = $manifest.config.sha256; actual = $configIdentity.sha256 },
        @{ name = 'executable'; expected = $manifest.executable.composite_sha256; actual = $executableIdentity.composite_sha256 },
        @{ name = 'model'; expected = $manifest.model.sha256; actual = $modelIdentity.sha256 }
    )
    if ($corpusIdentity -or $manifest.corpus) {
        $identityChecks += @{ name = 'corpus'; expected = $manifest.corpus.sha256; actual = $corpusIdentity.sha256 }
    }
    if ($baselineIdentity -or $manifest.baseline) {
        $identityChecks += @{ name = 'baseline'; expected = $manifest.baseline.sha256; actual = $baselineIdentity.sha256 }
    }
    foreach ($check in $identityChecks) {
        if (-not $check.expected -or -not $check.actual -or $check.expected -ne $check.actual) {
            $runnerLock.Dispose()
            throw "Run identity mismatch for $($check.name); create a new isolated run."
        }
    }
}

$semanticPattern = '(?i)(access violation|assertion failed|ggml_abort|cuda error|nan|\binf\b|critical error|failed to (load|allocate|compute))'
foreach ($coordinate in $selected) {
    $existing = @(Get-ChildItem -LiteralPath (Join-Path $RunPath 'results') -Filter "$($coordinate.id).attempt-*.json" -ErrorAction SilentlyContinue)
    $completed = $existing | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json } |
        Where-Object { $_.classification -eq 'pass' } | Select-Object -First 1
    if ($completed) { continue }
    $attempt = $existing.Count + 1
    $stem = "$($coordinate.id).attempt-$attempt"
    $stdout = Join-Path $RunPath "logs\$stem.stdout.log"
    $stderr = Join-Path $RunPath "logs\$stem.stderr.log"
    $resultPath = Join-Path $RunPath "results\$stem.json"
    $status = [ordered]@{
        id = $coordinate.id
        task = $coordinate.task
        attempt = $attempt
        classification = 'running'
        started_utc = [DateTime]::UtcNow.ToString('o')
        command = Quote-Command $exe $coordinate.resolved_arguments
        stdout = $stdout
        stderr = $stderr
    }
    Write-JsonAtomic $resultPath $status

    # Identity hashing may take several minutes. Recheck immediately before
    # launch so a process started during that interval cannot overlap this run.
    Assert-NoComputeProcess
    $processArguments = ConvertTo-WindowsArgumentString $coordinate.resolved_arguments
    $process = Start-Process -FilePath $exe -ArgumentList $processArguments `
        -WorkingDirectory (Split-Path -Parent $exe) -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    # Retain the native process handle before an asynchronously redirected
    # child exits; otherwise Windows PowerShell can leave ExitCode unset.
    $null = $process.Handle
    $timeoutMs = 1000 * [int]$coordinate.timeout_seconds
    $exited = $process.WaitForExit($timeoutMs)
    if (-not $exited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    } else {
        # A second parameterless wait flushes redirected streams and populates
        # ExitCode reliably on Windows PowerShell.
        $process.WaitForExit()
    }
    $process.Refresh()
    # Large llama.cpp logs are intentionally not streamed or read in full.
    $combined = ((Get-Content -LiteralPath $stdout -Tail 2000 -ErrorAction SilentlyContinue) -join "`n") + "`n" +
                 ((Get-Content -LiteralPath $stderr -Tail 2000 -ErrorAction SilentlyContinue) -join "`n")
    $semanticFailure = [regex]::Match($combined, $semanticPattern).Value
    $classification = if (-not $exited) { 'timeout' } elseif ($process.ExitCode -ne 0) { 'exit-failure' } `
        elseif ($semanticFailure) { 'semantic-failure' } elseif ([string]::IsNullOrWhiteSpace($combined)) { 'empty-output' } else { 'pass' }
    $status.classification = $classification
    $status.exit_code = if ($exited) { $process.ExitCode } else { $null }
    $status.semantic_failure = $semanticFailure
    $status.finished_utc = [DateTime]::UtcNow.ToString('o')
    Write-JsonAtomic $resultPath $status
}

$results = @(Get-ChildItem -LiteralPath (Join-Path $RunPath 'results') -Filter '*.json' |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
$summary = $results | Select-Object id, task, attempt, classification, exit_code, started_utc, finished_utc
$summary | Export-Csv -LiteralPath (Join-Path $RunPath 'summaries\summary.csv') -NoTypeInformation
Write-JsonAtomic (Join-Path $RunPath 'summaries\summary.json') @($summary)
$runnerLock.Dispose()
if (@($results | Where-Object classification -ne 'pass').Count -gt 0) { exit 1 }
