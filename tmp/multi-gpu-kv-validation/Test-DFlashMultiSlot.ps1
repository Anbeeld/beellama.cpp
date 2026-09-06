[CmdletBinding()]
param(
    [string] $Executable = 'C:\Users\anbee\projects\beellama.cpp\build-local-rtx3090-cuda-13.1\bin\llama-server.exe',
    [string] $TargetModel = 'D:\models\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q5_K_S.gguf',
    [string] $DraftModel = 'D:\models\Qwen3.6-27B-DFlash-GGUF\Qwen3.6-27B-DFlash-Q3_K_M.gguf',
    [ValidateSet(1, 2, 4)] [int] $Parallel = 2,
    [string] $DraftCacheK = 'kvarn4',
    [string] $DraftCacheV = 'kvarn2',
    [ValidateSet('off', 'profit')] [string] $DmController = 'profit',
    [int] $Context = 16384,
    [string] $OutputRoot = (Join-Path $PSScriptRoot 'runs\dflash-server'),
    [int] $Port = 18190
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Net.Http
$busy = @(Get-Process -Name 'llama-cli','llama-bench','llama-perplexity','llama-server' -ErrorAction SilentlyContinue)
if ($busy.Count -gt 0) { throw 'A llama model process is already active.' }
$null = New-Item -ItemType Directory -Path $OutputRoot -Force
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$stdout = Join-Path $OutputRoot "$stamp.stdout.log"
$stderr = Join-Path $OutputRoot "$stamp.stderr.log"
$summary = Join-Path $OutputRoot "$stamp.summary.json"
$args = @(
    '--load-mode', 'dio',
    '-m', (Get-Item -LiteralPath $TargetModel).FullName,
    '--spec-type', 'draft-dflash',
    '--spec-draft-model', (Get-Item -LiteralPath $DraftModel).FullName,
    '--spec-draft-device', 'CUDA0', '--spec-draft-ngl', 'all', '--spec-draft-n-max', '16',
    '--spec-draft-type-k', $DraftCacheK, '--spec-draft-type-v', $DraftCacheV,
    '--spec-dm-controller', $DmController,
    '-c', [string]$Context, '-b', '2048', '-ub', '512', '-fa', 'on',
    '-ctk', 'f16', '-ctv', 'f16',
    '--device', 'CUDA0', '-ngl', 'all',
    '-np', [string]$Parallel, '--no-kv-unified', '--slots', '--host', '127.0.0.1', '--port', [string]$Port
)
$process = Start-Process -FilePath (Get-Item -LiteralPath $Executable).FullName -ArgumentList $args `
    -WorkingDirectory (Split-Path -Parent $Executable) -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$null = $process.Handle
try {
    if ($process.WaitForExit(30000)) {
        throw "DFlash server exited during startup with code $($process.ExitCode)."
    }
    $health = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 10
    if ($health.status -ne 'ok') { throw "DFlash server health was '$($health.status)'." }

    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(5)
    try {
        $uri = "http://127.0.0.1:$Port/v1/chat/completions"
        $topics = @('consensus safety', 'consensus liveness', 'quorum intersection', 'leader leases')
        $requests = @(for ($i = 0; $i -lt $Parallel; ++$i) {
            @{ model='local'; seed=(11 + $i); temperature=0; max_tokens=160; stream=$false;
               messages=@(@{role='user';content="DFLASH_SLOT_$($i)_SENTINEL. Explain $($topics[$i])."}) }
        })
        $tasks = @($requests | ForEach-Object {
            $json = $_ | ConvertTo-Json -Depth 8 -Compress
            $client.PostAsync($uri, [Net.Http.StringContent]::new($json, [Text.Encoding]::UTF8, 'application/json'))
        })
        if (-not [Threading.Tasks.Task]::WaitAll($tasks, 300000)) { throw 'Concurrent DFlash requests timed out.' }
        $texts = @()
        foreach ($task in $tasks) {
            if (-not $task.Result.IsSuccessStatusCode) { throw "DFlash request failed: $($task.Result.StatusCode)" }
            $body = $task.Result.Content.ReadAsStringAsync().Result | ConvertFrom-Json
            $text = [string]$body.choices[0].message.content
            if ([string]::IsNullOrWhiteSpace($text)) { $text = [string]$body.choices[0].message.reasoning_content }
            if ([string]::IsNullOrWhiteSpace($text)) { throw 'DFlash request returned empty content and reasoning.' }
            $texts += $text
        }
        for ($i = 0; $i -lt $Parallel; ++$i) {
            for ($j = 0; $j -lt $Parallel; ++$j) {
                if ($i -ne $j -and $texts[$i].Contains("DFLASH_SLOT_$($j)_SENTINEL")) {
                    throw "DFlash draft cache leaked slot $j content into slot $i."
                }
            }
        }
    } finally {
        $client.Dispose()
    }

    $logTail = ((Get-Content -LiteralPath $stderr -Tail 2000) -join "`n")
    if ($logTail -notmatch '(?i)dflash' -or $logTail -notmatch '(?i)accepted|draft') {
        throw 'DFlash route/acceptance evidence is missing from the bounded server log tail.'
    }
    if ($logTail -match '(?i)access violation|assertion failed|cuda error|nan|critical error') {
        throw 'DFlash server log contains a semantic failure.'
    }
    [ordered]@{
        status = 'pass'; context = $Context; slots = $Parallel; target_cache = 'f16/f16'
        draft_cache = "$DraftCacheK/$DraftCacheV"; target_device = 'CUDA0'; draft_device = 'CUDA0'
        dm_controller = $DmController
        output_chars = @($texts | ForEach-Object { $_.Length })
        output_sha256 = @($texts | ForEach-Object {
            $bytes = [Text.Encoding]::UTF8.GetBytes($_)
            $hasher = [Security.Cryptography.SHA256]::Create()
            try { ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
            finally { $hasher.Dispose() }
        })
        stdout = $stdout; stderr = $stderr
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summary -Encoding utf8
    Get-Content -LiteralPath $summary
} finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    $process.WaitForExit()
}
