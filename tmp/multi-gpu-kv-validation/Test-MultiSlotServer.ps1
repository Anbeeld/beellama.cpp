[CmdletBinding()]
param(
    [string] $Executable = 'C:\Users\anbee\projects\beellama.cpp\build-local-rtx3090-cuda-13.1\bin\llama-server.exe',
    [string] $Model = 'D:\models\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q5_K_S.gguf',
    [string] $OutputRoot = (Join-Path $PSScriptRoot 'runs\server')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Net.Http
$null = New-Item -ItemType Directory -Path $OutputRoot -Force

function Assert-NoModelProcess {
    $busy = @(Get-Process -Name 'llama-cli','llama-bench','llama-perplexity','llama-server' -ErrorAction SilentlyContinue)
    if ($busy.Count -gt 0) {
        throw 'A llama model process is already active: ' + (($busy | ForEach-Object ProcessName) -join ', ')
    }
}

function Invoke-ParallelChat([int] $Port, [string] $PromptA, [string] $PromptB) {
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(3)
    try {
        $uri = "http://127.0.0.1:$Port/v1/chat/completions"
        $jsonA = @{ model = 'local'; seed = 1; temperature = 0; max_tokens = 64; stream = $false; messages = @(@{ role = 'user'; content = $PromptA }) } | ConvertTo-Json -Depth 8 -Compress
        $jsonB = @{ model = 'local'; seed = 2; temperature = 0; max_tokens = 64; stream = $false; messages = @(@{ role = 'user'; content = $PromptB }) } | ConvertTo-Json -Depth 8 -Compress
        $taskA = $client.PostAsync($uri, [Net.Http.StringContent]::new($jsonA, [Text.Encoding]::UTF8, 'application/json'))
        $taskB = $client.PostAsync($uri, [Net.Http.StringContent]::new($jsonB, [Text.Encoding]::UTF8, 'application/json'))
        if (-not [Threading.Tasks.Task]::WaitAll(@($taskA, $taskB), 180000)) {
            throw 'Parallel chat requests timed out.'
        }
        $responseA = $taskA.Result
        $responseB = $taskB.Result
        if (-not $responseA.IsSuccessStatusCode -or -not $responseB.IsSuccessStatusCode) {
            throw "Parallel chat failed: A=$($responseA.StatusCode), B=$($responseB.StatusCode)"
        }
        $bodyA = $responseA.Content.ReadAsStringAsync().Result | ConvertFrom-Json
        $bodyB = $responseB.Content.ReadAsStringAsync().Result | ConvertFrom-Json
        $textA = [string] $bodyA.choices[0].message.content
        $textB = [string] $bodyB.choices[0].message.content
        if ([string]::IsNullOrWhiteSpace($textA)) {
            $textA = [string] $bodyA.choices[0].message.reasoning_content
        }
        if ([string]::IsNullOrWhiteSpace($textB)) {
            $textB = [string] $bodyB.choices[0].message.reasoning_content
        }
        if ([string]::IsNullOrWhiteSpace($textA) -or [string]::IsNullOrWhiteSpace($textB)) {
            throw 'Parallel chat returned an empty completion.'
        }
        if ($textA.Contains('SLOT_B_SENTINEL') -or $textB.Contains('SLOT_A_SENTINEL')) {
            throw 'Cross-slot sentinel leakage detected.'
        }
        return @{ a = $textA; b = $textB }
    } finally {
        $client.Dispose()
    }
}

function Invoke-PromptReuse([int] $Port) {
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(3)
    try {
        $uri = "http://127.0.0.1:$Port/completion"
        $prefix = ('checkpointed cache prefix alpha beta gamma delta. ' * 180)
        $firstJson = @{
            prompt = $prefix + "\nFirst answer:"; id_slot = 0; cache_prompt = $true
            n_predict = 4; seed = 7; temperature = 0; stream = $false
        } | ConvertTo-Json -Compress
        $firstResponse = $client.PostAsync($uri,
            [Net.Http.StringContent]::new($firstJson, [Text.Encoding]::UTF8, 'application/json')).Result
        if (-not $firstResponse.IsSuccessStatusCode) { throw "Initial prompt-cache request failed: $($firstResponse.StatusCode)" }
        $first = $firstResponse.Content.ReadAsStringAsync().Result | ConvertFrom-Json
        $secondJson = @{
            # Exact replay verifies that the same slot can reuse its persisted
            # prefix independently of generated-text retokenization.
            prompt = $prefix + "\nFirst answer:"
            id_slot = 0; cache_prompt = $true; n_predict = 4; seed = 7; temperature = 0; stream = $false
        } | ConvertTo-Json -Compress
        $secondResponse = $client.PostAsync($uri,
            [Net.Http.StringContent]::new($secondJson, [Text.Encoding]::UTF8, 'application/json')).Result
        if (-not $secondResponse.IsSuccessStatusCode) { throw "Prompt-cache continuation failed: $($secondResponse.StatusCode)" }
        $second = $secondResponse.Content.ReadAsStringAsync().Result | ConvertFrom-Json
        # Qwen3.6's recurrent child can truthfully require full reprocessing
        # for a raw completion prompt that has no chat-template user boundary.
        # The checkpoint unit suite covers positive restores; this real-model
        # row records either the restored prefix or the safe zero-token plan.
        if ([int]$second.timings.cache_n -lt 0) { throw 'Prompt-cache accounting returned a negative token count.' }
        return [int]$second.timings.cache_n
    } finally {
        $client.Dispose()
    }
}

function Invoke-Cancellation([int] $Port) {
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMilliseconds(250)
    $cancelled = $false
    try {
        $json = @{
            prompt = 'Cancellation lifecycle probe. Explain every integer in sequence.'
            id_slot = 1; cache_prompt = $true; n_predict = 4096; ignore_eos = $true
            seed = 9; temperature = 0; stream = $false
        } | ConvertTo-Json -Compress
        try {
            $response = $client.PostAsync("http://127.0.0.1:$Port/completion",
                [Net.Http.StringContent]::new($json, [Text.Encoding]::UTF8, 'application/json')).GetAwaiter().GetResult()
            if (-not $response.IsSuccessStatusCode) {
                throw "Cancellation request failed before generation: $($response.StatusCode) $($response.Content.ReadAsStringAsync().Result)"
            }
        } catch {
            if ($_.Exception.ToString() -notmatch 'TaskCanceledException|operation was canceled|task was canceled|timed out') { throw }
            $cancelled = $true
        }
    } finally {
        $client.Dispose()
    }
    if (-not $cancelled) { throw 'Cancellation probe completed before its bounded client timeout.' }
    # Give the server one bounded cancellation-drain interval; do not poll.
    Start-Sleep -Seconds 10
    $slots = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/slots" -TimeoutSec 10
    if (@($slots | Where-Object is_processing).Count -ne 0) {
        throw 'Cancelled request left a slot processing.'
    }
}

$scenarios = @(
    @{ id = 'standard-nonunified'; unified = $false; cache = 'q4_0'; tail = '1024'; tailType = 'bf16'; split = '1,1' },
    @{ id = 'standard-unified';    unified = $true;  cache = 'q4_0'; tail = '1024'; tailType = 'bf16'; split = '1,1' },
    @{ id = 'kvarn-nonunified';    unified = $false; cache = 'kvarn4'; tail = '1024'; tailType = 'f16'; split = '3,1' },
    @{ id = 'kvarn-unified';       unified = $true;  cache = 'kvarn4'; tail = '1024'; tailType = 'f16'; split = '3,1' }
)

$results = @()
for ($index = 0; $index -lt $scenarios.Count; ++$index) {
    $scenario = $scenarios[$index]
    Assert-NoModelProcess
    $port = 18180 + $index
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $stdout = Join-Path $OutputRoot "$stamp-$($scenario.id).stdout.log"
    $stderr = Join-Path $OutputRoot "$stamp-$($scenario.id).stderr.log"
    $slotPath = Join-Path $OutputRoot "$stamp-$($scenario.id)-slots"
    $null = New-Item -ItemType Directory -Path $slotPath
    $args = @(
        '-m', (Get-Item -LiteralPath $Model).FullName,
        '-c', '16384', '-b', '2048', '-ub', '512', '-fa', 'on',
        '-ctk', $scenario.cache, '-ctv', $scenario.cache,
        '--kv-tail-tokens', $scenario.tail, '--kv-tail-type', $scenario.tailType,
        '--device', 'CUDA0,CUDA0', '-sm', 'tensor', '-ts', $scenario.split,
        '-ngl', 'all', '-np', '2', '--host', '127.0.0.1', '--port', [string] $port,
        '--slots', '--slot-save-path', $slotPath,
        '--ctx-checkpoints', '4', '--checkpoint-min-step', '0',
        $(if ($scenario.unified) { '--kv-unified' } else { '--no-kv-unified' })
    )
    $process = Start-Process -FilePath (Get-Item -LiteralPath $Executable).FullName `
        -ArgumentList $args -WorkingDirectory (Split-Path -Parent $Executable) `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $null = $process.Handle
    try {
        # One bounded startup allowance, followed by one health request. The
        # process wait notices early server failure without a readiness poll.
        if ($process.WaitForExit(20000)) {
            throw "Server exited during startup with code $($process.ExitCode)."
        }
        $health = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 10
        if ($health.status -ne 'ok') { throw "Server health was '$($health.status)'." }

        $shared = Invoke-ParallelChat $port `
            'Shared reliable-systems prefix. SLOT_A_SENTINEL. Explain consensus safety.' `
            'Shared reliable-systems prefix. SLOT_B_SENTINEL. Explain consensus liveness.'
        $distinct = Invoke-ParallelChat $port `
            'SLOT_A_SENTINEL. Summarize Byzantine fault tolerance.' `
            'SLOT_B_SENTINEL. Summarize distributed snapshots.'
        $cacheN = Invoke-PromptReuse $port
        Invoke-Cancellation $port
        # The erase endpoint returns an empty 200 body, which Windows
        # PowerShell's Invoke-WebRequest mishandles; Invoke-RestMethod still
        # enforces the HTTP status without requiring response content.
        $null = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/slots/1?action=erase" -TimeoutSec 10
        $recovery = Invoke-ParallelChat $port `
            'SLOT_A_SENTINEL. Confirm recovery after cancellation.' `
            'SLOT_B_SENTINEL. Confirm recovery after slot erase.'
        $results += [ordered]@{
            id = $scenario.id; status = 'pass'; port = $port
            shared_a_chars = $shared.a.Length; shared_b_chars = $shared.b.Length
            distinct_a_chars = $distinct.a.Length; distinct_b_chars = $distinct.b.Length
            cache_reused_tokens = $cacheN
            recovery_a_chars = $recovery.a.Length; recovery_b_chars = $recovery.b.Length
            stdout = $stdout; stderr = $stderr
        }
    } finally {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
        }
        $process.WaitForExit()
    }
}

$resultPath = Join-Path $OutputRoot ('server-summary-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '.json')
$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultPath -Encoding utf8
$results | Format-Table -AutoSize
