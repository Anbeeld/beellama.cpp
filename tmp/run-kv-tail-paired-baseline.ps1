[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BaselineRuntime,
    [string]$CandidateRuntime = 'C:\Users\anbee\projects\beellama.cpp\build-local-rtx3090-cuda-13.1\bin',
    [ValidateSet('Iteration', 'StdQuant', 'Prefill', 'Server')][string]$Mode = 'Iteration',
    [ValidateSet(7, 21, 35)][int]$Repetitions = 7,
    [string]$HarnessPath = 'C:\Users\anbee\projects\decode-benchmarks\Start-DecodeTailBenchmarks.ps1',
    [string]$ServerDriver = 'C:\Users\anbee\projects\beellama.cpp\tmp\bench-kv-tail-server.py',
    [string]$ServerModel = '',
    [ValidateSet(1, 2)][int]$ServerSlots = 2,
    [int]$ServerRequests = 6,
    [int]$ServerPromptTokens = 768,
    [string]$ServerCacheType = 'q4_0',
    [int]$ServerTailTokens = 128,
    [string]$ServerTailType = 'bf16',
    [switch]$ServerUnified,
    [string]$OnlyModel = '',
    [string]$OnlyCache = '',
    [string]$OutputRoot = '',
    [int]$TimeoutMinutes = 120,
    [switch]$DryRun,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $PSScriptRoot ("kv-tail-paired-{0}" -f $Mode.ToLowerInvariant())
}
if (-not (Test-Path -LiteralPath $HarnessPath)) {
    throw "canonical benchmark harness missing: $HarnessPath"
}

function Resolve-RuntimeBinary {
    param(
        [Parameter(Mandatory = $true)][string]$Runtime,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $item = Get-Item -LiteralPath $Runtime -ErrorAction Stop
    $binary = if ($item.PSIsContainer) { Join-Path $item.FullName $Name } else { $item.FullName }
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        throw "$Name missing from runtime: $Runtime"
    }
    return (Resolve-Path -LiteralPath $binary).Path
}

function Get-Median {
    param([Parameter(Mandatory = $true)][double[]]$Values)
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count % 2 -eq 1) {
        return [double]$sorted[[int][Math]::Floor($sorted.Count / 2.0)]
    }
    $hi = [int][Math]::Floor($sorted.Count / 2.0)
    return 0.5 * ([double]$sorted[$hi - 1] + [double]$sorted[$hi])
}

function Get-PairedBootstrap {
    param(
        [Parameter(Mandatory = $true)][double[]]$Ratios,
        [Parameter(Mandatory = $true)][int]$Seed,
        [int]$Iterations = 10000
    )
    if ($Ratios.Count -eq 0) {
        throw 'cannot bootstrap an empty paired sample'
    }
    $random = [System.Random]::new($Seed)
    $medians = [double[]]::new($Iterations)
    for ($iteration = 0; $iteration -lt $Iterations; $iteration++) {
        $sample = [double[]]::new($Ratios.Count)
        for ($i = 0; $i -lt $sample.Count; $i++) {
            $sample[$i] = $Ratios[$random.Next($Ratios.Count)]
        }
        $medians[$iteration] = Get-Median -Values $sample
    }
    [Array]::Sort($medians)
    return [pscustomobject]@{
        median = Get-Median -Values $Ratios
        lower95 = $medians[[int][Math]::Floor(0.025 * ($Iterations - 1))]
        upper95 = $medians[[int][Math]::Floor(0.950 * ($Iterations - 1))]
    }
}

function Get-RecordKey {
    param([Parameter(Mandatory = $true)][object]$Record)
    return @(
        [string]$Record.mode,
        [string]$Record.model,
        [string]$Record.cache,
        (@($Record.contexts) -join ','),
        (@($Record.tails) -join ','),
        [string]$Record.tailType,
        [string]$Record.promptTokens,
        [string]$Record.generatedTokens,
        [string]$Record.repetitions,
        [string]$Record.batchSize,
        [string]$Record.ubatchSize
    ) -join '|'
}

function Get-RowKey {
    param([Parameter(Mandatory = $true)][object]$Row)
    return @(
        [string]$Row.n_depth,
        [string]$Row.n_prompt,
        [string]$Row.n_gen,
        [string]$Row.type_k,
        [string]$Row.type_v,
        [string]$Row.kv_tail_tokens_requested,
        [string]$Row.kv_tail_tokens_effective,
        [string]$Row.kv_tail_type
    ) -join '|'
}

function Read-CompletedRecords {
    param([Parameter(Mandatory = $true)][string]$SideRoot)
    $records = @()
    foreach ($file in Get-ChildItem -LiteralPath $SideRoot -Filter '*.json' -File -Recurse |
            Where-Object { $_.Directory.Name -eq 'results' }) {
        $record = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
        if ([string]$record.status -eq 'completed' -and [string]$record.mode -eq $Mode) {
            $pairDirectory = $file.Directory.Parent.Name
            if ($pairDirectory -notmatch '^pair-(\d+)$') {
                throw "paired result is not under a pair-N directory: $($file.FullName)"
            }
            $record | Add-Member -NotePropertyName pairIndex -NotePropertyValue ([int]$matches[1])
            $records += $record
        }
    }
    return @($records)
}

if ($Mode -eq 'Server') {
    if ($DryRun) {
        for ($pair = 0; $pair -lt $Repetitions; $pair++) {
            $order = if ($pair % 2 -eq 0) { 'baseline,candidate' } else { 'candidate,baseline' }
            Write-Host "[DRY-PAIR] serverPair=$pair order=$order"
        }
        Write-Host "[DRY-PAIR] mode=Server pairs=$Repetitions requests=$ServerRequests slots=$ServerSlots"
        return
    }
    if (-not (Test-Path -LiteralPath $ServerDriver -PathType Leaf)) {
        throw "server benchmark driver missing: $ServerDriver"
    }
    if (-not $ServerModel -or -not (Test-Path -LiteralPath $ServerModel -PathType Leaf)) {
        throw "Server mode requires an existing -ServerModel"
    }
    if ($ServerRequests -lt 2 -or $ServerPromptTokens -le 0 -or $ServerTailTokens -lt 0) {
        throw 'Server mode requires at least two requests and valid prompt/tail lengths'
    }
    $serverBinaries = @{
        baseline = Resolve-RuntimeBinary -Runtime $BaselineRuntime -Name 'llama-server.exe'
        candidate = Resolve-RuntimeBinary -Runtime $CandidateRuntime -Name 'llama-server.exe'
    }
    $serverRows = @{ baseline = @(); candidate = @() }
    $serverIdentities = @{ baseline = @(); candidate = @() }
    $sequence = @()
    for ($pair = 0; $pair -lt $Repetitions; $pair++) {
        $order = if ($pair % 2 -eq 0) { @('baseline', 'candidate') } else { @('candidate', 'baseline') }
        foreach ($side in $order) {
            $sideRoot = Join-Path $OutputRoot "server\$side"
            New-Item -ItemType Directory -Path $sideRoot -Force | Out-Null
            $output = Join-Path $sideRoot ("rep-{0:D2}.jsonl" -f $pair)
            $driverArguments = @(
                $ServerDriver,
                '--server-bin', $serverBinaries[$side],
                '--model', $ServerModel,
                '--output', $output,
                '--slots', [string]$ServerSlots,
                '--requests', [string]$ServerRequests,
                '--prompt-tokens', [string]$ServerPromptTokens,
                '--cache-type', $ServerCacheType,
                '--tail-tokens', [string]$ServerTailTokens,
                '--tail-type', $ServerTailType,
                '--startup-timeout', [string]($TimeoutMinutes * 60)
            )
            if ($ServerUnified) { $driverArguments += '--unified' }
            $sequence += "pair-$pair`:$side"
            Write-Host "[PAIR] serverPair=$pair side=$side"
            & python @driverArguments
            $rows = @(Get-Content -LiteralPath $output | ForEach-Object { $_ | ConvertFrom-Json })
            if ($rows.Count -ne $ServerRequests) {
                throw "server driver wrote $($rows.Count) rows, expected $ServerRequests"
            }
            $serverRows[$side] += ,@($rows)
            $manifest = Get-Content -LiteralPath "$output.manifest.json" -Raw | ConvertFrom-Json
            $serverIdentities[$side] += [string]$manifest.server_sha256
        }
    }
    foreach ($side in @('baseline', 'candidate')) {
        $identities = @($serverIdentities[$side] | Sort-Object -Unique)
        if ($identities.Count -ne 1) {
            throw "mixed $side server composite identities: $($identities -join ',')"
        }
        $serverIdentities[$side] = $identities[0]
    }
    $pairedRows = @()
    $seed = 20260716
    foreach ($request in 1..($ServerRequests - 1)) {
        for ($pair = 0; $pair -lt $Repetitions; $pair++) {
            $baselineRow = $serverRows.baseline[$pair][$request]
            $candidateRow = $serverRows.candidate[$pair][$request]
            if ([string]$baselineRow.response_sha256 -ne [string]$candidateRow.response_sha256) {
                throw "server response mismatch at pair=$pair request=$request"
            }
        }
        foreach ($metric in @('ttft_ms', 'handoff_ms')) {
            $slowdowns = [double[]]::new($Repetitions)
            for ($pair = 0; $pair -lt $Repetitions; $pair++) {
                $baselineValue = [double]$serverRows.baseline[$pair][$request].$metric
                $candidateValue = [double]$serverRows.candidate[$pair][$request].$metric
                if ($baselineValue -le 0 -or $candidateValue -le 0) {
                    throw "non-positive server latency at pair=$pair request=$request metric=$metric"
                }
                $slowdowns[$pair] = $candidateValue / $baselineValue
            }
            $bootstrap = Get-PairedBootstrap -Ratios $slowdowns -Seed $seed
            $seed++
            $pairedRows += [pscustomobject]@{
                mode = 'Server'
                request = $request
                slot = [int]$serverRows.candidate[0][$request].slot
                metric = $metric
                repetitions = $Repetitions
                baselinePromptN = Get-Median -Values ([double[]]@($serverRows.baseline | ForEach-Object { $_[$request].prompt_n }))
                candidatePromptN = Get-Median -Values ([double[]]@($serverRows.candidate | ForEach-Object { $_[$request].prompt_n }))
                baselineTokensReused = Get-Median -Values ([double[]]@($serverRows.baseline | ForEach-Object { $_[$request].tokens_reused }))
                candidateTokensReused = Get-Median -Values ([double[]]@($serverRows.candidate | ForEach-Object { $_[$request].tokens_reused }))
                medianSlowdown = [double]$bootstrap.median
                bootstrapLower95Slowdown = [double]$bootstrap.lower95
                bootstrapUpper95Slowdown = [double]$bootstrap.upper95
                neutral = [double]$bootstrap.median -le 1.005 -and [double]$bootstrap.upper95 -le 1.010
                sampleSlowdowns = $slowdowns -join '/'
                baselineBinarySha256 = $serverIdentities.baseline
                candidateBinarySha256 = $serverIdentities.candidate
            }
        }
    }
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    $reportPath = Join-Path $OutputRoot 'paired-server.csv'
    $pairedRows | Export-Csv -LiteralPath $reportPath -NoTypeInformation -Encoding UTF8
    [ordered]@{
        mode = 'Server'
        repetitions = $Repetitions
        sequence = $sequence
        rowCount = $pairedRows.Count
        neutralRows = @($pairedRows | Where-Object { $_.neutral }).Count
        failedRows = @($pairedRows | Where-Object { -not $_.neutral }).Count
        report = $reportPath
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputRoot 'paired-server-manifest.json') -Encoding UTF8
    $failed = @($pairedRows | Where-Object { -not $_.neutral })
    if ($failed.Count -gt 0) {
        throw "$($failed.Count) server latency rows exceeded the neutrality gate"
    }
    Write-Host "[OK] $($pairedRows.Count) paired server latency rows are neutral; report=$reportPath"
    return
}

$dryArguments = @{
    Mode = $Mode
    Repetitions = 1
    DryRun = $true
}
if ($OnlyModel) { $dryArguments.OnlyModel = $OnlyModel }
if ($OnlyCache) { $dryArguments.OnlyCache = $OnlyCache }
$dryLines = @(& $HarnessPath @dryArguments 6>&1 | ForEach-Object { [string]$_ })
$groups = @($dryLines | Where-Object { $_.StartsWith('[DRY] model=', [System.StringComparison]::Ordinal) } |
    ForEach-Object {
        if ($_ -notmatch '^\[DRY\] model=([^ ]+) .* cache=([^ ]+) ') {
            throw "unrecognized canonical dry-run row: $_"
        }
        [pscustomobject]@{ Model = $matches[1]; Cache = $matches[2] }
    } | Sort-Object Model, Cache -Unique)
if ($groups.Count -eq 0) {
    throw "canonical harness selected no groups for mode $Mode"
}
if ($DryRun) {
    for ($index = 0; $index -lt $groups.Count; $index++) {
        $group = $groups[$index]
        for ($pair = 0; $pair -lt $Repetitions; $pair++) {
            $order = if (($index + $pair) % 2 -eq 0) { 'baseline,candidate' } else { 'candidate,baseline' }
            Write-Host "[DRY-PAIR] group=$($group.Model)/$($group.Cache) pair=$pair order=$order"
        }
    }
    $rowCount = @($dryLines | Where-Object { $_.StartsWith('[DRY] model=', [System.StringComparison]::Ordinal) }).Count
    Write-Host "[DRY-PAIR] mode=$Mode groups=$($groups.Count) rows=$rowCount pairs=$Repetitions invocations=$($groups.Count*$Repetitions*2)"
    return
}

$baselineBinary = Resolve-RuntimeBinary -Runtime $BaselineRuntime -Name 'llama-bench.exe'
$candidateBinary = Resolve-RuntimeBinary -Runtime $CandidateRuntime -Name 'llama-bench.exe'

$baselineRoot = Join-Path $OutputRoot 'baseline'
$candidateRoot = Join-Path $OutputRoot 'candidate'
New-Item -ItemType Directory -Path $baselineRoot, $candidateRoot -Force | Out-Null
$sequence = @()
for ($index = 0; $index -lt $groups.Count; $index++) {
    $group = $groups[$index]
    for ($pair = 0; $pair -lt $Repetitions; $pair++) {
        $order = if (($index + $pair) % 2 -eq 0) { @('baseline', 'candidate') } else { @('candidate', 'baseline') }
        foreach ($side in $order) {
            $binary = if ($side -eq 'baseline') { $baselineBinary } else { $candidateBinary }
            $sideBase = if ($side -eq 'baseline') { $baselineRoot } else { $candidateRoot }
            $sideRoot = Join-Path $sideBase ("pair-{0:D2}" -f $pair)
            $sequence += "$($group.Model)/$($group.Cache):pair-$pair`:$side"
            $harnessArguments = @{
                Mode = $Mode
                Repetitions = 1
                BinaryPath = $binary
                OnlyModel = $group.Model
                OnlyCache = $group.Cache
                OutputRoot = $sideRoot
                TimeoutMinutes = $TimeoutMinutes
                Force = [bool]$Force
            }
            Write-Host "[PAIR] group=$($group.Model)/$($group.Cache) pair=$pair side=$side"
            & $HarnessPath @harnessArguments
        }
    }
}

$baselineRecords = @(Read-CompletedRecords -SideRoot $baselineRoot)
$candidateRecords = @(Read-CompletedRecords -SideRoot $candidateRoot)
$baselineIdentities = @($baselineRecords | ForEach-Object { [string]$_.binarySha256 } | Sort-Object -Unique)
$candidateIdentities = @($candidateRecords | ForEach-Object { [string]$_.binarySha256 } | Sort-Object -Unique)
if ($baselineIdentities.Count -ne 1 -or $candidateIdentities.Count -ne 1) {
    throw "mixed composite identities: baseline=$($baselineIdentities -join ',') candidate=$($candidateIdentities -join ',')"
}

$expectedRecordCount = $groups.Count*$Repetitions
if ($baselineRecords.Count -ne $expectedRecordCount -or $candidateRecords.Count -ne $expectedRecordCount) {
    throw "paired run has incomplete group records: expected=$expectedRecordCount baseline=$($baselineRecords.Count) candidate=$($candidateRecords.Count)"
}

$baselineByPairKey = @{}
$candidateByPairKey = @{}
foreach ($item in @(
        [pscustomobject]@{ Name = 'baseline'; Records = $baselineRecords; Map = $baselineByPairKey },
        [pscustomobject]@{ Name = 'candidate'; Records = $candidateRecords; Map = $candidateByPairKey })) {
    foreach ($record in $item.Records) {
        $key = "$($record.pairIndex)|$(Get-RecordKey -Record $record)"
        if ($item.Map.ContainsKey($key)) { throw "duplicate $($item.Name) paired group: $key" }
        $item.Map[$key] = $record
    }
}

$groupKeys = @($baselineRecords | ForEach-Object { Get-RecordKey -Record $_ } | Sort-Object -Unique)
if ($groupKeys.Count -ne $groups.Count) {
    throw "paired run selected $($groupKeys.Count) unique result groups; expected $($groups.Count)"
}

$pairedRows = @()
$seed = 20260716
foreach ($recordKey in $groupKeys) {
    $baselinePairRecords = @()
    $candidatePairRecords = @()
    for ($pair = 0; $pair -lt $Repetitions; $pair++) {
        $pairKey = "$pair|$recordKey"
        if (-not $baselineByPairKey.ContainsKey($pairKey) -or -not $candidateByPairKey.ContainsKey($pairKey)) {
            throw "missing matched execution pair: $pairKey"
        }
        $baselineRecord = $baselineByPairKey[$pairKey]
        $candidateRecord = $candidateByPairKey[$pairKey]
        if ([string]$baselineRecord.modelSha256 -ne [string]$candidateRecord.modelSha256) {
            throw "model identity differs within execution pair: $pairKey"
        }
        $baselinePairRecords += $baselineRecord
        $candidatePairRecords += $candidateRecord
    }

    $baselineRecord = $baselinePairRecords[0]
    foreach ($baselineRow in $baselineRecord.rows) {
        $rowKey = Get-RowKey -Row $baselineRow
        $slowdowns = [double[]]::new($Repetitions)
        for ($pair = 0; $pair -lt $Repetitions; $pair++) {
            $baselineRows = @($baselinePairRecords[$pair].rows | Where-Object { (Get-RowKey -Row $_) -eq $rowKey })
            $candidateRows = @($candidatePairRecords[$pair].rows | Where-Object { (Get-RowKey -Row $_) -eq $rowKey })
            if ($baselineRows.Count -ne 1 -or $candidateRows.Count -ne 1) {
                throw "paired row is missing or duplicated: pair=$pair $recordKey/$rowKey"
            }
            $baselineSamples = [double[]]@($baselineRows[0].samples_ts)
            $candidateSamples = [double[]]@($candidateRows[0].samples_ts)
            if ($baselineSamples.Count -ne 1 -or $candidateSamples.Count -ne 1 -or
                    $baselineSamples[0] -le 0 -or $candidateSamples[0] -le 0) {
                throw "execution pair does not contain one positive throughput sample: pair=$pair $recordKey/$rowKey"
            }
            $slowdowns[$pair] = $baselineSamples[0] / $candidateSamples[0]
        }
        $bootstrap = Get-PairedBootstrap -Ratios $slowdowns -Seed $seed
        $seed++
        $pairedRows += [pscustomobject]@{
            mode = $Mode
            model = [string]$baselineRecord.model
            cacheTypeK = [string]$baselineRow.type_k
            cacheTypeV = [string]$baselineRow.type_v
            contextTokens = [int]$baselineRow.n_depth
            promptTokens = [int]$baselineRow.n_prompt
            generatedTokens = [int]$baselineRow.n_gen
            requestedTailTokens = [int]$baselineRow.kv_tail_tokens_requested
            effectiveTailTokens = [int]$baselineRow.kv_tail_tokens_effective
            tailType = [string]$baselineRow.kv_tail_type
            repetitions = $Repetitions
            medianSlowdown = [double]$bootstrap.median
            bootstrapLower95Slowdown = [double]$bootstrap.lower95
            bootstrapUpper95Slowdown = [double]$bootstrap.upper95
            neutral = [double]$bootstrap.median -le 1.005 -and [double]$bootstrap.upper95 -le 1.010
            sampleSlowdowns = $slowdowns -join '/'
            baselineBinarySha256 = $baselineIdentities[0]
            candidateBinarySha256 = $candidateIdentities[0]
            modelSha256 = [string]$baselineRecord.modelSha256
        }
    }
}

$expectedRows = @($dryLines | Where-Object { $_.StartsWith('[DRY] model=', [System.StringComparison]::Ordinal) }).Count
if ($pairedRows.Count -ne $expectedRows) {
    throw "paired report has $($pairedRows.Count) rows; canonical mode selected $expectedRows"
}
$reportPath = Join-Path $OutputRoot ("paired-{0}.csv" -f $Mode.ToLowerInvariant())
$manifestPath = Join-Path $OutputRoot ("paired-{0}-manifest.json" -f $Mode.ToLowerInvariant())
$pairedRows | Sort-Object model, cacheTypeK, contextTokens, requestedTailTokens |
    Export-Csv -LiteralPath $reportPath -NoTypeInformation -Encoding UTF8
[ordered]@{
    mode = $Mode
    repetitions = $Repetitions
    baselineBinary = $baselineBinary
    candidateBinary = $candidateBinary
    baselineBinarySha256 = $baselineIdentities[0]
    candidateBinarySha256 = $candidateIdentities[0]
    sequence = $sequence
    rowCount = $pairedRows.Count
    neutralRows = @($pairedRows | Where-Object { $_.neutral }).Count
    failedRows = @($pairedRows | Where-Object { -not $_.neutral }).Count
    report = $reportPath
    createdAt = (Get-Date).ToString('o')
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$failed = @($pairedRows | Where-Object { -not $_.neutral })
if ($failed.Count -gt 0) {
    $failed | Sort-Object -Property bootstrapUpper95Slowdown -Descending |
        Select-Object -First 20 model, cacheTypeK, contextTokens, requestedTailTokens, medianSlowdown, bootstrapUpper95Slowdown |
        Format-Table -AutoSize | Out-String | Write-Host
    throw "$($failed.Count) paired rows exceeded the 0.5% median / 1.0% upper-bound neutrality gate"
}
Write-Host "[OK] $($pairedRows.Count) paired $Mode rows are neutral; report=$reportPath"
