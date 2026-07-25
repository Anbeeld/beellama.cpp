[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [Parameter(Mandatory = $true)][string]$StopEventName,
    [ValidateRange(100, 10000)][int]$SampleIntervalMs = 1000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$stop = [System.Threading.EventWaitHandle]::new(
    $false,
    [System.Threading.EventResetMode]::ManualReset,
    $StopEventName
)

$maxima = [ordered]@{
    dedicatedBytes = [uint64]0
    sharedBytes = [uint64]0
    localBytes = [uint64]0
    nonLocalBytes = [uint64]0
    totalCommittedBytes = [uint64]0
}
$samples = New-Object System.Collections.Generic.List[object]
$errors = New-Object System.Collections.Generic.List[string]

function Read-GpuProcessMemorySample {
    $prefix = "\GPU Process Memory(pid_${ProcessId}_"
    $paths = @(Get-Counter -ListSet 'GPU Process Memory' -ErrorAction Stop |
        Select-Object -ExpandProperty PathsWithInstances |
        Where-Object { $_.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase) })
    if ($paths.Count -eq 0) {
        return $null
    }
    $counter = Get-Counter -Counter $paths -MaxSamples 1 -ErrorAction Stop
    $row = [ordered]@{
        timestampUtc = (Get-Date).ToUniversalTime().ToString('o')
        dedicatedBytes = [uint64]0
        sharedBytes = [uint64]0
        localBytes = [uint64]0
        nonLocalBytes = [uint64]0
        totalCommittedBytes = [uint64]0
    }
    foreach ($sample in $counter.CounterSamples) {
        $value = [uint64][math]::Max([double]0, [double]$sample.CookedValue)
        switch -Wildcard ($sample.Path) {
            '*\Dedicated Usage'  { $row.dedicatedBytes += $value }
            '*\Shared Usage'     { $row.sharedBytes += $value }
            '*\Local Usage'      { $row.localBytes += $value }
            '*\Non Local Usage'  { $row.nonLocalBytes += $value }
            '*\Total Committed'  { $row.totalCommittedBytes += $value }
        }
    }
    return [pscustomobject]$row
}

try {
    while (-not $stop.WaitOne(0)) {
        try {
            $sample = Read-GpuProcessMemorySample
            if ($null -ne $sample) {
                $samples.Add($sample)
                foreach ($field in @($maxima.Keys)) {
                    if ([uint64]$sample.$field -gt [uint64]$maxima[$field]) {
                        $maxima[$field] = [uint64]$sample.$field
                    }
                }
            }
        }
        catch {
            $errors.Add($_.Exception.Message)
        }
        $null = $stop.WaitOne($SampleIntervalMs)
    }
}
finally {
    $stop.Dispose()
    $result = [ordered]@{
        processId = $ProcessId
        sampleIntervalMs = $SampleIntervalMs
        sampleCount = $samples.Count
        maxima = $maxima
        samples = @($samples | ForEach-Object { $_ })
        errors = @($errors | Select-Object -Unique)
        completedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    }
    $parent = Split-Path -Parent $OutputPath
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $temporary = "$OutputPath.tmp"
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $OutputPath -Force
}
