[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $Destination,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string[]] $Urls
)

$ErrorActionPreference = 'Stop'

# Keep each invocation isolated; runner temp cleanup owns these files.
$workDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("cuda-archives-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $workDirectory | Out-Null
$archives = @()
foreach ($url in $Urls) {
    $archivePath = Join-Path $workDirectory ("archive-{0}.zip" -f $archives.Count)
    & curl.exe --fail --location --retry 3 --output $archivePath $url
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA archive download failed (curl exit code $LASTEXITCODE)."
    }
    $archives += $archivePath
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
for ($index = 0; $index -lt $archives.Count; $index++) {
    $stagingDirectory = Join-Path $workDirectory ("staging-{0}" -f $index)
    Expand-Archive -LiteralPath $archives[$index] -DestinationPath $stagingDirectory
    $roots = @(Get-ChildItem -LiteralPath $stagingDirectory -Force)
    if ($roots.Count -ne 1 -or -not $roots[0].PSIsContainer) {
        throw "CUDA archive must contain exactly one root directory: $($archives[$index])"
    }
    $files = @(Get-ChildItem -LiteralPath $roots[0].FullName -File -Recurse -Force)
    if ($files.Count -eq 0) {
        throw "CUDA archive root must not be empty: $($archives[$index])"
    }
    foreach ($child in Get-ChildItem -LiteralPath $roots[0].FullName -Force) {
        Copy-Item -LiteralPath $child.FullName -Destination $Destination -Recurse -Force
    }
}
