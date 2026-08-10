param(
    [string]$OutputDir = "release-packages",
    [string]$PackageName = "build-win-vulkan",
    [string]$BuildName = "build-win-vulkan",
    [string]$Target = "",
    [int]$Parallel = 16,
    [switch]$Package = $false,
    [switch]$AllTests = $false,
    [switch]$SkipStage = $false,
    [switch]$ConfigureOnly = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:MSBUILDDISABLENODEREUSE = "1"

$repoRoot = $PSScriptRoot | Split-Path -Parent
$ninjaExe = Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $ninjaExe) {
    $ninjaExe = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
}

if (Test-Path $ninjaExe) {
    $env:PATH = "$(Split-Path -Parent $ninjaExe);$env:PATH"
} else {
    Write-Host "[FAIL] ninja.exe not found at $ninjaExe"
    exit 1
}

$sccacheExe = Get-Command sccache.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $sccacheExe) {
    $wingetPackages = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    $sccachePackage = Get-ChildItem $wingetPackages -Directory -Filter "Mozilla.sccache_*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($sccachePackage) {
        $sccacheExe = Get-ChildItem $sccachePackage.FullName -Recurse -Filter sccache.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    }
}
if ($sccacheExe) {
    $sccachePath = if ($sccacheExe.Source) {
        [string]$sccacheExe.Source
    } else {
        [string]$sccacheExe.FullName
    }
    $sccacheDir = Split-Path -Parent $sccachePath
    $env:PATH = "$sccacheDir;$env:PATH"
    $env:SCCACHE_IDLE_TIMEOUT = "7200"
    Write-Host "[ENV] sccache on PATH: $sccachePath"
    Write-Host "[ENV] sccache idle timeout: $($env:SCCACHE_IDLE_TIMEOUT)s"
} else {
    Write-Host "[WARN] sccache.exe not found under WinGet packages; compiler cache disabled"
}

$vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if (Test-Path (Join-Path $vsInstaller "vswhere.exe")) {
    $env:PATH = "$vsInstaller;$env:PATH"
}

$vcvarsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $vcvarsPath) {
    Write-Host "[ENV] Activating MSVC via vcvarsall.bat x64"
    cmd /c "`"$vcvarsPath`" x64 > nul && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
} else {
    Write-Host "[WARN] vcvarsall.bat not found - MSVC may not be available"
}

$vulkanSdk = $env:VULKAN_SDK
if (-not $vulkanSdk -or -not (Test-Path -LiteralPath $vulkanSdk)) {
    $installedSdks = @(
        Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
            Sort-Object { [version]$_.Name } -Descending
    )
    if ($installedSdks.Count -gt 0) {
        $vulkanSdk = $installedSdks[0].FullName
    }
}
if (-not $vulkanSdk -or -not (Test-Path -LiteralPath $vulkanSdk)) {
    Write-Host "[FAIL] Vulkan SDK not found. Set VULKAN_SDK or install it under C:\VulkanSDK."
    exit 1
}

$glslcExe = Join-Path $vulkanSdk "Bin\glslc.exe"
if (-not (Test-Path -LiteralPath $glslcExe)) {
    Write-Host "[FAIL] glslc.exe not found at $glslcExe"
    exit 1
}

$env:VULKAN_SDK = $vulkanSdk
$env:PATH = "$vulkanSdk\Bin;$env:PATH"

if ($AllTests) {
    $SkipStage = $true
}

$buildDir = Join-Path $repoRoot $BuildName
$pkgDir = Join-Path $repoRoot "$OutputDir\$PackageName"
$binDir = Join-Path $buildDir "bin"

$commonFlags = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_VULKAN=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=$(if ($AllTests) { 'OFF' } else { 'ON' })",
    "-DGGML_RPC=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_TESTS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON"
)

if ($AllTests) {
    $commonFlags += "-DBUILD_SHARED_LIBS=OFF"
}

Write-Host "========================================"
Write-Host "BeeLlama.cpp Windows Vulkan Build"
Write-Host "SDK:     $vulkanSdk"
Write-Host "Build:   $buildDir"
Write-Host "Package: $(if ($Package) { $pkgDir } else { 'disabled (use -Package to enable)' })"
Write-Host "Target:  $(if ($Target) { $Target } elseif ($AllTests) { 'every test-* target (static, backend-dl off)' } else { 'all' })"
Write-Host "Jobs:    $Parallel"
Write-Host "Zip:     disabled"
Write-Host "========================================"

Write-Host "`n[CONFIGURE] cmake -S $repoRoot -B $buildDir"
$cmakeArgs = @("-S", $repoRoot, "-B", $buildDir)
$cmakeArgs += $commonFlags
$cmakeArgs += "-DCMAKE_MAKE_PROGRAM=$ninjaExe"

$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& cmake @cmakeArgs 2>&1 | ForEach-Object { Write-Host $_ }
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] CMake configure failed (exit code $LASTEXITCODE)"
    exit 1
}
Write-Host "[OK] CMake configured"

if ($ConfigureOnly) {
    Write-Host "[DONE] ConfigureOnly requested; build/package skipped"
    exit 0
}

$buildArgs = @("--build", $buildDir, "--parallel", "$Parallel")
if ($Target) {
    $buildArgs += @("--target", $Target)
} elseif ($AllTests) {
    $ninjaTargets = & $ninjaExe -C $buildDir -t targets all 2>$null
    $testTargets = $ninjaTargets |
        ForEach-Object { if ($_ -match "^(test-[A-Za-z0-9_.-]+):") { $matches[1] } } |
        Where-Object { $_ -notlike "*.exe" } |
        Sort-Object -Unique
    if ($ninjaTargets -match "(?m)^llama-eval-callback:") {
        $testTargets += "llama-eval-callback"
    }

    if ($testTargets) {
        Write-Host "[TESTS] Building $($testTargets.Count) test targets: $($testTargets -join ', ')"
        $buildArgs += @("--target") + $testTargets
    } else {
        Write-Host "[WARN] No test-* targets discovered in the ninja graph; falling back to 'all'"
    }
    $buildArgs += @("--", "-k", "0")
}
Write-Host "`n[BUILD] cmake $($buildArgs -join ' ')"
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& cmake @buildArgs 2>&1 | ForEach-Object { Write-Host $_ }
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Build failed (exit code $LASTEXITCODE)"
    exit 1
}
Write-Host "[OK] Build complete"

if (-not $Package -or $SkipStage) {
    if ($AllTests) {
        Write-Host "[DONE] All test targets built; package staging skipped"
        Write-Host "       Run them with: ctest --test-dir $buildDir --output-on-failure"
    } elseif ($SkipStage) {
        Write-Host "[DONE] SkipStage requested; package staging skipped"
    } else {
        Write-Host "[DONE] Package staging disabled; use -Package to enable"
    }
    exit 0
}

Write-Host "`n[STAGE] Copying binaries to $pkgDir"
New-Item -ItemType Directory -Path $pkgDir -Force | Out-Null

$toStage = @(
    "$binDir\ggml*.dll",
    "$binDir\llama.dll",
    "$binDir\llama-common.dll",
    "$binDir\*-impl.dll",
    "$binDir\mtmd.dll",
    "$binDir\*.exe",
    "$binDir\libomp*.dll"
)

$stageCount = 0
foreach ($pattern in $toStage) {
    $files = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        Copy-Item -LiteralPath $file.FullName -Destination $pkgDir -Force
        $stageCount++
    }
}

$licensePath = Join-Path $repoRoot "LICENSE"
if (Test-Path $licensePath) {
    Copy-Item -LiteralPath $licensePath -Destination $pkgDir -Force
    $stageCount++
}

Write-Host "[OK] Staged $stageCount built files"
Write-Host "`n========================================"
Write-Host "Done. Package folder: $pkgDir"
Write-Host "No zip was created. Folder name is persistent across commits."
Write-Host "========================================"
