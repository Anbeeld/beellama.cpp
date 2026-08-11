param(
    [string]$OutputDir = "release-packages",
    [string]$PackageName = "build-win-cuda-13.1-sm_86-default",
    [string]$BuildName = "build-win-cuda-13.1-sm_86-default",
    [string]$Target = "",
    [int]$Parallel = 16,
    [switch]$Package = $false,
    [switch]$AllTests = $false,
    [switch]$SkipStage = $false,
    [switch]$SkipConfigure = $false,
    [switch]$ConfigureOnly = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:MSBUILDDISABLENODEREUSE = "1"

$repoRoot = $PSScriptRoot | Split-Path -Parent
$cudaVer = "13.1"
$cudaArch = "86" # RTX 3090 / GA102 / Ampere
$cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cudaPath = Join-Path $cudaBase "v$cudaVer"
$ninjaExe = Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $ninjaExe) {
    $ninjaExe = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
}

# Ensure ninja is in PATH.
if (Test-Path $ninjaExe) {
    $env:PATH = "$(Split-Path -Parent $ninjaExe);$env:PATH"
} else {
    Write-Host "[FAIL] ninja.exe not found at $ninjaExe"
    exit 1
}

# Keep sccache enabled, but prevent its request-based idle timer from expiring
# while a long CUDA template translation unit is still compiling. The daemon
# retains a bounded lifetime after the build becomes truly inactive.
$sccacheExe = Get-Command sccache.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($sccacheExe) {
    $env:PATH = "$($sccacheExe.DirectoryName);$env:PATH"
    $env:SCCACHE_IDLE_TIMEOUT = "7200"
    Write-Host "[ENV] sccache on PATH: $($sccacheExe.Source)"
    Write-Host "[ENV] sccache idle timeout: $($env:SCCACHE_IDLE_TIMEOUT)s"
} else {
    Write-Host "[WARN] sccache.exe not found; compiler cache disabled"
}

# Activate MSVC environment for Ninja (cl.exe, link.exe, etc.).
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

if (-not (Test-Path $cudaPath)) {
    Write-Host "[FAIL] CUDA $cudaVer not found at $cudaPath"
    exit 1
}

$env:CUDA_PATH = $cudaPath
$env:PATH = "$cudaPath\bin;$env:PATH"

# -AllTests compiles every test target, including tests hidden from Windows shared-library
# builds or backend-dynamic-loading builds. Use a separate static build tree and skip packaging.
if ($AllTests) {
    $SkipStage = $true
}

# Persistent Windows build/package folders. These names do not depend on git commit/hash.
$buildDir = Join-Path $repoRoot $BuildName
$pkgDir = Join-Path $repoRoot "$OutputDir\$PackageName"
$binDir = Join-Path $buildDir "bin"

# Release flags narrowed to CUDA 13.1 + RTX 3090 with the default FA matrices.
# Ninja generator avoids MSBuild CUDA targets interference (lets us pick nvcc per toolkit).
$commonFlags = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
    "-DGGML_CUDA_KVARN=ON",
    "-DGGML_CUDA_CUB_3DOT2=ON",
    "-DGGML_CCACHE=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=$(if ($AllTests) { 'OFF' } else { 'ON' })",
    "-DBUILD_SHARED_LIBS=$(if ($AllTests) { 'OFF' } else { 'ON' })",
    "-DGGML_RPC=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_TESTS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$cudaArch"
)

Write-Host "========================================"
Write-Host "BeeLlama.cpp Windows CUDA 13.1 sm_86 Default-Pairs Build"
Write-Host "FA mode: DEFAULT standard pairs; 15 default KVarN fast-decode pairs"
Write-Host "CUDA:    $cudaVer"
Write-Host "Arch:    sm_$cudaArch"
Write-Host "Build:   $buildDir"
Write-Host "Package: $(if ($Package) { $pkgDir } else { 'disabled (use -Package to enable)' })"
Write-Host "Target:  $(if ($Target) { $Target } elseif ($AllTests) { 'every test-* target (static, backend-dl off)' } else { 'all' })"
Write-Host "Jobs:    $Parallel"
Write-Host "Zip:     disabled"
Write-Host "========================================"

if ($SkipConfigure) {
    if (-not (Test-Path (Join-Path $buildDir "build.ninja"))) {
        Write-Host "[FAIL] SkipConfigure requested but build.ninja is missing in $buildDir"
        exit 1
    }
    Write-Host "`n[CONFIGURE] skipped; reusing $buildDir\build.ninja"
} else {
    # --- CMake Configure ---
    Write-Host "`n[CONFIGURE] cmake -S $repoRoot -B $buildDir"
    $cmakeArgs = @("-S", $repoRoot, "-B", $buildDir)
    $cmakeArgs += $commonFlags
    $cmakeArgs += "-DCUDAToolkit_ROOT=$cudaPath"
    $cmakeArgs += "-DCMAKE_CUDA_COMPILER=$cudaPath\bin\nvcc.exe"
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
}

if ($ConfigureOnly) {
    Write-Host "[DONE] ConfigureOnly requested; build/package skipped"
    exit 0
}

# --- Build ---
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

# --- Stage package folder, no zip ---
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
    foreach ($f in $files) {
        Copy-Item -LiteralPath $f.FullName -Destination $pkgDir -Force
        $stageCount++
    }
}

$licensePath = Join-Path $repoRoot "LICENSE"
if (Test-Path $licensePath) {
    Copy-Item -LiteralPath $licensePath -Destination $pkgDir -Force
    $stageCount++
}

Write-Host "[OK] Staged $stageCount built files"

# Copy CUDA runtime DLLs directly into the package folder so the binaries are easy to run.
Write-Host "`n[STAGE] Copying CUDA runtime DLLs into package folder"
$cudaBin = Join-Path $cudaPath "bin"
$cudaBinX64 = Join-Path $cudaPath "bin\x64"
$cudaLib = Join-Path $cudaPath "lib"

$dllPatterns = @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
$dllCount = 0
foreach ($searchPath in @($cudaBin, $cudaBinX64, $cudaLib)) {
    if (-not (Test-Path $searchPath)) { continue }
    foreach ($pattern in $dllPatterns) {
        $files = Get-ChildItem -LiteralPath $searchPath -Filter $pattern -ErrorAction SilentlyContinue
        foreach ($f in $files) {
            Copy-Item -LiteralPath $f.FullName -Destination $pkgDir -Force
            $dllCount++
        }
    }
}

if ($dllCount -gt 0) {
    Write-Host "[OK] Staged $dllCount CUDA runtime DLLs"
} else {
    Write-Host "[WARN] No CUDA runtime DLLs found in toolkit"
}

Write-Host "`n========================================"
Write-Host "Done. Package folder: $pkgDir"
Write-Host "No zip was created. Folder name is persistent across commits."
Write-Host "========================================"
