param(
    [string]$OutputDir = "release-packages",
    [string]$PackageName = "build-win-cuda-13.1-sm_86",
    [string]$BuildName = "build-win-cuda-13.1-sm_86",
    [string]$Target = "",
    [int]$Parallel = 16,
    [switch]$Package = $false,
    [switch]$StdQuantIteration = $false,
    [switch]$AllTests = $false,
    [switch]$SkipStage = $false,
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

# Ensure sccache (compiler cache) is on PATH so the GGML_CCACHE wiring activates and the baked
# `sccache` launcher resolves at build time. Installed via `winget install Mozilla.sccache`.
# Restrict the fallback search to the package directory: recursively walking every WinGet
# package adds roughly fifteen minutes to every incremental target build on this host.
$sccacheExe = Get-Command sccache.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $sccacheExe) {
    $wingetPackages = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    $sccachePackage = Get-ChildItem $wingetPackages -Directory -Filter "Mozilla.sccache_*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($sccachePackage) {
        $sccacheExe = Get-ChildItem $sccachePackage.FullName -Recurse -Filter sccache.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    }
}
if ($sccacheExe) {
    $env:PATH = "$($sccacheExe.DirectoryName);$env:PATH"
    # A single CUDA template compilation can run for more than sccache's
    # 10-minute default inactivity timeout. The server resets this timer on
    # incoming requests, not compiler progress, and gives active compilers only
    # 10 seconds after it expires. Keep a bounded but comfortably long timeout
    # so fresh parallel CUDA builds cannot lose the daemon mid-compilation.
    $env:SCCACHE_IDLE_TIMEOUT = "7200"
    Write-Host "[ENV] sccache on PATH: $($sccacheExe.FullName)"
    Write-Host "[ENV] sccache idle timeout: $($env:SCCACHE_IDLE_TIMEOUT)s"
} else {
    Write-Host "[WARN] sccache.exe not found under WinGet packages; compiler cache disabled"
}

# Ensure vswhere.exe is locatable; vcvarsall.bat calls it and the script aborts under
# -ErrorAction Stop when it is not on PATH. Standard Visual Studio Installer location.
$vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if (Test-Path (Join-Path $vsInstaller "vswhere.exe")) {
    $env:PATH = "$vsInstaller;$env:PATH"
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

# -AllTests compiles every test target, including the ones tests/CMakeLists.txt hides from the
# default local configuration:
#   - `if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)` skips test-reasoning-budget, test-sampling, the
#     grammar/tokenizer-1 tests, test-chat, test-json-schema-to-grammar, test-llama-archs and
#     test-gbnf-validator on a Windows shared-library build.
#   - `if (NOT GGML_BACKEND_DL)` skips test-quantize-stats, test-barrier, test-quantize-fns,
#     test-quantize-perf, test-rope and test-col2im-1d.
# Both gates are configuration-level, so the only way to compile those sources is a static
# build. That needs its own build tree: reusing the shared-library one would relink every
# staged artifact. Staging is meaningless here, so it is forced off.
if ($AllTests) {
    $SkipStage = $true
}

# Persistent Windows build/package folders. These names do not depend on git commit/hash.
$buildDir = Join-Path $repoRoot $BuildName
$pkgDir = Join-Path $repoRoot "$OutputDir\$PackageName"
$binDir = Join-Path $buildDir "bin"

# Same release flags as build-release.ps1, narrowed to CUDA 13.1 + RTX 3090 only.
# Ninja generator avoids MSBuild CUDA targets interference (lets us pick nvcc per toolkit).
$commonFlags = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DGGML_CUDA_FA_ALL_QUANTS=ON",
    "-DGGML_CUDA_KVARN=$(if ($StdQuantIteration) { 'OFF' } else { 'ON' })",
    "-DGGML_CUDA_CUB_3DOT2=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=$(if ($AllTests) { 'OFF' } else { 'ON' })",
    "-DGGML_RPC=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_TESTS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$cudaArch"
)

if ($AllTests) {
    # GGML_BACKEND_DL requires BUILD_SHARED_LIBS (ggml/src/CMakeLists.txt), so both flip together.
    $commonFlags += "-DBUILD_SHARED_LIBS=OFF"
}

Write-Host "========================================"
Write-Host "BeeLlama.cpp Windows CUDA 13.1 sm_86 Build"
Write-Host "CUDA:    $cudaVer"
Write-Host "Arch:    sm_$cudaArch"
Write-Host "Build:   $buildDir"
Write-Host "Package: $(if ($Package) { $pkgDir } else { 'disabled (use -Package to enable)' })"
Write-Host "Target:  $(if ($Target) { $Target } elseif ($AllTests) { 'every test-* target (static, backend-dl off)' } else { 'all' })"
Write-Host "Jobs:    $Parallel"
Write-Host "Zip:     disabled"
Write-Host "========================================"

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

if ($ConfigureOnly) {
    Write-Host "[DONE] ConfigureOnly requested; build/package skipped"
    exit 0
}

# --- Build ---
$buildArgs = @("--build", $buildDir, "--parallel", "$Parallel")
if ($Target) {
    $buildArgs += @("--target", $Target)
} elseif ($AllTests) {
    # Ask the generated ninja graph which test targets exist rather than hardcoding a list that
    # would silently rot as tests/CMakeLists.txt grows. Each CMake target appears as a phony
    # target under its own name plus a `bin/<name>.exe` path rule; keep the former only.
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

    # Keep going after a failure so one run reports every broken target instead of only the
    # first. The build still exits non-zero, so the [FAIL] path below is unaffected.
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
