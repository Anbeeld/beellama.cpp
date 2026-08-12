#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    expected_scripts = {
        "build-win-cuda-13.1-sm_86.ps1": "build-win-cuda-13.1-sm_86",
        "build-win-cuda-13.1-sm_86-default.ps1": "build-win-cuda-13.1-sm_86-default",
        "build-win-vulkan.ps1": "build-win-vulkan",
    }
    old_scripts = (
        "build-local-3090-cuda13.1.ps1",
        "build-local-3090-cuda13.1-default.ps1",
        "build-local-3090-cuda13.1-v0.5.0.ps1",
        "build-local-3090-vulkan.ps1",
        "build-win-vulkan-sm_86.ps1",
    )

    for old_script in old_scripts:
        if (ROOT / "tmp/build" / old_script).exists():
            raise AssertionError(f"obsolete build script still exists: tmp/build/{old_script}")
        if (ROOT / "scripts" / old_script).exists():
            raise AssertionError(f"obsolete build script still exists: scripts/{old_script}")

    for script_name, artifact_name in expected_scripts.items():
        script_path = ROOT / "scripts" / script_name
        if not script_path.is_file():
            raise AssertionError(f"required Windows build script is missing: scripts/{script_name}")
        source = script_path.read_text(encoding="utf-8")

        if f'[string]$BuildName = "{artifact_name}"' not in source:
            raise AssertionError(f"{script_name} does not use the required build directory name")
        if f'[string]$PackageName = "{artifact_name}"' not in source:
            raise AssertionError(f"{script_name} does not use the required package name")
        if any(old_name in source for old_name in ("build-local-", "local-test-", "rtx3090")):
            raise AssertionError(f"{script_name} retains an obsolete local/test/RTX-3090 artifact name")
        if "$repoRoot = $PSScriptRoot | Split-Path -Parent" not in source:
            raise AssertionError(f"{script_name} does not resolve the repository root from scripts/")
        if '$BuildName = "$BuildName-all-tests"' in source:
            raise AssertionError(f"{script_name} changes its build directory name for -AllTests")

        for required in (
            "[switch]$AllTests = $false",
            '"-DGGML_BACKEND_DL=$(if ($AllTests) { \'OFF\' } else { \'ON\' })"',
            "$ninjaExe -C $buildDir -t targets all",
            '"^(test-[A-Za-z0-9_.-]+):"',
            '$buildArgs += @("--target") + $testTargets',
            '$buildArgs += @("--", "-k", "0")',
            "if (-not $Package -or $SkipStage)",
        ):
            if required not in source:
                raise AssertionError(f"{script_name} lacks full-test behavior: {required}")

        shared_libs_test_mode = (
            '$commonFlags += "-DBUILD_SHARED_LIBS=OFF"' in source
            or '"-DBUILD_SHARED_LIBS=$(if ($AllTests) { \'OFF\' } else { \'ON\' })"' in source
        )
        if not shared_libs_test_mode:
            raise AssertionError(f"{script_name} does not select static libraries for -AllTests")


if __name__ == "__main__":
    main()
