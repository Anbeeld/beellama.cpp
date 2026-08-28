#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/verify-windows-package.py"
SPEC = importlib.util.spec_from_file_location("verify_windows_package", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "libomp.dll").write_bytes(b"runtime")
        (root / "cudart64_13.dll").write_bytes(b"runtime")

        failures = VERIFY.expected_file_failures(
            root,
            required_names=["libomp.dll"],
            required_globs=["cudart64_*.dll"],
            forbidden_names=["libomp140.x86_64.dll"],
        )
        require(not failures, f"valid expected-file set failed: {failures}")

        failures = VERIFY.expected_file_failures(
            root,
            required_names=["amdhip64_7.dll"],
            required_globs=["cublas64_*.dll"],
            forbidden_names=["libomp.dll"],
        )
        require(
            failures == [
                "missing required file amdhip64_7.dll",
                "missing required file matching cublas64_*.dll",
                "forbidden file present: libomp.dll",
            ],
            f"unexpected expected-file failures: {failures}",
        )

        failures = VERIFY.required_local_import_failures(
            external_imports={"kernel32.dll", "rocm_kpack.dll"},
            required_local_imports=["rocm_kpack.dll", "amd_comgr_3.dll"],
        )
        require(
            failures == ["required local import is missing: rocm_kpack.dll"],
            f"unexpected required-local-import failures: {failures}",
        )
        require(
            not VERIFY.required_local_import_failures(
                external_imports={"kernel32.dll"},
                required_local_imports=["rocm_kpack.dll"],
            ),
            "an SDK that does not import KPack must not require it",
        )

        archive_path = root / "duplicates.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("bin/ggml-cuda.dll", b"first")
            archive.writestr("BIN/GGML-CUDA.DLL", b"second")
        duplicates = VERIFY.duplicate_archive_entries(archive_path)
        require(
            duplicates == ["BIN/GGML-CUDA.DLL"],
            f"case-insensitive duplicate ZIP entry was not detected: {duplicates}",
        )


if __name__ == "__main__":
    main()
