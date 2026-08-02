#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import tempfile


def run_validator(repo: Path, configure_text: str) -> subprocess.CompletedProcess[str]:
    validator = repo / "scripts" / "check-cmake-configure-vars.py"
    if not validator.is_file():
        raise AssertionError("release CMake configure validator is missing")
    with tempfile.TemporaryDirectory() as tmp:
        configure_log = Path(tmp) / "configure.log"
        configure_log.write_text(configure_text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(validator), str(configure_log)],
            text=True,
            capture_output=True,
            check=False,
        )


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    accepted = run_validator(repo, "-- Configuring done\n-- Generating done\n")
    if accepted.returncode != 0:
        raise AssertionError(f"validator rejected a clean configure: {accepted.stderr}")

    rejected = run_validator(
        repo,
        "CMake Warning:\n  Manually-specified variables were not used by the project:\n\n"
        "    GGML_UNUSED_RELEASE_SWITCH\n    LLAMA_UNKNOWN_RELEASE_SWITCH\n"
        "    THIRD_PARTY_UNUSED\n\n-- Build files have been written\n",
    )
    if rejected.returncode == 0:
        raise AssertionError("validator accepted unused project-prefixed CMake variables")
    for expected in ("GGML_UNUSED_RELEASE_SWITCH", "LLAMA_UNKNOWN_RELEASE_SWITCH"):
        if expected not in rejected.stderr:
            raise AssertionError(f"validator diagnostic omitted {expected}")
    if "THIRD_PARTY_UNUSED" in rejected.stderr:
        raise AssertionError("validator rejected a non-project configure variable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
