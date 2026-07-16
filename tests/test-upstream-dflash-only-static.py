from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETIRED_ARCH = "dflash" + "-draft"


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [ROOT / path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def main() -> None:
    needle = RETIRED_ARCH.encode("utf-8")
    hits: list[str] = []

    for path in tracked_files():
        if not path.is_file():
            continue

        relative = path.relative_to(ROOT).as_posix()
        if RETIRED_ARCH in relative.lower():
            hits.append(relative)
            continue

        if needle in path.read_bytes().lower():
            hits.append(relative)

    assert not hits, "retired DFlash architecture remains in tracked files: " + ", ".join(hits)


if __name__ == "__main__":
    main()
