#!/usr/bin/env python3
"""Require nonempty UI output from the layout selected by the source tree."""
from pathlib import Path
import sys


def verify_source_ui(source: Path, build: Path) -> Path:
    if (source / "ui/apps/chat/CMakeLists.txt").is_file():
        relative = Path("ui/apps/chat/dist/index.html")
    elif (source / "tools/ui/CMakeLists.txt").is_file():
        relative = Path("tools/ui/dist/index.html")
    else:
        raise RuntimeError("Cannot identify the source UI CMake layout")
    output = build / relative
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Nonempty source-built UI output is missing: {output}")
    return output


def main() -> int:
    try:
        output = verify_source_ui(Path.cwd(), Path("build"))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Verified source-built UI output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
