#!/usr/bin/env python3
from pathlib import Path
src=(Path(__file__).resolve().parents[1]/"CMakeLists.txt").read_text(encoding="utf-8")
assert '/STACK:16777216' in src
assert 'MSVC' in src
print("Windows tensor-parallel graph stack invariant OK")
