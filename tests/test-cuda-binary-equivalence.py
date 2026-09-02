#!/usr/bin/env python3
"""Tests for the CUDA KVarN binary-equivalence verifier."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify-cuda-binary-equivalence.py"


def fixture(
    kernel,
    instruction="MOV R1, R2 ;",
    instruction_word="0x0000000000000001",
    control_word="0x0000000000000002",
    resources="REG:32 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:8",
):
    return f"""Function : {kernel}
        /*0000*/ {instruction} /* {instruction_word} */
                                  /* {control_word} */
 Function {kernel}:
  {resources}
"""


class BinaryEquivalenceTest(unittest.TestCase):
    def run_verifier(self, baseline, refactored, common=None):
        common = common or {}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, text in {**baseline, **refactored, **{f"common/{k}": v for k, v in common.items()}}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            dump = root / "mock-cuobjdump.py"
            dump.write_text(
                "import pathlib, sys\nprint(pathlib.Path(sys.argv[-1]).read_text())\n",
                encoding="utf-8",
            )
            command = [sys.executable, str(SCRIPT), str(root / "baseline"),
                       str(root / "refactored"), "--cuobjdump", str(dump)]
            for path in common:
                command += ["--common", str(root / "common" / path)]
            return subprocess.run(command, text=True, capture_output=True)

    def test_retained_kernel_and_resources_must_match(self):
        result = self.run_verifier(
            {"baseline/decode-pair-k0.o": fixture("keep")},
            {"refactored/decode-pair-k0.o": fixture("keep")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        result = self.run_verifier(
            {"baseline/decode-pair-k0.o": fixture("keep", resources="REG:33 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:8")},
            {"refactored/decode-pair-k0.o": fixture("keep")},
        )
        self.assertNotEqual(result.returncode, 0)

        result = self.run_verifier(
            {"baseline/decode-pair-k0.o": fixture("keep", control_word="0x0000000000000003")},
            {"refactored/decode-pair-k0.o": fixture("keep")},
        )
        self.assertNotEqual(result.returncode, 0)

    def test_each_object_is_compared_independently(self):
        result = self.run_verifier(
            {
                "baseline/decode-pair-a.o": fixture("same", resources="REG:31 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:8"),
                "baseline/decode-pair-b.o": fixture("same"),
            },
            {
                "refactored/decode-pair-a.o": fixture("same"),
                "refactored/decode-pair-b.o": fixture("same"),
            },
        )
        self.assertNotEqual(result.returncode, 0)

    def test_only_canonical_decode_removal_is_allowed(self):
        removed = fixture("ggml_cuda_fattn_kvarn_decode_combine_kernel<128>")
        result = self.run_verifier(
            {"baseline/decode-pair-k0.o": removed},
            {"refactored/decode-pair-k0.o": ""},
            common={"decode-common.o": removed},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_added_kernel_and_uncanonical_removal_fail(self):
        result = self.run_verifier(
            {"baseline/foo.o": fixture("keep")},
            {"refactored/foo.o": fixture("keep") + fixture("added")},
        )
        self.assertNotEqual(result.returncode, 0)

        result = self.run_verifier(
            {"baseline/decode-pair-k0.o": fixture("ggml_cuda_fattn_kvarn_decode_combine_kernel<128>")},
            {"refactored/decode-pair-k0.o": ""},
        )
        self.assertNotEqual(result.returncode, 0)

    def test_windows_paths_and_geometry_removal(self):
        result = self.run_verifier(
            {"baseline/geometry-case.o": fixture("ggml_cuda_fattn_kvarn_window_dequant_kernel<128>")},
            {"refactored/geometry-case.o": ""},
            common={"window-common.o": fixture("ggml_cuda_fattn_kvarn_window_dequant_kernel<128>")}
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
