#!/usr/bin/env python3
"""Exercise the compilation-slice CLI with real, short-lived process trees."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/cache-build-deadline.py"


class CompilationSliceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.env = {**os.environ, "GITHUB_OUTPUT": str(self.output)}

    def run_slice(self, code, seconds="5"):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--seconds", seconds, "--", sys.executable, "-c", code],
            env=self.env, capture_output=True, text=True, timeout=20,
        )
        outputs = self.output.read_text() if self.output.exists() else ""
        return result, outputs

    def test_default_interval_is_110_minutes(self):
        spec = importlib.util.spec_from_file_location("cache_build_deadline", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.CHECKPOINT_SECONDS, 110 * 60)

    def test_success_is_complete(self):
        result, outputs = self.run_slice("print('compiled')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compiled", result.stdout)
        self.assertEqual(outputs, "checkpoint=false\n")

    def test_compiler_errors_are_not_checkpoints(self):
        for code in (1, 2, 75):
            with self.subTest(code=code):
                self.output.write_text("")
                result, outputs = self.run_slice(f"raise SystemExit({code})")
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(outputs, "checkpoint=false\n")

    def test_interval_stops_compiler_descendants_before_checkpoint(self):
        heartbeat = self.root / "heartbeat"
        child = (
            "import pathlib, time, signal; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"p = pathlib.Path({str(heartbeat)!r}); "
            "exec('while True:\\n p.write_text(str(time.time_ns()))\\n time.sleep(0.02)')"
        )
        parent = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
            "time.sleep(60)"
        )
        result, outputs = self.run_slice(parent, "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs, "checkpoint=true\n")
        self.assertTrue(heartbeat.exists(), "descendant must run before the interval expires")
        final = heartbeat.read_text()
        time.sleep(0.2)
        self.assertEqual(heartbeat.read_text(), final, "compiler child continued writing during checkpoint")

    def test_incremental_build_can_finish_after_checkpoint(self):
        state = self.root / "finished-object"
        code = (
            "from pathlib import Path; import time; "
            f"p = Path({str(state)!r}); "
            "done = p.exists(); p.write_text('finished'); "
            "time.sleep(0 if done else 60)"
        )
        first, outputs = self.run_slice(code, "0.5")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(outputs, "checkpoint=true\n")
        self.assertTrue(state.exists())
        self.output.write_text("")
        second, outputs = self.run_slice(code)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(outputs, "checkpoint=false\n")

    def test_invalid_interval_cannot_launch_a_build(self):
        for seconds in ("0", "-1", "nan", "inf"):
            with self.subTest(seconds=seconds):
                result, outputs = self.run_slice("raise SystemExit(99)", seconds)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(outputs, "")

    def test_missing_command_is_rejected(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], env=self.env, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
