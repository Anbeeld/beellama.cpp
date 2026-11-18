#!/usr/bin/env python3
"""Execute the workflow's VS discovery lines under CMD, including quoted arguments."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "Windows CMD required")
class VswhereCommandTests(unittest.TestCase):
    def test_configure_and_slice_find_vs2022_with_spaces_and_parentheses(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        cpu = workflow.split("\n  windows-cpu:\n", 1)[1].split("\n  windows-sycl:\n", 1)[0]
        commands = re.findall(r'^\s*(for /f .+do set "VS_INSTALL_DIR=%%i")$', cpu, re.MULTILINE)
        self.assertEqual(len(commands), 2)
        with tempfile.TemporaryDirectory(prefix="VS discovery (test) ") as directory:
            root = Path(directory)
            installer = root / "Microsoft Visual Studio/Installer"
            installer.mkdir(parents=True)
            # Substitute only the executable, preserving the workflow's CMD
            # invocation, path quoting, version range, and FOR /F parsing.
            (installer / "vswhere.cmd").write_text('@echo off\necho C:\\Program Files\\VS 2022\n')
            for index, command in enumerate(commands):
                with self.subTest(invocation=index):
                    command = command.replace("vswhere.exe", "vswhere.cmd")
                    batch = root / f"discovery-{index}.cmd"
                    batch.write_text('@echo off\nset "VS_INSTALL_DIR="\n' + command +
                                     '\nif not defined VS_INSTALL_DIR exit /b 1\necho FOUND=%VS_INSTALL_DIR%\n')
                    result = subprocess.run(
                        [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(batch)],
                        env={**os.environ, "ProgramFiles(x86)": str(root)},
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("FOUND=C:\\Program Files\\VS 2022", result.stdout)
                    self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
