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

    def test_configure_and_slice_override_bundled_clang_after_vcvars(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        cpu = workflow.split("\n  windows-cpu:\n", 1)[1].split("\n  windows-sycl:\n", 1)[0]
        configure = cpu.split("      - name: Configure\n", 1)[1].split("        run: |\n", 1)[1]
        configure = configure.split("          cmake -S", 1)[0]
        compile_command = cpu.split("          command: |\n", 1)[1].split("              python ", 1)[0]
        with tempfile.TemporaryDirectory(prefix="LLVM selection (test) ") as directory:
            root = Path(directory)
            vs = root / "VS 2022"
            vcvars = vs / "VC/Auxiliary/Build"
            vcvars.mkdir(parents=True)
            old = root / "bundled clang"
            standalone = root / "LLVM/bin"
            old.mkdir()
            standalone.mkdir(parents=True)
            (vcvars / "vcvarsall.bat").write_text(f'@set "PATH={old};%PATH%"\n')
            for tool in ("clang", "clang++"):
                (old / f"{tool}.cmd").write_text('@echo Clang 19\n')
                (standalone / f"{tool}.cmd").write_text('@echo Clang 20.1.8\n')
            installer = root / "Microsoft Visual Studio/Installer"
            installer.mkdir(parents=True)
            (installer / "vswhere.cmd").write_text(f'@echo off\necho {vs}\n')
            for index, command in enumerate((configure, compile_command)):
                with self.subTest(invocation=index):
                    command = "\n".join(line.strip() for line in command.splitlines())
                    command = command.replace("vswhere.exe", "vswhere.cmd")
                    # A .cmd fixture needs CALL to return, unlike the real .exe.
                    command = command.replace('"%ProgramFiles%\\LLVM\\bin\\clang.exe"',
                                              'call "%ProgramFiles%\\LLVM\\bin\\clang.cmd"')
                    # CMD restores ProgramFiles from Windows on this host; use
                    # a fixture variable without changing the path-prepend logic.
                    command = command.replace("%ProgramFiles%", "%TEST_PROGRAMFILES%")
                    command = command.replace("%ProgramFiles(x86)%", "%TEST_PROGRAMFILES%")
                    batch = root / f"toolchain-{index}.cmd"
                    batch.write_text('@echo off\n' + command + '\ncall clang.cmd\ncall clang++.cmd\n')
                    result = subprocess.run(
                        [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(batch)],
                        env={**os.environ, "TEST_PROGRAMFILES": str(root)},
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("Clang 19", result.stdout)
                    self.assertEqual(result.stdout.count("Clang 20.1.8"), 3, result.stdout)
                    self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
