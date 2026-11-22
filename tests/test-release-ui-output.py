#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_source_ui", ROOT / ".github/scripts/verify-source-ui.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceUIOutputTests(unittest.TestCase):
    def check_layout(self, layout: str, content: bytes | None) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            cmake = source / layout / "CMakeLists.txt"
            cmake.parent.mkdir(parents=True)
            cmake.write_text("# fixture", encoding="utf-8")
            output = source / "build" / layout / "dist/index.html"
            if content is not None:
                output.parent.mkdir(parents=True)
                output.write_bytes(content)
            if content:
                self.assertEqual(MODULE.verify_source_ui(source, source / "build"), output)
            else:
                with self.assertRaisesRegex(RuntimeError, "Nonempty source-built UI"):
                    MODULE.verify_source_ui(source, source / "build")

    def test_modern_nonempty(self) -> None:
        self.check_layout("ui/apps/chat", b"<html>built</html>")

    def test_legacy_nonempty(self) -> None:
        self.check_layout("tools/ui", b"<html>built</html>")

    def test_missing_and_empty_outputs(self) -> None:
        for layout in ("ui/apps/chat", "tools/ui"):
            for content in (None, b""):
                with self.subTest(layout=layout, content=content):
                    self.check_layout(layout, content)

    def test_stale_legacy_output_cannot_satisfy_modern_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for layout in ("ui/apps/chat", "tools/ui"):
                cmake = source / layout / "CMakeLists.txt"
                cmake.parent.mkdir(parents=True)
                cmake.touch()
            stale = source / "build/tools/ui/dist/index.html"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Nonempty source-built UI"):
                MODULE.verify_source_ui(source, source / "build")

    def test_cli_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            cmake = source / "ui/apps/chat/CMakeLists.txt"
            cmake.parent.mkdir(parents=True)
            cmake.touch()
            command = [sys.executable, str(ROOT / ".github/scripts/verify-source-ui.py")]
            missing = subprocess.run(command, cwd=source, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 1)
            self.assertIn("Nonempty source-built UI", missing.stderr)
            output = source / "build/ui/apps/chat/dist/index.html"
            output.parent.mkdir(parents=True)
            output.write_text("built", encoding="utf-8")
            present = subprocess.run(command, cwd=source, capture_output=True, text=True)
            self.assertEqual(present.returncode, 0, present.stderr)

    def test_unknown_layout(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "Cannot identify"):
                MODULE.verify_source_ui(source, source / "build")


if __name__ == "__main__":
    unittest.main()
