"""Exercise the CUDA archive installer with local HTTP fixtures, never NVIDIA downloads."""

import ctypes
import functools
import http.server
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/install-cuda-archives.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def ps_string(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(POWERSHELL and shutil.which("curl.exe"), "PowerShell and curl.exe required")
class InstallCudaArchivesTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="cuda installer test ")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        self.downloads = self.root / "downloads"
        self.downloads.mkdir()
        self.runner_temp = self.root / "runner temp"
        self.runner_temp.mkdir()
        self.destination = self.root / "CUDA destination with spaces"
        handler = functools.partial(QuietHandler, directory=str(self.downloads))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def archive(self, name, entries):
        with zipfile.ZipFile(self.downloads / name, "w") as archive:
            for path, content in entries.items():
                archive.writestr(path, content)
        return name

    def install(self, *names):
        urls = [f"http://127.0.0.1:{self.server.server_port}/{name}" for name in names]
        command = (
            f"& {ps_string(SCRIPT)} -Destination {ps_string(self.destination)} "
            f"-Urls @({', '.join(ps_string(url) for url in urls)})"
        )
        env = os.environ.copy()
        env.update(TEMP=str(self.runner_temp), TMP=str(self.runner_temp))
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True, text=True, env=env, timeout=60,
        )

    def assert_failed(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def test_multiple_archives_merge_shared_directories_and_preserve_spaces(self):
        first = self.archive("first.zip", {"first root/bin/first tool.exe": "first", "first root/include/shared.h": "old"})
        second = self.archive("second.zip", {"second root/bin/second.exe": "second", "second root/include/shared.h": "new"})
        # Existing destination contents and unrelated ZIPs must not be removed or extracted.
        self.destination.mkdir()
        (self.destination / "keep.txt").write_text("keep")
        self.archive("unrequested.zip", {"unexpected/bin/bad.exe": "bad"})
        result = self.install(first, second)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = {"keep.txt": "keep", "bin/first tool.exe": "first", "bin/second.exe": "second", "include/shared.h": "new"}
        actual = {path.relative_to(self.destination).as_posix(): path.read_text() for path in self.destination.rglob("*") if path.is_file()}
        self.assertEqual(actual, expected)
        workdirs = list(self.runner_temp.glob("cuda-archives-*"))
        self.assertEqual(len(workdirs), 1)
        self.assertTrue((workdirs[0] / "archive-0.zip").is_file())
        self.assertTrue((workdirs[0] / "archive-1.zip").is_file())

    def test_invalid_zip_fails(self):
        (self.downloads / "bad.zip").write_text("not a zip")
        result = self.install("bad.zip")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(self.destination.rglob("*")))

    def test_http_error_stops_before_extraction(self):
        valid = self.archive("valid.zip", {"root/bin/tool.exe": "ok"})
        result = self.install("missing.zip", valid)
        self.assert_failed(result, "CUDA archive download failed")
        self.assertFalse(self.destination.exists())
        self.assertFalse(any(self.runner_temp.rglob("archive-1.zip")))

    def test_multiple_roots_fail(self):
        name = self.archive("roots.zip", {"one/a": "a", "two/b": "b"})
        self.assert_failed(self.install(name), "exactly one root directory")

    def test_root_file_fails(self):
        name = self.archive("file.zip", {"file.txt": "a"})
        self.assert_failed(self.install(name), "exactly one root directory")

    def test_empty_zip_fails(self):
        name = self.archive("empty.zip", {})
        self.assert_failed(self.install(name), "exactly one root directory")

    def test_empty_root_fails(self):
        name = self.archive("empty-root.zip", {"root/": ""})
        self.assert_failed(self.install(name), "root must not be empty")

    @unittest.skipUnless(os.name == "nt", "Windows exclusive file lock required")
    def test_copy_failure_stops_before_later_archive(self):
        first = self.archive("first.zip", {"root/bin/tool.exe": "first"})
        second = self.archive("second.zip", {"root/later.txt": "later"})
        (self.destination / "bin").mkdir(parents=True)
        target = self.destination / "bin/tool.exe"
        target.write_text("locked original")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.CreateFileW(str(target), 0x80000000, 0, None, 3, 0, None)
        self.assertNotEqual(handle, ctypes.c_void_p(-1).value, ctypes.get_last_error())
        try:
            result = self.install(first, second)
        finally:
            kernel32.CloseHandle(handle)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.destination / "later.txt").exists())
        self.assertEqual(target.read_text(), "locked original")


if __name__ == "__main__":
    unittest.main()
