#!/usr/bin/env python3
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMMON = (
    "index.html",
    "manifest.webmanifest",
    "sw.js",
    "build.json",
    "_app/version.json",
)
LEGACY = COMMON + ("bundle.hash.js", "bundle.hash.css", "workbox-hash.js")
SVELTEKIT = COMMON + (
    "_app/immutable/entry/start.hash.js",
    "_app/immutable/assets/app.hash.css",
    "service-worker.js",
    "_app/immutable/assets/pdf.worker.min.hash.mjs",
)


class UIAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "source"
        self.output = self.folder / "output"

    def emit(self, files: tuple[str, ...], gzip: bool = False) -> subprocess.CompletedProcess:
        for name in files:
            asset = self.source / "dist" / name
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_text("fixture", encoding="utf-8")
        return subprocess.run(
            [
                "cmake",
                f"-DUI_SOURCE_DIR={self.source.as_posix()}",
                f"-DUI_BINARY_DIR={self.output.as_posix()}",
                f"-DLLAMA_SOURCE_DIR={ROOT.as_posix()}",
                "-DHF_ENABLED=OFF",
                "-DBUILD_UI=OFF",
                f"-DLLAMA_UI_GZIP={'ON' if gzip else 'OFF'}",
                "-P", str(ROOT / "scripts/ui-assets.cmake"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def check_embedded(self, files: tuple[str, ...], gzip: bool = False) -> str:
        result = self.emit(files, gzip)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"embedded {len(files)} assets", result.stdout)
        generated = (self.output / "ui.cpp").read_text(encoding="utf-8")
        for name in files:
            self.assertIn(f'{{ "{name}",', generated)
        return generated

    def test_legacy_workbox_layout(self) -> None:
        self.check_embedded(LEGACY)

    def test_single_bundle_service_worker_layout(self) -> None:
        files = COMMON + (
            "_app/immutable/bundle.hash.js",
            "_app/immutable/assets/bundle.hash.css",
            "service-worker.js",
        )
        self.check_embedded(files)

    def test_mjs_mime_with_legacy_layout(self) -> None:
        generated = self.check_embedded(LEGACY + ("pdf.worker.min.hash.mjs",))
        row = next(line for line in generated.splitlines() if '"pdf.worker.min.hash.mjs"' in line)
        self.assertIn('"application/javascript"', row)

    def test_split_sveltekit_layout_and_mjs_mime(self) -> None:
        generated = self.check_embedded(SVELTEKIT)
        row = next(
            line for line in generated.splitlines()
            if '"_app/immutable/assets/pdf.worker.min.hash.mjs"' in line
        )
        self.assertIn('"application/javascript"', row)

    def test_split_sveltekit_gzip(self) -> None:
        generated = self.check_embedded(SVELTEKIT, gzip=True)
        self.assertIn("return true;", generated)
        worker = self.output / "ui-gzip/_gzip/_app/immutable/assets/pdf.worker.min.hash.mjs"
        self.assertEqual(worker.read_bytes()[:2], b"\x1f\x8b")

    def test_incomplete_layout_is_rejected(self) -> None:
        result = self.emit(("index.html",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UI: missing required asset(s)", result.stderr)
        self.assertIn("_app/immutable/entry/start[hash].js (or bundle[hash].js)", result.stderr)
        self.assertIn("_app/immutable/assets/[name][hash].css (or bundle[hash].css)", result.stderr)
        self.assertFalse((self.output / "ui.cpp").exists())

    def test_no_assets_still_supported(self) -> None:
        result = self.emit(())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("embedded 0 assets", result.stdout)
        self.assertNotIn("#define LLAMA_UI_HAS_ASSETS", (self.output / "ui.h").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
