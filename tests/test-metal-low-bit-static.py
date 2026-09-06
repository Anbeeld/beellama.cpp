#!/usr/bin/env python3

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
METAL_DIR = ROOT / "ggml/src/ggml-metal"
DIGEST_SCRIPT = METAL_DIR / "cmake/append-source-digest.cmake"


def function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"missing Metal function: {signature}")
    end = source.find("\n}", start)
    if end < 0:
        raise AssertionError(f"unterminated Metal function: {signature}")
    return source[start : end + 2]


class MetalLowBitTests(unittest.TestCase):
    def test_low_bit_quantizers_are_defined(self):
        source = (METAL_DIR / "kernels/quantize.h").read_text(encoding="utf-8")

        q2_0s = function_body(source, "void quantize_q2_0s(")
        self.assertIn("quantize_planar_values<4, false>", q2_0s)
        self.assertIn("dst.d = d;", q2_0s)
        self.assertIn("dst.qs[j] = qs[j];", q2_0s)

        q2_1 = function_body(source, "void quantize_q2_1(")
        self.assertIn("quantize_planar_values<4, true>", q2_1)
        self.assertIn("dst.d = d;", q2_1)
        self.assertIn("dst.m = m;", q2_1)
        self.assertIn("dst.qs[j] = qs[j];", q2_1)

    def test_bfloat_paths_use_explicit_conversions(self):
        dequantize = (METAL_DIR / "kernels/dequantize.h").read_text(encoding="utf-8")
        iq4_nl = function_body(dequantize, "void dequantize_iq4_nl(")
        self.assertIn("float4x4 tmp;", iq4_nl)
        self.assertIn("reg = (type4x4) tmp;", iq4_nl)

        quantize = (METAL_DIR / "kernels/quantize.metal").read_text(encoding="utf-8")
        get_rows = function_body(quantize, "kernel void kernel_get_rows_f(")
        self.assertIn("pdst[ind] = (T) psrc[ind];", get_rows)

    def test_embedded_source_digest_is_content_sensitive_and_space_safe(self):
        self.assertTrue(DIGEST_SCRIPT.is_file(), f"missing digest helper: {DIGEST_SCRIPT}")

        cmake_source = (METAL_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('set(METALLIB_APPEND_SOURCE_DIGEST "${CMAKE_CURRENT_SOURCE_DIR}/cmake/append-source-digest.cmake")', cmake_source)
        self.assertIn('"-DGGML_METAL_EMBED_SOURCE=${EMBED}"', cmake_source)
        self.assertIn('"-DGGML_METAL_EMBED_ASM=${ASM}"', cmake_source)
        self.assertIn('kernels/${kind}.metal "${METALLIB_APPEND_SOURCE_DIGEST}"', cmake_source)

        with tempfile.TemporaryDirectory(prefix="metal embed digest ") as directory:
            work = Path(directory)
            source = work / "kernel source.metal"
            assembly = work / "kernel embed.s"

            digests = []
            for contents in ("kernel v1\n", "kernel v2\n"):
                source.write_text(contents, encoding="utf-8")
                assembly.write_text(".incbin \"kernel source.metal\"\n", encoding="utf-8")
                subprocess.run(
                    [
                        "cmake",
                        f"-DGGML_METAL_EMBED_SOURCE={source}",
                        f"-DGGML_METAL_EMBED_ASM={assembly}",
                        "-P",
                        str(DIGEST_SCRIPT),
                    ],
                    check=True,
                )
                output = assembly.read_text(encoding="utf-8")
                digest = hashlib.md5(source.read_bytes()).hexdigest()
                self.assertIn(f"/* src-md5: {digest} */", output)
                digests.append(digest)

            self.assertNotEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
