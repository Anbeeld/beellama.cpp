#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/resolve-release-cache-parent.py"


def load_resolver():
    if not SCRIPT.is_file():
        raise AssertionError(f"missing release cache parent resolver: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("release_cache_parent", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load release cache parent resolver: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseCacheParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = load_resolver()

    def test_selects_nearest_lower_ancestor(self):
        branches = {
            "v0.4.0": "sha-040",
            "v0.4.1": "sha-041",
            "v0.4.2": "sha-042",
            "v0.5.0": "sha-050",
        }

        parent = self.resolver.select_cache_parent(
            "v0.4.2",
            branches,
            lambda sha: sha in {"sha-040", "sha-041"},
        )

        self.assertEqual(parent, "v0.4.1")

    def test_skips_non_ancestor_predecessor(self):
        branches = {
            "v0.3.2": "sha-032",
            "v0.4.0": "sha-040",
            "v0.4.1": "sha-041",
            "v0.4.2": "sha-042",
        }

        parent = self.resolver.select_cache_parent(
            "v0.4.2",
            branches,
            lambda sha: sha == "sha-040",
        )

        self.assertEqual(parent, "v0.4.0")

    def test_compares_numeric_version_components(self):
        branches = {
            "v0.9.9": "sha-099",
            "v0.10.0": "sha-0100",
            "v0.10.1": "sha-0101",
        }

        parent = self.resolver.select_cache_parent(
            "v0.10.1",
            branches,
            lambda _sha: True,
        )

        self.assertEqual(parent, "v0.10.0")

    def test_ignores_non_release_names_and_missing_predecessors(self):
        branches = {
            "v0.4.1-preview": "sha-preview",
            "v0.4": "sha-short",
            "feature": "sha-feature",
        }

        self.assertEqual(
            self.resolver.select_cache_parent("v0.4.2", branches, lambda _sha: True),
            "",
        )
        self.assertEqual(
            self.resolver.select_cache_parent("v0.4.2-rc1", branches, lambda _sha: True),
            "",
        )


if __name__ == "__main__":
    unittest.main()
