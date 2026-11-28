#!/usr/bin/env python3
"""Regression tests for immutable ONNX Runtime dependency pinning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import check_ort_dependency as checker


class DependencyProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "ios/Podspecs").mkdir(parents=True)
        (self.root / ".github/workflows").mkdir(parents=True)
        c_spec = {
            "name": "onnxruntime-c",
            "version": "1.24.3",
            "license": {"type": "MIT"},
            "source": {
                "http": checker.EXPECTED_C_URL,
                "sha256": checker.EXPECTED_C_SHA256,
            },
        }
        objc_spec = {
            "name": "onnxruntime-objc",
            "version": "1.24.3",
            "license": {"type": "MIT"},
            "source": {"http": checker.EXPECTED_OBJC_URL, "flatten": True},
            "subspecs": [{"dependencies": {"onnxruntime-c": ["1.24.3"]}}],
        }
        (self.root / "ios/Podspecs/onnxruntime-c.podspec.json").write_text(
            json.dumps(c_spec), encoding="utf-8"
        )
        (self.root / "ios/Podspecs/onnxruntime-objc.podspec.json").write_text(
            json.dumps(objc_spec), encoding="utf-8"
        )
        (self.root / "ios/Podfile").write_text(
            "Podspecs/onnxruntime-c.podspec.json\n"
            "Podspecs/onnxruntime-objc.podspec.json\n",
            encoding="utf-8",
        )
        (self.root / ".github/workflows/ios-build.yml").write_text(
            'url="' + checker.EXPECTED_OBJC_URL + '"\n'
            'curl --fail --location "$url" -o build/onnxruntime-objc-source.zip\n'
            'unzip -tq build/onnxruntime-objc-source.zip\n'
            "grep objectivec/include/onnxruntime.h\n"
            "grep /LICENSE\n",
            encoding="utf-8",
        )
        (self.root / "THIRD_PARTY_NOTICES.md").write_text(
            "The Objective-C API sources are pinned to the immutable ONNX Runtime commit\n"
            + "`" + checker.EXPECTED_OBJC_COMMIT + "` in the Microsoft repository.\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accepts_immutable_commit_archive(self) -> None:
        checker.validate(self.root)

    def test_rejects_mutable_release_branch(self) -> None:
        path = self.root / "ios/Podspecs/onnxruntime-objc.podspec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["source"]["http"] = (
            "https://codeload.github.com/microsoft/onnxruntime/legacy.zip/"
            "refs/heads/rel-1.24.3"
        )
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "immutable upstream commit"):
            checker.validate(self.root)

    def test_rejects_different_commit_in_ci_archive_url(self) -> None:
        path = self.root / ".github/workflows/ios-build.yml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                checker.EXPECTED_OBJC_COMMIT, "0" * 40
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "immutable Objective-C source commit"):
            checker.validate(self.root)

    def test_rejects_legacy_commit_archive_endpoint(self) -> None:
        path = self.root / "ios/Podspecs/onnxruntime-objc.podspec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["source"]["http"] = (
            "https://codeload.github.com/microsoft/onnxruntime/legacy.zip/"
            + checker.EXPECTED_OBJC_COMMIT
        )
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "immutable upstream commit archive"):
            checker.validate(self.root)

    def test_rejects_github_api_preflight(self) -> None:
        path = self.root / ".github/workflows/ios-build.yml"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "curl https://api.github.com/repos/microsoft/onnxruntime/commits/"
            + checker.EXPECTED_OBJC_COMMIT
            + "\\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "must not depend on the GitHub REST API"):
            checker.validate(self.root)

    def test_rejects_unpinned_c_archive(self) -> None:
        path = self.root / "ios/Podspecs/onnxruntime-c.podspec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["source"]["sha256"] = "0" * 64
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
            checker.validate(self.root)


if __name__ == "__main__":
    unittest.main()
