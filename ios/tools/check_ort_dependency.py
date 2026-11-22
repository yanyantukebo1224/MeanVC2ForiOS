#!/usr/bin/env python3
"""Check that ONNX Runtime pods are pinned to reproducible upstream artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
c_spec = json.loads((ROOT / "ios/Podspecs/onnxruntime-c.podspec.json").read_text())
objc_spec = json.loads((ROOT / "ios/Podspecs/onnxruntime-objc.podspec.json").read_text())
podfile = (ROOT / "ios/Podfile").read_text()

expected_c_url = (
    "https://github.com/baijum/ukulele-companion/releases/download/"
    "vendor-onnxruntime-c-1.24.3/pod-archive-onnxruntime-c-1.24.3.zip"
)
expected_c_sha = "b7eedc45932bac758ffd057cac0feb3f682269e47750b159e4c865145cbf0a8e"
expected_objc_ref = "rel-1.24.3"
expected_objc_commit = "3a728b75062256951b6e19ce718907cf1a1d4cf0"
expected_objc_url = (
    "https://codeload.github.com/microsoft/onnxruntime/legacy.zip/refs/heads/"
    + expected_objc_ref
)

assert c_spec["name"] == "onnxruntime-c" and c_spec["version"] == "1.24.3"
assert c_spec["license"]["type"] == "MIT"
assert c_spec["source"]["http"] == expected_c_url
assert c_spec["source"]["sha256"] == expected_c_sha

assert objc_spec["name"] == "onnxruntime-objc" and objc_spec["version"] == "1.24.3"
assert objc_spec["license"]["type"] == "MIT"
assert objc_spec["source"] == {"http": expected_objc_url, "flatten": True}
assert "v1.24.3" not in objc_spec["source"]["http"]

workflow = (ROOT / ".github/workflows/ios-build.yml").read_text()
assert f"commits/{expected_objc_ref}" in workflow
assert f"refs/heads/{expected_objc_ref}" in workflow
assert expected_objc_commit in workflow
assert objc_spec["subspecs"][0]["dependencies"]["onnxruntime-c"] == ["1.24.3"]

assert "Podspecs/onnxruntime-c.podspec.json" in podfile
assert "Podspecs/onnxruntime-objc.podspec.json" in podfile
assert "download.onnxruntime.ai" not in podfile
assert "onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-objc" not in json.dumps(objc_spec)

print(
    "ONNX Runtime dependency provenance OK: C XCFramework SHA-256 pinned; "
    f"Objective-C wrapper pinned to upstream branch {expected_objc_ref} ({expected_objc_commit})."
)
