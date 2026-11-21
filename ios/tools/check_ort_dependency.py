#!/usr/bin/env python3
"""Check that the local ONNX Runtime podspecs remain version- and hash-pinned."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
c_spec = json.loads((ROOT / "ios/Podspecs/onnxruntime-c.podspec.json").read_text())
objc_spec = json.loads((ROOT / "ios/Podspecs/onnxruntime-objc.podspec.json").read_text())
podfile = (ROOT / "ios/Podfile").read_text()

expected_url = (
    "https://github.com/baijum/ukulele-companion/releases/download/"
    "vendor-onnxruntime-c-1.24.3/pod-archive-onnxruntime-c-1.24.3.zip"
)
expected_sha = "b7eedc45932bac758ffd057cac0feb3f682269e47750b159e4c865145cbf0a8e"

assert c_spec["name"] == "onnxruntime-c" and c_spec["version"] == "1.24.3"
assert c_spec["license"]["type"] == "MIT"
assert c_spec["source"]["http"] == expected_url
assert c_spec["source"]["sha256"] == expected_sha
assert objc_spec["name"] == "onnxruntime-objc" and objc_spec["version"] == "1.24.3"
assert objc_spec["license"]["type"] == "MIT"
assert objc_spec["source"]["http"] == "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-objc-1.24.3.zip"
objc_sha = objc_spec["source"].get("sha256")
assert objc_sha is None or (
    len(objc_sha) == 64 and all(character in "0123456789abcdef" for character in objc_sha)
)
assert objc_spec["subspecs"][0]["dependencies"]["onnxruntime-c"] == ["1.24.3"]
assert "Podspecs/onnxruntime-c.podspec.json" in podfile
assert "Podspecs/onnxruntime-objc.podspec.json" in podfile
assert "download.onnxruntime.ai" not in podfile
assert "download.onnxruntime.ai" not in json.dumps(c_spec)
assert "download.onnxruntime.ai" not in json.dumps(objc_spec)

print("ONNX Runtime dependency provenance OK: v1.24.3 pod archives pinned; ObjC SHA pending CI mirror verification.")
