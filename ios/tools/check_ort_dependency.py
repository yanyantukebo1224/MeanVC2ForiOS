#!/usr/bin/env python3
"""Validate reproducible ONNX Runtime CocoaPods source provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_C_URL = (
    "https://github.com/baijum/ukulele-companion/releases/download/"
    "vendor-onnxruntime-c-1.24.3/pod-archive-onnxruntime-c-1.24.3.zip"
)
EXPECTED_C_SHA256 = "b7eedc45932bac758ffd057cac0feb3f682269e47750b159e4c865145cbf0a8e"
EXPECTED_OBJC_COMMIT = "3a728b75062256951b6e19ce718907cf1a1d4cf0"
EXPECTED_OBJC_URL = (
    "https://codeload.github.com/microsoft/onnxruntime/legacy.zip/"
    + EXPECTED_OBJC_COMMIT
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(root: Path) -> None:
    c_spec = json.loads(
        (root / "ios/Podspecs/onnxruntime-c.podspec.json").read_text(encoding="utf-8")
    )
    objc_spec = json.loads(
        (root / "ios/Podspecs/onnxruntime-objc.podspec.json").read_text(encoding="utf-8")
    )
    podfile = (root / "ios/Podfile").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/ios-build.yml").read_text(encoding="utf-8")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    require(c_spec["name"] == "onnxruntime-c" and c_spec["version"] == "1.24.3",
            "C/C++ pod identity or version changed")
    require(c_spec["license"]["type"] == "MIT", "C/C++ pod license is not MIT")
    require(c_spec["source"]["http"] == EXPECTED_C_URL,
            "C/C++ pod archive URL changed")
    require(c_spec["source"]["sha256"] == EXPECTED_C_SHA256,
            "C/C++ pod archive SHA-256 changed")

    require(objc_spec["name"] == "onnxruntime-objc" and objc_spec["version"] == "1.24.3",
            "Objective-C pod identity or version changed")
    require(objc_spec["license"]["type"] == "MIT", "Objective-C pod license is not MIT")
    require(objc_spec["source"] == {"http": EXPECTED_OBJC_URL, "flatten": True},
            "Objective-C source must use the immutable upstream commit archive")
    require(objc_spec["subspecs"][0]["dependencies"]["onnxruntime-c"] == ["1.24.3"],
            "Objective-C pod must depend on the matching C/C++ runtime")

    require("Podspecs/onnxruntime-c.podspec.json" in podfile,
            "Podfile does not include the pinned C/C++ podspec")
    require("Podspecs/onnxruntime-objc.podspec.json" in podfile,
            "Podfile does not include the pinned Objective-C podspec")
    require("download.onnxruntime.ai" not in podfile,
            "Podfile still references the unavailable download host")

    require("/commits/" + EXPECTED_OBJC_COMMIT in workflow,
            "CI must verify the immutable Objective-C source commit")
    require(EXPECTED_OBJC_URL in workflow,
            "CI archive verification URL differs from the podspec")
    require("refs/heads/" not in workflow and "refs/tags/" not in workflow,
            "CI must not download a mutable branch or tag archive")
    require(EXPECTED_OBJC_COMMIT in notices,
            "Third-party notices must document the exact Objective-C source commit")

    print(
        "ONNX Runtime provenance OK: C/C++ archive SHA-256 pinned; "
        "Objective-C sources pinned to immutable commit " + EXPECTED_OBJC_COMMIT + "."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    try:
        validate(args.repo_root.resolve())
    except (AssertionError, KeyError, OSError, ValueError) as error:
        raise SystemExit("ONNX Runtime provenance check failed: " + str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
