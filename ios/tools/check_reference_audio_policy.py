#!/usr/bin/env python3
"""Verify reference-audio duration limits are enforced before large allocations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "ios/MeanVC2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"reference audio policy check failed: {message}")


def main() -> int:
    policy = (APP / "MeanVC2ReferenceAudioPolicy.swift").read_text(encoding="utf-8")
    audio_file = (APP / "MeanVC2AudioFile.swift").read_text(encoding="utf-8")
    encoder = (APP / "MeanVC2SpeakerEncoder.swift").read_text(encoding="utf-8")
    view = (APP / "ContentView.swift").read_text(encoding="utf-8")
    project = (APP / "MeanVC2.xcodeproj/project.pbxproj").read_text(encoding="utf-8")
    tests = (ROOT / "ios/tools/ReferenceAudioPolicyTests.swift").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ios-build.yml").read_text(encoding="utf-8")

    require("minimumDurationSeconds = 3.0" in policy, "three-second minimum is missing")
    require("maximumDurationSeconds = 20.0" in policy, "twenty-second maximum is missing")
    require("MeanVC2ReferenceAudioPolicy.validationError(" in audio_file,
            "file duration must be checked before decoding")
    require(
        audio_file.index("MeanVC2ReferenceAudioPolicy.validationError(")
        < audio_file.index("AVAudioPCMBuffer("),
        "duration must be checked before allocating the full source buffer",
    )
    require("MeanVC2ReferenceAudioPolicy.validationError(" in encoder,
            "speaker encoder must validate the resampled sample count")
    require(
        encoder.index("MeanVC2ReferenceAudioPolicy.validationError(")
        < encoder.index("ORTTensor.float(waveform"),
        "sample count must be checked before constructing the ONNX tensor",
    )
    require("3〜20秒" in view, "reference picker guidance must state the supported range")
    require("MeanVC2ReferenceAudioPolicy.swift in Sources" in project,
            "duration policy must be compiled into the app")
    for marker in ("3.0", "20.0", "2.999", "20.001", ".nan", ".infinity", "47_999", "320_001"):
        require(marker in tests, f"missing boundary test: {marker}")
    require("ReferenceAudioPolicyTests.swift" in workflow,
            "macOS CI must compile and run the duration tests")

    print("Reference audio policy integration OK: 3–20s checked before decode allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
