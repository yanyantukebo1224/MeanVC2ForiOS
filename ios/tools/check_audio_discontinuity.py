#!/usr/bin/env python3
"""Ensure microphone overload cannot silently corrupt streaming cache continuity."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "ios/MeanVC2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"audio discontinuity check failed: {message}")


def main() -> int:
    controller = (APP / "MeanVC2LiveAudioController.swift").read_text(encoding="utf-8")
    helper = (APP / "MeanVC2InputDiscontinuity.swift").read_text(encoding="utf-8")
    project = (APP / "MeanVC2.xcodeproj/project.pbxproj").read_text(encoding="utf-8")
    tests = (ROOT / "ios/tools/InputDiscontinuityTests.swift").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ios-build.yml").read_text(encoding="utf-8")

    require("private let inputDiscontinuity = MeanVC2InputDiscontinuity()" in controller,
            "audio controller must own the gap signal")
    require("inputDiscontinuity.markDroppedBuffer()" in controller,
            "the tap must mark a dropped input buffer")
    require("inputDiscontinuity.consumePendingGap()" in controller,
            "the serial processing queue must consume the gap")
    require("try stream.reset()" in controller,
            "ASR/DiT/Vocos caches must reset after an input gap")
    require("pendingInput.removeAll(keepingCapacity: true)" in controller,
            "partial samples spanning a missing buffer must be discarded")
    require("playbackGeneration &+= 1" in controller and "player.stop()" in controller,
            "queued audio from the pre-gap stream must be invalidated and flushed")
    require("private let lock = NSLock()" in helper,
            "cross-thread gap state must be protected by a lock")
    require("DispatchQueue.concurrentPerform" in tests,
            "gap coalescing must have a concurrent regression test")
    require("MeanVC2InputDiscontinuity.swift in Sources" in project,
            "gap helper must be compiled into the app")
    require("InputDiscontinuityTests.swift" in workflow,
            "macOS CI must compile and run the gap tests")

    print("Audio discontinuity integration OK: dropped buffers reset caches and stale playback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
