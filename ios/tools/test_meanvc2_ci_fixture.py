#!/usr/bin/env python3
"""Validate deterministic non-degenerate CI inputs and optional Swift parity."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from meanvc2_ci_fixture import (
    CHUNK_COUNT,
    CHUNK_SAMPLES,
    float32_bytes,
    noise_values,
    speaker_embedding,
    waveform_chunk,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swift-fixture-dir", type=Path)
    args = parser.parse_args()

    chunks = [waveform_chunk(index) for index in range(CHUNK_COUNT)]
    waveform = [sample for chunk in chunks for sample in chunk]
    speaker = speaker_embedding()
    noise = noise_values(16)

    assert len(waveform) == CHUNK_COUNT * CHUNK_SAMPLES
    assert all(math.isfinite(value) for value in waveform)
    assert max(abs(value) for value in waveform) > 0.01
    assert chunks[0] != chunks[1]
    assert len(speaker) == 256 and all(math.isfinite(value) for value in speaker)
    speaker_norm = math.sqrt(sum(value * value for value in speaker))
    assert abs(speaker_norm - 1.0) < 1e-6
    assert len(noise) == 16 * 80 and any(value != 0 for value in noise)
    assert waveform_chunk(0) == waveform_chunk(0)
    assert speaker_embedding() == speaker_embedding()
    assert noise_values(16) == noise_values(16)

    if args.swift_fixture_dir:
        expected = {
            "waveform.f32": float32_bytes(waveform),
            "speaker.f32": float32_bytes(speaker),
            "noise.f32": float32_bytes(noise),
        }
        for name, payload in expected.items():
            actual = (args.swift_fixture_dir / name).read_bytes()
            if actual != payload:
                raise SystemExit(f"Swift/Python fixture mismatch: {name}")
        print("Swift/Python fixture byte parity OK (waveform, speaker, noise)")

    print(
        "CI fixture OK: "
        f"{CHUNK_COUNT} non-zero chunks, normalized 256-D conditioning, "
        f"{len(noise)} non-zero noise values."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
