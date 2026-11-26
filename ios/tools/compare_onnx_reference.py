#!/usr/bin/env python3
"""Compare the Swift simulator output with the pinned Python ONNX host loop."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from meanvc2_ci_fixture import (
    CHUNK_COUNT,
    noise_values,
    speaker_embedding,
    waveform_chunk,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--swift-output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reference_root / "tool"))
    from onnx_stream import OnnxStreamingVC

    runner = OnnxStreamingVC(
        args.model_dir,
        suffix={"asr": ".int8", "dit": ".int8", "vocos": ".int8"},
        threads=2,
        steps=2,
    )
    runner.noise_fn = lambda n: np.asarray(
        noise_values(n), dtype=np.float32
    ).reshape(1, n, 80)
    runner.set_target(
        np.asarray(speaker_embedding(), dtype=np.float32).reshape(1, 256)
    )

    pieces = []
    for index in range(CHUNK_COUNT):
        chunk = np.asarray(waveform_chunk(index), dtype=np.float32)
        pieces.append(runner.process(chunk))
    expected = np.concatenate(pieces).astype(np.float32, copy=False)
    actual = np.fromfile(args.swift_output, dtype=np.float32)

    if expected.size != actual.size:
        print(f"sample-count mismatch: Swift={actual.size}, reference={expected.size}")
        return 1
    if expected.size == 0:
        print("reference produced no waveform")
        return 1
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        print("non-finite samples found in Swift or reference output")
        return 1
    if not np.any(expected != 0) or not np.any(actual != 0):
        print("degenerate all-zero output: non-zero CI fixture did not exercise synthesis")
        return 1

    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    denom = np.linalg.norm(actual) * np.linalg.norm(expected)
    if denom == 0:
        print("cannot compute cosine similarity for zero-energy output")
        return 1
    cosine = float(np.dot(actual, expected) / denom)
    print(
        f"Swift/Python ONNX parity: samples={actual.size}, "
        f"max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}, cosine={cosine:.9f}"
    )
    if not np.allclose(actual, expected, atol=1e-3, rtol=1e-3):
        print("ONNX streaming parity check failed")
        return 1
    print("ONNX streaming parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
