#!/usr/bin/env python3
"""Compare the Swift simulator output with the pinned Python ONNX host loop."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


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
        steps=3,
    )
    runner.noise_fn = lambda n: np.zeros((1, n, 80), dtype=np.float32)
    runner.set_target(np.zeros((1, 256), dtype=np.float32))

    pieces = []
    for _ in range(8):
        pieces.append(runner.process(np.zeros(2560, dtype=np.float32)))
    expected = np.concatenate(pieces).astype(np.float32, copy=False)
    actual = np.fromfile(args.swift_output, dtype=np.float32)

    if expected.size != actual.size:
        print(f"sample-count mismatch: Swift={actual.size}, reference={expected.size}")
        return 1
    if expected.size == 0:
        print("reference produced no waveform")
        return 1

    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    denom = np.linalg.norm(actual) * np.linalg.norm(expected)
    cosine = float(np.dot(actual, expected) / denom) if denom else 1.0
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
