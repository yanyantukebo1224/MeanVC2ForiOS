#!/usr/bin/env python3
"""Verify that the Swift ONNX host loop matches the published 120ms Python profile."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def integer(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"Missing {label} in reference profile")
    return int(match.group(1).replace("_", ""))


swift = (ROOT / "ios/MeanVC2/MeanVC2ONNXStream.swift").read_text()
reference = (ROOT / "runtime/run_rt.py").read_text()
config = json.loads((ROOT / "src/config/config_120ms_40ms.json").read_text())["model"]

profile_match = re.search(
    r'"120ms"\s*:\s*\{(?P<body>.*?)\n\s*\},\s*"40ms"\s*:',
    reference,
    re.DOTALL,
)
if profile_match is None:
    raise SystemExit("Could not locate the Python 120ms model profile")
profile = profile_match.group("body")


def swift_constant(name: str) -> int:
    return integer(
        rf"private static let {re.escape(name)}\s*=\s*([0-9_]+)",
        swift,
        f"Swift {name}",
    )


def swift_state(name: str) -> int:
    return integer(
        rf"private var {re.escape(name)}\s*=\s*([0-9_]+)",
        swift,
        f"Swift {name}",
    )


def python_profile(name: str) -> int:
    return integer(
        rf'"{re.escape(name)}"\s*:\s*(\d+)',
        profile,
        f"Python {name}",
    )


checks: list[str] = []


def equal(label: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: Swift={actual}, Python reference={expected}")
    checks.append(label)


reference_rate = integer(r"if sr != (\d+)", reference, "Python sample rate")
equal("sample rate", swift_constant("sampleRate"), reference_rate)
equal(
    "input chunk samples",
    swift_constant("inputChunkSamples"),
    integer(r"self\.CHUNK\s*=\s*(\d+)", reference, "Python input chunk"),
)
equal(
    "fbank frame samples",
    swift_constant("fbankFrameSamples"),
    integer(r"frame_length\s*=\s*(\d+)", reference, "Python fbank frame length")
    * reference_rate
    // 1000,
)
equal(
    "fbank stride samples",
    swift_constant("fbankStride"),
    integer(r"frame_shift\s*=\s*(\d+)", reference, "Python fbank frame shift")
    * reference_rate
    // 1000,
)
equal("BN window", swift_constant("bnWindow"), python_profile("bn_window"))
equal("BN stride", swift_constant("bnStride"), python_profile("bn_stride"))
equal(
    "ASR required cache",
    swift_constant("requiredCache"),
    python_profile("required_cache_size"),
)
equal("ASR initial offset", swift_state("asrOffset"), python_profile("asr_offset_init"))
equal(
    "ASR offset step",
    integer(r"asrOffset\s*\+=\s*(\d+)", swift, "Swift ASR offset step"),
    python_profile("asr_offset_step"),
)
equal("DiT chunk frames", swift_constant("chunkFrames"), config["chunk_size"])
equal(
    "DiT chunk frames in runtime",
    integer(
        r'self\.chunk_size\s*=\s*(\d+)\s*if model == "120ms"',
        reference,
        "Python 120ms chunk size",
    ),
    config["chunk_size"],
)
equal("look-ahead frames", swift_constant("lookaheadFrames"), config["block_size"])
equal(
    "look-ahead frames in runtime",
    integer(r"self\.block_size\s*=\s*(\d+)", reference, "Python look-ahead size"),
    config["block_size"],
)
equal("DiT layer count", swift_constant("layerCount"), config["depth"])
equal(
    "mean-flow ODE steps",
    integer(r"odeSteps: Int = (\d+)", swift, "Swift default ODE steps"),
    integer(r"\bN\s*=\s*(\d+)\s*#\s*ODE steps", reference, "Python ODE steps"),
)
equal(
    "vocoder overlap frames",
    swift_constant("vocoderOverlapFrames"),
    integer(r"self\.vocoder_overlap\s*=\s*(\d+)", reference, "Python vocoder overlap"),
)
equal(
    "waveform overlap samples",
    swift_constant("waveformOverlapSamples"),
    integer(r"self\.upsample_factor\s*=\s*(\d+)", reference, "Python waveform overlap"),
)

print(
    "MeanVC2 streaming profile parity OK: "
    f"{len(checks)} invariants; 120ms chunks, 40ms look-ahead, 2 ODE steps."
)
