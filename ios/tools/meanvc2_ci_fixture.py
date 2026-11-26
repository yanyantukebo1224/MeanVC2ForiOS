"""Cross-language deterministic fixture shared by Swift Simulator and Python ONNX."""

from __future__ import annotations

import struct

CHUNK_COUNT = 8
CHUNK_SAMPLES = 2_560
SPEAKER_NORM = 1_158.3781765900117


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def waveform_chunk(index: int) -> list[float]:
    if not 0 <= index < CHUNK_COUNT:
        raise ValueError(f"chunk index out of range: {index}")
    start = index * CHUNK_SAMPLES
    values = []
    for offset in range(CHUNK_SAMPLES):
        sample = start + offset
        low = (sample * 37 + 11) % 257 - 128
        high = (sample * 13 + 7) % 127 - 63
        values.append(_float32(low / 2_048.0 + high / 4_096.0))
    return values


def speaker_embedding() -> list[float]:
    return [
        _float32(((index * 29 + 11) % 251 - 125) / SPEAKER_NORM)
        for index in range(256)
    ]


def noise_values(frame_count: int) -> list[float]:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    return [
        _float32(((index * 17 + 3) % 101 - 50) / 512.0)
        for index in range(frame_count * 80)
    ]


def float32_bytes(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)
