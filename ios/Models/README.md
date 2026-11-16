# MeanVC2 iOS model assets

## Active runtime: ONNX Runtime

The current SwiftUI app uses the published **120ms + 40ms look-ahead ONNX graph set**.
It is not the legacy 40ms Core ML export profile.

The five conversion graphs are downloaded from
[`kiarina/MeanVC2-ONNX`](https://huggingface.co/kiarina/MeanVC2-ONNX) at the
immutable revision `2db97430c6e315a819daa30e255e14615205a3d1`:

| Asset | Purpose | Size |
|---|---|---:|
| `fbank.onnx` | 16 kHz waveform framing and log-mel features | 908,339 bytes |
| `asr.int8.onnx` | Streaming Fast-U2++ BN extraction | 28,655,587 bytes |
| `gtm.onnx` | UTTE speaker-memory conditioning | 17,441,893 bytes |
| `dit.int8.onnx` | DiT / CFM flow step with FRC KV caches | 13,849,138 bytes |
| `vocos.int8.onnx` | Mel-to-waveform synthesis | 12,998,461 bytes |

The conversion set totals **73,853,418 bytes**. The app verifies the pinned
byte count and SHA-256 before accepting each downloaded file. The model
manifest consistency check also verifies that the Swift asset table, CI
download list, and CI digests agree.

The optional reference-speaker encoder is
`speaker/speaker.int8.onnx` (387,106,790 bytes): WavLM-Large + ECAPA-TDNN
producing a 256-dimensional embedding. It is downloaded only when a user
registers a reference voice, and the resulting embedding is stored locally.
This asset is **CC BY-SA 3.0**, not Apache-2.0; see the root
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md). Use only recordings
whose speaker has given permission.

The first model download requires an internet connection. Once all required
graphs and the selected speaker embedding are present and verified, conversion
inference does not require a network connection.

## Streaming profile

- Input: mono Float32, 16 kHz, 160 ms chunks (2,560 samples)
- Output block: 120 ms, with 40 ms future look-ahead
- Content: Fast-U2++ streaming BN features
- Speaker: WavLM-Large + ECAPA embedding, transformed by GTM / UTTE
- Synthesis: DiT mean-flow steps with per-layer KV caches and overlapping
  noise state, followed by Vocos waveform synthesis

The Swift implementation is intended to match the pinned ONNX reference
runtime. CI feeds synthetic zero audio and a zero speaker vector to exercise
the graph chain and compare the produced samples with the Python ONNX
reference. That is a deterministic execution/parity test only: it does not
measure perceptual voice quality, speaker similarity, or live microphone
latency.

## Core ML export status

The manual workflow named **Legacy Core ML export diagnostics** is exploratory
and separate from the active ONNX runtime. It exports selected fixed-shape
Fast-U2++ ASR and Vocos components from the original 40ms profile and checks
the native Accelerate iSTFT path. It does **not** export or integrate the full
WavLM + ECAPA speaker encoder, GTM / UTTE, streaming DiT / CFM with cache/FRC
semantics, or a complete real-time audio pipeline. Its artifacts must not be
treated as a runnable Core ML voice-conversion app.

A complete Core ML replacement is not currently available. The ONNX Runtime
path remains the active implementation until each replacement component has
numerical parity and end-to-end streaming verification.

## Verification still required

- Successful Simulator graph execution and numerical comparison against the
  pinned Python reference
- Physical iPhone/iPad microphone-to-speaker conversion using a real registered
  target voice
- Audio interruption, route-change, queue underrun/backpressure, and recovery
  tests
- Peak memory, sustained latency, thermal behavior, and listening-based quality
  measurements on target devices
- Signed IPA and distribution; signing is separate from the unsigned CI build

Do not describe the app as complete until these checks are recorded.
