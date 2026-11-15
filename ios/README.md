# MeanVC2 for iOS

This directory contains the first native SwiftUI app shell for MeanVC2.

## Current status

- SwiftUI iOS app project: prototype
- Minimal dark UI and reference-audio file picker: implemented
- MeanVC2 inference on iOS: not implemented yet
- Core ML model conversion / Core Audio streaming: not implemented yet
- IPA / device build: not produced or tested

## Open

Open `MeanVC2/MeanVC2.xcodeproj` in Xcode.

The app shell is intentionally honest about the inference engine being under development. The Python implementation in the repository is not an iOS runtime: the Fast-U2++ TorchScript encoder, WavLM/ECAPA speaker encoder, DiT checkpoint, and Vocos TorchScript vocoder need an iOS-compatible conversion and streaming implementation.

## Planned native pipeline

1. Convert and validate the ASR encoder for Core ML.
2. Port speaker embedding extraction and UTTE conditioning.
3. Convert the DiT/CFM decoder and preserve its streaming caches.
4. Port Vocos decoding.
5. Connect the stages through a low-latency AVAudioEngine pipeline.
6. Benchmark end-to-end latency, memory, thermal behavior, and audio quality on-device.
