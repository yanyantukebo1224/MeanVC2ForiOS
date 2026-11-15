# MeanVC2 for iOS

Native SwiftUI app with a Swift streaming host loop and ONNX Runtime for the
published MeanVC2 120ms_40ms graph set.

## Active implementation

- SwiftUI reference-audio picker and model status UI
- ONNX Runtime through Microsoft's Swift Package Manager distribution
- SHA-256-verified model downloader pinned to a fixed Hugging Face revision
- Native Swift streaming host loop: fbank, Fast-U2++ ASR, BN interpolation,
  DiT flow-matching steps with per-layer KV caches, and Vocos waveform output
- Optional WavLM-Large + ECAPA speaker encoder for reference-voice registration
- AVFoundation audio-file decoding, mono downmix and 16 kHz resampling
- iPhone Simulator smoke test that loads the graph set and runs the streaming
  inference loop on synthetic audio

## Model profile

The current ONNX graph set is the 120ms_40ms profile: 160 ms input chunks,
120 ms output chunks and 40 ms look-ahead. The first converted audio is expected
after the initial streaming context has accumulated. This is not the earlier
40ms_40ms Core ML prototype.

The conversion graphs are downloaded on first launch (roughly 74 MB total,
including the GTM graph). The optional speaker encoder is downloaded only when
a reference voice is registered (roughly 387 MB). After these assets are present,
model inference is offline.

## Build

Open `MeanVC2/MeanVC2.xcodeproj` in Xcode. Xcode resolves the official
`onnxruntime-swift-package-manager` package automatically.

## Verification status

- Earlier Vocos Core ML spectral export and native iSTFT parity: passed; retained
  as diagnostics, no longer the active runtime path
- ONNX model checksums: pinned and checked before model files are accepted
- Native Swift ONNX streaming implementation: added; awaiting successful
  Xcode/Simulator compilation and numerical smoke-test confirmation
- Target speaker extraction: implemented; awaiting device-side verification
- Live microphone capture, resampling and playback via AVAudioEngine: not yet
  integrated
- Real-time end-to-end conversion from microphone to speaker: not yet verified
- Physical iPhone performance, thermal behavior, memory use and audio quality:
  not yet verified
- Signed IPA: not yet produced

## Model provenance and licenses

See the repository root `THIRD_PARTY_NOTICES.md`. The conversion graphs are
Apache-2.0. The optional WavLM-Large + ECAPA speaker encoder is separately
licensed CC BY-SA 3.0. Only register voices with the speaker's permission.
