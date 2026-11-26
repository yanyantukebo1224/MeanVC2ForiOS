# MeanVC2 for iOS

Native SwiftUI app with a Swift streaming host loop and ONNX Runtime for the
published MeanVC2 120ms_40ms graph set.

## Active implementation

- SwiftUI reference-audio picker and model status UI
- ONNX Runtime 1.24.3: C/C++ XCFramework pinned to a SHA-256-verified GitHub release mirror; Objective-C wrapper sources pinned to Microsoft's immutable commit archive (3a728b75062256951b6e19ce718907cf1a1d4cf0)
- SHA-256-verified model downloader pinned to a fixed Hugging Face revision
- Native Swift streaming host loop: fbank, Fast-U2++ ASR, BN interpolation,
  DiT flow-matching steps with per-layer KV caches, and Vocos waveform output
- Optional WavLM-Large + ECAPA speaker encoder for reference-voice registration
- AVFoundation audio-file decoding, mono downmix and 16 kHz resampling
- CI-only iPhone Simulator smoke test, enabled by a one-shot marker file
  created in the installed app's Documents directory; it uses a deterministic
  non-zero non-speech fixture, synthetic normalized speaker conditioning, and
  non-zero diffusion noise (not a perceptual or real-speaker quality test)
- DiT mean-flow defaults to two ODE steps, matching the Python reference
  runtime's `N = 2` schedule for the 120ms_40ms profile

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

Install CocoaPods (if needed), then run `cd ios && pod install`. Open
`ios/MeanVC2/MeanVC2.xcworkspace` in Xcode. Local podspecs avoid the broken
`download.onnxruntime.ai` host and Git clone dependency: Objective-C sources
come from the immutable upstream commit archive
`3a728b75062256951b6e19ce718907cf1a1d4cf0`; the C/C++ XCFramework comes
from a GitHub mirror with a pinned SHA-256. The bridging header imports the official
Objective-C API.

## Verification status

- Earlier Vocos Core ML spectral export and native iSTFT parity: passed; retained
  as diagnostics, no longer the active runtime path
- ONNX model checksums: pinned and checked before model files are accepted
- Native Swift ONNX streaming implementation: integrated; the earlier SPM-based
  Simulator build passed, but the CocoaPods migration and current streaming
  smoke test / Python numerical-parity check are not yet verified
- Target speaker extraction: integrated; awaiting device-side verification
- Live microphone capture, resampling and playback via AVAudioEngine: wired to
  the UI; stream caches reset on each start, input work is bounded, and the
  scheduled playback queue is capped at 600 ms
- Audio interruption and route-change notifications: automatic restart is
  implemented when iOS indicates resumption is allowed; not yet device-tested
- Audio interruption / route-change recovery, queue-drop behavior, sustained
  latency, peak memory and thermal behavior: not yet verified on hardware
- Real-time end-to-end conversion with a real registered speaker embedding:
  not yet verified
- Physical iPhone performance, thermal behavior, memory use and audio quality:
  not yet verified
- Signed IPA: not yet produced

## Model provenance and licenses

See the repository root `THIRD_PARTY_NOTICES.md`. The conversion graphs are
Apache-2.0. The optional WavLM-Large + ECAPA speaker encoder is separately
licensed CC BY-SA 3.0. Only register voices with the speaker's permission.
