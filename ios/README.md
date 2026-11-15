# MeanVC2 for iOS

Native SwiftUI iOS app shell and the first validated Core ML component export.

## Verified status

- SwiftUI dark UI and reference-audio picker: implemented
- Xcode iOS Simulator build: successful in GitHub Actions
- Vocos spectral network: exported as a Core ML ML Program
- Vocos spectral output vs. original PyTorch iSTFT waveform: numerical parity passed in GitHub Actions
- Native Accelerate iSTFT + Core ML model loader: added to the iOS target; compile status is checked by the iOS build workflow
- Vocos checkpoint parameters: FFT 640, hop 160, centered iSTFT
- Fast-U2++ ASR: Core ML conversion still blocked by dynamic scalar-to-integer operations in its streaming positional-encoding path
- WavLM + ECAPA speaker encoder: not ported
- DiT/CFM decoder and streaming KV cache: not ported
- AVAudioEngine microphone-to-output pipeline: not integrated
- End-to-end voice conversion on device: not verified
- IPA and physical-device test: not produced / not tested

## Build

Open `MeanVC2/MeanVC2.xcodeproj` in Xcode.

## Model conversion

Run the **iOS Core ML model export** workflow from GitHub Actions. It downloads the official checkpoints and creates separate artifacts for Vocos and Fast-U2++ so one conversion failure does not hide the other.

The Vocos artifact contains the learned backbone and spectral projection. The iOS app performs the final centered inverse STFT with Accelerate. The exported model is not yet connected to the app's microphone pipeline.

The Fast-U2++ exporter deliberately does not freeze the stream offset to a constant: a four-chunk parity test showed that doing so changes the output and is not a valid streaming implementation.

## Remaining native pipeline

1. Make Fast-U2++ export accept the positional-encoding window as a tensor input while preserving the rolling attention and convolution caches.
2. Port WavLM + ECAPA speaker embedding extraction and UTTE conditioning.
3. Export the DiT/CFM decoder with per-layer streaming KV cache.
4. Connect ASR, speaker conditioning, DiT, and Vocos through AVAudioEngine.
5. Validate waveform parity, end-to-end latency, memory, thermal behavior, and audio quality on a physical iPhone.
6. Produce a signed IPA only after the complete pipeline passes the checks above.
