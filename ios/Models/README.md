# MeanVC2 iOS model conversion

Run the **iOS Core ML model export** workflow from the Actions tab (or push a
change under `ios/tools/`). It downloads the official 40ms ASR encoder and
Vocos checkpoint and attempts fixed-shape Core ML exports on a macOS runner.

Important: this first export stage does **not** produce a complete VC engine.
The speaker encoder (WavLM + ECAPA), DiT/CFM decoder with streaming KV-cache,
and native AVAudioEngine orchestration are still missing. The workflow's
artifact is labelled partial and must not be presented as a working voice
converter. The exporter fails rather than claiming success when conversion
fails.
