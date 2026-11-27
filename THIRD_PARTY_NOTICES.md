# Third-party notices

## MeanVC2 ONNX conversion graphs

The ONNX conversion graphs (fbank, ASR, GTM, DiT and Vocos) are sourced from
[kiarina/MeanVC2-ONNX](https://huggingface.co/kiarina/MeanVC2-ONNX), pinned to
revision `2db97430c6e315a819daa30e255e14615205a3d1`. These graph files are
licensed under Apache License 2.0. The streaming host loop is an independent
Swift implementation of the published MeanVC2 runtime algorithm.

The ONNX graph export and runtime reference implementation are maintained at
[kiarina/meanvc2_onnx](https://github.com/kiarina/meanvc2_onnx). This project is
not affiliated with or endorsed by the MeanVC2 authors or the ONNX exporter
maintainer. Please cite the original MeanVC2 paper when redistributing results.

## Optional speaker encoder

The optional `speaker/speaker.int8.onnx` WavLM-Large + ECAPA-TDNN speaker
verification model is licensed separately under Creative Commons
Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0). It is not covered by the
Apache-2.0 license for the conversion graphs. Any redistribution of that
model must comply with CC BY-SA 3.0 and preserve attribution and share-alike
requirements.

## ONNX Runtime

The Objective-C API sources are pinned to the immutable ONNX Runtime commit
`3a728b75062256951b6e19ce718907cf1a1d4cf0` in Microsoft's repository. CocoaPods
downloads the codeload archive by full commit SHA rather than a mutable tag or
branch. The podspec retains Microsoft's upstream MIT license. CI downloads the
same immutable URL directly, validates ZIP integrity, and checks for the public
Objective-C header and upstream license. It deliberately avoids a separate
GitHub REST API lookup because hosted-runner API 403/rate-limit responses had
blocked the build before the archive could be tested.

The iOS C/C++ XCFramework is ONNX Runtime `v1.24.3`. Its transport mirror is
the [`vendor-onnxruntime-c-1.24.3` release](https://github.com/baijum/ukulele-companion/releases/tag/vendor-onnxruntime-c-1.24.3)
in `baijum/ukulele-companion`. The release metadata identifies Microsoft's
upstream `v1.24.3` archive, records the original and Azure fallback URLs, and
publishes SHA-256
`b7eedc45932bac758ffd057cac0feb3f682269e47750b159e4c865145cbf0a8e`.
The local CocoaPods specification pins that GitHub asset and digest; TLS
verification is not disabled. The mirror release identifies the binary under
the MIT License, and the archive's upstream license file is preserved.

## Responsible use

Only register reference voices when you have the speaker's permission. Do not
use voice conversion to impersonate someone or mislead listeners about who is
speaking.

## Upstream revisions and component attribution

The pinned ONNX export's own [NOTICE](https://huggingface.co/kiarina/MeanVC2-ONNX/blob/2db97430c6e315a819daa30e255e14615205a3d1/NOTICE)
records the following provenance:

- MeanVC2 source code: commit
  `13acf84c1bf135ea5edad9c245b345289b06b33e`; its README states Apache-2.0.
- MeanVC2 model weights: Hugging Face revision
  `39cdd19522fe896c227da691314d9a0e3b995486`, Apache-2.0.
- WavLM base weights: `s3prl/converted_ckpts` revision
  `8cad0b370e7e35f8d56951d95d2be036ea85510c`; WavLM is MIT.
- s3prl model code: commit
  `ec8064b5889f81ca460fbe2c094ce576a6f120b7`, Apache-2.0.
- The fbank export incorporates Kaldi-compatible mel filter values from
  torchaudio, BSD-2-Clause.
- Vocos is Copyright (c) 2023 Charactr Inc., MIT.
- Fast-U2++ uses the WeNet architecture, Apache-2.0.
- The DiT backbone references F5-TTS architecture/code, whose upstream code
  is MIT-licensed. F5-TTS separately distributes pretrained checkpoints under
  CC-BY-NC; those checkpoints are not part of this app. The active graph uses
  the MeanVC2 weights listed above, not an F5-TTS pretrained checkpoint.

The export changes the model format and quantizes MatMul/Gemm weights in the
`*.int8.onnx` files; it does not retrain the models. Each exported component
retains the license of its source. The optional speaker encoder remains
CC BY-SA 3.0, including the modified INT8 ONNX file; it must not be represented
as Apache-2.0.

The iOS app downloads these assets from the pinned export revision at runtime.
Their exact byte counts and SHA-256 digests are maintained in
`MeanVC2ModelStore.swift` and checked by `check_model_manifest.py`.
