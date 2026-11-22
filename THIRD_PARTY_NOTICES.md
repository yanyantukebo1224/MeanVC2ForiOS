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

The Objective-C API sources are pinned to Microsoft's ONNX Runtime
`rel-1.24.3` release branch, whose release metadata identifies commit
`3a728b75062256951b6e19ce718907cf1a1d4cf0`. CocoaPods obtains the source from
the GitHub codeload branch archive, avoiding a Git clone in CI. The podspec retains
Microsoft's upstream MIT license; CI verifies that the release branch resolves
to the documented commit, checks the public header and license in the archive,
and records its SHA-256.

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
