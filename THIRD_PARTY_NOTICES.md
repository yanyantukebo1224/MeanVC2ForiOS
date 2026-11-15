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

The iOS runtime is provided by Microsoft's
[onnxruntime-swift-package-manager](https://github.com/microsoft/onnxruntime-swift-package-manager),
which is distributed under the MIT License. The ONNX Runtime native library
and Objective-C bindings retain their upstream licenses.

## Responsible use

Only register reference voices when you have the speaker's permission. Do not
use voice conversion to impersonate someone or mislead listeners about who is
speaking.
