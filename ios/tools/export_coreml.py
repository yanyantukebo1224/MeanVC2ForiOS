#!/usr/bin/env python3
"""Export the fixed-shape MeanVC2 40ms front-end and vocoder to Core ML.

This is deliberately fail-closed: a model is only marked exported after a
numerical parity check. Dynamic DiT KV-cache export is handled by a separate
stage and is not silently replaced with a stateless model.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ios" / "Models"
CKPT = ROOT / "preprocess" / "ckpts"
VOCOS = ROOT / "ckpts" / "vocos" / "vocos.pt"

class ASRChunk(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, fbank, offset, required_cache_size, att_cache, cnn_cache):
        bn, next_att, next_cnn = self.model(
            fbank, offset.reshape(()), required_cache_size.reshape(()), att_cache, cnn_cache
        )
        return bn, next_att, next_cnn

class VocosDecode(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, mel):
        return self.model.decode(mel)

def export_coreml(module, inputs, name, ct):
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, inputs, strict=False)
        expected = traced(*inputs)
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=n, shape=tuple(x.shape)) for n, x in zip(
            (["fbank", "offset", "required_cache_size", "att_cache", "cnn_cache"]
             if name == "MeanVC2ASR" else ["mel"]), inputs)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{name}.mlpackage"
    mlmodel.save(str(destination))
    # Reload and run a numerical parity check on CPU before accepting the artifact.
    loaded = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_ONLY)
    input_names = [item.name for item in mlmodel.input_description]
    prediction = loaded.predict({n: x.detach().cpu().numpy() for n, x in zip(input_names, inputs)})
    output_names = list(mlmodel.output_description)
    expected_values = expected if isinstance(expected, (tuple, list)) else (expected,)
    if len(output_names) != len(expected_values):
        raise RuntimeError(f"{name}: output count mismatch: {len(output_names)} != {len(expected_values)}")
    for output_name, reference in zip(output_names, expected_values):
        actual = np.asarray(prediction[output_name])
        reference = reference.detach().cpu().numpy()
        if actual.shape != reference.shape:
            raise RuntimeError(f"{name}/{output_name}: shape mismatch {actual.shape} != {reference.shape}")
        if not np.allclose(actual, reference, rtol=5e-2, atol=5e-2, equal_nan=False):
            error = float(np.max(np.abs(actual - reference)))
            raise RuntimeError(f"{name}/{output_name}: Core ML parity failed (max abs error {error})")
    print(f"EXPORTED + PARITY OK {destination}")
    return str(destination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-vocoder", action="store_true")
    args = parser.parse_args()
    import coremltools as ct

    exported = {}
    if not args.skip_asr:
        asr_path = CKPT / "fastu2pp_80ms.pt"
        if not asr_path.exists():
            raise FileNotFoundError(asr_path)
        asr = torch.jit.load(str(asr_path), map_location="cpu").eval()
        sample = (
            torch.zeros(1, 11, 80),
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int64),
            torch.zeros(6, 4, 4, 128),
            torch.zeros(6, 1, 256, 8),
        )
        exported["asr"] = export_coreml(ASRChunk(asr), sample, "MeanVC2ASR", ct)

    if not args.skip_vocoder:
        if not VOCOS.exists():
            raise FileNotFoundError(VOCOS)
        vocos = torch.jit.load(str(VOCOS), map_location="cpu").eval()
        mel = torch.zeros(1, 80, 16)
        exported["vocoder"] = export_coreml(VocosDecode(vocos), (mel,), "MeanVC2Vocos", ct)

    manifest = {
        "format_version": 1,
        "model": "MeanVC2",
        "profile": "40ms_40ms",
        "deployment_target": "iOS17",
        "exported_components": exported,
        "not_exported": [
            "WavLM+ECAPA speaker encoder",
            "DiT/CFM decoder with per-layer streaming KV cache",
            "end-to-end native streaming orchestration"
        ],
        "note": "Only components listed in exported_components are available. This is not a complete voice-conversion model."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
