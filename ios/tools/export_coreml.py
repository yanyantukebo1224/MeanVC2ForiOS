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

class ASRChunkWithPos(torch.nn.Module):
    """Fast-U2++ chunk encoder with the relative-position window supplied as data.

    The official TorchScript wrapper casts its scalar offset to Python int, which
    Core ML cannot represent as a dynamic tensor input. The offset is used only
    to select the relative positional window. Supplying that exact window as an
    input preserves the streaming math and keeps the attention/CNN caches live.
    """
    def __init__(self, model):
        super().__init__()
        self.encoder = model.m.encoder

    def forward(self, fbank, pos_emb, att_cache, cnn_cache):
        xs = fbank
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)

        embed = self.encoder.embed
        xs = xs.unsqueeze(1)
        xs = embed.conv(xs)
        batch, channels, time, freq = xs.size()
        xs = embed.out(
            xs.transpose(1, 2).contiguous().view(batch, time, channels * freq)
        )
        xs = embed.pos_enc.dropout(xs * embed.pos_enc.xscale)
        pos_emb = embed.pos_enc.dropout(pos_emb)

        elayers = att_cache.size(0)
        cache_t1 = att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        # The 40ms_40ms runtime profile uses a four-frame rolling attention cache.
        next_cache_start = max(attention_key_size - 4, 0)

        empty_mask = torch.zeros((0, 0, 0), dtype=torch.bool, device=xs.device)
        new_att_caches = []
        new_cnn_caches = []
        # TorchScript's ModuleList wrapper does not implement __getitem__ in Python.
        # Its named child modules remain addressable by their constant numeric names.
        layers = (
            getattr(self.encoder.encoders, "0"),
            getattr(self.encoder.encoders, "1"),
            getattr(self.encoder.encoders, "2"),
            getattr(self.encoder.encoders, "3"),
            getattr(self.encoder.encoders, "4"),
            getattr(self.encoder.encoders, "5")
        )
        for index, layer in enumerate(layers):
            if elayers > 0:
                layer_cache = att_cache[index:index + 1]
            else:
                layer_cache = att_cache

            if cnn_cache.size(0) > 0:
                layer_cnn_cache = cnn_cache[index]
            else:
                layer_cnn_cache = cnn_cache

            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                empty_mask,
                pos_emb,
                empty_mask,
                layer_cache,
                layer_cnn_cache
            )
            new_att_caches.append(new_att_cache[:, :, next_cache_start:, :])
            new_cnn_caches.append(new_cnn_cache.unsqueeze(0))

        if self.encoder.normalize_before and self.encoder.final_norm:
            xs = self.encoder.after_norm(xs)

        return (
            xs,
            torch.cat(new_att_caches, dim=0),
            torch.cat(new_cnn_caches, dim=0)
        )

class VocosSpectrogram(torch.nn.Module):
    """Export Vocos' learned spectral head; native iOS performs the final iSTFT."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, mel):
        hidden = self.model.backbone(mel)
        projected = self.model.head.out(hidden).transpose(1, 2)
        magnitude, phase = projected.chunk(2, dim=1)
        magnitude = torch.exp(magnitude).clamp(max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.stack((real, imag), dim=-1)

def export_coreml(module, inputs, name, ct):
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, inputs, strict=False)
        expected = traced(*inputs)
        # Freeze specializes Optional arguments (e.g. Vocos conditioning=None)
        # and removes TorchScript prim::Uninitialized branches before Core ML.
        traced = torch.jit.freeze(traced.eval())
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=n, shape=tuple(x.shape)) for n, x in zip(
            (["fbank", "pos_emb", "att_cache", "cnn_cache"]
             if name == "MeanVC2ASR" else ["mel"]), inputs)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{name}.mlpackage"
    mlmodel.save(str(destination))
    # Reload and run a numerical parity check on CPU before accepting the artifact.
    loaded = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_ONLY)
    input_names = list(mlmodel.input_description)
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
        OUT.mkdir(parents=True, exist_ok=True)
        pos_table = asr.m.encoder.embed.pos_enc.pe.detach().cpu().numpy().astype(np.float32)
        pos_table[0].tofile(OUT / "MeanVC2ASRPositionalEncoding.f32")
        (OUT / "MeanVC2ASRPositionalEncoding.json").write_text(json.dumps({
            "shape": list(pos_table[0].shape),
            "dtype": "float32",
            "layout": "time_major",
            "profile": "40ms_40ms",
            "initial_offset": 4,
            "attention_cache_frames": 4,
            "frames_per_chunk": 2,
            "window_start": "current_offset - attention_cache_frames"
        }, indent=2) + "\\n")

        asr_chunk = ASRChunkWithPos(asr).eval()
        fbank = torch.zeros(1, 11, 80)
        att0 = torch.zeros(6, 4, 4, 128)
        cnn0 = torch.zeros(6, 1, 256, 8)

        # Verify the externally supplied positional window against the original
        # streaming TorchScript entrypoint over several successive chunks.
        att_reference, cnn_reference = att0.clone(), cnn0.clone()
        att_candidate, cnn_candidate = att0.clone(), cnn0.clone()
        offset = 4
        with torch.no_grad():
            for chunk_index in range(4):
                chunk_fbank = torch.randn_like(fbank)
                cache_frames = att_reference.size(2)
                projected_frames = 2
                start = offset - cache_frames
                pos_input = torch.from_numpy(
                    pos_table[:, start:start + cache_frames + projected_frames].copy()
                )
                reference = asr(
                    chunk_fbank,
                    torch.tensor(offset, dtype=torch.int64),
                    torch.tensor(4, dtype=torch.int64),
                    att_reference,
                    cnn_reference
                )
                candidate = asr_chunk(
                    chunk_fbank, pos_input, att_candidate, cnn_candidate
                )
                for label, actual, expected_value in zip(
                    ("bnf", "attention_cache", "convolution_cache"),
                    candidate, reference
                ):
                    if actual.shape != expected_value.shape or not torch.allclose(
                        actual, expected_value, rtol=1e-5, atol=1e-6
                    ):
                        difference = float((actual - expected_value).abs().max())
                        raise RuntimeError(
                            f"ASR injected-pos parity failed at chunk {chunk_index} "
                            f"for {label}: max_abs_diff={difference}"
                        )
                att_reference, cnn_reference = reference[1], reference[2]
                att_candidate, cnn_candidate = candidate[1], candidate[2]
                offset += reference[0].size(1)

        sample = (
            fbank,
            torch.from_numpy(pos_table[:, 0:6].copy()),
            att0,
            cnn0
        )
        exported["asr"] = export_coreml(asr_chunk, sample, "MeanVC2ASR", ct)

    if not args.skip_vocoder:
        if not VOCOS.exists():
            raise FileNotFoundError(VOCOS)
        vocos = torch.jit.load(str(VOCOS), map_location="cpu").eval()
        mel = torch.zeros(1, 80, 16)
        spectral = VocosSpectrogram(vocos).eval()
        with torch.no_grad():
            real_imag = spectral(mel)
            reconstructed = vocos.head.istft(torch.complex(real_imag[..., 0], real_imag[..., 1]))
            reference = vocos.decode(mel)
            if reconstructed.shape != reference.shape or not torch.allclose(
                reconstructed, reference, rtol=1e-5, atol=1e-6
            ):
                raise RuntimeError("Vocos spectral-head + iSTFT parity check failed")
        exported["vocoder_spectrogram"] = export_coreml(
            spectral, (mel,), "MeanVC2VocosSpectrogram", ct
        )

    manifest = {
        "format_version": 1,
        "model": "MeanVC2",
        "profile": "40ms_40ms",
        "deployment_target": "iOS17",
        "exported_components": exported,
        "not_exported": [
            "WavLM+ECAPA speaker encoder",
            "DiT/CFM decoder with per-layer streaming KV cache",
            "native iSTFT synthesis and end-to-end streaming orchestration"
        ],
        "note": "Only components listed in exported_components are available. This is not a complete voice-conversion model."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
