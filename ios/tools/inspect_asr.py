#!/usr/bin/env python3
"""Emit a text-only diagnostic of the official Fast-U2++ TorchScript interface."""
from pathlib import Path
import torch

checkpoint = Path("preprocess/ckpts/fastu2pp_80ms.pt")
model = torch.jit.load(str(checkpoint), map_location="cpu").eval()
lines = [
    "=== methods ===",
    str(model._c._method_names()),
    "=== forward schema ===",
    str(model.forward.schema),
    "=== forward graph ===",
    str(model.forward.graph),
    "=== inlined graph ===",
    str(model.inlined_graph),
    "=== nested module methods ===",
]
for name, module in model.named_modules():
    try:
        methods = module._c._method_names()
    except Exception:
        methods = []
    if methods or "encoder" in name or "embed" in name or "pos_enc" in name:
        lines.append(f"{name}: type={type(module).__name__}, methods={methods}")
        for method_name in methods:
            try:
                method = module._c._get_method(method_name)
                lines.append(f"  {method_name} schema: {method.schema}")
            except Exception as error:
                lines.append(f"  {method_name} schema error: {error}")

lines.append("=== forward_encoder_chunk_j3 graph ===")
lines.append(str(model.m._c._get_method("forward_encoder_chunk_j3").graph))
lines.append("=== encoder.forward_chunk_j3 graph ===")
lines.append(str(model.m.encoder._c._get_method("forward_chunk_j3").graph))

# Verify whether the encoder's positional offset can be reset without changing
# the streaming representation. If this fails, a fixed-offset Core ML export
# would be invalid and must not be used.
torch.manual_seed(20261115)
att0 = torch.zeros(6, 4, 4, 128)
cnn0 = torch.zeros(6, 1, 256, 8)
fbanks = [torch.randn(1, 11, 80) for _ in range(4)]
def run_sequence(reset_offset: bool):
    att, cnn = att0.clone(), cnn0.clone()
    outs = []
    for index, fb in enumerate(fbanks):
        offset = 4 if reset_offset else 4 + index * 2
        with torch.no_grad():
            out, att, cnn = model(
                fb, torch.tensor(offset, dtype=torch.int64),
                torch.tensor(4, dtype=torch.int64), att, cnn
            )
        outs.append(out.detach().clone())
    return outs
reference = run_sequence(False)
reset = run_sequence(True)
lines.append("=== fixed-offset streaming parity ===")
for index, (a, b) in enumerate(zip(reference, reset)):
    lines.append(f"chunk {index}: max_abs_diff={float((a-b).abs().max()):.9g}, allclose={bool(torch.allclose(a,b,rtol=1e-5,atol=1e-6))}")
out = Path("ios/Diagnostics")
out.mkdir(parents=True, exist_ok=True)
target = out / "fastu2pp_torchscript_graph.txt"
target.write_text("\n\n".join(lines), encoding="utf-8")
print(f"Wrote {target} ({target.stat().st_size} bytes)")
