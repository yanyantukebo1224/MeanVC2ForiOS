#!/usr/bin/env python3
"""Inspect the official TorchScript Vocos graph for unsupported lazy/uninitialized ops."""
from pathlib import Path
import torch

model = torch.jit.load("ckpts/vocos/vocos.pt", map_location="cpu").eval()
mel = torch.zeros(1, 80, 16)
class Decode(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, x):
        return self.inner.decode(x)
traced = torch.jit.trace(Decode(model), (mel,), strict=False)
istft = model.head.istft
lines = [
    "=== model metadata ===",
    f"n_fft={istft.n_fft}",
    f"hop_length={istft.hop_length}",
    f"win_length={istft.win_length}",
    f"padding={istft.padding}",
    f"window_shape={tuple(istft.window.shape)}",
    "=== module tree ===",
    "\n".join(f"{name}: {type(module).__name__} {module}" for name, module in model.named_modules()),
    "=== parameters ===",
    "\n".join(f"{name}: {tuple(value.shape)}" for name, value in model.named_parameters()),
    "=== traced graph ===",
    str(traced.inlined_graph),
    "=== uninitialized nodes ===",
]
for node in traced.inlined_graph.nodes():
    if "Uninitialized" in node.kind() or "uninitialized" in node.kind().lower():
        lines.append(f"{node.kind()} :: {node.sourceRange()} :: {node}")
out = Path("ios/Diagnostics")
out.mkdir(parents=True, exist_ok=True)
target = out / "vocos_torchscript_graph.txt"
target.write_text("\n\n".join(lines), encoding="utf-8")
print(f"Wrote {target} ({target.stat().st_size} bytes)")
