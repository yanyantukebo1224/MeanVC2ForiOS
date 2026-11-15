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
]
out = Path("ios/Diagnostics")
out.mkdir(parents=True, exist_ok=True)
target = out / "fastu2pp_torchscript_graph.txt"
target.write_text("\n\n".join(lines), encoding="utf-8")
print(f"Wrote {target} ({target.stat().st_size} bytes)")
