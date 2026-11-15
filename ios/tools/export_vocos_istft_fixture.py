#!/usr/bin/env python3
"""Create a deterministic Vocos spectral fixture for the Swift Accelerate iSTFT test."""
from pathlib import Path
import json
import numpy as np
import torch

torch.manual_seed(20261115)
model = torch.jit.load("ckpts/vocos/vocos.pt", map_location="cpu").eval()
mel = torch.rand(1, 80, 16)
with torch.no_grad():
    hidden = model.backbone(mel)
    projected = model.head.out(hidden).transpose(1, 2)
    magnitude, phase = projected.chunk(2, dim=1)
    magnitude = torch.exp(magnitude).clamp(max=1e2)
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    real_imag = torch.stack((real, imag), dim=-1)
    waveform = model.decode(mel)

out = Path("ios/Diagnostics")
out.mkdir(parents=True, exist_ok=True)
np.asarray(real_imag[0].cpu(), dtype=np.float32).tofile(out / "vocos_real_imag.f32")
np.asarray(waveform[0].cpu(), dtype=np.float32).tofile(out / "vocos_waveform.f32")
metadata = {
    "fft_size": int(model.head.istft.n_fft),
    "hop_length": int(model.head.istft.hop_length),
    "win_length": int(model.head.istft.win_length),
    "padding": str(model.head.istft.padding),
    "frames": int(mel.shape[-1]),
    "frequency_bins": int(real_imag.shape[1]),
    "waveform_samples": int(waveform.shape[-1]),
    "layout": "[frequency, time, real_imag]"
}
(out / "vocos_istft_fixture.json").write_text(json.dumps(metadata, indent=2) + "\\n")
print(json.dumps(metadata, indent=2))
