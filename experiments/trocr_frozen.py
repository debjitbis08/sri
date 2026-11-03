"""The image encoder of TrOCR-small, stage 1 (Li et al. 2021, arXiv 2109.10282), as a frozen glyph encoder, written here from its published shapes so that no downloaded code runs. The text decoder is not used.
The encoder is a DeiT-small: pre-norm transformer, width 384, 12 layers, 6 heads, patch 16 on 384 x 384, with a class and a distillation token. Stage 1 is TrOCR's pretraining on printed text lines, before any fine-tuning.

Fetch, from outside the repository:
  mkdir -p /data/external-downloads/trocr
  curl -sSL -o /data/external-downloads/trocr/pytorch_model.bin https://huggingface.co/microsoft/trocr-small-stage1/resolve/main/pytorch_model.bin
Revision d94c4048159050bb51e4105bad0a45dbfd5f3a02, sha256 35ee4bee78f7686e15e5eaf9de5b0788ae28c53fa756883afd3405924f6c49c5; encoder 21,812,352 parameters.
Loaded as plain tensors (`weights_only=True`), encoder keys only.

Inputs: grey (n, h, w) in [0, 1], resized to 384 (bicubic, as its preprocessor) and normalized with mean and std 0.5. TrOCR saw dark text on light paper; INVERT (TROCR_INVERT=1) turns every input over, whatever its kind, so it needs no label.
Outputs: the class token and the mean of patch tokens, both after the final norm.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path(os.environ.get('TROCR_WEIGHTS', '/data/external-downloads/trocr/pytorch_model.bin'))
SHA256 = '35ee4bee78f7686e15e5eaf9de5b0788ae28c53fa756883afd3405924f6c49c5'
SIZE, DIM, DEPTH, HEADS, PATCH, EPS = 384, 384, 12, 6, 16, 1e-12
INVERT = bool(int(os.environ.get('TROCR_INVERT', 0)))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm_before, self.layernorm_after = nn.LayerNorm(DIM, eps=EPS), nn.LayerNorm(DIM, eps=EPS)
        self.q, self.k, self.v, self.o = (nn.Linear(DIM, DIM) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(DIM, 4 * DIM), nn.Linear(4 * DIM, DIM)

    def forward(self, x):
        n, t, _ = x.shape
        h = self.layernorm_before(x)
        q, k, v = (f(h).reshape(n, t, HEADS, DIM // HEADS).transpose(1, 2) for f in (self.q, self.k, self.v))
        a = ((q @ k.transpose(-2, -1)) * (DIM // HEADS) ** -.5).softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(n, t, DIM))
        return x + self.fc2(F.gelu(self.fc1(self.layernorm_after(x))))


class DeiTSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token, self.distillation_token = nn.Parameter(torch.zeros(1, 1, DIM)), nn.Parameter(torch.zeros(1, 1, DIM))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 2 + (SIZE // PATCH) ** 2, DIM))
        self.projection = nn.Conv2d(3, DIM, PATCH, PATCH)
        self.layers = nn.ModuleList([Layer() for _ in range(DEPTH)])
        self.layernorm = nn.LayerNorm(DIM, eps=EPS)

    def forward(self, x):
        p = self.projection(x).flatten(2).transpose(1, 2)
        h = torch.cat([self.cls_token.expand(len(x), -1, -1), self.distillation_token.expand(len(x), -1, -1), p], 1) + self.position_embeddings
        for layer in self.layers:
            h = layer(h)
        h = self.layernorm(h)
        return {'cls': h[:, 0], 'patch': h[:, 2:].mean(1)}


def renamed(state):
    """The checkpoint's encoder keys, renamed to this module's."""
    out = {}
    for k, v in state.items():
        if not k.startswith('encoder.'):
            continue
        k = k[len('encoder.'):]
        k = k.replace('embeddings.patch_embeddings.projection.', 'projection.').replace('embeddings.', '').replace('encoder.layer.', 'layers.')
        k = k.replace('attention.attention.query.', 'q.').replace('attention.attention.key.', 'k.').replace('attention.attention.value.', 'v.').replace('attention.output.dense.', 'o.')
        k = k.replace('intermediate.dense.', 'fc1.').replace('output.dense.', 'fc2.')
        out[k] = v
    return out


def prepare(x, invert=INVERT):
    """[0, 1] grey images to the network's input."""
    if invert:
        x = 1 - x
    x = F.interpolate(x[:, None], size=(SIZE, SIZE), mode='bicubic', align_corners=False).clamp(0, 1)
    return ((x - .5) / .5).expand(-1, 3, -1, -1)


def load(device):
    model = DeiTSmall()
    model.load_state_dict(renamed(torch.load(WEIGHTS, map_location='cpu', weights_only=True)), strict=True)
    return model.to(device).eval().requires_grad_(False)


def layers(model, chunk=32, invert=INVERT):
    """For experiments.general_encoder: images -> {'cls', 'patch'}, in chunks."""
    def fn(x):
        dev = next(model.parameters()).device
        parts = [model(prepare(x[i:i + chunk].to(dev), invert)) for i in range(0, len(x), chunk)]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return fn


def gate(seeds=(392,)):
    """Does the encoder read glyphs? The transfer reads of experiments.general_encoder (glyph identity from one or 16 examples, across fonts, and each kind from a strong view), both polarities, beside DINOv2.
    There is no reference output to match exactly, since the reference code is not run; a wrong implementation would read glyphs near chance."""
    import time
    from experiments import device as DEV
    from experiments import dinov2_frozen as DINO
    from experiments import general_encoder as GE
    from experiments import shared_core_stage0 as S0
    S0.prepare()
    DEV.install()
    dev = DEV.NAME if DEV.ACTIVE else 'cpu'
    S0.set_up_task('sparse48')
    m = load(dev)
    nets = {'trocr-s': layers(m, invert=False), 'trocr-s-inverted': layers(m, invert=True), 'dinov2-s': DINO.layers(DINO.load(dev))}
    out = {}
    for seed in seeds:
        for name, fn in nets.items():
            t = time.perf_counter()
            out[f'{seed}/{name}'] = GE.transfer_scores(fn, seed)
            print(seed, name, f'{time.perf_counter() - t:.0f} s', {lay: {k: round(v, 2) for k, v in r.items()} for lay, r in out[f'{seed}/{name}'].items()}, flush=True)
    return out


if __name__ == '__main__':
    gate()  # NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.trocr_frozen
