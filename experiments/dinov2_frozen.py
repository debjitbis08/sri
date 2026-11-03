"""DINOv2 ViT-S/14 (Oquab et al. 2023) as a frozen encoder, written here from its published shapes so that no downloaded code runs. Weights: the official backbone file, loaded as plain tensors (`weights_only=True`).

Fetch, from outside the repository:
  mkdir -p /data/external-downloads/dinov2
  curl -sSL -o /data/external-downloads/dinov2/dinov2_vits14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
sha256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9, 22,056,576 parameters.

Inputs: grey (n, h, w) or colour (n, 3, h, w) in [0, 1], resized to SIZE and normalized as in ImageNet training. Outputs: the class token and the mean of patch tokens, both after the final norm.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path(os.environ.get('DINO_WEIGHTS', '/data/external-downloads/dinov2/dinov2_vits14_pretrain.pth'))
SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
SIZE = int(os.environ.get('DINO_SIZE', 224))
DIM, DEPTH, HEADS, PATCH = 384, 12, 6, 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(DIM, eps=1e-6), nn.LayerNorm(DIM, eps=1e-6)
        self.attn = nn.Module()
        self.attn.qkv, self.attn.proj = nn.Linear(DIM, 3 * DIM), nn.Linear(DIM, DIM)
        self.ls1, self.ls2 = nn.Module(), nn.Module()
        self.ls1.gamma, self.ls2.gamma = nn.Parameter(torch.ones(DIM)), nn.Parameter(torch.ones(DIM))
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(DIM, 4 * DIM), nn.Linear(4 * DIM, DIM)

    def attention(self, x):
        n, t, _ = x.shape
        q, k, v = self.attn.qkv(x).reshape(n, t, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1)) * (DIM // HEADS) ** -.5
        return self.attn.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(n, t, DIM))

    def forward(self, x):
        x = x + self.ls1.gamma * self.attention(self.norm1(x))
        return x + self.ls2.gamma * self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))


class ViTS14(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, DIM))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, DIM, PATCH, PATCH)
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM, eps=1e-6)

    def positions(self, side):
        grid = self.pos_embed[:, 1:].reshape(1, 37, 37, DIM).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(side, side), mode='bicubic', align_corners=False)
        return torch.cat([self.pos_embed[:, :1], grid.permute(0, 2, 3, 1).reshape(1, side * side, DIM)], 1)

    def forward(self, x):
        p = self.patch_embed.proj(x)
        side = p.shape[-1]
        h = torch.cat([self.cls_token.expand(len(x), -1, -1), p.flatten(2).transpose(1, 2)], 1) + self.positions(side)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return {'cls': h[:, 0], 'patch': h[:, 1:].mean(1)}


def prepare(x, size=SIZE):
    """[0, 1] images to the network's input: three channels, resized, normalized."""
    if x.dim() == 3:
        x = x[:, None].expand(-1, 3, -1, -1)
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor(MEAN, device=x.device)[None, :, None, None]
    std = torch.tensor(STD, device=x.device)[None, :, None, None]
    return (x - mean) / std


def load(device):
    model = ViTS14()
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def layers(model, chunk=32):
    """For experiments.general_encoder: images -> {'cls', 'patch'}, in chunks, since 224 x 224 inputs fill this GPU fast."""
    def fn(x):
        parts = [model(prepare(x[i:i + chunk])) for i in range(0, len(x), chunk)]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return fn


def gate():
    """Is this implementation right? Nearest-neighbour class reads on CIFAR-10 test images, split as in experiments.natural_memory: colour 32 and grey 28, class token and patch mean, each centred."""
    import numpy as np
    from experiments import device as DEV
    from experiments import natural_memory as NM
    DEV.install()
    fn = layers(load(DEV.NAME if DEV.ACTIVE else 'cpu'))
    sp, d = NM.split_test(392), NM.data()
    x, y = d['x_test'], d['y_test']
    def codes(imgs):
        with torch.no_grad():
            parts = [fn(torch.from_numpy(np.ascontiguousarray(imgs[i:i + 64]))) for i in range(0, len(imgs), 64)]
        return {k: torch.cat([p[k] for p in parts]).cpu().numpy() for k in parts[0]}
    for name, prep in (('colour 32', NM.colour32), ('grey 28', NM.grey28)):
        W, Q = codes(prep(x[sp['write']])), codes(prep(x[sp['query']]))
        for k in W:
            c = W[k].mean(0)
            print(name, k, NM.knn_class(NM.unit(W[k] - c), y[sp['write']], NM.unit(Q[k] - c), y[sp['query']]), flush=True)


if __name__ == '__main__':
    gate()  # NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.dinov2_frozen
