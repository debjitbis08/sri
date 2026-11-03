"""One interface for sense models, so established models can be used, combined and replaced without redesign (docs/sense-integration-plan.md, step 1).

A sense model is a frozen network that turns inputs into one code per input. Each is described by a `SenseModel`:
what it is (name, sense, layer read), where it comes from (module, weights, sha256), what it costs, and its licence.
Models are registered by name. Nothing here trains a sense model.

Memory combines models through `Joint`. It keeps each model's code as a separate part of a key, centred on that model's centre for the input's group (a domain, say).
Stored keys weight every part equally. A query weights the parts as it likes, by fixed weights or a learned `Gate`, and its similarity to a stored key is then
    sum_i w_i cos_i
over models i, with weights summing to 1. Keys never move when query weights change. A part missing from a key (a model added later) is zero there, so it adds nothing.

Model modules (experiments/dinov2_frozen.py, experiments/trocr_frozen.py) implement the networks; this module only needs from each:
    load(device) -> model,  layers(model, **options) -> fn(images tensor) -> {layer: codes tensor},  WEIGHTS, SHA256.
"""
import hashlib
import importlib
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn


@dataclass(frozen=True)
class Licence:
    """Licence of a model's code and weights. `verified` stays False until someone has read the terms at the source (plan step 2)."""
    code: str = 'unknown'
    weights: str = 'unknown'
    source: str = ''
    verified: bool = False
    notes: str = ''


@dataclass(frozen=True)
class SenseModel:
    name: str  # unique, used in keys and reports
    sense: str  # 'sight', ...
    module: str  # import path of the module implementing the network
    layer: str  # which of the module's outputs is the code
    dim: int
    options: dict = field(default_factory=dict)  # passed to module.layers
    inputs: str = ''  # what the encoder expects
    licence: Licence = Licence()
    reference: str = ''

    def impl(self):
        return importlib.import_module(self.module)

    def weights(self):
        m = self.impl()
        return {'path': str(m.WEIGHTS), 'sha256': m.SHA256}

    def verify_weights(self):
        w = self.weights()
        h = hashlib.sha256()
        with open(w['path'], 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest() == w['sha256']

    def describe(self):
        return {'name': self.name, 'sense': self.sense, 'module': self.module, 'layer': self.layer, 'dim': self.dim, 'options': dict(self.options),
                'inputs': self.inputs, 'licence': vars(self.licence), 'reference': self.reference, **self.weights()}

    def load(self, device='cpu', chunk=64):
        m = self.impl()
        net = m.load(device)
        return Encoder(self, m.layers(net, **self.options), chunk, sum(p.numel() for p in net.parameters()))


class Encoder:
    """A loaded sense model: numpy inputs in, raw float32 codes (n, dim) out, in chunks, with its cost counted."""
    def __init__(self, spec, fn, chunk, parameters):
        self.spec, self.fn, self.chunk, self.parameters = spec, fn, chunk, parameters
        self.seconds, self.inputs = 0., 0

    def __call__(self, x):
        t = time.perf_counter()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.chunk):
                r = self.fn(torch.from_numpy(np.ascontiguousarray(x[i:i + self.chunk], dtype=np.float32)))
                out.append(r[self.spec.layer].detach().cpu().numpy().astype(np.float32))
        self.seconds += time.perf_counter() - t
        self.inputs += len(x)
        return np.concatenate(out)

    def cost(self):
        return {'parameters': int(self.parameters), 'inputs': self.inputs, 'seconds': self.seconds,
                'seconds_per_input': self.seconds / self.inputs if self.inputs else None, 'key_bytes': 4 * self.spec.dim}


REGISTRY = {}


def register(spec):
    if spec.name in REGISTRY and REGISTRY[spec.name] != spec:
        raise ValueError(f'{spec.name} is already registered differently')
    REGISTRY[spec.name] = spec
    return spec


def get(name):
    return REGISTRY[name]


GREY = 'grey (n, h, w) float32 in [0, 1]; colour (n, 3, h, w) also accepted by DINOv2'

register(SenseModel(
    'dinov2-s', 'sight', 'experiments.dinov2_frozen', 'patch', 384, inputs=GREY,
    licence=Licence('Apache-2.0 (believed)', 'Apache-2.0 (believed)', 'https://github.com/facebookresearch/dinov2', notes='not yet checked at the source'),
    reference='Oquab et al. 2023, arXiv 2304.07193; ViT-S/14, mean of patch tokens after the final norm'))
register(SenseModel(
    'trocr-s', 'sight', 'experiments.trocr_frozen', 'cls', 384, options={'invert': True}, inputs=GREY,
    licence=Licence('MIT (believed, microsoft/unilm)', 'unknown (Hugging Face card microsoft/trocr-small-stage1)', 'https://huggingface.co/microsoft/trocr-small-stage1', notes='not yet checked at the source'),
    reference='Li et al. 2021, arXiv 2109.10282; stage 1 encoder, class token after the final norm, inputs inverted'))


class Sense:
    """Several loaded models of one sense, each input encoded once per model."""
    def __init__(self, names, device='cpu', chunk=64):
        self.names = tuple(names)
        self.encoders = {n: get(n).load(device, chunk) for n in self.names}

    def __call__(self, x):
        return {n: e(x) for n, e in self.encoders.items()}

    def describe(self):
        return {n: {**e.spec.describe(), 'cost': e.cost()} for n, e in self.encoders.items()}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def centred(z, c):
    """Subtract a centre, then back to unit length. A code at the centre stays zero, which a read can never match."""
    y = z - c
    return y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)


def join(parts, weights=None):
    """Unit codes from n models joined as sqrt(n) [w_1 z_1, ..., w_n z_n]. Weights are scalars or one per row; by default 1/n each, which gives keys unit length.
    A query with weights w then has dot product sum_i w_i cos_i with a key."""
    n = len(parts)
    weights = [1. / n] * n if weights is None else weights
    w = lambda x: np.asarray(x, np.float32)[..., None] if np.ndim(x) else np.float32(x)
    return np.sqrt(float(n)) * np.concatenate([w(wi) * z for wi, z in zip(weights, parts)], -1).astype(np.float32)


class Joint:
    """How memory combines models: a fixed order of models, one key part each, and a centre per (model, group)."""
    def __init__(self, names):
        self.names = tuple(names)
        self.dims = {n: get(n).dim for n in self.names}
        self.centres = {n: {} for n in self.names}

    @property
    def width(self):
        return sum(self.dims.values())

    def set_centre(self, name, group, c):
        self.centres[name][group] = np.asarray(c, np.float32)

    def code(self, name, group, raw):
        """One model's raw codes as its key part: unit length, centred on the group's centre unless group is None."""
        return unit(raw) if group is None else centred(raw, self.centres[name][group])

    def key(self, codes, group, weights=None):
        """codes: {model: raw codes}. A model missing from codes gives a zero part (a model added after the key was written)."""
        rows = len(next(iter(codes.values())))
        parts = [self.code(n, group, codes[n]) if n in codes else np.zeros((rows, self.dims[n]), np.float32) for n in self.names]
        return join(parts, weights)


class Gate:
    """Per-query weights over models, from their raw unit codes: softmax of one linear layer. Trained with no labels: one view of an image should match
    another view of the same image better than other images, under the weighted similarity sum_i w_i cos_i of the centred codes. No balancing term,
    since inputs need not use the models equally."""
    def __init__(self, layer):
        self.layer = layer

    @classmethod
    def train(cls, raw, views, seed, steps=300, batch=512, lr=1e-2, temperature=.1):
        """raw: list of (n, d_i) raw unit codes of the first view, one per model. views: list of (2, n, d_i) centred codes of two views, one per model."""
        g = torch.Generator(device='cpu').manual_seed(seed)
        W = torch.nn.Linear(sum(r.shape[1] for r in raw), len(raw))
        with torch.no_grad():
            W.weight.copy_(.01 * torch.randn(W.weight.shape, generator=g, device='cpu'))
            W.bias.zero_()
        opt = torch.optim.Adam(W.parameters(), lr=lr)
        X = torch.as_tensor(np.concatenate(raw, 1))
        V = [(torch.as_tensor(v[0]), torch.as_tensor(v[1])) for v in views]
        n = len(X)
        for _ in range(steps):
            idx = torch.randperm(n, generator=g, device='cpu')[:batch].to(X.device)
            w = W(X[idx]).softmax(1)
            S = w[:, :1] * (V[0][0][idx] @ V[0][1][idx].T)
            for i in range(1, len(V)):
                S = S + w[:, i:i + 1] * (V[i][0][idx] @ V[i][1][idx].T)
            loss = Fn.cross_entropy(S / temperature, torch.arange(len(idx), device=X.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        W.requires_grad_(False)
        return cls(W), float(loss.detach())

    def __call__(self, raw):
        return self.layer(torch.as_tensor(np.concatenate(raw, 1))).softmax(1).detach().cpu().numpy()

    def parameters(self):
        return sum(p.numel() for p in self.layer.parameters())
