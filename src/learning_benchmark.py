"""Tasks, encoders, objectives and probe metric for the single-module learning benchmark v1."""
import copy

import numpy as np
import torch
from torch import func, nn


D, M, BLOCK = 32, 4, 8
BUDGETS = (4, 8, 16, 32, 64, 128, 256)
REGIMES = ('aligned', 'shifted', 'dense')
CONDITIONS = ('in_distribution', 'heldout_combination', 'changed_nuisance', 'new_detail')
ISO, TEXTURE, NEW_DETAIL = .25, 4., 2.
CCA_RIDGE = .1
PROBE, TEST, STEPS = 64, 1024, 1500
SEED = 20261201


def dct_basis():
    j = np.arange(BLOCK)+.5
    out = np.stack([np.cos(np.pi*k*j/BLOCK) for k in range(BLOCK)], 1)*np.sqrt(2/BLOCK)
    out[:, 0] = 1/np.sqrt(BLOCK)
    return out


def task_world(index, seed=SEED):
    """Smooth broad templates (DCT modes 0-2) and view-independent texture
    nuisance (modes 3-7) in each eight-site block, plus the regime rotations."""
    rng = np.random.default_rng([seed, 0, index])
    basis = dct_basis()
    u, v = np.zeros((D, M)), np.zeros((D, M))
    for k in range(M):
        sites = slice(k*BLOCK, (k+1)*BLOCK)
        t = basis[:, :3]@rng.standard_normal(3)
        n = basis[:, 3:]@rng.standard_normal(5)
        u[sites, k], v[sites, k] = t/np.linalg.norm(t), n/np.linalg.norm(n)
    q, r = np.linalg.qr(rng.standard_normal((D, D)))
    q = q*np.where(np.diag(r) < 0, -1., 1.)[None, :]
    return u, v, dict(aligned=np.eye(D), shifted=np.roll(np.eye(D), 4, axis=0), dense=q)


def coefficients(rng, n, heldout=False):
    """c ~ N(0,I) conditioned on sign(c0 c1) > 0 (training) or < 0 (held out)."""
    c = rng.standard_normal((n, M))
    sign = np.sign(c[:, 0])*(-1 if heldout else 1)
    c[:, 1] = np.abs(c[:, 1])*np.where(sign == 0, 1, sign)
    return c


def observe(rng, world, regime, c, condition='in_distribution'):
    """One view per row of c. Rotation applies to signal and nuisance together."""
    u, v, rotations = world
    n = len(c)
    texture = TEXTURE*(4 if condition == 'changed_nuisance' else 1)
    h = c@u.T+np.sqrt(texture)*rng.standard_normal((n, M))@v.T+np.sqrt(ISO)*rng.standard_normal((n, D))
    if condition == 'new_detail':
        basis = np.hstack([u, v])
        extra = rng.standard_normal((n, D))
        extra -= extra@basis@basis.T
        h += np.sqrt(NEW_DETAIL)*extra
    return h@rotations[regime].T


def training_pairs(world, regime, index, seed=SEED, size=max(BUDGETS)):
    """Nested-prefix pair stream; both views share only c."""
    rng = np.random.default_rng([seed, 1, index, REGIMES.index(regime)])
    c = coefficients(rng, size)
    return observe(rng, world, regime, c), observe(rng, world, regime, c)


def evaluation_sets(world, regime, index, seed=SEED):
    """Labelled probe set (counted supervision) and one test set per condition."""
    rng = np.random.default_rng([seed, 2, index, REGIMES.index(regime)])
    c = coefficients(rng, PROBE)
    probe = (observe(rng, world, regime, c), c)
    tests = {}
    for condition in CONDITIONS:
        c = coefficients(rng, TEST, heldout=condition == 'heldout_combination')
        tests[condition] = (observe(rng, world, regime, c, condition), c)
    return probe, tests


def probe_r2(code_probe, c_probe, code_tests, ridge=1e-3):
    """Ridge linear probe fitted on the counted labelled set only.

    R2 = 1 - sum||c_hat-c||^2 / sum||c||^2; a constant code scores <=0."""
    mean, scale = code_probe.mean(0), code_probe.std(0)+1e-8
    x = np.hstack([(code_probe-mean)/scale, np.ones((len(code_probe), 1))])
    w = np.linalg.solve(x.T@x+ridge*len(x)*np.eye(x.shape[1]), x.T@c_probe)
    out = {}
    for name, (code, c) in code_tests.items():
        pred = np.hstack([(code-mean)/scale, np.ones((len(code), 1))])@w
        out[name] = float(1-np.sum((pred-c)**2)/np.sum(c**2))
    return out


# ---------------------------------------------------------------- spectral references

def cross_covariance(h1, h2):
    k = h1.T@h2/len(h1)
    return (k+k.T)/2


def global_spectral(h1, h2):
    return np.linalg.eigh(cross_covariance(h1, h2))[1][:, -M:]


def local_spectral(h1, h2):
    k, out = cross_covariance(h1, h2), np.zeros((D, M))
    for b in range(M):
        s = slice(b*BLOCK, (b+1)*BLOCK)
        out[s, b] = np.linalg.eigh(k[s, s])[1][:, -1]
    return out


def fixed_averaging(h1=None, h2=None):
    out = np.zeros((D, M))
    for b in range(M):
        out[b*BLOCK:(b+1)*BLOCK, b] = 1/np.sqrt(BLOCK)
    return out


def _cca(k, pooled, rank):
    """Top generalized eigenvectors of (k, pooled + ridge); ridge is CCA_RIDGE * mean variance."""
    reg = pooled+CCA_RIDGE*np.trace(pooled)/len(pooled)*np.eye(len(pooled))
    l = np.linalg.cholesky(reg)
    li = np.linalg.inv(l)
    vecs = np.linalg.eigh(li@k@li.T)[1][:, -rank:]
    w = li.T@vecs
    return w/np.linalg.norm(w, axis=0)


def pooled_covariance(h1, h2):
    both = np.vstack([h1, h2])
    return both.T@both/len(both)


def global_cca(h1, h2):
    return _cca(cross_covariance(h1, h2), pooled_covariance(h1, h2), M)


def local_cca(h1, h2):
    k, p, out = cross_covariance(h1, h2), pooled_covariance(h1, h2), np.zeros((D, M))
    for b in range(M):
        s = slice(b*BLOCK, (b+1)*BLOCK)
        out[s, b] = _cca(k[s, s], p[s, s], 1)[:, 0]
    return out


REFERENCES = dict(local_spectral=local_spectral, global_spectral=global_spectral,
                  local_cca=local_cca, global_cca=global_cca, fixed_averaging=fixed_averaging)


# ---------------------------------------------------------------- encoders

class CoarseLocal(nn.Module):
    """Learned coarse module: 4-site 32->8 and pairwise 8->4 restrictions.

    Its settled core equals B A h exactly (preflight), so this is the module."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(8, 4, dtype=torch.float32)/2)
        self.b = nn.Parameter(torch.randn(M, 2, dtype=torch.float32)/np.sqrt(2))

    def forward(self, h):
        z = (h.reshape(-1, 8, 4)*self.a).sum(-1)
        return (z.reshape(-1, M, 2)*self.b).sum(-1)


class DenseLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(D, M, bias=False, dtype=torch.float32)

    def forward(self, h):
        return self.w(h)


def mlp(width):
    return nn.Sequential(nn.Linear(D, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, M))


class ConvResidual(nn.Module):
    """Circular 1-D convolutional encoder with one residual block."""
    def __init__(self, width):
        super().__init__()
        conv = lambda i, o: nn.Conv1d(i, o, 5, padding=2, padding_mode='circular', dtype=torch.float32)
        self.stem, self.c1, self.c2 = conv(1, width), conv(width, width), conv(width, width)
        self.head = nn.Linear(width*D, M, dtype=torch.float32)

    def forward(self, h):
        x = torch.relu(self.stem(h[:, None, :]))
        x = torch.relu(x+self.c2(torch.relu(self.c1(x))))
        return self.head(x.flatten(1))


class PairedDAE(nn.Module):
    """Denoising autoencoder trained across views: encode one view, reconstruct the other."""
    def __init__(self, width):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, width), nn.ReLU(), nn.Linear(width, M))
        self.dec = nn.Sequential(nn.Linear(M, width), nn.ReLU(), nn.Linear(width, D))

    def forward(self, h, decode=False):
        return self.dec(self.enc(h)) if decode else self.enc(h)


def vicreg(z1, z2, inv=25., var=25., cov=1.):
    """VICReg applied directly to the 4-dim code (no expander head)."""
    loss = inv*torch.mean((z1-z2)**2)
    for z in (z1, z2):
        std = torch.sqrt(z.var(0)+1e-4)
        loss = loss+var*torch.mean(torch.relu(1-std))
        zc = z-z.mean(0)
        c = zc.T@zc/(len(z)-1)
        loss = loss+cov*(c.pow(2).sum()-c.diagonal().pow(2).sum())/M
    return loss


ARMS = {
    'coarse_local_vicreg': dict(build=lambda cfg: CoarseLocal(), objective='vicreg',
                                grid=[dict(lr=lr, wd=wd) for lr in (3e-3, 1e-2, 3e-2) for wd in (0., 1e-2)]),
    'dense_linear_vicreg': dict(build=lambda cfg: DenseLinear(), objective='vicreg',
                                grid=[dict(lr=lr, wd=wd) for lr in (3e-3, 1e-2, 3e-2) for wd in (0., 1e-2)]),
    'mlp_vicreg': dict(build=lambda cfg: mlp(cfg['width']), objective='vicreg',
                       grid=[dict(lr=lr, wd=1e-4, width=w) for lr in (1e-3, 3e-3, 1e-2) for w in (16, 64)]),
    'conv_vicreg': dict(build=lambda cfg: ConvResidual(cfg['width']), objective='vicreg',
                        grid=[dict(lr=lr, wd=1e-4, width=w) for lr in (1e-3, 3e-3, 1e-2) for w in (4, 8)]),
    'dae_paired': dict(build=lambda cfg: PairedDAE(cfg['width']), objective='dae',
                       grid=[dict(lr=lr, wd=1e-4, width=w) for lr in (1e-3, 3e-3, 1e-2) for w in (16, 64)]),
}


def parameter_count(arm, cfg):
    return sum(p.numel() for p in ARMS[arm]['build'](cfg).parameters())


def group_key(cfg):
    """Configurations sharing an architecture train together as one ensemble."""
    return cfg.get('width')


def train_ensemble(arm, cfgs, h1, h2, init_seeds, steps=STEPS):
    """Train K independent models in one vectorized pass.

    cfgs, init_seeds: length K (same architecture); h1, h2: (K, n, D) arrays.
    Full-batch AdamW, written out explicitly so each member has its own lr/wd.
    Returns stacked parameters, the meta-model, and per-member convergence."""
    spec = ARMS[arm]
    models = []
    for cfg, seed in zip(cfgs, init_seeds):
        torch.manual_seed(seed)
        models.append(spec['build'](cfg))
    params, _ = func.stack_module_state(models)
    base = copy.deepcopy(models[0]).to('meta')
    lr = torch.tensor([c['lr'] for c in cfgs], dtype=torch.float32)
    wd = torch.tensor([c['wd'] for c in cfgs], dtype=torch.float32)
    t1, t2 = torch.from_numpy(h1).float(), torch.from_numpy(h2).float()

    def loss(p, a, b):
        if spec['objective'] == 'vicreg':
            return vicreg(func.functional_call(base, p, (a,)), func.functional_call(base, p, (b,)))
        return (torch.mean((func.functional_call(base, p, (a,), dict(decode=True))-b)**2)
                + torch.mean((func.functional_call(base, p, (b,), dict(decode=True))-a)**2))

    step_fn = func.vmap(func.grad_and_value(loss))
    m = {k: torch.zeros_like(v) for k, v in params.items()}
    v2 = {k: torch.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = .9, .999, 1e-8
    history = torch.zeros(steps, len(cfgs))
    for t in range(1, steps+1):
        grads, values = step_fn(params, t1, t2)
        history[t-1] = values.detach()
        with torch.no_grad():
            for k, g in grads.items():
                shape = (-1,)+(1,)*(g.dim()-1)
                params[k].mul_(1-lr.view(shape)*wd.view(shape))
                m[k].mul_(beta1).add_(g, alpha=1-beta1)
                v2[k].mul_(beta2).addcmul_(g, g, value=1-beta2)
                update = (m[k]/(1-beta1**t))/(torch.sqrt(v2[k]/(1-beta2**t))+eps)
                params[k].sub_(lr.view(shape)*update)
    tail = history[-steps//10:]
    records = [dict(final_loss=float(history[-1, i]), finite=bool(torch.isfinite(history[-1, i])),
                    tail_relative_change=float((tail[:, i].max()-tail[:, i].min())/(tail[:, i].mean().abs()+1e-12)))
               for i in range(len(cfgs))]
    return params, base, records


def encode(params, base, member, h):
    with torch.no_grad():
        p = {k: v[member] for k, v in params.items()}
        return func.functional_call(base, p, (torch.from_numpy(h).float(),)).double().numpy()
