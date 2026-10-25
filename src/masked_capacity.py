"""Privileged capacity fitting for the unchanged routed module with observation masks."""
import numpy as np
import torch
from torch import nn

from src import abstraction_preflight as pre
from src import single_module_plasticity as v1

ROOT_SEED = 2026102503
RATIOS = (1/3, 1., 3.)
CHECKPOINTS = (0, 100, 250, 500, 1000)
RATES = (.01, .05)


def task():
    labels = np.array([(r, q, t) for r in (1, 2, 3) for q in range(3) for t in range(8)])
    angles = (labels[:, 2, None]+labels[:, 0, None]*np.arange(4)) % 8
    q = np.array(RATIOS)[labels[:, 1]]
    a = np.sqrt(2/(1+q*q))
    amplitudes = np.stack([a, q*a, a, q*a], axis=1)
    clean = np.zeros((72, 4, 8))
    clean[:, :, 0] = amplitudes*np.cos(angles*np.pi/4)
    clean[:, :, 1] = amplitudes*np.sin(angles*np.pi/4)
    held = labels[:, 1] == labels[:, 0]-1
    return labels, angles, clean, held


def rendering(world):
    if world == 0:
        return np.repeat(np.eye(8)[None], 4, axis=0)
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, world, 0]))
    return np.stack([v1.qf(rng.standard_normal((8, 8))) for _ in range(4)])


def initial_parameters(world, start):
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, world, 1, start]))
    return pre.random_params(rng)


def as_numpy_parameters(p):
    return dict(phase=0, maps=[np.asarray(d, dtype=np.float64) for d in p['maps']],
                v=np.asarray(p['v'], dtype=np.float64))


def scores(pred, x, matrices, subset=None):
    labels, angles, _, _ = task()
    if subset is None:
        subset = np.ones(72, bool)
    loss = np.sum((pred-x[:, 16:])**2, axis=1)/2
    decoded = pre.decode(pred, matrices, (2, 3))
    joint = np.all(decoded == angles[:, 2:], axis=1)
    return dict(mean_loss=float(loss[subset].mean()), worst_loss=float(loss[subset].max()),
        joint_accuracy=float(joint[subset].mean()),
        per_rule={str(r): float(loss[subset & (labels[:, 0] == r)].mean()) for r in (1, 2, 3)},
        per_ratio_index={str(q): float(loss[subset & (labels[:, 1] == q)].mean()) for q in range(3)},
        scene_indices=np.flatnonzero(subset), per_scene_loss=loss[subset],
        decoded_hidden=decoded[subset], all_pass=bool(np.all(loss[subset] <= .10) and np.all(joint[subset])))


def prediction_record(pred, x, matrices):
    _, _, _, held = task()
    return dict(prediction_hidden=pred, all=scores(pred, x, matrices),
                train=scores(pred, x, matrices, ~held), held_out=scores(pred, x, matrices, held))


def latent(x, matrices):
    coordinates = np.einsum('jab,nja->njb', matrices, x.reshape(len(x), 4, 8))
    return coordinates[:, :, 0]+1j*coordinates[:, :, 1]


def encode_hidden(z, matrices):
    values = np.zeros((len(z), 4, 8))
    values[:, 2:, 0], values[:, 2:, 1] = z.real, z.imag
    return pre.render(values, matrices)[:, 16:]


def template_prediction(x, matrices, train, kind):
    indices = np.flatnonzero(train)
    if kind in ('lookup', 'signed_lookup'):
        candidates = x[train]
        if kind == 'signed_lookup':
            candidates = np.stack([candidates, -candidates], axis=1).reshape(-1, 32)
        distance = np.sum((x[:, None, :16]-candidates[None, :, :16])**2, axis=-1)
        chosen = pre.tied_argmin(distance)
        return candidates[chosen, 16:], dict(min_visible_squared_distance=distance.min(axis=1),
            chosen_scene=indices[chosen//2 if kind == 'signed_lookup' else chosen],
            candidate_count=len(candidates), chosen_sign=np.where(chosen % 2 == 0, 1, -1)
            if kind == 'signed_lookup' else np.ones(len(x), dtype=int))
    z = latent(x, matrices)
    query, source = z[:, :2], z[train, :2]
    if kind == 'factored_template':
        query, source = query/np.abs(query), source/np.abs(source)
    c = np.sum(source[None].conj()*query[:, None], axis=-1)/np.sum(abs(source)**2, axis=-1)[None]
    zero = abs(c) <= 1e-15
    if kind == 'factored_template':
        c = np.where(zero, 1+0j, c/np.maximum(abs(c), 1e-300))
    distance = np.sum(abs(query[:, None]-c[:, :, None]*source[None])**2, axis=-1)
    chosen = pre.tied_argmin(distance)
    selected_c = c[np.arange(len(x)), chosen]
    targets = z[train, 2:][chosen]
    if kind == 'factored_template':
        targets = targets/np.abs(targets)*np.abs(z[:, :2])
    predicted = selected_c[:, None]*targets
    return encode_hidden(predicted, matrices), dict(min_visible_squared_distance=distance.min(axis=1),
        chosen_scene=indices[chosen], candidate_count=len(indices), phase_degeneracies=int(zero.sum()),
        fitted_complex_scalar_real=selected_c.real, fitted_complex_scalar_imag=selected_c.imag)


def audit_world(world, meter):
    labels, angles, clean, held = task()
    matrices = rendering(world)
    x = pre.render(clean, matrices)
    train = ~held
    arms = {}
    for name, pred in [('zero', np.zeros((72, 16))),
                       ('training_mean', np.broadcast_to(x[train, 16:].mean(axis=0), (72, 16)))]:
        arms[name] = prediction_record(pred, x, matrices)
    for name, cols, fit in [('first_part_affine', np.arange(8), train),
                            ('two_part_affine', np.arange(16), train),
                            ('all_scene_affine_oracle', np.arange(16), np.ones(72, bool)),
                            ('equal_amplitude_affine_oracle', np.arange(16), labels[:, 1] == 1)]:
        weights, metadata = pre.affine_fit(x[fit][:, cols], x[fit, 16:], meter)
        pred = np.column_stack([x[:, cols], np.ones(72)])@weights
        arms[name] = dict(prediction_record(pred, x, matrices), weights=weights,
                          fit_indices=np.flatnonzero(fit), fit=metadata)
        if name == 'equal_amplitude_affine_oracle':
            value = np.mean(np.sum((pred[fit]-x[fit, 16:])**2, axis=1)/2)
            pre.require(abs(value-pre.AFFINE_FLOOR) < 1e-10, 'equal-amplitude affine floor')
            arms[name]['fit_subset_mean_loss'] = value
    for kind in ('lookup', 'signed_lookup', 'rotation_scale_lookup', 'factored_template'):
        pred, metadata = template_prediction(x, matrices, train, kind)
        arms[kind] = dict(prediction_record(pred, x, matrices), metadata=metadata)
        meter.counts[f'{kind}_candidate_scores'] += len(x)*metadata['candidate_count']
    # This reference receives the known renderer and modular rule, for task identification only.
    z = latent(x, matrices)
    vis_angle = np.round(np.angle(z[:, :2])*4/np.pi).astype(int) % 8
    r = (vis_angle[:, 1]-vis_angle[:, 0]) % 8
    predicted_angles = (vis_angle[:, 0, None]+r[:, None]*np.array([2, 3])) % 8
    oracle = encode_hidden(np.abs(z[:, :2])*np.exp(1j*predicted_angles*np.pi/4), matrices)
    arms['modular_rule_oracle'] = prediction_record(oracle, x, matrices)
    pre.require(arms['modular_rule_oracle']['all']['worst_loss'] < 1e-20, 'oracle identification')
    pre.require(arms['factored_template']['all']['worst_loss'] < 1e-20, 'factored reference')
    no_matches = all(np.min(arms[k]['metadata']['min_visible_squared_distance'][held]) > 1e-10
                     for k in ('signed_lookup', 'rotation_scale_lookup'))
    no_shortcut_pass = all(not arms[k]['held_out']['all_pass'] for k in ('signed_lookup', 'rotation_scale_lookup'))
    meter.counts.update(scene_renderings=72, constant_mean_fits=1, oracle_scene_predictions=72,
                        comparator_scene_scores=72*len(arms))
    return dict(world=world, renderers=matrices, arms=arms,
                narrow_split_gate=bool(no_matches and no_shortcut_pass))


class MaskedModule(nn.Module):
    """Same energy/gates as v1, with explicit masking and full differentiation."""
    def __init__(self, p, meter=None):
        super().__init__()
        self.maps = nn.ParameterList([nn.Parameter(torch.tensor(d, dtype=torch.float64)) for d in p['maps']])
        self.v = nn.Parameter(torch.tensor(p['v'], dtype=torch.float64))
        base = v1.H_BASE.copy()
        base[np.arange(32), np.arange(32)] -= v1.ANCHOR*(1-pre.MASK)
        self.register_buffer('base', torch.tensor(base, dtype=torch.float64))
        self.register_buffer('mask', torch.tensor(pre.MASK, dtype=torch.float64))
        self.indices = [torch.tensor(ix) for ix in v1.sector_indices(0)]
        for l, ix in enumerate(v1.sector_indices(0)):
            selector = np.eye(80)[ix]
            self.register_buffer(f'selector_{l}', torch.tensor(selector, dtype=torch.float64))
        self.meter = meter or pre.Meter()

    def numpy_parameters(self):
        return dict(phase=0, maps=[d.detach().numpy().copy() for d in self.maps],
                    v=self.v.detach().numpy().copy())

    def curvatures(self):
        blocks = []
        for l, d in enumerate(self.maps):
            r = getattr(self, f'selector_{l}')-torch.einsum('jab,jbm->jam', d, getattr(self, f'selector_{l+1}'))
            blocks.append(r.transpose(-1, -2)@r)
        q = torch.eye(2, dtype=torch.float64)[None]-self.v[:, :, None]*self.v[:, None, :]
        r = q@self.selector_3
        blocks.append(r.transpose(-1, -2)@r)
        self.meter.counts['curvature_assemblies'] += 1
        return torch.cat(blocks), q

    def forward(self, observed):
        n = len(observed)
        k, q = self.curvatures()
        rhs = torch.cat([v1.ANCHOR*observed*self.mask, torch.zeros((n, 48), dtype=torch.float64)], dim=1)
        hp = self.base+k.sum(dim=0)
        lp = torch.linalg.cholesky(hp)
        pilot = torch.cholesky_solve(rhs.T, lp).T
        gs = []
        for l, d in enumerate(self.maps):
            zl, zn = pilot[:, self.indices[l]], pilot[:, self.indices[l+1]]
            dz = torch.einsum('jab,njb->nja', d, zn)
            ratio = ((zl-dz)**2).sum(dim=-1)/((zl*zl+dz*dz).sum(dim=-1)+v1.EPSILON)
            gs.append(.1+.9*torch.exp(-ratio/.25))
        z3 = pilot[:, self.indices[3]]
        qz = torch.einsum('jab,njb->nja', q, z3)
        gs.append(.1+.9*torch.exp(-(qz*qz).sum(dim=-1)/((z3*z3).sum(dim=-1)+v1.EPSILON)/.25))
        gates = torch.stack(gs, dim=1)
        routes = .25+.75*gates
        h = self.base+torch.einsum('nk,kij->nij', routes.reshape(n, 16), k)
        l = torch.linalg.cholesky(h)
        state = torch.cholesky_solve(rhs[:, :, None], l).squeeze(-1)
        with torch.no_grad():
            denom = torch.clamp(torch.linalg.vector_norm(rhs, dim=1), min=1.)
            rp = torch.linalg.vector_norm(pilot@hp.T-rhs, dim=1)/denom
            rs = torch.linalg.vector_norm((h@state[:, :, None]).squeeze(-1)-rhs, dim=1)/denom
            pre.require(bool(torch.isfinite(state).all()), 'nonfinite equilibrium')
            pre.require(float(torch.max(torch.cat([rp, rs]))) <= 1e-10, 'Torch solve residual')
        self.meter.counts.update(pilot_factorizations=1, pilot_vector_solves=n,
            second_factorizations=n, second_vector_solves=n, route_evaluations=n, forward_batches=1)
        return state[:, :32]

    def loss(self, observed, targets):
        predicted = self(observed)[:, 16:]
        return ((predicted-targets)**2).sum(dim=1).mean()/2

    @torch.no_grad()
    def project_gradients(self):
        for d in self.maps:
            g = d.grad
            dtg = d.transpose(-1, -2)@g
            g.sub_(d@((dtg+dtg.transpose(-1, -2))/2))
        self.v.grad.sub_(self.v*(self.v*self.v.grad).sum(dim=1, keepdim=True))
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10., error_if_nonfinite=True)

    @torch.no_grad()
    def retract(self):
        for d in self.maps:
            q, r = torch.linalg.qr(d, mode='reduced')
            diag = torch.diagonal(r, dim1=-2, dim2=-1)
            pre.require(bool(torch.isfinite(diag).all()) and float(abs(diag).min()) > 1e-10, 'QR rank failure')
            d.copy_(q*torch.where(diag < 0, -1., 1.)[:, None, :])
        norm = torch.linalg.vector_norm(self.v, dim=1, keepdim=True)
        pre.require(bool(torch.isfinite(norm).all()) and float(norm.min()) > 0, 'terminal normalization')
        self.v.div_(norm)
        v1.check_constraints(self.numpy_parameters())
        self.meter.counts.update(qr_retractions=12, unit_retractions=4)


def numpy_prediction(p, x, meter=None, checked=True):
    """Independent frozen NumPy gates and Schur checks; no autodiff code reused."""
    p = as_numpy_parameters(p)
    meter = meter or pre.Meter()
    k = v1.block_curvatures(p)
    out, checks = [], []
    for scene in x:
        observed = scene*pre.MASK
        if checked:
            pilot, acheck = pre.solve_checked(p, k, np.ones((4, 4)), pre.MASK, observed, meter)
        else:
            rhs = np.r_[v1.ANCHOR*observed, np.zeros(48)]
            pilot = np.linalg.solve(pre.masked_hessian(k, np.ones((4, 4)), pre.MASK), rhs)
        routes, _ = v1.routes(v1.gates(p, pilot[None]), 'matched')
        if checked:
            state, bcheck = pre.solve_checked(p, k, routes[0], pre.MASK, observed, meter)
            checks.extend([acheck, bcheck])
        else:
            state = np.linalg.solve(pre.masked_hessian(k, routes[0], pre.MASK), rhs)
        out.append(state[16:32])
    summary = {key: dict(min=min(c[key] for c in checks), max=max(c[key] for c in checks))
               for key in checks[0]} if checks else {}
    return np.array(out), summary
