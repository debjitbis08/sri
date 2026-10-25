"""Relational capacity Stage B v1: rank-two, parallel-mean and terminal-coupled capacity search.

Contract: docs/relational-capacity-stage-b-v1-contract.md. The task, renderings, scoring, decoding,
radial geometry and v1 gate formula are imported unchanged. Hessians are assembled here because the
frozen helpers hard-code one module and anchor 0.1. Targets enter only through the loss and scores.
"""
import numpy as np
import torch
from torch import nn
from scipy.linalg import cho_factor, cho_solve

from src import abstraction_preflight as pre
from src import masked_capacity as c
from src import single_module_plasticity as v1

ROOT_SEED = 2026102506
ALPHA = .1
MUS = (.001, .01, .1)
RATES = (.003, .01)
STARTS = (0, 1)
WORLDS = (0, 1)
CHECKPOINTS = (0, 100, 250, 500, 1000)
UPDATES = 1000
ARMS = (('S2', (None,)), ('P0', (None,)), ('D1', MUS))
A_MIN = .325
AFFINE_FLOOR = (8-2*np.sqrt(2))/21
TAU = float(np.sqrt(2*AFFINE_FLOOR)-np.sqrt(.2))
SMALL_DENOMINATOR = 1e-8
TERMINAL = np.arange(72, 80)
HIDDEN = np.arange(16, 32)
CYCLES = v1.H_BASE-np.diag(np.r_[v1.ANCHOR*np.ones(32), np.zeros(48)])   # lambda cycle Laplacians only

# Each system is solved jointly; slots refer to stored parameter modules.
LAYOUTS = dict(
    S2=[dict(modules=(0,), anchors=(ALPHA,))],
    P0=[dict(modules=(0,), anchors=(ALPHA/2,)), dict(modules=(1,), anchors=(ALPHA/2,))],
    D1=[dict(modules=(0, 1), anchors=(ALPHA, 0.))])
# Output is the arithmetic mean of these (system, slot) outer hidden states.
OUTPUTS = dict(S2=((0, 0),), P0=((0, 0), (1, 0)), D1=((0, 1),))


def weak_coupling_threshold(alpha=ALPHA, lam=v1.LATERAL, a_min=A_MIN):
    """mu below this value is excluded by the supplementary weak-coupling bound."""
    return .5*(a_min**2*(1-np.sqrt(.2))**2/(60*alpha)-4*lam)


def initial_parameters(name, world, start):
    """SeedSequence([root, world, start, module]); S2 keeps module 0's maps and drops its terminal vectors."""
    ps = [pre.random_params(np.random.default_rng(np.random.SeedSequence([ROOT_SEED, world, start, m])))
          for m in (0, 1)]
    if name == 'S2':
        return [dict(maps=ps[0]['maps'])]
    return [dict(maps=p['maps'], v=p['v']) for p in ps]


def check_constraints(params):
    for p in params:
        v1.check_constraints(dict(phase=0, maps=p['maps'], v=p.get('v', np.tile([1., 0.], (4, 1)))))


def stored_floats(params):
    return int(sum(m.size for p in params for m in p['maps'])+sum(p['v'].size for p in params if 'v' in p))


# ---------------------------------------------------------------- independent NumPy reference

def numpy_curvatures(p):
    v = p.get('v', np.tile([1., 0.], (4, 1)))
    k = v1.block_curvatures(dict(phase=0, maps=p['maps'], v=v))
    return k if 'v' in p else k[:12]


def numpy_gates(p, z):
    """Existing v1 gate formula from one module's 80-state slice; rank two omits terminal gates."""
    v = p.get('v', np.tile([1., 0.], (4, 1)))
    g = v1.gates(dict(phase=0, maps=p['maps'], v=v), z).reshape(len(z), 16)
    return g if 'v' in p else g[:, :12]


def system_size(system):
    return 80*len(system['modules'])


def numpy_hessian(params, system, routes, mu, curvatures=None):
    """routes: one route vector per slot."""
    n = system_size(system)
    h = np.zeros((n, n))
    for slot, (module, anchor) in enumerate(zip(system['modules'], system['anchors'])):
        k = numpy_curvatures(params[module]) if curvatures is None else curvatures[module]
        s = slice(80*slot, 80*slot+80)
        h[s, s] = CYCLES+np.tensordot(routes[slot], k, axes=1)
        h[80*slot+np.arange(32), 80*slot+np.arange(32)] += anchor*pre.MASK
    if len(system['modules']) == 2:
        i, j = TERMINAL, 80+TERMINAL
        h[i, i] += mu
        h[j, j] += mu
        h[i, j] -= mu
        h[j, i] -= mu
    return h


def numpy_rhs(system, observed):
    b = np.zeros(system_size(system))
    for slot, anchor in enumerate(system['anchors']):
        b[80*slot:80*slot+32] = anchor*pre.MASK*observed
    return b


def numpy_energy(state, observed, params, system, routes, mu):
    """Direct residual expansion, independent of every Hessian assembly."""
    idx = v1.sector_indices(0)
    e = 0.
    for slot, (module, anchor) in enumerate(zip(system['modules'], system['anchors'])):
        x, p, a = state[80*slot:80*slot+80], params[module], routes[slot]
        e += anchor/2*np.sum(pre.MASK*(x[:32]-observed)**2)
        for l in range(3):
            for j in range(4):
                r = x[idx[l][j]]-p['maps'][l][j]@x[idx[l+1][j]]
                e += a[4*l+j]/2*r@r
        if 'v' in p:
            for j in range(4):
                r = (np.eye(2)-np.outer(p['v'][j], p['v'][j]))@x[idx[3][j]]
                e += a[12+j]/2*r@r
        for offset, width in zip(v1.OFFSETS, v1.WIDTHS):
            ring = x[offset:offset+width]
            e += v1.LATERAL/2*np.sum((ring-np.roll(ring, -1))**2)
    if len(system['modules']) == 2:
        e += mu/2*np.sum((state[TERMINAL]-state[80+TERMINAL])**2)
    return float(e)


def checked_solve(h, b, output, meter):
    """Cholesky solve with symmetry, residual and Schur-onto-output checks."""
    pre.require(np.abs(h-h.T).max() <= 1e-12*max(1., np.abs(h).max()), 'Hessian symmetry')
    factor = cho_factor(h, lower=True, check_finite=True)
    x = cho_solve(factor, b)
    residual = np.linalg.norm(h@x-b)/max(1., np.linalg.norm(b))
    pre.require(np.all(np.isfinite(x)) and residual <= 1e-10, 'NumPy solve residual')
    rest = np.setdiff1d(np.arange(len(b)), output)
    hrr = cho_factor(h[np.ix_(rest, rest)], lower=True)
    schur = h[np.ix_(output, output)]-h[np.ix_(output, rest)]@cho_solve(hrr, h[np.ix_(rest, output)])
    reduced = b[output]-h[np.ix_(output, rest)]@cho_solve(hrr, b[rest])
    y = np.linalg.solve(schur, reduced)
    disagreement = float(np.abs(y-x[output]).max())
    pre.require(disagreement <= 1e-9, 'Schur/full disagreement')
    diag = np.diagonal(factor[0])
    meter.counts[f'numpy_cholesky_d{len(b)}'] += 1
    return x, dict(relative_residual=float(residual), schur_disagreement=disagreement,
                   condition_lower_bound=float((diag.max()/diag.min())**2))


def numpy_prediction(params, name, mu, x, meter=None, checked=True):
    """Two-pass inference for every scene; returns routed and fixed-route (pilot) outputs."""
    meter = meter or pre.Meter()
    systems, curvatures = LAYOUTS[name], [numpy_curvatures(p) for p in params]
    out, pilot_out, checks = np.zeros((len(x), 16)), np.zeros((len(x), 16)), []
    for n, scene in enumerate(x):
        observed = scene[:32]*pre.MASK
        states, pilots = [], []
        for s, system in enumerate(systems):
            b = numpy_rhs(system, observed)
            ones = [np.ones(len(curvatures[m])) for m in system['modules']]
            hp = numpy_hessian(params, system, ones, mu, curvatures)
            output = np.concatenate([80*slot+HIDDEN for owner, slot in OUTPUTS[name] if owner == s])
            if checked:
                z, ca = checked_solve(hp, b, output, meter)
            else:
                z = np.linalg.solve(hp, b)
            routes = [.25+.75*numpy_gates(params[m], z[None, 80*slot:80*slot+80])[0]
                      for slot, m in enumerate(system['modules'])]
            h = numpy_hessian(params, system, routes, mu, curvatures)
            if checked:
                y, cb = checked_solve(h, b, output, meter)
                checks.extend([ca, cb])
            else:
                y = np.linalg.solve(h, b)
            states.append(y)
            pilots.append(z)
        out[n] = np.mean([states[s][80*slot+HIDDEN] for s, slot in OUTPUTS[name]], axis=0)
        pilot_out[n] = np.mean([pilots[s][80*slot+HIDDEN] for s, slot in OUTPUTS[name]], axis=0)
    summary = {key: dict(min=min(ch[key] for ch in checks), max=max(ch[key] for ch in checks))
               for key in checks[0]} if checks else {}
    return out, pilot_out, summary


# ---------------------------------------------------------------- differentiable Torch model

class Composition(nn.Module):
    """Collective pilot with all routes one, local gates per module slice, one collective routed solve."""

    def __init__(self, name, mu, params, meter=None):
        super().__init__()
        self.name, self.mu = name, 0. if mu is None else float(mu)
        self.systems, self.outputs = LAYOUTS[name], OUTPUTS[name]
        self.rank_one = 'v' in params[0]
        self.maps = nn.ParameterList([nn.Parameter(torch.tensor(d, dtype=torch.float64))
                                      for p in params for d in p['maps']])
        self.vs = nn.ParameterList([nn.Parameter(torch.tensor(p['v'], dtype=torch.float64))
                                    for p in params] if self.rank_one else [])
        self.count = len(params)
        self.register_buffer('cycles', torch.tensor(CYCLES, dtype=torch.float64))
        self.register_buffer('mask', torch.tensor(pre.MASK, dtype=torch.float64))
        port = np.zeros((160, 160))
        port[np.r_[TERMINAL, 80+TERMINAL], np.r_[TERMINAL, 80+TERMINAL]] = 1.
        port[TERMINAL, 80+TERMINAL] = port[80+TERMINAL, TERMINAL] = -1.
        self.register_buffer('port', torch.tensor(port, dtype=torch.float64))
        self.indices = [torch.tensor(ix) for ix in v1.sector_indices(0)]
        for l, ix in enumerate(v1.sector_indices(0)):
            self.register_buffer(f'selector_{l}', torch.tensor(np.eye(80)[ix], dtype=torch.float64))
        self.meter = meter or pre.Meter()
        self.stats = {}

    def module_maps(self, m):
        return list(self.maps[3*m:3*m+3])

    def numpy_parameters(self):
        out = []
        for m in range(self.count):
            p = dict(maps=[d.detach().numpy().copy() for d in self.module_maps(m)])
            if self.rank_one:
                p['v'] = self.vs[m].detach().numpy().copy()
            out.append(p)
        return out

    def curvatures(self, m):
        blocks = []
        for l, d in enumerate(self.module_maps(m)):
            r = getattr(self, f'selector_{l}')-torch.einsum('jab,jbm->jam', d, getattr(self, f'selector_{l+1}'))
            blocks.append(r.transpose(-1, -2)@r)
        q = None
        if self.rank_one:
            v = self.vs[m]
            q = torch.eye(2, dtype=torch.float64)[None]-v[:, :, None]*v[:, None, :]
            r = q@self.selector_3
            blocks.append(r.transpose(-1, -2)@r)
        self.meter.counts['curvature_assemblies'] += 1
        return torch.cat(blocks), q

    def hessian(self, system, ks, routes):
        """routes: per slot (n, blocks); returns (n, D, D)."""
        slots = []
        for slot, (module, anchor) in enumerate(zip(system['modules'], system['anchors'])):
            h = self.cycles+torch.diag(torch.cat([anchor*self.mask, torch.zeros(48, dtype=torch.float64)]))
            slots.append(h+torch.einsum('nk,kij->nij', routes[slot], ks[module][0]))
        if len(slots) == 1:
            return slots[0]
        zero = torch.zeros_like(slots[0])
        top, bottom = torch.cat([slots[0], zero], dim=-1), torch.cat([zero, slots[1]], dim=-1)
        return torch.cat([top, bottom], dim=-2)+self.mu*self.port

    def gates(self, m, q, z):
        gs, denominators = [], []
        for l, d in enumerate(self.module_maps(m)):
            zl, zn = z[:, self.indices[l]], z[:, self.indices[l+1]]
            dz = torch.einsum('jab,njb->nja', d, zn)
            denominator = (zl*zl+dz*dz).sum(dim=-1)
            gs.append(.1+.9*torch.exp(-((zl-dz)**2).sum(dim=-1)/(denominator+v1.EPSILON)/.25))
            denominators.append(denominator)
        if self.rank_one:
            z3 = z[:, self.indices[3]]
            qz = torch.einsum('jab,njb->nja', q, z3)
            denominator = (z3*z3).sum(dim=-1)
            gs.append(.1+.9*torch.exp(-(qz*qz).sum(dim=-1)/(denominator+v1.EPSILON)/.25))
            denominators.append(denominator)
        smallest = float(torch.stack(denominators).detach().min())
        self.stats['min_gate_denominator'] = min(self.stats.get('min_gate_denominator', np.inf), smallest)
        if smallest < SMALL_DENOMINATOR:
            self.meter.counts['small_gate_denominator_batches'] += 1
        return torch.cat(gs, dim=1)

    def forward(self, observed):
        """Returns (routed output, fixed-route pilot output), each (n, 16)."""
        n = len(observed)
        ks = [self.curvatures(m) for m in range(self.count)]
        routed, fixed = [], []
        for system in self.systems:
            size = system_size(system)
            rhs = torch.cat([torch.cat([anchor*observed*self.mask, torch.zeros((n, 48), dtype=torch.float64)], dim=1)
                             for anchor in system['anchors']], dim=1)
            ones = [torch.ones((1, len(ks[m][0])), dtype=torch.float64) for m in system['modules']]
            hp = self.hessian(system, ks, ones)[0]
            lp = torch.linalg.cholesky(hp)
            pilot = torch.cholesky_solve(rhs.T, lp).T
            routes = [.25+.75*self.gates(m, ks[m][1], pilot[:, 80*slot:80*slot+80])
                      for slot, m in enumerate(system['modules'])]
            h = self.hessian(system, ks, routes)
            l = torch.linalg.cholesky(h)
            state = torch.cholesky_solve(rhs[:, :, None], l).squeeze(-1)
            with torch.no_grad():
                denom = torch.clamp(torch.linalg.vector_norm(rhs, dim=1), min=1.)
                rp = torch.linalg.vector_norm(pilot@hp.T-rhs, dim=1)/denom
                rs = torch.linalg.vector_norm((h@state[:, :, None]).squeeze(-1)-rhs, dim=1)/denom
                pre.require(bool(torch.isfinite(state).all() and torch.isfinite(pilot).all()), 'nonfinite equilibrium')
                pre.require(float(torch.max(torch.cat([rp, rs]))) <= 1e-10, 'Torch solve residual')
            self.meter.counts.update({f'pilot_factorizations_d{size}': 1, f'pilot_vector_solves_d{size}': n,
                                      f'second_factorizations_d{size}': n, f'second_vector_solves_d{size}': n,
                                      'route_evaluations': n*len(system['modules'])})
            routed.append(state)
            fixed.append(pilot)
        hidden = torch.tensor(HIDDEN)
        pick = lambda states: torch.stack([states[s][:, 80*slot+hidden] for s, slot in self.outputs]).mean(dim=0)
        self.meter.counts['forward_batches'] += 1
        return pick(routed), pick(fixed)

    @staticmethod
    def scene_losses(predicted, targets):
        return ((predicted-targets)**2).sum(dim=1)/2

    def objective(self, observed, targets):
        """Exact maximum per-scene loss; torch.amax averages gradients over exact ties."""
        losses = self.scene_losses(self(observed)[0], targets)
        return torch.amax(losses), losses

    @torch.no_grad()
    def project_gradients(self):
        for d in self.maps:
            dtg = d.transpose(-1, -2)@d.grad
            d.grad.sub_(d@((dtg+dtg.transpose(-1, -2))/2))
        for v in self.vs:
            v.grad.sub_(v*(v*v.grad).sum(dim=1, keepdim=True))
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10., error_if_nonfinite=True)

    @torch.no_grad()
    def retract(self):
        for d in self.maps:
            q, r = torch.linalg.qr(d, mode='reduced')
            diag = torch.diagonal(r, dim1=-2, dim2=-1)
            pre.require(bool(torch.isfinite(diag).all()) and float(abs(diag).min()) > 1e-10, 'QR rank failure')
            d.copy_(q*torch.where(diag < 0, -1., 1.)[:, None, :])
        for v in self.vs:
            norm = torch.linalg.vector_norm(v, dim=1, keepdim=True)
            pre.require(bool(torch.isfinite(norm).all()) and float(norm.min()) > 0, 'terminal normalization')
            v.div_(norm)
        check_constraints(self.numpy_parameters())
        self.meter.counts.update(qr_retractions=len(self.maps)*4, unit_retractions=len(self.vs)*4)


# ---------------------------------------------------------------- scores and screen

def route_screen(routed, fixed, targets):
    """Necessary condition on the 24 q=1 scenes for any witness (contract section 7)."""
    labels, _, _, _ = c.task()
    q1 = labels[:, 1] == 1
    d = float(np.sqrt(np.mean(np.sum((routed[q1]-fixed[q1])**2, axis=1))))
    fixed_rms = float(np.sqrt(np.mean(np.sum((fixed[q1]-targets[q1])**2, axis=1))))
    specific = max(0., fixed_rms-float(np.sqrt(.2)))
    return dict(d=d, tau=TAU, parameter_specific_threshold=specific, fixed_route_rms_error=fixed_rms,
                excludes_setting=bool(d < TAU or d < specific))


def selection_key(summary, step):
    return (summary['all']['worst_loss'], summary['all']['mean_loss'], step)
