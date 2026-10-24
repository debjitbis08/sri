"""One circular four-ring module: energy inference, familiarity routes and routed ring learning.

Contract: docs/single-module-plasticity-v1-contract.md. Source acquisition, CCA and the direct
references are imported unchanged from the frozen v1/v2/reuse modules. Learners receive observed
pairs and the acquired phase only; world truth (U, V, W, j*, true phase) is used only by the evaluator.
"""
import numpy as np
from scipy.linalg import cho_solve

from src.locality_adaptation import B, D, PHASES, Meter, ScoringFailure, dct_basis, groups, local_basis, shift_matrix
from src.representation_reuse import PhaseReuseTarget, PooledLocal, acquire_local


ROOT_SEED = 2026102410
RANK = D//B
WIDTHS, SECTORS = (32, 24, 16, 8), (8, 6, 4, 2)
OFFSETS = (0, 32, 56, 72)
M = sum(WIDTHS)
RINGS = len(WIDTHS)
BLOCKS = RINGS*RANK
ANCHOR, LATERAL, EPSILON = .1, .001, 1e-12
ETA, PAIRS_PER_BATCH, UPDATES_PER_BATCH, STREAM = .1, 8, 4, 256
DRIFT_DEGREES = 60
TEXTURE_SD, ISO_SD = 2., .5
PROFILES = dict(graded=(32/15, 16/15, 8/15, 4/15), uniform=(1., 1., 1., 1.), reversed=(4/15, 8/15, 16/15, 32/15))
LAYERED_ARMS = dict(graded_matched=('graded', 'matched'), uniform_matched=('uniform', 'matched'),
                    reversed_matched=('reversed', 'matched'), graded_mean=('graded', 'mean'),
                    graded_shifted=('graded', 'shifted'), frozen_layered=(None, 'matched'))
DIRECT_ARMS = ('phase_target_cca', 'pooled_local_cca')
ARMS = (*LAYERED_ARMS, *DIRECT_ARMS)
ADAPTING_ARMS = tuple(a for a, (profile, _) in LAYERED_ARMS.items() if profile)
BRANCHES = ('stable', 'regional', 'global')
B_CHECKPOINTS = (0, 8, 16, 32, 64, 128, 256)
RETURN_CHECKPOINTS = (8, 16, 32, 64, 128, 256)
PRIMARY_WINDOW = (16, 32, 64, 128, 256)
SOLVE_TOL, CONSTRAINT_TOL, SCHUR_TOL, BASIS_FLOOR, RANK_FLOOR = 1e-10, 1e-9, 1e-9, 1e-10, 1e-10
SCHUR_INPUTS = 8


# ---------------------------------------------------------------- generator (reuse v1 recipe, new root)

def rng_for(stage, world, part, extra=(), root=ROOT_SEED):
    return np.random.default_rng(np.random.SeedSequence([root, stage, world, part, *extra]))


def templates(stage, world, root=ROOT_SEED):
    """Part 0: per block 3 signal then 5 texture normals on DCT-II modes (reuse v1 / v1 recipe)."""
    basis, rng = dct_basis(B), rng_for(stage, world, 0, root=root)
    u, v = np.zeros((D, RANK)), np.zeros((D, RANK))
    for j in range(RANK):
        sites = slice(j*B, (j+1)*B)
        t = basis[:, :3]@rng.standard_normal(3)
        n = basis[:, 3:]@rng.standard_normal(5)
        u[sites, j], v[sites, j] = t/np.linalg.norm(t), n/np.linalg.norm(n)
    return u, v


def drift_directions(u, v, stage, world, root=ROOT_SEED):
    """Part 5: one (k, b) Gaussian array; w_j = normalize((I - u_j u_j^T - v_j v_j^T) a_j) per block."""
    a = rng_for(stage, world, 5, root=root).standard_normal((RANK, B))
    w = np.zeros((D, RANK))
    for j in range(RANK):
        sites = slice(j*B, (j+1)*B)
        uj, vj = u[sites, j], v[sites, j]
        r = a[j]-uj*(uj@a[j])-vj*(vj@a[j])
        norm = np.linalg.norm(r)
        if not np.isfinite(norm) or norm == 0:
            raise ScoringFailure(f'zero or nonfinite drift residual in block {j}')
        w[sites, j] = r/norm
    return w


def make_world(stage, world, root=ROOT_SEED):
    """Part 1 reserved; part 2 phase via integers(0, 8); part 6 drifted sector via integers(0, 4)."""
    u, v = templates(stage, world, root)
    phase = int(rng_for(stage, world, 2, root=root).integers(0, PHASES))
    jstar = int(rng_for(stage, world, 6, root=root).integers(0, RANK))
    return dict(u=u, v=v, w=drift_directions(u, v, stage, world, root), phase=phase, jstar=jstar,
                rotation=shift_matrix(D, phase))


def branch_signal(world, branch):
    """Stable U; regional drift rotates column j* by 60 degrees toward w; global rotates all columns."""
    c, s = np.cos(np.deg2rad(DRIFT_DEGREES)), np.sin(np.deg2rad(DRIFT_DEGREES))
    u = world['u'].copy()
    if branch == 'regional':
        j = world['jstar']
        u[:, j] = c*world['u'][:, j]+s*world['w'][:, j]
    elif branch == 'global':
        u = c*world['u']+s*world['w']
    return u


def stream(stage, world, part, n=STREAM, root=ROOT_SEED):
    """Parts 3 source, 4 B, 7 A-return, 8 evaluator A, 9 evaluator B: c, g1, g2, e1, e2 in that order."""
    rng = rng_for(stage, world, part, root=root)
    return tuple(rng.standard_normal((n, size)) for size in (RANK, RANK, RANK, D, D))


def observe(world, signal, draws):
    c, g1, g2, e1, e2 = draws
    r = world['rotation']
    view = lambda g, e: (c@signal.T+TEXTURE_SD*g@world['v'].T+ISO_SD*e)@r.T
    return view(g1, e1), view(g2, e2)


def clean(world, signal, draws):
    return draws[0]@signal.T@world['rotation'].T


# ---------------------------------------------------------------- geometry

def sector_indices(phase):
    """Per ring l, a (4, n_l/4) array of activity indices: outer sector j is groups(phase)[j];
    inner sector j is the contiguous range [j n_l/4, (j+1) n_l/4) of its ring."""
    out = [groups(phase, D, B)]
    for l in range(1, RINGS):
        s = SECTORS[l]
        out.append(OFFSETS[l]+np.arange(WIDTHS[l]).reshape(RANK, s))
    return out


def base_hessian():
    """Anchor 0.1 on the outer ring plus 0.001 times the cycle Laplacian of every ring."""
    h = np.zeros((M, M))
    h[np.arange(D), np.arange(D)] += ANCHOR
    for l, n in enumerate(WIDTHS):
        i = OFFSETS[l]+np.arange(n)
        j = OFFSETS[l]+(np.arange(n)+1) % n
        np.add.at(h, (i, i), LATERAL)
        np.add.at(h, (j, j), LATERAL)
        np.add.at(h, (i, j), -LATERAL)
        np.add.at(h, (j, i), -LATERAL)
    return h


H_BASE = base_hessian()


# ---------------------------------------------------------------- parameters

def complete_basis(w):
    """w first, then coordinate axes in order, two-pass MGS into the complement; skip norms <= 1e-10."""
    cols = [w/np.linalg.norm(w)]
    for axis in np.eye(len(w)):
        if len(cols) == len(w):
            break
        e = axis.copy()
        for _ in range(2):
            for c in cols:
                e -= (c@e)*c
        norm = np.linalg.norm(e)
        if norm > BASIS_FLOOR:
            cols.append(e/norm)
    if len(cols) != len(w):
        raise ScoringFailure('basis completion did not reach full rank')
    return np.stack(cols, 1)


def initial_params(phase, weights):
    d0 = np.stack([complete_basis(w)[:, :SECTORS[1]] for w in weights])
    d1 = np.tile(np.eye(SECTORS[1])[:, :SECTORS[2]], (RANK, 1, 1))
    d2 = np.tile(np.eye(SECTORS[2])[:, :SECTORS[3]], (RANK, 1, 1))
    v = np.tile(np.array([1., 0.]), (RANK, 1))
    return dict(phase=int(phase), maps=[d0, d1, d2], v=v)


def copy_params(p):
    return dict(phase=p['phase'], maps=[m.copy() for m in p['maps']], v=p['v'].copy())


def check_constraints(p):
    for l, maps in enumerate(p['maps']):
        gram = np.einsum('jab,jac->jbc', maps, maps)
        if not np.all(np.isfinite(maps)) or np.abs(gram-np.eye(maps.shape[2])).max() > CONSTRAINT_TOL:
            raise ScoringFailure(f'Stiefel constraint violated for ring map {l}')
    if not np.all(np.isfinite(p['v'])) or np.abs(np.linalg.norm(p['v'], axis=1)-1).max() > CONSTRAINT_TOL:
        raise ScoringFailure('terminal unit-vector constraint violated')


def terminal_q(v):
    """Q_j = I - v_j v_j^T (not assumed idempotent: v may be off the unit sphere in derivative checks)."""
    return np.eye(2)[None]-v[:, :, None]*v[:, None, :]


def block_curvatures(p):
    """K[4l+j] = R^T R for each residual operator; H(a) = H_BASE + sum_b a_b K_b."""
    idx, k = sector_indices(p['phase']), np.zeros((BLOCKS, M, M))
    for l in range(RINGS-1):
        for j in range(RANK):
            r = np.zeros((SECTORS[l], M))
            r[np.arange(SECTORS[l]), idx[l][j]] = 1.
            r[:, idx[l+1][j]] -= p['maps'][l][j]
            k[RANK*l+j] = r.T@r
    q = terminal_q(p['v'])
    for j in range(RANK):
        r = np.zeros((2, M))
        r[:, idx[3][j]] = q[j]
        k[RANK*3+j] = r.T@r
    return k


def energy(x, h, p, a):
    """Independent residual expansion of E_a(x; h, Theta) for one state; a has shape (4, 4)."""
    idx = sector_indices(p['phase'])
    e = ANCHOR/2*np.sum((x[:D]-h)**2)
    for l in range(RINGS-1):
        for j in range(RANK):
            r = x[idx[l][j]]-p['maps'][l][j]@x[idx[l+1][j]]
            e += a[l, j]/2*r@r
    q = terminal_q(p['v'])
    for j in range(RANK):
        r = q[j]@x[idx[3][j]]
        e += a[3, j]/2*r@r
    for l, n in enumerate(WIDTHS):
        ring = x[OFFSETS[l]:OFFSETS[l]+n]
        e += LATERAL/2*np.sum((ring-np.roll(ring, -1))**2)
    return float(e)


# ---------------------------------------------------------------- direct solves

def hessians(k, a):
    """a: (n, 4, 4) route field ordered [ring, sector] -> (n, M, M)."""
    return H_BASE+np.tensordot(a.reshape(len(a), BLOCKS), k, axes=1)


def rhs(x):
    out = np.zeros((len(x), M))
    out[:, :D] = ANCHOR*x
    return out


def factor(h, meter, name):
    try:
        l = np.linalg.cholesky(h)
    except np.linalg.LinAlgError as error:
        raise ScoringFailure(f'{name}: Hessian not positive definite: {error}') from None
    meter.counts[f'{name}_factorizations'] += 1 if h.ndim == 2 else len(h)
    return l


def solve(l, h, b, meter, name):
    """Cholesky solve with relative residual and symmetry checks; l, h may be (M, M) or (n, M, M)."""
    if l.ndim == 2:
        x = cho_solve((l, True), b.T).T
        res = b-x@h.T
    else:
        x = cho_solve((l, True), b[:, :, None])[:, :, 0]
        res = b-np.einsum('nij,nj->ni', h, x)
    rel = np.linalg.norm(res, axis=1)/np.maximum(1., np.linalg.norm(b, axis=1))
    if not np.all(np.isfinite(x)) or rel.max() > SOLVE_TOL:
        raise ScoringFailure(f'{name}: relative solve residual {rel.max():.3g} exceeds {SOLVE_TOL}')
    meter.counts[f'{name}_solves'] += len(b)
    meter.stats[f'{name}_max_relative_residual'] = max(meter.stats.get(f'{name}_max_relative_residual', 0.),
                                                        float(rel.max()))
    return x


def check_symmetric(h, name):
    if np.abs(h-np.swapaxes(h, -1, -2)).max() > 1e-12*max(1., np.abs(h).max()):
        raise ScoringFailure(f'{name}: Hessian is not symmetric')


def cholesky_condition_bound(l):
    """Lower bound on cond(H) from the Cholesky diagonal: (max l_ii / min l_ii)^2."""
    diag = np.diagonal(l, axis1=-2, axis2=-1)
    return float(np.max((diag.max(-1)/diag.min(-1))**2))


class Meter2(Meter):
    """v1 Meter plus max/record statistics (runtime JSON only)."""
    def __init__(self):
        super().__init__()
        self.stats = {}


# ---------------------------------------------------------------- routes

def ring_view(x, idx, l):
    return x[:, idx[l]]


def gates(p, z):
    """g[n, ring, sector] from the pilot state z (n, M)."""
    idx, g = sector_indices(p['phase']), np.zeros((len(z), RINGS, RANK))
    for l in range(RINGS-1):
        zl, zn = ring_view(z, idx, l), ring_view(z, idx, l+1)
        dz = np.einsum('jab,njb->nja', p['maps'][l], zn)
        r = np.sum((zl-dz)**2, -1)/(np.sum(zl**2, -1)+np.sum(dz**2, -1)+EPSILON)
        g[:, l] = .1+.9*np.exp(-r/.25)
    z3 = ring_view(z, idx, 3)
    qz = np.einsum('jab,njb->nja', terminal_q(p['v']), z3)
    r = np.sum(qz**2, -1)/(np.sum(z3**2, -1)+EPSILON)
    g[:, 3] = .1+.9*np.exp(-r/.25)
    return g


def routes(g, policy):
    """(a, b) from g under the arm's policy; b has mean one over sectors of each ring."""
    if policy == 'mean':
        g = np.repeat(g.mean(2, keepdims=True), RANK, 2)
        return .25+.75*g, np.ones_like(g)
    if policy == 'shifted':
        g = np.roll(g, -1, axis=2)
    elif policy != 'matched':
        raise ValueError(policy)
    return .25+.75*g, g/g.mean(2, keepdims=True)


class Module:
    """Persistent state: the 328 map/terminal floats and the phase. Everything else is transient."""

    def __init__(self, params, profile=None, policy='matched', meter=None):
        self.p, self.profile, self.policy = params, profile, policy
        self.meter = meter or Meter2()
        self._k = self._pilot = None

    # cached per parameter state; invalidated by every update
    def curvatures(self):
        if self._k is None:
            with self.meter.time('hessian_assembly'):
                self._k = block_curvatures(self.p)
            self.meter.counts['curvature_assemblies'] += 1
        return self._k

    def pilot_factor(self):
        if self._pilot is None:
            h = hessians(self.curvatures(), np.ones((1, RINGS, RANK)))[0]
            check_symmetric(h, 'pilot')
            with self.meter.time('pilot_factor'):
                l = factor(h, self.meter, 'pilot')
            cond = float(np.linalg.cond(h))
            self.meter.stats['pilot_max_condition'] = max(self.meter.stats.get('pilot_max_condition', 0.), cond)
            self._pilot = (l, h)
        return self._pilot

    def route(self, x):
        """Pilot solve with all a = 1, then routes from the pilot state (inputs only)."""
        l, h = self.pilot_factor()
        with self.meter.time('pilot_solve'):
            z = solve(l, h, rhs(x), self.meter, 'pilot')
        with self.meter.time('routes'):
            g = gates(self.p, z)
            a, b = routes(g, self.policy)
        self.meter.counts['route_evaluations'] += len(x)
        return g, a, b, z

    def second(self, x, a):
        h = hessians(self.curvatures(), a)
        check_symmetric(h, 'second')
        with self.meter.time('second_factor'):
            l = factor(h, self.meter, 'second')
        self.meter.stats['second_max_condition_lower_bound'] = max(
            self.meter.stats.get('second_max_condition_lower_bound', 0.), cholesky_condition_bound(l))
        with self.meter.time('second_solve'):
            state = solve(l, h, rhs(x), self.meter, 'second')
        return state, l, h

    def infer(self, x):
        """Two-pass routed inference; returns the settled outer field, full states and gates."""
        g, a, _, z = self.route(x)
        state, _, _ = self.second(x, a)
        return state[:, :D], dict(state=state, pilot=z, g=g, a=a)

    def invalidate(self):
        self._k = self._pilot = None

    # ------------------------------------------------------------ learning
    def gradient(self, x, t, a, shared=False):
        """Fixed-route implicit gradient of J = (1/(4dB)) sum ||f(x)-t||^2 through the second solve.
        `shared`: all routes are one, so the input-independent pilot factor serves every direction."""
        if shared:
            h = hessians(self.curvatures(), np.ones((1, RINGS, RANK)))[0]
            check_symmetric(h, 'fixed_route')
            with self.meter.time('fixed_route_factor'):
                l = factor(h, self.meter, 'fixed_route')
            with self.meter.time('second_solve'):
                state = solve(l, h, rhs(x), self.meter, 'second')
        else:
            state, l, h = self.second(x, a)
        n_pairs = len(x)//2
        dj = np.zeros_like(state)
        dj[:, :D] = 2*(state[:, :D]-t)/(4*D*n_pairs)
        with self.meter.time('adjoint_solve'):
            u = solve(l, h, dj, self.meter, 'adjoint')
        with self.meter.time('gradient_assembly'):
            grads = implicit_gradients(self.p, state, u, a)
        loss = float(np.sum((state[:, :D]-t)**2)/(4*D*n_pairs))
        return grads, loss

    def update(self, x, t, fixed_routes=False):
        """One optimizer update on the 2B directions (x -> t); routes recomputed unless fixed."""
        if fixed_routes:
            a = b = np.ones((len(x), RINGS, RANK))
        else:
            _, a, b, _ = self.route(x)
        grads, loss = self.gradient(x, t, a, shared=fixed_routes)
        bbar = b.mean(0)
        with self.meter.time('retraction'):
            changes = retract(self.p, grads, self.profile, bbar)
        check_constraints(self.p)
        self.invalidate()
        self.meter.counts['parameter_updates'] += 1
        self.meter.counts['retractions'] += BLOCKS
        return loss, changes

    def learn_batch(self, h1, h2, fixed_routes=False):
        """Exactly four updates on the current minibatch, both prediction directions."""
        x, t = np.concatenate([h1, h2]), np.concatenate([h2, h1])
        out = [self.update(x, t, fixed_routes) for _ in range(UPDATES_PER_BATCH)]
        return [o[0] for o in out], np.sum([o[1] for o in out], 0)


def implicit_gradients(p, x, u, a):
    """Euclidean gradients: -u^T d(grad_x E)/dTheta at the second-solve state x and adjoint u.

    For r = x_l - D x_{l+1}, rho = u_l - D u_{l+1}: dJ/dD = a (r u_{l+1}^T + rho x_{l+1}^T).
    For Q = I - v v^T (general v): with alpha = v.u3, beta = v.x3,
    dJ/dv = -a [-2(beta u3 + alpha x3) + |v|^2 (beta u3 + alpha x3) + 2 alpha beta v]."""
    idx, out = sector_indices(p['phase']), []
    for l in range(RINGS-1):
        maps = p['maps'][l]
        xl, xn, ul, un = (ring_view(y, idx, i) for y, i in ((x, l), (x, l+1), (u, l), (u, l+1)))
        r = xl-np.einsum('jab,njb->nja', maps, xn)
        rho = ul-np.einsum('jab,njb->nja', maps, un)
        w = a[:, l]
        out.append(np.einsum('nj,nja,njb->jab', w, r, un)+np.einsum('nj,nja,njb->jab', w, rho, xn))
    v, x3, u3 = p['v'], ring_view(x, idx, 3), ring_view(u, idx, 3)
    alpha, beta = np.einsum('jb,njb->nj', v, u3), np.einsum('jb,njb->nj', v, x3)
    s = beta[..., None]*u3+alpha[..., None]*x3
    vv = np.sum(v**2, 1)
    df = -2*s+vv[None, :, None]*s+2*(alpha*beta)[..., None]*v[None]
    out.append(-np.einsum('nj,nja->ja', a[:, 3], df))
    return out


def qf(a):
    """Thin QR with positive R diagonal; rank loss stops the run."""
    q, r = np.linalg.qr(a)
    diag = np.diagonal(r, axis1=-2, axis2=-1)
    if not np.all(np.isfinite(diag)) or np.abs(diag).min() <= RANK_FLOOR:
        raise ScoringFailure('retraction rank loss')
    return q*np.where(diag < 0, -1., 1.)[..., None, :]


def tangent(p, grads):
    out = []
    for maps, g in zip(p['maps'], grads[:3]):
        dtg = np.einsum('jab,jac->jbc', maps, g)
        out.append(g-np.einsum('jab,jbc->jac', maps, (dtg+np.swapaxes(dtg, 1, 2))/2))
    v, g = p['v'], grads[3]
    out.append(g-v*np.sum(v*g, 1, keepdims=True))
    return out


def clip(g):
    """Each sector block to Frobenius norm at most 1 (clipping, not normalization)."""
    norm = np.sqrt(np.sum(g**2, axis=tuple(range(1, g.ndim)), keepdims=True))
    return g*np.minimum(1., 1/np.maximum(norm, 1e-300))


def retract(p, grads, profile, bbar):
    """In place: D <- qf(D - eta d_l bbar_lj G_tan); v <- normalize(v - eta d_3 bbar_3j G_tan).
    Returns the Frobenius norm of the applied change per ring group."""
    rates, changes = PROFILES[profile], np.zeros(RINGS)
    for l, g in enumerate(clip(t) for t in tangent(p, grads)):
        step = ETA*rates[l]*bbar[l]
        if l < RINGS-1:
            new = qf(p['maps'][l]-step[:, None, None]*g)
            changes[l] = np.linalg.norm(new-p['maps'][l])
            p['maps'][l] = new
        else:
            v = p['v']-step[:, None]*g
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            if not np.all(np.isfinite(norm)) or norm.min() == 0:
                raise ScoringFailure('zero or nonfinite terminal normalization')
            changes[l] = np.linalg.norm(v/norm-p['v'])
            p['v'] = v/norm
    return changes


def parameter_distance(p, q):
    return np.array([*(np.linalg.norm(a-b) for a, b in zip(p['maps'], q['maps'])), np.linalg.norm(p['v']-q['v'])])


# ---------------------------------------------------------------- Schur reference

def schur_states(k, a, x):
    """Independent direct-map reference: eliminate the 48 interior coordinates by LU solves,
    then reconstruct them from the outer field."""
    out = np.zeros((len(x), M))
    for n, (an, xn) in enumerate(zip(a, x)):
        h = hessians(k, an[None])[0]
        hoo, hoi, hii = h[:D, :D], h[:D, D:], h[D:, D:]
        schur = hoo-hoi@np.linalg.solve(hii, hoi.T)
        outer = np.linalg.solve(schur, ANCHOR*xn)
        out[n, :D], out[n, D:] = outer, -np.linalg.solve(hii, hoi.T@outer)
    return out


def schur_check(module, x):
    """Both passes on x against the Cholesky states; disagreement stops scoring."""
    k = module.curvatures()
    g, a, _, z = module.route(x)
    state, _, _ = module.second(x, a)
    worst = 0.
    for ref, got in ((schur_states(k, np.ones_like(a), x), z), (schur_states(k, a, x), state)):
        scale = np.maximum(1., np.abs(ref).max(1, keepdims=True))
        worst = max(worst, float((np.abs(ref-got)/scale).max()))
    if worst > SCHUR_TOL:
        raise ScoringFailure(f'Schur and full solves disagree by {worst:.3g}')
    module.meter.counts['schur_checked_inputs'] += len(x)
    return worst


# ---------------------------------------------------------------- source acquisition and warm-up

def acquire(xs, ys, meter=None):
    """Frozen v2 CV + refit via reuse v1 `acquire_local` (33 local fits, sums exposed), then the
    initial module from basis completion. Source pairs are not retained by the returned objects."""
    meter = meter or Meter2()
    local = acquire_local(xs, ys, meter=meter)
    with meter.time('basis_completion'):
        params = initial_params(local['phase'], local['weights'])
    return local, params


def warm_up(params, xs, ys, meter=None):
    """One chronological source pass: 32 batches x 4 updates, uniform rates, a = b = 1, no pilot."""
    module = Module(copy_params(params), 'uniform', 'matched', meter)
    losses = []
    for i in range(0, len(xs), PAIRS_PER_BATCH):
        batch = slice(i, i+PAIRS_PER_BATCH)
        losses += module.learn_batch(xs[batch], ys[batch], fixed_routes=True)[0]
    return module.p, losses


# ---------------------------------------------------------------- direct references

def direct_reference(arm, local, meter=None):
    """Frozen reuse v1 learners at the acquired phase; never reset at the return boundary."""
    return dict(phase_target_cca=PhaseReuseTarget, pooled_local_cca=PooledLocal)[arm](local, meter)


def projector_output(learner, x):
    kind, phase, w = learner.deploy()
    basis = local_basis(w, phase, D)
    return x@basis@basis.T


# ---------------------------------------------------------------- evaluator only

def errors(output, signal, phase):
    """Per-example squared error per true block (n, 4) under the true phase groups."""
    return np.stack([np.sum((output[:, g]-signal[:, g])**2, 1) for g in groups(phase, D, B)], 1)


def score(output, signal, phase):
    e = errors(output, signal, phase)
    return dict(E=float(e.sum(1).mean()/RANK), sector_E=e.mean(0))


def linear_trace_risk(m, s, n):
    """Population risk of a fixed linear map m: [tr((M-I)S(M-I)^T) + tr(M N M^T)]/k."""
    i = np.eye(len(m))
    return float((np.trace((m-i)@s@(m-i).T)+np.trace(m@n@m.T))/RANK)


def linear_error_map_risk(m, s, n):
    """Independent check: trace of the full error covariance via an explicit stacked map."""
    i = np.eye(len(m))
    a = np.hstack([m-i, m])
    cov = np.block([[s, np.zeros_like(s)], [np.zeros_like(n), n]])
    return float(np.trace(a@cov@a.T)/RANK)


def fixed_route_map(module, a):
    """Constant linear map h -> x0 for fixed routes a (one (4, 4) field)."""
    h = hessians(module.curvatures(), a[None])[0]
    return np.linalg.solve(h, np.vstack([ANCHOR*np.eye(D), np.zeros((M-D, D))]))[:D]
