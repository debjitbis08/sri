"""Generator, counted source acquisition, reuse learners and evaluator for representation reuse v1.

Contract: docs/representation-reuse-v1-contract.md. Estimators, ties, ridge and metrics are
imported unchanged from the frozen v1/v2 modules. Learners receive observed pairs, sufficient
statistics and a declared or acquired phase only; world truth is used only by the evaluator.
"""
import numpy as np

from src.locality_adaptation import (B, D, PHASES, RANK, Meter, ScoringFailure, cca_top, dct_basis,
    finite, groups, local_basis, nmse, orthonormal_basis, phase_floors, projected_noise, select,
    shift_matrix, signal_loss)
from src.locality_adaptation_v2 import FOLDS, cv_scores, fit_cv_refit


ROOT_SEED = 2026102409
TEXTURE_SD, ISO_SD = 2., .5
ANGLES = (0, 15, 30, 60, 90)
SOURCE_BUDGETS = (16, 64, 256)
TARGET_BUDGETS = (0, 4, 8, 16, 32, 64, 128, 256)
CHECKPOINTS = TARGET_BUDGETS[1:]
PRIMARY_SOURCE, PRIMARY_TARGET = 256, (16, 32, 64)
STREAM = 256
CONDITIONS = ('in_distribution', 'low_noise', 'texture_16')
ARMS = ('frozen_local', 'phase_reuse_target', 'pooled_local', 'frozen_global', 'pooled_global',
        'fresh_local_cv', 'fresh_global')
SOURCE_ARMS = ARMS[:5]
FRESH_ARMS = ARMS[5:]
LOCAL_ACQUISITION_ARMS = ARMS[:3]
GLOBAL_ACQUISITION_ARMS = ARMS[3:5]
PROJECTOR_TOL = 1e-9


# ---------------------------------------------------------------- generator

def rng_for(stage, world, part, extra=()):
    return np.random.default_rng(np.random.SeedSequence([ROOT_SEED, stage, world, part, *extra]))


def templates(stage, world, d=D, b=B, extra=()):
    """Part 0: v1's interleaved recipe, per block 3 signal then 5 texture normals."""
    k, basis = d//b, dct_basis(b)
    rng = rng_for(stage, world, 0, extra)
    u, v = np.zeros((d, k)), np.zeros((d, k))
    for j in range(k):
        sites = slice(j*b, (j+1)*b)
        t = basis[:, :3]@rng.standard_normal(3)
        n = basis[:, 3:]@rng.standard_normal(5)
        u[sites, j], v[sites, j] = t/np.linalg.norm(t), n/np.linalg.norm(n)
    return u, v


def drift_directions(u, v, stage, world, b=B, extra=()):
    """Part 5: one (k, b) Gaussian array; w_j = normalize((I - u_j u_j^T - v_j v_j^T) a_j) per block."""
    d, k = u.shape
    a = rng_for(stage, world, 5, extra).standard_normal((k, b))
    w = np.zeros((d, k))
    for j in range(k):
        sites = slice(j*b, (j+1)*b)
        uj, vj = u[sites, j], v[sites, j]
        r = a[j]-uj*(uj@a[j])-vj*(vj@a[j])
        norm = np.linalg.norm(r)
        if not np.isfinite(norm) or norm == 0:
            raise ScoringFailure(f'zero or nonfinite drift residual in block {j}')
        w[sites, j] = r/norm
    return w


def make_world(stage, world, d=D, b=B, extra=()):
    """Part 1 is reserved and never drawn; part 2 is the source phase via integers(0, 8)."""
    u, v = templates(stage, world, d, b, extra)
    phase = int(rng_for(stage, world, 2, extra).integers(0, PHASES))
    return dict(u=u, v=v, w=drift_directions(u, v, stage, world, b, extra), phase=phase,
                rotation=shift_matrix(d, phase), d=d, b=b)


def drifted_signal(world, theta):
    """U_theta = cos(theta) U + sin(theta) W, theta in degrees."""
    angle = np.deg2rad(theta)
    return np.cos(angle)*world['u']+np.sin(angle)*world['w']


def stream(stage, world, part, d=D, b=B, n=STREAM, extra=()):
    """Part 3 source, part 4 target: c, g1, g2, e1, e2 drawn in that order."""
    k, rng = d//b, rng_for(stage, world, part, extra)
    return tuple(rng.standard_normal((n, size)) for size in (k, k, k, d, d))


def observe(world, signal, draws):
    c, g1, g2, e1, e2 = draws
    view = lambda g, e: (c@signal.T+TEXTURE_SD*g@world['v'].T+ISO_SD*e)@world['rotation'].T
    return view(g1, e1), view(g2, e2)


# ---------------------------------------------------------------- sufficient statistics and solves

def block_sums(x, y, phase, b=B):
    """C_j = sum (x y^T + y x^T)/2 and A_j = sum (x x^T + y y^T)/2 on each group of the phase."""
    gs = groups(phase, x.shape[1], b)
    c = np.stack([(x[:, g].T@y[:, g]+y[:, g].T@x[:, g])/2 for g in gs])
    a = np.stack([(x[:, g].T@x[:, g]+y[:, g].T@y[:, g])/2 for g in gs])
    return c, a


def full_sums(x, y):
    return (x.T@y+y.T@x)/2, (x.T@x+y.T@y)/2


def add_pair_local(c, a, x, y, phase, b=B):
    """One arriving pair: rank-one updates of every block sum at the phase, in place."""
    g = groups(phase, len(x), b)
    xb, yb = x[g], y[g]
    cross = xb[:, :, None]*yb[:, None, :]
    c += (cross+cross.transpose(0, 2, 1))/2
    a += (xb[:, :, None]*xb[:, None, :]+yb[:, :, None]*yb[:, None, :])/2


def add_pair_full(c, a, x, y):
    cross = np.outer(x, y)
    c += (cross+cross.T)/2
    a += (np.outer(x, x)+np.outer(y, y))/2


def solve_local(c, a, count, meter):
    """K = C/q, T = A/q per block; v1 ridge, whitening, ties and orientation; returns (k, b) weights."""
    with meter.time('solve'):
        w = np.stack([cca_top(cj/count, aj/count, 1, meter)[:, 0] for cj, aj in zip(c, a)])
    meter.counts['local_solves'] += 1
    return finite(w)


def solve_global(c, a, count, rank, meter):
    with meter.time('solve'):
        basis = orthonormal_basis(finite(cca_top(c/count, a/count, rank, meter)))
    meter.counts['global_solves'] += 1
    return basis


# ---------------------------------------------------------------- source acquisition

def acquire_local(x, y, b=B, meter=None):
    """v2 four-fold CV over eight phases, v1 select, then one all-s refit solved from exposed block sums."""
    meter = meter or Meter()
    with meter.time('source_selection'):
        scores = cv_scores(x, y, b, meter)
        phase = select(scores)
    with meter.time('source_statistics'):
        c, a = block_sums(x, y, phase, b)
    weights = solve_local(c, a, len(x), meter)
    meter.counts['fits'] += 1
    meter.counts['refits'] += 1
    meter.counts['source_pairs'] += len(x)
    meter.counts['fit_row_exposures'] += len(x)
    return dict(phase=phase, weights=weights, c=c, a=a, count=len(x), scores=[float(s) for s in scores])


def acquire_global(x, y, b=B, meter=None):
    meter = meter or Meter()
    with meter.time('source_statistics'):
        c, a = full_sums(x, y)
    basis = solve_global(c, a, len(x), x.shape[1]//b, meter)
    meter.counts['fits'] += 1
    meter.counts['source_pairs'] += len(x)
    meter.counts['fit_row_exposures'] += len(x)
    return dict(basis=basis, c=c, a=a, count=len(x))


# ---------------------------------------------------------------- learners after acquisition

class Learner:
    """Holds only declared retained state; `deploy` returns (kind, phase, weights or basis)."""
    kind = 'local'
    defined_at_zero = True

    def __init__(self, meter=None, b=B):
        self.meter, self.b, self.count = meter or Meter(), b, 0

    def observe(self, x, y):
        self.count += 1
        self.meter.counts['target_pairs'] += 1

    def checkpoint(self):
        pass

    def state(self):
        return {}

    def state_bytes(self):
        return {name: int(np.asarray(value).nbytes) for name, value in self.state().items()}


class FrozenLocal(Learner):
    def __init__(self, source, meter=None, b=B):
        super().__init__(meter, b)
        self.phase, self.weights = source['phase'], source['weights'].copy()

    def deploy(self):
        return 'local', self.phase, self.weights

    def state(self):
        return dict(weights=self.weights, phase=np.int64(self.phase))


class PhaseReuseTarget(Learner):
    """Source phase only; target block sums from zero; the source basis is released."""
    defined_at_zero = False

    def __init__(self, source, meter=None, b=B):
        super().__init__(meter, b)
        self.phase, k = source['phase'], len(source['weights'])
        self.c, self.a, self.weights = np.zeros((k, b, b)), np.zeros((k, b, b)), None

    def observe(self, x, y):
        super().observe(x, y)
        with self.meter.time('target_update'):
            add_pair_local(self.c, self.a, x, y, self.phase, self.b)

    def checkpoint(self):
        self.weights = solve_local(self.c, self.a, self.count, self.meter)

    def deploy(self):
        return 'local', self.phase, self.weights

    def state(self):
        out = dict(c=self.c, a=self.a, count=np.int64(self.count), phase=np.int64(self.phase))
        if self.weights is not None:
            out['weights'] = self.weights
        return out


class PooledLocal(PhaseReuseTarget):
    """Source block sums at the source phase plus target pairs; m=0 deploys the source refit."""
    defined_at_zero = True

    def __init__(self, source, meter=None, b=B):
        Learner.__init__(self, meter, b)
        self.phase, self.source_count = source['phase'], source['count']
        self.c, self.a, self.weights = source['c'].copy(), source['a'].copy(), source['weights'].copy()

    def checkpoint(self):
        self.weights = solve_local(self.c, self.a, self.source_count+self.count, self.meter)

    def state(self):
        return dict(super().state(), source_count=np.int64(self.source_count))


class FrozenGlobal(Learner):
    kind = 'global'

    def __init__(self, source, meter=None, b=B):
        super().__init__(meter, b)
        self.basis = source['basis'].copy()

    def deploy(self):
        return 'global', None, self.basis

    def state(self):
        return dict(basis=self.basis)


class FreshGlobal(Learner):
    kind = 'global'
    defined_at_zero = False

    def __init__(self, d, meter=None, b=B):
        super().__init__(meter, b)
        self.c, self.a, self.basis = np.zeros((d, d)), np.zeros((d, d)), None

    def observe(self, x, y):
        super().observe(x, y)
        with self.meter.time('target_update'):
            add_pair_full(self.c, self.a, x, y)

    def checkpoint(self):
        self.basis = solve_global(self.c, self.a, self.total(), len(self.c)//self.b, self.meter)

    def total(self):
        return self.count

    def deploy(self):
        return 'global', None, self.basis

    def state(self):
        out = dict(c=self.c, a=self.a, count=np.int64(self.count))
        if self.basis is not None:
            out['basis'] = self.basis
        return out


class PooledGlobal(FreshGlobal):
    defined_at_zero = True

    def __init__(self, source, meter=None, b=B):
        Learner.__init__(self, meter, b)
        self.source_count = source['count']
        self.c, self.a, self.basis = source['c'].copy(), source['a'].copy(), source['basis'].copy()

    def total(self):
        return self.source_count+self.count

    def state(self):
        return dict(super().state(), source_count=np.int64(self.source_count))


class FreshLocalCV(Learner):
    """v2 CV and refit rerun from scratch on the target prefix; raw target pairs are workspace."""
    defined_at_zero = False

    def __init__(self, d, meter=None, b=B):
        super().__init__(meter, b)
        self.x, self.y = np.zeros((STREAM, d)), np.zeros((STREAM, d))
        self.phase = self.weights = self.scores = None

    def observe(self, x, y):
        self.x[self.count], self.y[self.count] = x, y
        super().observe(x, y)

    def checkpoint(self):
        with self.meter.time('target_cv_refit'):
            self.phase, self.weights, self.scores = fit_cv_refit(self.x[:self.count], self.y[:self.count],
                                                                 self.b, self.meter)
        self.meter.counts['cv_runs'] += 1

    def deploy(self):
        return 'local', self.phase, self.weights

    def state(self):
        out = dict(raw_target_buffer_x=self.x, raw_target_buffer_y=self.y, count=np.int64(self.count))
        if self.weights is not None:
            out.update(weights=self.weights, phase=np.int64(self.phase))
        return out


def make_learner(arm, local_source, global_source, d, meter=None, b=B):
    """Each call is an independent branch from the same source checkpoint (state is copied)."""
    return dict(frozen_local=lambda: FrozenLocal(local_source, meter, b),
                phase_reuse_target=lambda: PhaseReuseTarget(local_source, meter, b),
                pooled_local=lambda: PooledLocal(local_source, meter, b),
                frozen_global=lambda: FrozenGlobal(global_source, meter, b),
                pooled_global=lambda: PooledGlobal(global_source, meter, b),
                fresh_local_cv=lambda: FreshLocalCV(d, meter, b),
                fresh_global=lambda: FreshGlobal(d, meter, b))[arm]()


def dense_basis(model, d):
    kind, phase, w = model
    return local_basis(w, phase, d) if kind == 'local' else w


def check_projector(basis, rank):
    p = basis@basis.T
    if not (np.abs(p-p.T).max() <= PROJECTOR_TOL and np.abs(p@p-p).max() <= PROJECTOR_TOL
            and abs(np.trace(p)-rank) <= PROJECTOR_TOL and np.linalg.matrix_rank(basis) == rank):
        raise ScoringFailure('deployed projector is not a rank-k orthogonal projector')


# ---------------------------------------------------------------- evaluator only

def signal_covariance(world, signal):
    r = world['rotation']
    return r@signal@signal.T@r.T


def noise_covariances(world):
    v, r, d = world['v'], world['rotation'], world['d']
    base = dict(in_distribution=(4., .25), low_noise=(4., .01), texture_16=(16., .25))
    return {c: r@(t*v@v.T+i*np.eye(d))@r.T for c, (t, i) in base.items()}


def metrics(basis, s, noises):
    return dict(signal_loss=signal_loss(basis, s),
                projected_noise={c: projected_noise(basis, s, n) for c, n in noises.items()},
                nmse={c: nmse(basis, s, n) for c, n in noises.items()})


def error_map_nmse(basis, s, n):
    """Independent check: trace of the full error covariance (P-I)S(P-I)^T + P N P^T over tr(S)."""
    p = basis@basis.T
    e = (p-np.eye(len(p)))@s@(p-np.eye(len(p))).T+p@n@p.T
    return float(np.trace(e)/np.trace(s))


def principal_angle_cosines(a, b):
    return np.linalg.svd(a.T@b, compute_uv=False)

