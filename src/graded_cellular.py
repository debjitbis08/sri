"""Small nonlinear graded-cellular mechanism test, not a multilayer SRI model."""
from dataclasses import dataclass
import math


KAPPA = 2.0
GRADIENT_TOLERANCE = 1e-7
RECOVERY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Model:
    name: str
    cellular: bool
    beta: float
    face: float
    gamma: float
    context: float

    def __post_init__(self):
        if (self.beta not in (0, 1) or not 0 <= self.face <= 1 or
                not 0 <= self.gamma <= 5 or self.context not in (-1, 1)):
            raise ValueError('outside the declared prototype parameter domain')
        if not self.cellular and self.face:
            raise ValueError('a disabled cellular operator cannot carry a face term')


VARIANTS = (
    ('cellular_only', True, 0., 0.),
    ('relation_only', False, 1., 0.),
    ('combined', True, 1., 0.),
    ('disk_quarter', True, 1., .25),
    ('disk_half', True, 1., .5),
    ('disk_full', True, 1., 1.),
)


def mean(state):
    return sum(state[:3]) / 3


def energy(model, state):
    a, b = mean(state), state[3]
    cellular = (1.5*sum((x-a)**2 for x in state[:3]) + 4.5*model.face*a*a
                if model.cellular else 0.)
    return (cellular + model.beta*(a*a-1)**2 + model.gamma*(a-b)**2/2
            + KAPPA*(b-model.context)**2/2)


def gradient(model, state):
    a, b = mean(state), state[3]
    common = (4*model.beta*a*(a*a-1) + model.gamma*(a-b))/3
    return tuple((3*(x-a)+3*model.face*a if model.cellular else 0.) + common
                 for x in state[:3]) + (KAPPA*(b-model.context)+model.gamma*(b-a),)


def mean_flow(model, state):
    a, b = state
    da = -(4*model.beta*a*(a*a-1)+model.gamma*(a-b))/3
    if model.cellular:
        da -= 3*model.face*a
    return da, -KAPPA*(b-model.context)-model.gamma*(b-a)


def mean_energy(model, state):
    a, b = state
    return (4.5*model.face*a*a + model.beta*(a*a-1)**2
            + model.gamma*(a-b)**2/2 + KAPPA*(b-model.context)**2/2)


def rk4(flow, state, dt):
    k1 = flow(state)
    k2 = flow(tuple(x+dt*v/2 for x, v in zip(state, k1)))
    k3 = flow(tuple(x+dt*v/2 for x, v in zip(state, k2)))
    k4 = flow(tuple(x+dt*v for x, v in zip(state, k3)))
    return tuple(x+dt*(a+2*b+2*c+d)/6 for x, a, b, c, d in zip(state, k1, k2, k3, k4))


def integrate(flow, start, duration, dt, potential=None):
    if dt <= 0:
        raise ValueError('dt must be positive')
    steps = round(duration/dt)
    if steps < 0 or not math.isclose(steps*dt, duration):
        raise ValueError('duration must be a nonnegative multiple of positive dt')
    state = tuple(map(float, start))
    previous_energy = potential(state) if potential else None
    max_increase = 0.
    for _ in range(steps):
        state = rk4(flow, state, dt)
        if not all(math.isfinite(x) and abs(x) < 1e6 for x in state):
            raise ArithmeticError('trajectory left numerical safety bounds')
        if potential:
            current = potential(state)
            max_increase = max(max_increase, current-previous_energy)
            previous_energy = current
    return dict(state=state, steps=steps, flow_evaluations=4*steps,
                max_energy_increase=max_increase)


class ReleasedFlow:
    """Cache identical mean/context solves; contrasts are evolved independently."""
    def __init__(self, model, duration=128., dt=1/16):
        self.model, self.duration, self.dt = model, duration, dt
        self.cache = {}

    def run(self, start):
        a0, b0 = mean(start), start[3]
        key = (a0, b0)
        if key not in self.cache:
            self.cache[key] = integrate(lambda s: mean_flow(self.model, s), key,
                self.duration, self.dt, lambda s: mean_energy(self.model, s))
        trajectory = self.cache[key]
        a, b = trajectory['state']
        decay = math.exp(-3*self.duration) if self.model.cellular else 1.
        state = tuple(a+(x-a0)*decay for x in start[:3]) + (b,)
        residual = max(map(abs, gradient(self.model, state)))
        return dict(initial=tuple(start), final=state,
            status='converged_numerically' if residual <= GRADIENT_TOLERANCE else 'unresolved',
            gradient_residual=residual, output=state[2],
            energy_initial=energy(self.model, start), energy_final=energy(self.model, state),
            max_energy_increase=trajectory['max_energy_increase'],
            duration=self.duration, dt=self.dt)

    def work(self):
        return dict(unique_reduced_trajectories=len(self.cache),
            flow_evaluations=sum(row['flow_evaluations'] for row in self.cache.values()),
            rk4_steps=sum(row['steps'] for row in self.cache.values()))


def pulse(model, evidence, duration=16., dt=1/16):
    if evidence not in (-1, 1):
        raise ValueError('pulse evidence must be -1 or +1')

    def flow(state):
        u, b = state
        a = (evidence+2*u)/3
        du = -(4*model.beta*a*(a*a-1)+model.gamma*(a-b))/3
        if model.cellular:
            du -= (1+2*model.face)*u+(model.face-1)*evidence
        return du, -KAPPA*(b-model.context)-model.gamma*(b-a)

    result = integrate(flow, (0., model.context), duration, dt)
    u, b = result['state']
    return dict(evidence=evidence, duration=duration, dt=dt,
                state=(float(evidence), u, u, b),
                flow_evaluations=result['flow_evaluations'])


def cubic_coefficients(model):
    lam = KAPPA*model.gamma/(KAPPA+model.gamma)
    return (4*model.beta, 9*model.face+lam-4*model.beta, -lam*model.context)


def real_roots(cubic, linear, constant):
    """Bracket on monotone cubic intervals, retaining double roots explicitly."""
    if not cubic:
        return (None if not constant else ()) if not linear else (-constant/linear,)
    radius = 1+max(abs(linear/cubic), abs(constant/cubic))
    cuts = [-radius, radius]
    if linear < 0:
        critical = math.sqrt(-linear/(3*cubic))
        cuts = [-radius, -critical, critical, radius]
    evaluate = lambda x: (cubic*x*x+linear)*x+constant
    roots = [x for x in cuts if abs(evaluate(x)) <= 1e-12]
    for lower, upper in zip(cuts, cuts[1:]):
        fl, fu = evaluate(lower), evaluate(upper)
        if abs(fl) <= 1e-12 or abs(fu) <= 1e-12 or fl*fu >= 0:
            continue
        for _ in range(80):
            midpoint = (lower+upper)/2
            fm = evaluate(midpoint)
            if fl*fm <= 0:
                upper = midpoint
            else:
                lower, fl = midpoint, fm
        roots.append((lower+upper)/2)
    unique = []
    for root in sorted(roots):
        if not unique or abs(root-unique[-1]) > 1e-8:
            unique.append(root)
    return tuple(unique)


def stationary_modes(model):
    coefficients = cubic_coefficients(model)
    roots = real_roots(*coefficients)
    flat = roots is None
    if flat:
        roots = (-1., 1.)  # declared representatives, not enumeration of a continuum
    modes = []
    for a in roots:
        b = (model.gamma*a+KAPPA*model.context)/(model.gamma+KAPPA)
        # Hessian in orthonormal mean coordinate sqrt(3)*a and context b.
        h11 = (9*model.face+4*model.beta*(3*a*a-1)+model.gamma)/3
        h22, h12 = KAPPA+model.gamma, -model.gamma/math.sqrt(3)
        discriminant = math.sqrt((h11-h22)**2+4*h12*h12)
        eigenvalues = ((h11+h22-discriminant)/2, (h11+h22+discriminant)/2)
        smallest = eigenvalues[0]
        kind = ('flat' if flat else 'unstable' if smallest < -1e-9 else
                'marginal' if smallest <= 1e-9 else
                'isolated_minimum' if model.cellular else 'minimum_manifold')
        state = (a, a, a, b)
        modes.append(dict(state=state, mean=a, kind=kind,
            mean_context_eigenvalues=eigenvalues,
            contrast_eigenvalue=3. if model.cellular else 0.,
            gradient_residual=max(map(abs, gradient(model, state)))))
    return modes


def classify(model, trajectory, modes=None):
    if trajectory['status'] != 'converged_numerically':
        return dict(kind='unresolved', mean=mean(trajectory['final']))
    modes = stationary_modes(model) if modes is None else modes
    if any(mode['kind'] == 'flat' for mode in modes):
        return dict(kind='flat', mean=mean(trajectory['final']))
    a = mean(trajectory['final'])
    nearest = min(modes, key=lambda mode: abs(mode['mean']-a))
    if abs(nearest['mean']-a) > 1e-4:
        return dict(kind='unresolved_mode', mean=a)
    return dict(kind=nearest['kind'], mean=nearest['mean'])
