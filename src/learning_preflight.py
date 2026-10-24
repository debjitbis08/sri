"""Linear equivalences and spectral estimators for the learning preflight."""
import numpy as np


D, M = 32, 4
BLOCK = D // M
BUDGETS = (4, 8, 16, 32, 64, 128, 256)
NOISE = (.25, 1.)
REGIMES = ('aligned', 'shifted', 'dense')
METHODS = ('global_spectral', 'local_spectral', 'fixed_averaging')
SEED = 20261101


def task_world(index):
    rng = np.random.default_rng([SEED, 0, index])
    j = np.arange(BLOCK) + .5
    basis = np.column_stack([np.ones(BLOCK)/np.sqrt(BLOCK),
                             np.sqrt(2/BLOCK)*np.cos(np.pi*j/BLOCK),
                             np.sqrt(2/BLOCK)*np.cos(2*np.pi*j/BLOCK)])
    coefficients = rng.standard_normal((M, 3))
    u = np.zeros((D, M))
    for k in range(M):
        column = basis @ coefficients[k]
        u[k*BLOCK:(k+1)*BLOCK, k] = column/np.linalg.norm(column)
    q, r = np.linalg.qr(rng.standard_normal((D, D)))
    signs = np.where(np.diag(r) < 0, -1., 1.)
    q = q*signs[None, :]
    return u, (np.eye(D), np.roll(np.eye(D), 4, axis=0), q)


def training_stream(world, stream, size=max(BUDGETS)):
    rng = np.random.default_rng([SEED, 1, world, stream])
    return (rng.standard_normal((size, M)), rng.standard_normal((size, D)),
            rng.standard_normal((size, D)))


def cross_covariance(first, second, pairs):
    if first.shape != second.shape or pairs < 1 or pairs > len(first):
        raise ValueError('matching views and a valid prefix length are required')
    product = first[:pairs].T @ second[:pairs] / pairs
    return (product + product.T)/2


def global_spectral(matrix, rank=M):
    return np.linalg.eigh(matrix)[1][:, -rank:]


def local_spectral(matrix):
    out = np.zeros((D, M))
    for k in range(M):
        block = slice(k*BLOCK, (k+1)*BLOCK)
        out[block, k] = np.linalg.eigh(matrix[block, block])[1][:, -1]
    return out


def fixed_averaging():
    out = np.zeros((D, M))
    for k in range(M):
        out[k*BLOCK:(k+1)*BLOCK, k] = 1/np.sqrt(BLOCK)
    return out


def subspace_loss(estimate, truth):
    value = 1 - np.linalg.norm(estimate.T@truth, 'fro')**2 / truth.shape[1]
    if not -1e-10 <= value <= 1+1e-10:
        raise ValueError('bases must have orthonormal columns of matching rank')
    return float(np.clip(value, 0., 1.))


def population_local(truth):
    s = truth@truth.T
    values = [np.linalg.eigvalsh(s[k*BLOCK:(k+1)*BLOCK, k*BLOCK:(k+1)*BLOCK])
              for k in range(M)]
    floor = 1 - sum(v[-1] for v in values)/M
    basis = local_spectral(s)
    assert abs(subspace_loss(basis, truth)-floor) < 1e-10
    return float(max(0., floor)), [float(v[-1]-v[-2]) for v in values]


def covariance_error_expectation(signal_covariance, noise_variance, pairs):
    s = signal_covariance
    t = s + noise_variance*np.eye(len(s))
    return float((np.trace(t)**2 + np.sum(t*t) + np.trace(s)**2 + np.sum(s*s))/(2*pairs))


def factor_local(compact):
    """Factor orthonormal, disjoint eight-site rows into 32->8->4 restrictions."""
    if compact.shape != (M, D):
        raise ValueError('expected a 4 by 32 compact restriction')
    a, b = np.zeros((8, D)), np.zeros((M, 8))
    for k in range(M):
        outside = compact[k].copy()
        outside[k*8:(k+1)*8] = 0
        if np.linalg.norm(outside) > 1e-12 or abs(np.linalg.norm(compact[k])-1) > 1e-12:
            raise ValueError('rows must be normalized and have their declared local support')
        for part in range(2):
            i = 2*k+part
            segment = compact[k, i*4:(i+1)*4]
            length = np.linalg.norm(segment)
            b[k, i] = length
            if length == 0:
                a[i, i*4] = 1.
            else:
                a[i, i*4:(i+1)*4] = segment/length
    return a, b


def parameter_restrictions(parameters):
    ar = np.asarray(parameters[:32]).reshape(8, 4)
    br = np.asarray(parameters[32:]).reshape(4, 2)
    a, b = np.zeros((8, D)), np.zeros((4, 8))
    for k in range(8):
        a[k, k*4:(k+1)*4] = ar[k]/np.linalg.norm(ar[k])
    for k in range(4):
        b[k, k*2:(k+1)*2] = br[k]/np.linalg.norm(br[k])
    return a, b


def spatial_response(a, b, inputs, context, lam=1.3, kappa=.8, eta=.7):
    h = np.zeros((44, 44))
    h[:32, :32] = lam*np.eye(32)+kappa*a.T@a
    h[32:40, 32:40] = kappa*(np.eye(8)+b.T@b)
    h[40:, 40:] = (kappa+eta)*np.eye(4)
    h[:32, 32:40] = -kappa*a.T
    h[32:40, :32] = -kappa*a
    h[32:40, 40:] = -kappa*b.T
    h[40:, 32:40] = -kappa*b
    forcing = np.vstack([lam*inputs.T, np.zeros((8, len(inputs))), eta*context.T])
    return np.linalg.solve(h, forcing).T


def explicit_response(a, b, inputs, context, lam=1.3, kappa=.8, eta=.7):
    c = b@a
    q = kappa*eta/(kappa+2*eta)
    gain = lam/(lam+q)
    ch = inputs@c.T
    cx = gain*ch+(1-gain)*context
    x = inputs+(cx-ch)@c
    y = (kappa/(kappa+2*eta))*cx + (2*eta/(kappa+2*eta))*context
    z = inputs@a.T - ch@b + .5*(cx+y)@b
    return np.column_stack([x, z, y])


def run_gates():
    rng = np.random.default_rng([SEED, 2])
    pars = rng.standard_normal(40)
    a, b = parameter_restrictions(pars)
    aa, bb = factor_local(b@a)
    factor_error = float(np.max(np.abs(bb@aa-b@a)))
    edge = np.zeros((4, 32))
    for k in range(4):
        edge[k, 8*k] = 1
    ae, be = factor_local(edge)
    factor_error = max(factor_error, float(np.max(np.abs(be@ae-edge))))
    first, second = rng.standard_normal((2, 5, D))
    context = rng.standard_normal((5, M))
    response_error = 0.
    for lam, kappa, eta in ((1., 1., 0.), (1., 1., 1.), (1.3, .8, .7)):
        response_error = max(response_error, float(np.max(np.abs(
            spatial_response(a, b, first, context, lam, kappa, eta)
            - explicit_response(a, b, first, context, lam, kappa, eta)))))
    def loss(parameters, response):
        ra, rb = parameter_restrictions(parameters)
        one = response(ra, rb, first, context)
        two = response(ra, rb, second, context)
        # Same differentiable loss, including core and surface outputs.
        return np.mean((one[:, -4:]-two[:, -4:])**2)+.1*np.mean((one[:, :32]-two[:, :32])**2)
    gradients = []
    for response in (spatial_response, explicit_response):
        grad = np.zeros(len(pars))
        for i in range(len(pars)):
            plus, minus = pars.copy(), pars.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            grad[i] = (loss(plus, response)-loss(minus, response))/2e-5
        gradients.append(grad)
    gradient_error = float(np.max(np.abs(gradients[0]-gradients[1])))
    u, rotations = task_world(0)
    assert np.max(np.abs(u.T@u-np.eye(M))) < 1e-12
    for rotation in rotations:
        assert np.max(np.abs(rotation.T@rotation-np.eye(D))) < 1e-12
    # Local known-support structure is identifiable from a single noiseless pair.
    single = (u@np.array([1., 2., 3., 4.]))[None, :]
    noiseless_local = subspace_loss(local_spectral(cross_covariance(single, single, 1)), u)
    spanning = u.T.copy()
    noiseless_global = subspace_loss(global_spectral(cross_covariance(spanning, spanning, 4)), u)
    x, y = rng.standard_normal((2, 12, D))
    before = cross_covariance(x, y, 4)
    x[4:] = np.nan
    y[4:] = np.nan
    prefix_error = float(np.max(np.abs(before-cross_covariance(x, y, 4))))
    assert factor_error < 1e-9 and response_error < 1e-9
    assert gradient_error < 1e-6
    assert noiseless_local < 1e-10 and noiseless_global < 1e-10
    assert prefix_error == 0.
    return dict(factorization_max_error=factor_error, full_state_map_max_error=response_error,
                same_parameter_loss_gradient_max_error=gradient_error,
                noiseless_local_one_pair_loss=noiseless_local,
                noiseless_global_four_pair_loss=noiseless_global,
                future_rows_prefix_effect=prefix_error)
