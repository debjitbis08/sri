"""Four bounded modeling controls; see docs/comparison-v1-contract.md.

Responses and certificates use exact rationals or finite fields. Floating point
is confined to diagnostic eigendecomposition, not response correctness.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
import math


@dataclass
class Work:
    arithmetic: int = 0
    pivots: int = 0
    candidates: int = 0


def matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def transpose(matrix):
    return tuple(zip(*matrix))


def matmul(left, right):
    columns = transpose(right)
    return tuple(tuple(sum(a * b for a, b in zip(row, col)) for col in columns)
                 for row in left)


def norm(vector):
    return max(map(abs, vector), default=F(0))


def difference(left, right):
    return tuple(a - b for a, b in zip(left, right))


def row_reduce(matrix, rhs, modulus=None, work=None):
    """RREF with fixed column order; inconsistent systems return None."""
    work = work if work is not None else Work()
    n = len(matrix[0]) if matrix else 0
    if len(rhs) != len(matrix) or any(len(row) != n for row in matrix):
        raise ValueError('inconsistent matrix dimensions')
    cast = F if modulus is None else lambda value: int(value) % modulus
    rows = [[cast(v) for v in row] + [cast(b)] for row, b in zip(matrix, rhs)]
    pivots = []
    for col in range(n):
        pivot = next((r for r in range(len(pivots), len(rows)) if rows[r][col]), None)
        if pivot is None:
            continue
        index = len(pivots)
        rows[index], rows[pivot] = rows[pivot], rows[index]
        scale = (1 / rows[index][col] if modulus is None
                 else pow(rows[index][col], -1, modulus))
        rows[index] = [cast(v * scale) for v in rows[index]]
        work.arithmetic += n + 1
        for r in range(len(rows)):
            if r == index or not rows[r][col]:
                continue
            factor = rows[r][col]
            rows[r] = [cast(a - factor * b) for a, b in zip(rows[r], rows[index])]
            work.arithmetic += 2 * (n + 1)
        pivots.append(col)
        work.pivots += 1
    if any(not any(row[:n]) and row[n] for row in rows):
        return None
    return rows, tuple(pivots)


def solve(matrix, rhs, previous=None, modulus=None, work=None):
    reduced = row_reduce(matrix, rhs, modulus, work)
    if reduced is None:
        return None
    rows, pivots = reduced
    n = len(matrix[0])
    values = list(previous if previous is not None else (0,) * n)
    if len(values) != n:
        raise ValueError('previous configuration has wrong size')
    cast = F if modulus is None else lambda value: int(value) % modulus
    values = list(map(cast, values))
    free = [i for i in range(n) if i not in pivots]
    for row, pivot in zip(rows, pivots):
        values[pivot] = cast(row[-1] - sum(row[j] * values[j] for j in free))
    return tuple(values)


def rank(matrix, modulus=None):
    return len(row_reduce(matrix, (0,) * len(matrix), modulus)[1])


def step(matrix, forcing, state, rate):
    residual = difference(matvec(matrix, state), forcing)
    return tuple(v - rate * r for v, r in zip(state, residual))


def iterate(matrix, forcing, state, rate, count):
    for _ in range(count):
        state = step(matrix, forcing, state, rate)
    return state


def contraction(matrix, rate):
    return max(sum(abs((1 if i == j else 0) - rate * value)
                   for j, value in enumerate(row)) for i, row in enumerate(matrix))


def dominance(matrix):
    return min(row[i] - sum(abs(value) for j, value in enumerate(row) if j != i)
               for i, row in enumerate(matrix))


def residual_bound(matrix, forcing, approximate):
    delta = dominance(matrix)
    if delta <= 0:
        raise ValueError('this bound requires positive diagonal dominance')
    return norm(difference(matvec(matrix, approximate), forcing)) / delta


def spectral_matrix(g, core=True):
    g = F(g)
    if not 0 <= g <= 2:
        raise ValueError('coupling outside declared [0,2] interval')
    q = [[F(int(i == j)) for j in range(4)] for i in range(4)]
    if core:
        for i, j, weight in ((0, 1, F(1)), (1, 2, F(1)), (1, 3, g)):
            q[i][i] += weight
            q[j][j] += weight
            q[i][j] -= weight
            q[j][i] -= weight
    return tuple(map(tuple, q))


def eigendiagnostic(matrix, tolerance=1e-13, max_rotations=128):
    """Small symmetric Jacobi diagonalization, with explicit completion status."""
    n = len(matrix)
    if any(len(row) != n for row in matrix) or any(
            matrix[i][j] != matrix[j][i] for i in range(n) for j in range(n)):
        raise ValueError('expected a symmetric square matrix')
    a = [list(map(float, row)) for row in matrix]
    vectors = [[float(i == j) for j in range(n)] for i in range(n)]
    rotations = 0
    while n > 1:
        p, q = max(((i, j) for i in range(n) for j in range(i + 1, n)),
                   key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) <= tolerance or rotations >= max_rotations:
            break
        angle = .5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        for k in range(n):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p], vectors[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
        a[p][p] = c*c*app - 2*s*c*apq + s*s*aqq
        a[q][q] = s*s*app + 2*s*c*apq + c*c*aqq
        a[p][q] = a[q][p] = 0.
        rotations += 1
    offdiag = max((abs(a[i][j]) for i in range(n) for j in range(i + 1, n)), default=0.)
    order = sorted(range(n), key=lambda j: a[j][j])
    values = tuple(a[j][j] for j in order)
    modes = tuple(tuple(vectors[i][j] for i in range(n)) for j in order)
    residual = max((norm(difference(matvec(matrix, v), tuple(lam*x for x in v)))
                    for lam, v in zip(values, modes)), default=0.)
    return dict(status='complete' if offdiag <= tolerance else 'truncated',
                eigenvalues=values, eigenvectors=modes, max_residual=residual,
                rotations=rotations, tolerance=tolerance)


def algebra_system(evidence, context, core=True):
    if context not in (0, 1) or evidence not in (None, 1, 2):
        raise ValueError('outside declared evidence/context domain')
    rows, rhs = [], []
    if core:
        rows.extend(((-1, 1, 1, 0), (0, 1, 2, 1)))
        rhs.extend((0, 0))
    rows.append((0, 0, 0, 1))
    rhs.append(context)
    if evidence is not None:
        rows.append((1, 0, 0, 0))
        rhs.append(evidence)
    return tuple(rows), tuple(rhs)


def algebra_response(evidence, context, previous=(0, 0, 0, 0), core=True, work=None):
    rows, rhs = algebra_system(evidence, context, core)
    return solve(rows, rhs, previous, modulus=3, work=work)


def relation_minima(evidence, context, g, core=True, scale=F(1), work=None):
    if evidence not in (None, 0, 1) or context not in (0, 1) or g < 0 or scale <= 0:
        raise ValueError('outside declared relational domain')
    work = work if work is not None else Work()
    best, states = None, []
    for state in product((0, 1), repeat=4):
        work.candidates += 1
        i, k, o, s = state
        if s != context or (evidence is not None and i != evidence):
            continue
        if core and (i ^ k ^ o ^ s):
            continue
        value = scale * (k + F(g) * (o ^ i))
        if best is None or value < best:
            best, states = value, [state]
        elif value == best:
            states.append(state)
    return tuple(states), best


def relation_response(evidence, context, g, previous=(0, 0, 0, 0), core=True, scale=F(1)):
    states, _ = relation_minima(evidence, context, g, core, scale)
    return previous if previous in states else states[0]


D1 = ((-1, 0, 1), (1, -1, 0), (0, 1, -1))


def cell_boundary(attached):
    return ((1,), (1,), (1,)) if attached else ((0,), (0,), (0,))


def cell_matrix(g, attached):
    if not 0 <= g <= 2:
        raise ValueError('coupling outside declared [0,2] interval')
    lower = matmul(transpose(D1), D1)
    boundary = cell_boundary(attached)
    upper = matmul(boundary, transpose(boundary))
    return tuple(tuple(F(a) + F(g)*b for a, b in zip(row, up))
                 for row, up in zip(lower, upper))


def cell_driven(g, attached, evidence, work=None):
    matrix = cell_matrix(g, attached)
    q = tuple(tuple(row[1:]) for row in matrix[1:])
    forcing = tuple(-row[0]*evidence for row in matrix[1:])
    return (F(evidence),) + solve(q, forcing, work=work)


def cell_withdrawal_limit(g, attached, state):
    if attached and g > 0:
        return (F(0),) * 3
    return (sum(state) / 3,) * 3


def replacements(state, alphabet):
    for i, old in enumerate(state):
        for value in alphabet:
            if value != old:
                new = list(state)
                new[i] = value
                yield i, value, tuple(new)


def additive_disturbances(state):
    for i in range(len(state)):
        for delta in (-1, 1):
            new = list(state)
            new[i] += delta
            yield i, delta, tuple(new)
