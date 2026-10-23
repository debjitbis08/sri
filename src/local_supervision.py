"""Local learning supervision v1: observation keys, projected clamp readout, coverage and timelines.

See docs/local-learning-supervision-v1-contract.md. Learners, battery and dynamics are imported
unchanged from src/mechanism_learning.py. This module adds the IB condition's projected clamp keys
(B6x, B7x), the label-free coverage subset and O+ order, supervision-cost accounting, and the
per-item acquisition/forgetting timeline summaries.
"""
from collections import Counter
import hashlib

from src.mechanism_learning import Simulator, observation_keys as v1_observation_keys
from src.shared_input_mediation import A, B, Query, explore, start_state


CONDITIONS = ('O', 'I', 'IB', 'O+')
MODES = (-1, 0, 1)
SLOTS = {'B1': 9, 'B6': 27, 'B7': 27, 'B6x': 3, 'B7x': 3}     # nominal output alphabet per key type
COVERAGE_PER_COMBINATION = 3
MIXED_COMBINATIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def observation_keys(x, cond):
    """Typed feedback keys per condition; O/O+/I keep v1's order, IB is B1, B6x ascending, B7x ascending."""
    if cond in ('O', 'O+'):
        return v1_observation_keys(x, 'O')
    if cond == 'I':
        return v1_observation_keys(x, 'I')
    if cond == 'IB':
        return ([('B1',)] + [('B6x', v) for v in sorted(x.allowed(A))]
                + [('B7x', v) for v in sorted(x.allowed(B))])
    raise ValueError(f'unknown condition {cond}')


class ProjectedSimulator:
    """IB simulation service: B1 as v1, and clamp results projected to the partner satellite's mode.

    Internal states are computed privately; only deduplicated sets of partner modes cross the
    learner-facing boundary. Pure memoization; no teacher labels, battery or held-out results.
    """
    def __init__(self):
        self.memo = {}
        self.requests = 0

    def predict(self, w, x, key):
        if key not in observation_keys(x, 'IB'):
            raise PermissionError(f'observation key {key} not supplied in condition IB')
        self.requests += 1
        mk = (w, x, key)
        if mk not in self.memo:
            if key[0] == 'B1':
                self.memo[mk] = frozenset((t[A], t[B]) for t in explore(x, w).terminals)
            else:
                c, other = (A, B) if key[0] == 'B6x' else (B, A)
                terms = explore(x, w, start=start_state(x, ((c, key[1]),)), frozen=(c,)).terminals
                self.memo[mk] = frozenset(t[other] for t in terms)
        return self.memo[mk]


def make_simulator(cond):
    if cond == 'IB':
        return ProjectedSimulator()
    return Simulator('I' if cond == 'I' else 'O')


def project(triples):
    """B6/B7 triple set (i1, i2, partner) to the deduplicated set of partner modes."""
    return frozenset(t[2] for t in triples)


def feedback_cost(feedback):
    """Encoding-space counts for one feedback map (contract section 6.1)."""
    c = Counter(experiences=1)
    for key, f in feedback.items():
        c['entries'] += 1
        c['supervised_slots'] += SLOTS[key[0]]
        c['positive_tuples'] += len(f)
        if key[0] == 'B1':
            c['canonical_requests'] += 1
        else:
            c['clamp_runs'] += 1
    return c


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def coverage_subset(split_seed, mixed_ids, queries):
    """3 mixed-evidence queries per (e1, e2) combination, smallest label-free digest, tie by query ID."""
    chosen = []
    for e in MIXED_COMBINATIONS:
        pool = [qi for qi in mixed_ids if (queries[qi].e1, queries[qi].e2) == e]
        ranked = sorted(pool, key=lambda qi: (_digest(
            f'local-learning-supervision-v1:coverage:{split_seed}:'
            f'{queries[qi].e1},{queries[qi].e2},{queries[qi].da},{queries[qi].db}'), qi))
        chosen += ranked[:COVERAGE_PER_COMBINATION]
    return sorted(chosen)


def oplus_order(split_seed, order_seed, axis_order, coverage_ids):
    """O+ stream: the frozen axis order, then coverage queries by ascending digest, tie by query ID."""
    tail = sorted(coverage_ids, key=lambda qi: (_digest(
        f'local-learning-supervision-v1:order:{split_seed}:{order_seed}:{qi}'), qi))
    return list(axis_order) + tail


def timeline(before, after):
    """Summary of one item's post-presentation correctness timeline (contract section 6.2).

    `before` is correctness immediately before the item's own update; `after[i]` is correctness
    after update t_s + i. Categories are mutually exclusive.
    """
    if not after:
        raise ValueError('an item has at least its own post-update entry')
    first = next((i for i, v in enumerate(after) if v), None)
    losses = sum(after[i - 1] and not after[i] for i in range(1, len(after)))
    reacq = 0
    seen_true = after[0]
    for i in range(1, len(after)):
        if after[i] and not after[i - 1] and seen_true:
            reacq += 1
        seen_true = seen_true or after[i]
    if first is None:
        category = 'never_acquired'
    elif not after[-1]:
        category = 'forgotten'
    elif losses:
        category = 'recovered'
    else:
        category = 'held'
    return dict(correct_before=bool(before), acquired_at_presentation=bool(after[0]),
                broken_at_presentation=bool(before and not after[0]),
                ever_acquired=first is not None, latency=first, loss_events=losses, reacquisitions=reacq,
                ever_lost=losses > 0, final=bool(after[-1]), category=category)


def pool_timelines(summaries):
    """Pooled counts and rates with explicit denominators; zero denominators are null."""
    c = Counter()
    lat = []
    for s in summaries:
        c['presented'] += 1
        c['correct_before'] += s['correct_before']
        c['acquired_at_presentation'] += s['acquired_at_presentation']
        c['newly_acquired_at_presentation'] += s['acquired_at_presentation'] and not s['correct_before']
        c['broken_at_presentation'] += s['broken_at_presentation']
        c['ever_acquired'] += s['ever_acquired']
        c['ever_lost'] += s['ever_lost']
        c['loss_events'] += s['loss_events']
        c['reacquisitions'] += s['reacquisitions']
        c['final_correct'] += s['final']
        c[s['category']] += 1
        if s['latency'] is not None:
            lat.append(s['latency'])
    for k in ('never_acquired', 'held', 'recovered', 'forgotten'):
        c[k] += 0
    out = dict(c)
    out['latency'] = dict(acquired=len(lat), censored_never_acquired=c['never_acquired'],
                          histogram={str(k): v for k, v in sorted(Counter(lat).items())},
                          mean=round(sum(lat) / len(lat), 6) if lat else None)
    out['acquisition_rate'] = rate(c['ever_acquired'], c['presented'])
    out['forgetting_rate'] = rate(c['forgotten'], c['ever_acquired'])
    out['ever_lost_rate'] = rate(c['ever_lost'], c['ever_acquired'])
    out['final_retention'] = rate(c['final_correct'], c['presented'])
    return out


def axis_phase(at_15, during, at_end):
    """O+ axis items over the coverage phase: baseline at step 15, path over steps 16-27, state at 27.

    `at_15[i]` and `at_end[i]` are Booleans; `during[i]` is the list of correctness values after
    steps 16..27 for axis item i.
    """
    c = Counter()
    for b, path, e in zip(at_15, during, at_end):
        c['axis_items'] += 1
        c[f'{"correct" if b else "wrong"}_at_15_{"correct" if e else "wrong"}_at_27'] += 1
        if b:
            c['correct_at_15'] += 1
            c['correct_at_15_ever_lost'] += not all(path)
        else:
            c['wrong_at_15'] += 1
            c['wrong_at_15_acquired_during_coverage'] += any(path)
    for k in ('correct_at_15', 'correct_at_15_ever_lost', 'wrong_at_15', 'wrong_at_15_acquired_during_coverage',
              'correct_at_15_correct_at_27', 'correct_at_15_wrong_at_27', 'wrong_at_15_correct_at_27',
              'wrong_at_15_wrong_at_27'):
        c[k] += 0
    return dict(c)


def rate(a, b):
    return None if not b else round(a / b, 6)


__all__ = ['CONDITIONS', 'MIXED_COMBINATIONS', 'ProjectedSimulator', 'Query', 'SLOTS', 'axis_phase',
           'coverage_subset', 'feedback_cost', 'make_simulator', 'observation_keys', 'oplus_order',
           'pool_timelines', 'project', 'rate', 'timeline']
