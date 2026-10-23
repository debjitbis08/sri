"""Selective satellite mediation v1: frozen manifest, matched learners and controls, reports.

Run with `.venv/bin/python -m experiments.selective_mediation`. The split manifest is written
once and verified on later runs; scientific output is deterministic and volatile provenance
(timings, git revision, memory measurement) goes to a separate runtime file.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import unittest

from src.selective_mediation import (
    CHECKPOINTS, CLASSES, CONTEXT_NAMES, DOMAINS, GATES, N_HYPOTHESES, NO_MEDIATION, PERMISSIVE,
    PRIMITIVES, QUERIES, SCHEMA_VERSION, SPLIT_SEEDS, TEACHERS, CoreFreeLearner, ExactQueryMemory,
    VersionSpaceLearner, conflict_class, gate_open, hypothesis_gates, make_split, realize, verify_path,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reports/selective-satellite-mediation-v1-splits.json'
REPORT = ROOT / 'reports/selective-satellite-mediation-v1.json'
RUNTIME = ROOT / 'reports/selective-satellite-mediation-v1-runtime.json'
COMMAND = '.venv/bin/python -m experiments.selective_mediation'
SOURCES = ('src/selective_mediation.py', 'experiments/selective_mediation.py',
           'tests/test_selective_mediation.py', 'docs/selective-satellite-mediation-v1-contract.md')
LEARNERS = ('core', 'core_free', 'positive_only', 'exact_memory', 'no_mediation', 'permissive',
            'teacher_oracle')
VERSION_SPACE = ('core', 'core_free', 'positive_only')
EXECUTORS = ('core', 'positive_only', 'no_mediation', 'permissive', 'teacher_oracle')
PARTITIONS = ('seen_training', 'within_context', 'unseen_context')
HELD_OUT = PARTITIONS[1:]
STRATA = ('all',) + CLASSES
COUNTS = ('n', 'exact', 'tp', 'fp', 'fn', 'teacher_empty', 'false_feasible', 'teacher_nonempty',
          'missed_feasible', 'truth_subset_pred', 'resolvable', 'exact_repairs', 'repair_path_length_sum',
          'repair_changed_A', 'repair_changed_B', 'repair_changed_both', 'response_domain_violations',
          'path_domain_violations', 'gate_invalid_steps', 'step_violations', 'evidence_mutations',
          'teacher_impermissible_steps', 'changed_vs_checkpoint0', 'right_to_wrong_vs_checkpoint0',
          'wrong_to_right_vs_checkpoint0', 'survivor_agreement', 'uncertain',
          'certain_empty_possible_nonempty', 'certain_size_sum', 'possible_size_sum',
          'certain_possible_containment_violations', 'wrong_with_survivor_agreement')
PATH_ONLY = {'repair_path_length_sum', 'path_domain_violations', 'gate_invalid_steps', 'step_violations',
             'teacher_impermissible_steps'}
VS_ONLY = {'survivor_agreement', 'uncertain', 'certain_empty_possible_nonempty', 'certain_size_sum',
           'possible_size_sum', 'certain_possible_containment_violations', 'wrong_with_survivor_agreement'}
HELD_ONLY = {'changed_vs_checkpoint0', 'right_to_wrong_vs_checkpoint0', 'wrong_to_right_vs_checkpoint0'}
ROW_COLUMNS = ('learner', 'teacher', 'split', 'order', 'checkpoint', 'partition', 'class') + COUNTS


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def manifest():
    body = dict(schema=SCHEMA_VERSION, query_order='(context, domain_A, domain_B, a0, b0)',
                queries=[[q.context, q.domain_a, q.domain_b, q.a0, q.b0] for q in QUERIES],
                checkpoints=list(CHECKPOINTS), splits=[make_split(s) for s in SPLIT_SEEDS])
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest():
    """Write the manifest once; afterwards refuse to proceed if regeneration differs."""
    m = manifest()
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != m:
            sys.exit('split manifest differs from the frozen file; refusing to run')
    else:
        MANIFEST.write_text(json.dumps(m, indent=1) + '\n')
    return m


# ---------------------------------------------------------------- evaluator (holds the teacher)

def mask(s):
    return sum(1 << t for t in s)


def unmask(m):
    return frozenset(t for t in range(4) if m >> t & 1)


# PART[qi][r]: bitset of hypotheses whose response to query qi has 4-bit mask r. Evaluation cache only.
PART = []
for _q in QUERIES:
    _row = [0] * 16
    for _h in range(N_HYPOTHESES):
        _row[mask(realize(_q, _h).response)] |= 1 << _h
    PART.append(_row)
EVAL_CACHE_INFERENCES = len(QUERIES) * N_HYPOTHESES


def certain_possible(qi, bits):
    present = [r for r in range(16) if PART[qi][r] & bits]
    certain, possible = 15, 0
    for r in present:
        certain &= r
        possible |= r
    return unmask(certain), unmask(possible), len(present) == 1


def predictions(name, learner, teacher, indices):
    """{query index: (response set, Inference or None)} for one learner at one checkpoint."""
    out = {}
    for qi in indices:
        q = QUERIES[qi]
        if name in ('core', 'positive_only'):
            r = learner.predict(q)
            out[qi] = (r.response, r)
        elif name in ('core_free', 'exact_memory'):
            out[qi] = (learner.predict(q), None)
        else:
            h = dict(no_mediation=NO_MEDIATION, permissive=PERMISSIVE, teacher_oracle=teacher)[name]
            r = realize(q, h)
            out[qi] = (r.response, r)
    return out


def score(name, preds, truth, cls, teacher, base, bits, partition, witness_sink):
    """Per-stratum count rows for one partition."""
    rows = {s: dict.fromkeys(COUNTS, 0) for s in STRATA}
    for qi, (pred, inf) in preds.items():
        q, t = QUERIES[qi], truth[qi]
        c = {k: 0 for k in COUNTS}
        c['n'] = 1
        c['exact'] = pred == t
        c['tp'], c['fp'], c['fn'] = len(pred & t), len(pred - t), len(t - pred)
        c['teacher_empty'] = not t
        c['false_feasible'] = not t and bool(pred)
        c['teacher_nonempty'] = bool(t)
        c['missed_feasible'] = bool(t) and not pred
        c['truth_subset_pred'] = t <= pred
        c['response_domain_violations'] = not pred <= DOMAINS[q.domain_a] & DOMAINS[q.domain_b]
        if cls[qi] == 'resolvable':
            c['resolvable'] = 1
            ok = pred == t
            if inf is not None:
                v = verify_path(q, inf)
                ok = ok and not any(v.values())
            if ok:
                c['exact_repairs'] = 1
                rep = min(pred)
                a, b = rep != q.a0, rep != q.b0
                c['repair_changed_A'], c['repair_changed_B'], c['repair_changed_both'] = a and not b, b and not a, a and b
                if inf is not None:
                    c['repair_path_length_sum'] = len(inf.path)
        if inf is not None:
            v = verify_path(q, inf)
            c['path_domain_violations'], c['gate_invalid_steps'] = v['domain'], v['gate']
            c['step_violations'], c['evidence_mutations'] = v['step'], v['evidence']
            tg = hypothesis_gates(teacher)
            names = [p[0] for p in PRIMITIVES]
            c['teacher_impermissible_steps'] = sum(not gate_open(tg[names.index(s.primitive)], q.c)
                                                   for s in inf.path)
        if base is not None:
            b = base[qi]
            c['changed_vs_checkpoint0'] = pred != b
            c['right_to_wrong_vs_checkpoint0'] = b == t and pred != t
            c['wrong_to_right_vs_checkpoint0'] = b != t and pred == t
        if bits is not None:
            certain, possible, agree = certain_possible(qi, bits)
            c['survivor_agreement'] = agree
            c['uncertain'] = not agree
            c['certain_empty_possible_nonempty'] = not certain and bool(possible)
            c['certain_size_sum'], c['possible_size_sum'] = len(certain), len(possible)
            c['certain_possible_containment_violations'] = not (certain <= t <= possible)
            c['wrong_with_survivor_agreement'] = agree and pred != t
        for s in ('all', cls[qi]):
            for k in COUNTS:
                rows[s][k] += int(c[k])
        witness_sink(name, partition, qi, pred, inf, c)
    return rows


def null_fields(name, partition, row):
    out = dict(row)
    for k in COUNTS:
        if (k in PATH_ONLY and name not in EXECUTORS) or (k in VS_ONLY and name not in VERSION_SPACE) \
                or (k in HELD_ONLY and partition not in HELD_OUT):
            out[k] = None
    return out


def run_grid(man):
    rows, runs, checks = [], [], defaultdict(int)
    witnesses = defaultdict(list)
    final_bits = defaultdict(set)
    work = defaultdict(lambda: defaultdict(int))
    for tname, teacher in TEACHERS.items():
        truth = [realize(q, teacher).response for q in QUERIES]
        work['teacher_oracle'][f'{tname}_feedback_edge_checks'] = sum(realize(q, teacher).edge_checks for q in QUERIES)
        cls = [conflict_class(q, truth[i]) for i, q in enumerate(QUERIES)]
        for split in man['splits']:
            parts = dict(within_context=split['within_context'], unseen_context=split['unseen_context'])
            for o, order in enumerate(split['orders']):
                learners = dict(core=VersionSpaceLearner(), core_free=CoreFreeLearner(),
                                positive_only=VersionSpaceLearner(positive_only=True),
                                exact_memory=ExactQueryMemory())
                run_info = {n: dict(learner=n, teacher=tname, split=split['seed'], order=o, checkpoints=[])
                            for n in LEARNERS}
                base = {}
                trained = 0

                def sink(name, partition, qi, pred, inf, c, k=None):
                    ctx = dict(teacher=tname, split=split['seed'], order=o, query_index=qi,
                               query=QUERIES[qi].to_json(), teacher_response=sorted(truth[qi]),
                               predicted=sorted(pred), checkpoint=sink.k, partition=partition)
                    if inf is not None:
                        ctx['certificate'] = dict(hypothesis=inf.hypothesis, path=[
                            dict(primitive=s.primitive, gate=s.gate, source=s.source, target=s.target)
                            for s in inf.path])
                    if sink.k == 48 and partition in HELD_OUT:
                        if name == 'core' and c['exact_repairs'] and c['repair_changed_both']:
                            witnesses[f'{tname}/core_repair_changing_both'].append(ctx)
                        if name == 'core' and c['exact_repairs'] and c['teacher_impermissible_steps']:
                            witnesses[f'{tname}/core_exact_set_via_teacher_closed_gate'].append(ctx)
                        if name == 'core' and not c['exact']:
                            witnesses[f'{tname}/core_wrong_{cls[qi]}'].append(ctx)
                        if name == 'positive_only' and c['false_feasible']:
                            witnesses[f'{tname}/positive_only_false_feasible'].append(ctx)
                        if name == 'permissive' and c['false_feasible'] and split['seed'] == o == 0:
                            witnesses[f'{tname}/permissive_false_feasible'].append(ctx)
                    for k in ('path_domain_violations', 'gate_invalid_steps', 'step_violations',
                              'evidence_mutations', 'response_domain_violations',
                              'certain_possible_containment_violations', 'wrong_with_survivor_agreement'):
                        if c[k]:
                            witnesses[f'violation/{k}'].append(dict(ctx, learner=name))
                for k in CHECKPOINTS:
                    while trained < k:
                        qi = order[trained]
                        q, fb = QUERIES[qi], truth[order[trained]]
                        status = {n: l.update(q, fb) for n, l in learners.items()}
                        trained += 1
                        checks['updates'] += 1
                        checks['inconsistent_feedback'] += sum(s == 'inconsistent_feedback' for s in status.values())
                        for n in VERSION_SPACE:
                            checks[f'{n}_teacher_eliminated'] += not learners[n].bits >> teacher & 1
                        checks['core_free_V_mismatch'] += learners['core'].bits != learners['core_free'].bits
                    sink.k = k
                    seen = order[:k]
                    for name in LEARNERS:
                        learner = learners.get(name)
                        bits = learner.bits if name in VERSION_SPACE else None
                        all_idx = seen + parts['within_context'] + parts['unseen_context']
                        preds = predictions(name, learner, teacher, all_idx)
                        if k == 0:
                            base[name] = {qi: p for qi, (p, _) in preds.items()}
                        pred_sets = {qi: p for qi, (p, _) in preds.items()}
                        if name == 'core_free':
                            core_sets = run_info['core']['_preds']
                            checks['core_free_prediction_mismatch'] += sum(core_sets[qi] != pred_sets[qi] for qi in all_idx)
                        run_info[name]['_preds'] = pred_sets
                        if bits is not None:
                            work['evaluator']['survivor_responses_consulted'] += bin(bits).count('1') * len(all_idx)
                        for partition in PARTITIONS:
                            idx = seen if partition == 'seen_training' else parts[partition]
                            sub = {qi: preds[qi] for qi in idx}
                            per = score(name, sub, truth, cls, teacher,
                                        base[name] if partition in HELD_OUT else None, bits, partition, sink)
                            for stratum, counts in per.items():
                                counts = null_fields(name, partition, counts)
                                rows.append([name, tname, split['seed'], o, k, partition, stratum]
                                            + [counts[c] for c in COUNTS])
                        if name in ('core', 'core_free'):
                            for qi in seen:          # every survivor retains every observed set
                                certain, possible, agree = certain_possible(qi, learner.bits)
                                checks['observed_retention_checks'] += 1
                                checks['observed_retention_failures'] += not (agree and certain == truth[qi])
                        if name == 'positive_only':
                            for qi in seen:
                                checks['positive_retention_checks'] += bool(truth[qi])
                                checks['positive_retention_failures'] += bool(truth[qi]) and not truth[qi] <= pred_sets[qi]
                        info = dict(checkpoint=k)
                        if bits is not None:
                            info.update(V_size=bin(bits).count('1'), teacher_in_V=bool(bits >> teacher & 1),
                                        selected=learner.selected,
                                        selected_gates=[GATES[g] for g in hypothesis_gates(learner.selected)])
                            if name != 'core_free':
                                info['membership'] = learner.serialize()['membership']
                        if name == 'exact_memory':
                            info['stored_entries'] = len(learner.table)
                        run_info[name]['checkpoints'].append(info)
                for n in ('core', 'positive_only'):
                    final_bits[(n, tname, split['seed'])].add(learners[n].bits)
                    for key in ('updates', 'hypothesis_evaluations', 'update_edge_checks', 'inferences',
                                'inference_edge_checks'):
                        work[n][key] += getattr(learners[n], key)
                for n in LEARNERS:
                    run_info[n].pop('_preds', None)
                    runs.append(run_info[n])
    for (n, _, _), finals in final_bits.items():
        checks[f'{n}_order_dependent_final_V'] += len(finals) != 1
    return rows, runs, dict(checks), witnesses, {k: dict(v) for k, v in work.items()}


# ---------------------------------------------------------------- aggregation and targets

def ratio(a, b):
    return None if not b else round(a / b, 6)


def derived(c):
    return dict(exact_set_accuracy=ratio(c['exact'], c['n']),
                micro_precision=ratio(c['tp'], c['tp'] + c['fp']),
                micro_recall=ratio(c['tp'], c['tp'] + c['fn']),
                false_feasible_rate=ratio(c['false_feasible'], c['teacher_empty']),
                missed_feasible_rate=ratio(c['missed_feasible'], c['teacher_nonempty']),
                exact_repair_rate=ratio(c['exact_repairs'], c['resolvable']),
                mean_repair_path_length=None if c['repair_path_length_sum'] is None
                else ratio(c['repair_path_length_sum'], c['exact_repairs']),
                survivor_agreement_fraction=None if c['survivor_agreement'] is None
                else ratio(c['survivor_agreement'], c['n']))


def aggregate(rows):
    pooled, per_run = {}, defaultdict(list)
    for r in rows:
        d = dict(zip(ROW_COLUMNS, r))
        key = (d['learner'], d['teacher'], d['checkpoint'], d['partition'], d['class'])
        if key not in pooled:
            pooled[key] = {k: (None if d[k] is None else 0) for k in COUNTS}
        for k in COUNTS:
            if d[k] is not None:
                pooled[key][k] += d[k]
        per_run[key].append(d)
    out = []
    for key, c in sorted(pooled.items(), key=lambda kv: (LEARNERS.index(kv[0][0]), kv[0][1], kv[0][2],
                                                           PARTITIONS.index(kv[0][3]), STRATA.index(kv[0][4]))):
        runs = per_run[key]
        accs = [(ratio(d['exact'], d['n']), d['split'], d['order']) for d in runs if d['n']]
        ffs = [d['false_feasible'] for d in runs]
        dist = dict(runs=len(runs), runs_with_queries=len(accs))
        if accs:
            vals = sorted(a for a, _, _ in accs)
            worst = min(accs, key=lambda x: (x[0], x[1], x[2]))
            dist.update(accuracy_min=vals[0], accuracy_median=statistics.median(vals), accuracy_max=vals[-1],
                        worst_run=dict(split=worst[1], order=worst[2], accuracy=worst[0]),
                        runs_with_false_feasible=sum(f > 0 for f in ffs), max_false_feasible_in_run=max(ffs))
        out.append(dict(learner=key[0], teacher=key[1], checkpoint=key[2], partition=key[3], **{'class': key[4]},
                        counts=c, rates=derived(c), distribution=dist))
    return out


def targets(agg):
    idx = {(a['learner'], a['teacher'], a['checkpoint'], a['partition'], a['class']): a for a in agg}
    out = []
    for tname in TEACHERS:
        for partition in HELD_OUT:
            for learner in LEARNERS:          # controls are listed to show what each condition discriminates
                a = idx[(learner, tname, 48, partition, 'all')]
                mem = idx[('exact_memory', tname, 48, partition, 'all')]
                c = a['counts']
                entry = dict(teacher=tname, partition=partition, learner=learner,
                             primary=learner == 'core', queries=c['n'],
                             exact=c['exact'], exact_set_accuracy=a['rates']['exact_set_accuracy'],
                             teacher_empty=c['teacher_empty'], false_feasible=c['false_feasible'],
                             resolvable=c['resolvable'], exact_repairs=c['exact_repairs'],
                             memory_exact_repairs=mem['counts']['exact_repairs'],
                             worst_run=a['distribution'].get('worst_run'))
                if c['wrong_with_survivor_agreement'] is not None:
                    entry['wrong_with_survivor_agreement'] = c['wrong_with_survivor_agreement']
                if tname == 'T3':
                    entry['conditions'] = dict(
                        zero_false_feasible=(c['teacher_empty'] > 0 and c['false_feasible'] == 0))
                    entry['repair_stratum'] = 'not scored (negative control)'
                else:
                    entry['conditions'] = dict(
                        accuracy_at_least_0_9=(c['n'] > 0 and c['exact'] >= 0.9 * c['n']),
                        zero_false_feasible=(c['teacher_empty'] > 0 and c['false_feasible'] == 0),
                        repairs_exceed_exact_memory=(c['resolvable'] > 0
                                                     and c['exact_repairs'] > mem['counts']['exact_repairs']))
                entry['passed'] = all(entry['conditions'].values())
                out.append(entry)
    return out


def class_counts(man):
    out = {}
    for tname, teacher in TEACHERS.items():
        truth = [realize(q, teacher).response for q in QUERIES]
        cls = [conflict_class(q, truth[i]) for i, q in enumerate(QUERIES)]
        per = {}
        for split in man['splits']:
            for part, idx in (('train', split['train']), ('within_context', split['within_context']),
                              ('unseen_context', split['unseen_context'])):
                per[f"split{split['seed']}/{part}"] = {c: sum(cls[i] == c for i in idx) for c in CLASSES}
        per['all_576'] = {c: cls.count(c) for c in CLASSES}
        out[tname] = per
    return out


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_selective_mediation')
    result = unittest.TextTestRunner(stream=open('/dev/null', 'w'), verbosity=0).run(suite)
    return dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                passed=result.wasSuccessful())


def write_report(report):
    rows = report.pop('rows')
    head = json.dumps(report, indent=1)
    body = ',\n'.join('  ' + json.dumps(r) for r in rows)
    REPORT.write_text(head[:-2] + ',\n "rows": [\n' + body + '\n ]\n}\n')


def main():
    start = time.perf_counter()
    man = freeze_manifest()
    rows, runs, checks, witnesses, work = run_grid(man)
    agg = aggregate(rows)
    tg = targets(agg)
    tests = run_tests()
    gate = dict(
        hard_constraint_and_path_violations=sum(len(v) for k, v in witnesses.items() if k.startswith('violation/')),
        observed_retention_failures_full_feedback=checks['observed_retention_failures'],
        teacher_eliminated_full_feedback=checks['core_teacher_eliminated'] + checks['core_free_teacher_eliminated'],
        core_free_V_mismatch=checks['core_free_V_mismatch'],
        core_free_prediction_mismatch=checks['core_free_prediction_mismatch'],
        order_dependent_final_V=checks['core_order_dependent_final_V'] + checks['positive_only_order_dependent_final_V'],
        inconsistent_feedback=checks['inconsistent_feedback'],
        tests=tests)
    gate['passed'] = (tests['passed'] and all(v == 0 for k, v in gate.items() if k != 'tests'))
    primary = [t for t in tg if t['primary']]
    report = dict(
        experiment=SCHEMA_VERSION,
        contract='docs/selective-satellite-mediation-v1-contract.md',
        status=dict(correctness_gate_passed=gate['passed'],
                    primary_transfer_targets_passed=sum(t['passed'] for t in primary),
                    primary_transfer_targets=len(primary)),
        schema=dict(modes=[0, 1, 2, 3], domains=[sorted(d) for d in DOMAINS], contexts=list(CONTEXT_NAMES),
                    context_coordinates='(c0, c1)', gates=list(GATES), primitives=[p[0] for p in PRIMITIVES],
                    hypotheses=N_HYPOTHESES, hypothesis_index='g(A_low)*64 + g(A_high)*16 + g(B_low)*4 + g(B_high)',
                    teachers={k: dict(id=v, gates=[GATES[g] for g in hypothesis_gates(v)]) for k, v in TEACHERS.items()}),
        configuration=dict(learners=list(LEARNERS), checkpoints=list(CHECKPOINTS), split_seeds=list(SPLIT_SEEDS),
                           order_seeds=[0, 1, 2, 3, 4], partitions=list(PARTITIONS), strata=list(STRATA),
                           selection='smallest surviving hypothesis id', main_experiment_cap=None),
        split_manifest=dict(path=str(MANIFEST.relative_to(ROOT)), sha256=man['sha256']),
        class_counts=class_counts(man),
        correctness_gate=gate,
        proposition_checks=checks,
        transfer_targets=tg,
        work=dict(learner=work, logical_persistent_bits=dict(core=N_HYPOTHESES, core_free=N_HYPOTHESES,
                                                             positive_only=N_HYPOTHESES),
                  evaluation_cache_inferences=EVAL_CACHE_INFERENCES,
                  note='learner counts are logical (every requested evaluation counts, cached or not); '
                       'evaluator, oracle and cache costs are listed separately'),
        witnesses={k: dict(count=len(v), examples=v[:3]) for k, v in sorted(witnesses.items())},
        aggregates=agg,
        runs=runs,
        provenance=dict(python=platform.python_version(), implementation=platform.python_implementation(),
                        command=COMMAND,
                        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}),
        row_columns=list(ROW_COLUMNS),
        rows=rows,
    )
    write_report(report)
    elapsed = time.perf_counter() - start
    learner = VersionSpaceLearner()
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    RUNTIME.write_text(json.dumps(dict(
        seconds=round(elapsed, 3), python=platform.python_version(), machine=platform.machine(),
        platform=platform.platform(), command=COMMAND, git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain')),
        memory=dict(method='sys.getsizeof of the membership int and of the learner __dict__ values',
                    membership_int_bytes=sys.getsizeof(learner.bits),
                    learner_object_bytes=sys.getsizeof(learner) + sys.getsizeof(learner.__dict__)
                    + sum(sys.getsizeof(v) for v in learner.__dict__.values()))), indent=1) + '\n')
    print(json.dumps(report['status']), json.dumps(gate), round(elapsed, 1), 's')


if __name__ == '__main__':
    main()
