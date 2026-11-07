"""Example-based acquisition over frozen everyday knowledge. See the frozen contract in docs/."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src import everyday_world as W
from experiments import rule_guided_search as G
from experiments import durable as DUR

ROOT, nc = G.ROOT, G.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'everyday-evidence-transfer-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs/everyday-evidence-transfer-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
ARMS = ('codes', 'learned', 'oracle')
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
STEPS = 50 if SMOKE else 2000
CKPTS = (50,) if SMOKE else (500, 1000, 2000)
BATCH, H, DIM = 32, 64, 94
HELD = ('dog', 'whale', 'sparrow', 'penguin', 'salmon', 'bee', 'turtle', 'frog')
CONDITIONS = ('k0', 'k4', 'k8', 'shuffled', 'reversed', 'noisy', 'missing', 'exception', 'conflict')


def digest_state(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_bank():
    names = list(W.ANIMALS)
    attrs = [('is-a', a, 0) for a in W.CLASSES]
    for rel, vals, k in [('has', W.PARTS, 1), ('can', W.ABILITIES, 2),
                         ('lives-in', W.HOMES, 3), ('eats', W.FOODS, 4)]:
        attrs.extend((rel, a, k) for a in vals)
    truth = np.array([[a in W.ANIMALS[n][k].split() for _, a, k in attrs] for n in names], dtype=np.int8)
    train = np.array([i for i, n in enumerate(names) if n not in HELD])
    held = np.array([i for i, n in enumerate(names) if n in HELD])
    concepts, desc, keys = [], [], set()
    raw_counts = {}
    for form in ('single', 'and', 'or', 'xor'):
        count = 0
        choices = [(i,) for i in range(len(attrs))] if form == 'single' else itertools.combinations(range(len(attrs)), 2)
        for ids in choices:
            y = truth[:, ids[0]]
            if form != 'single':
                y = {'and': np.logical_and, 'or': np.logical_or, 'xor': np.logical_xor}[form](y, truth[:, ids[1]])
            y = y.astype(np.int8)
            if min(y[train].sum(), (1-y[train]).sum()) < 6 or min(y[held].sum(), (1-y[held]).sum()) < 2:
                continue
            count += 1
            key = min(y.tobytes(), (1-y).tobytes())
            if key in keys:
                continue
            keys.add(key); concepts.append(y); desc.append({'form': form, 'attributes': list(ids)})
        raw_counts[form] = count
    ordinary = np.array([i for i, d in enumerate(desc) if d['form'] != 'xor'])
    rng = np.random.default_rng(2041)
    rng.shuffle(ordinary)
    cut = int(.7 * len(ordinary))
    train_c, novel = ordinary[:cut], ordinary[cut:]
    structure = np.array([i for i, d in enumerate(desc) if d['form'] == 'xor'])
    groups = {}
    for label, ids in [('novel_combinations', novel), ('held_structure', structure)]:
        take = min(8 if SMOKE else 100, len(ids))
        groups[label] = np.sort(rng.choice(ids, take, replace=False))
    return dict(names=names, attributes=attrs, truth=truth, labels=np.stack(concepts), descriptions=desc,
                train_entities=train, held_entities=held, train_concepts=train_c, test_groups=groups,
                raw_eligible=raw_counts, available_test_counts={'novel_combinations': len(novel), 'held_structure': len(structure)})


@torch.no_grad()
def source_profiles(rd, bank):
    ids = {n: i for i, n in enumerate(rd.W.st.entities)}
    aa, rr, bb = [], [], []
    for animal in bank['names']:
        for rel, attr, _ in bank['attributes']:
            aa.append(ids[animal]); rr.append(rd.W.R[rel]); bb.append(ids[attr])
    inputs = [torch.tensor(v) for v in (aa, rr, bb)]
    scores, _ = rd.net.logic(*inputs)
    p = scores[:, 0].reshape(len(bank['names']), -1)
    e = F.normalize(rd.net.E_L[torch.tensor([ids[n] for n in bank['names']])], dim=1)
    masks = torch.ones_like(p)
    out = {'codes': torch.cat([e, torch.zeros_like(p), torch.zeros_like(p)], 1),
           'learned': torch.cat([e, p, masks], 1),
           'oracle': torch.cat([e, torch.tensor(bank['truth']).float(), masks], 1)}
    y = bank['truth']
    return out, {'accuracy': float(((p.numpy() > .5) == y).mean()),
                 'brier': float(((p.numpy()-y)**2).mean()), 'reader_queries': len(aa)}, inputs, p.clone()


class EvidenceLearner(nn.Module):
    def __init__(self, X):
        super().__init__()
        self.register_buffer('X', X.clone())
        self.project = nn.Sequential(nn.Linear(DIM, H), nn.ReLU(), nn.Linear(H, H))
        self.element = nn.Sequential(nn.Linear(H+1, H), nn.ReLU(), nn.Linear(H, H))
        self.attention = nn.MultiheadAttention(H, 4, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(3*H, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, ctx, labels, mask, query, X=None):
        X = self.X if X is None else X
        c, q = self.project(X[ctx]), self.project(X[query])
        values = self.element(torch.cat([c, (2*labels-1)[..., None]], -1))
        mean = (values * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        # Empty sets use a masked-out zero summary; expose one temporary slot to avoid softmax(all -inf).
        safe = mask.clone()
        empty = ~safe.any(1)
        safe[empty, 0] = True
        attended, _ = self.attention(q, c, values, key_padding_mask=~safe, need_weights=False)
        attended = attended * (~empty)[:, None, None]
        return self.decoder(torch.cat([q, mean[:, None].expand_as(q), attended], -1))[..., 0]


def train_batch(bank, rng, audit):
    ctx = np.zeros((BATCH, 9), dtype=np.int64)
    labs = np.zeros((BATCH, 9), dtype=np.float32)
    mask = np.zeros((BATCH, 9), dtype=bool)
    query = np.zeros((BATCH, 5), dtype=np.int64)
    target = np.zeros((BATCH, 5), dtype=np.float32)
    qm = np.zeros((BATCH, 5), dtype=bool)
    for i in range(BATCH):
        ci = int(rng.choice(bank['train_concepts']))
        flip = int(rng.integers(2))
        y = bank['labels'][ci] ^ flip
        k = int(rng.choice([0, 4, 8]))
        chosen, asks = [], []
        for v in (0, 1):
            pool = bank['train_entities'][y[bank['train_entities']] == v]
            pick = rng.choice(pool, k//2 + 2, replace=False)
            chosen.extend(pick[:k//2]); asks.extend(pick[k//2:])
        chosen = np.asarray(chosen, dtype=np.int64)
        if k:
            ctx[i, :k] = chosen
            labs[i, :k] = y[chosen] ^ (rng.random(k) < .1)
            mask[i, :k] = True
        query[i, :4] = asks; target[i, :4] = y[asks]; qm[i, :4] = True
        if rng.random() < .25:
            eligible = np.setdiff1d(bank['train_entities'], np.r_[chosen, asks])
            ex = int(rng.choice(eligible))
            ctx[i, k] = ex; labs[i, k] = 1-y[ex]; mask[i, k] = True
            query[i, 4] = ex; target[i, 4] = 1-y[ex]; qm[i, 4] = True
        payload = (np.asarray([ci, flip], dtype=np.int32).tobytes() + ctx[i].tobytes() + labs[i].tobytes()
                   + mask[i].tobytes() + query[i].tobytes() + target[i].tobytes() + qm[i].tobytes())
        audit.add(hashlib.sha256(payload).digest())
    return tuple(torch.from_numpy(x) for x in (ctx, labs, mask, query, target, qm))


def test_episodes(bank, seed):
    rng = np.random.default_rng([seed, 61])
    out = {}
    for group, ids in bank['test_groups'].items():
        rows = []
        for ci in ids:
            for rep in range(4):
                y = bank['labels'][ci] ^ (rep % 2)
                # Interleave classes so prefixes are balanced.
                picks = [rng.choice(bank['train_entities'][y[bank['train_entities']] == v], 4, replace=False) for v in (0,1)]
                ctx = np.stack(picks, 1).reshape(-1)
                rows.append(dict(concept=int(ci), rep=rep, context=ctx, labels=y[ctx].astype(np.float32),
                                 query=bank['held_entities'].copy(), target=y[bank['held_entities']].astype(np.float32),
                                 permutation=rng.permutation(8), corrupt=rng.choice(8, 2, replace=False),
                                 exception=int(rng.integers(8))))
        out[group] = rows
    return out


def pack_test(rows, condition):
    n = len(rows)
    ctx = np.zeros((n, 10), dtype=np.int64)
    labels = np.zeros((n, 10), dtype=np.float32)
    mask = np.zeros((n, 10), dtype=bool)
    query = np.stack([r['query'] for r in rows])
    y = np.stack([r['target'] for r in rows])
    k = 0 if condition=='k0' else 4 if condition=='k4' else 8
    for i, r in enumerate(rows):
        ctx[i, :k] = r['context'][:k]; labels[i, :k] = r['labels'][:k]; mask[i, :k] = True
        if condition == 'shuffled':
            labels[i, :8] = labels[i, r['permutation']]
        elif condition == 'reversed':
            labels[i, :8] = 1-labels[i, :8]; y[i] = 1-y[i]
        elif condition == 'noisy':
            labels[i, r['corrupt']] = 1-labels[i, r['corrupt']]
        elif condition in ('exception', 'conflict'):
            j = r['exception']
            ctx[i, 8] = query[i, j]; labels[i, 8] = 1-y[i, j]; mask[i, 8] = True
            if condition == 'exception':
                y[i, j] = 1-y[i, j]
            else:
                ctx[i, 9] = query[i, j]; labels[i, 9] = y[i, j]; mask[i, 9] = True
                y[i, j] = -1  # No single target truth for the conflicting observation diagnostic.
    return tuple(torch.from_numpy(x) for x in (ctx, labels, mask, query)), y


def metrics(p, y):
    valid = y >= 0
    pred = p >= .5
    ba = []
    for a, b, v in zip(pred, y, valid):
        ba.append(float(np.mean([np.mean(a[(b==c)&v] == c) for c in (0,1) if np.any((b==c)&v)])))
    wrong = (pred != y) & valid
    return {'balanced_accuracy': float(np.mean(ba)), 'brier': float(np.mean((p[valid]-y[valid])**2)),
            'confident_error_rate': float(np.sum(wrong & (np.maximum(p,1-p)>=.9))/valid.sum()),
            'query_count': int(valid.sum()), 'episode_balanced_accuracy': ba}


@torch.no_grad()
def predict(model, packed, X=None):
    return np.concatenate([model(*(x[i:i+64] for x in packed), X=X).sigmoid().numpy()
                           for i in range(0, len(packed[0]), 64)])


def ridge_predict(X, packed):
    ctx, labels, mask, query = (a.numpy() for a in packed)
    X = X.numpy().astype(np.float64)
    X = X/np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    X = np.c_[X, np.ones(len(X))]
    out, fits = [], 0
    for c, y, m, q in zip(ctx, labels, mask, query):
        if not m.any():
            out.append(np.full(len(q), .5)); continue
        C = X[c[m]]
        alpha = np.linalg.solve(C@C.T + .1*np.eye(len(C)), 2*y[m]-1)
        out.append(np.clip((X[q]@C.T@alpha+1)/2, 0, 1)); fits += 1
    return np.array(out), fits


def evaluate(model, X, episodes, seed, full=True, ridge=False):
    rng = np.random.default_rng([seed, 81])
    missing = X.clone()
    drops = torch.from_numpy(rng.random((len(X),31))<.5)
    missing[:,32:63][drops] = 0; missing[:,63:94][drops] = 0
    results, counts = {}, {'query_predictions':0, 'context_examples':0, 'ridge_fits':0}
    for group, rows in episodes.items():
        results[group] = {}
        predictions, targets = {}, {}
        for cond in CONDITIONS if full else ('k8',):
            packed, y = pack_test(rows, cond)
            features = missing if cond=='missing' else X
            if ridge:
                p, fits = ridge_predict(features, packed); counts['ridge_fits'] += fits
            else:
                p = predict(model, packed, features)
            counts['query_predictions'] += p.size
            counts['context_examples'] += int(packed[2].sum())
            results[group][cond] = metrics(p, y)
            predictions[cond], targets[cond] = p, y
        if full:
            idx = np.arange(len(rows)); ex = np.array([r['exception'] for r in rows])
            clean, changed = predictions['k8'], predictions['exception']
            new_y = targets['exception'][idx, ex]
            before, after = clean[idx, ex], changed[idx, ex]
            ordinary_y = targets['k8'].copy(); ordinary_y[idx, ex] = -1
            before_ba = metrics(clean, ordinary_y)['balanced_accuracy']
            after_ba = metrics(changed, ordinary_y)['balanced_accuracy']
            conflict = predictions['conflict'][idx, ex]
            ent = lambda p: -(p*np.log2(np.clip(p,1e-9,1))+(1-p)*np.log2(np.clip(1-p,1e-9,1)))
            results[group]['evidence_updates'] = {
                'exception_accuracy_before':float(np.mean((before>=.5)==new_y)),
                'exception_accuracy_after':float(np.mean((after>=.5)==new_y)),
                'exception_correct_probability_before':float(np.mean(np.where(new_y, before, 1-before))),
                'exception_correct_probability_after':float(np.mean(np.where(new_y, after, 1-after))),
                'ordinary_ba_before':before_ba, 'ordinary_ba_after':after_ba,
                'ordinary_ba_drop':before_ba-after_ba,
                'reversal_complement_error':float(np.mean(abs(predictions['reversed']-(1-clean)))),
                'conflict_mean_confidence':float(np.mean(np.maximum(conflict,1-conflict))),
                'conflict_mean_entropy_bits':float(np.mean(ent(conflict))),
                'clean_exception_location_confidence':float(np.mean(np.maximum(before,1-before)))}
    return results, counts


def describe_bank(bank):
    result = {k:v for k,v in bank.items() if k not in ('truth', 'labels')} | {
        'concepts':len(bank['labels']), 'truth_sha256':hashlib.sha256(bank['truth'].tobytes()).hexdigest(),
        'labels_sha256':hashlib.sha256(bank['labels'].tobytes()).hexdigest(),
        'data_bytes':bank['truth'].nbytes+bank['labels'].nbytes}
    return json.loads(json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)))


def verify():
    torch.set_num_threads(1)
    bank = make_bank()
    tr, te = set(bank['train_entities']), set(bank['held_entities'])
    assert not tr & te and len(tr)==26 and len(te)==8
    keys = [min(y.tobytes(), (1-y).tobytes()) for y in bank['labels']]
    assert len(keys)==len(set(keys))
    assert not set(bank['train_concepts']) & set(np.concatenate(list(bank['test_groups'].values())))
    audit = set(); batch = train_batch(bank, np.random.default_rng(1), audit)
    ctx, y, mask, query, target, qm = batch
    assert set(ctx[mask].tolist())<=tr and set(query[qm].tolist())<=tr
    for i in range(BATCH):
        assert not set(ctx[i, :8][mask[i,:8]].tolist()) & set(query[i,:4].tolist()) or bool(qm[i,4])
        # Explicit exception is distinct from ordinary questions even when it occupies a low context slot.
        assert not set(ctx[i][mask[i]].tolist()) & set(query[i,:4].tolist())
    X = torch.randn(34,DIM)
    model = EvidenceLearner(X)
    model.eval()
    with torch.no_grad():
        p = model(ctx,y,mask,query)
        perm = torch.randperm(ctx.shape[1])
        assert torch.allclose(p, model(ctx[:,perm],y[:,perm],mask[:,perm],query), atol=1e-6)
        blank = torch.zeros_like(mask)
        assert torch.isfinite(model(ctx,y,blank,query)).all()
        assert torch.allclose(model(ctx,y,blank,query),model(ctx,1-y,blank,query),atol=1e-6)
    loss = F.binary_cross_entropy_with_logits(model(ctx,y,mask,query)[qm],target[qm]); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    eps = test_episodes(bank,450)
    for rows in eps.values():
        for row in rows:
            assert not set(row['context']) & set(row['query'])
        normal, yt = pack_test(rows,'k8')
        reverse, yr = pack_test(rows,'reversed')
        assert np.array_equal(yr,1-yt)
        assert torch.equal(reverse[1][:,:8],1-normal[1][:,:8])
        _, conflict_y = pack_test(rows,'conflict')
        assert (conflict_y<0).sum()==len(rows)
    print('Verified data separation/deduplication, masked/permuted contexts, finite gradients, reversal/conflict labels.',flush=True)


def run(arm,seed):
    torch.set_num_threads(1)
    phases = {}
    @contextlib.contextmanager
    def phase(name):
        w,c=time.perf_counter(),time.process_time()
        yield
        phases[name]={'wall_seconds':time.perf_counter()-w,'cpu_seconds':time.process_time()-c}
    with phase('construction'):
        bank=make_bank(); rd=G.Readers(seed)
        old_hash=digest_state(rd.model)
        profiles,source,old_inputs,old_scores=source_profiles(rd,bank)
        X=profiles[arm]
        torch.manual_seed(seed)
        model=EvidenceLearner(X)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        episodes=test_episodes(bank,seed)
    rng=np.random.default_rng([seed,51])
    audit=set(); log=[]; ckpts={}; exposures={'episodes':0,'support_examples':0,'target_examples':0}
    sample_wall=sample_cpu=0.
    checkpoint_counts={}
    with phase('training_with_checkpoints'):
        for step in range(1,STEPS+1):
            w,c=time.perf_counter(),time.process_time()
            ctx,labels,mask,query,target,qm=train_batch(bank,rng,audit)
            sample_wall+=time.perf_counter()-w; sample_cpu+=time.process_time()-c
            exposures['episodes']+=BATCH
            exposures['support_examples']+=int(mask.sum()); exposures['target_examples']+=int(qm.sum())
            opt.zero_grad()
            pred=model(ctx,labels,mask,query)
            loss=F.binary_cross_entropy_with_logits(pred[qm],target[qm])
            loss.backward(); opt.step()
            if step%100==0 or step==STEPS:
                log.append({'step':step,'loss':float(loss.detach())})
            if step in CKPTS:
                with phase(f'checkpoint_{step}'):
                    model.eval()
                    ckpts[step],checkpoint_counts[step]=evaluate(model,X,episodes,seed,full=False)
                    model.train()
                print(f'{arm} seed{seed}: step {step}',flush=True)
    ckcost=[v for k,v in phases.items() if k.startswith('checkpoint_')]
    phases['sampling']={'wall_seconds':sample_wall,'cpu_seconds':sample_cpu}
    phases['training']={k:phases['training_with_checkpoints'][k]-sum(x[k] for x in ckcost)-phases['sampling'][k]
                        for k in ('wall_seconds','cpu_seconds')}
    del phases['training_with_checkpoints']
    model.eval()
    with phase('final_evaluation'):
        final,counts=evaluate(model,X,episodes,seed)
    baseline=None; ridge_counts=None
    if arm=='learned':
        with phase('ridge_baseline'):
            baseline,ridge_counts=evaluate(None,X,episodes,seed,ridge=True)
    with phase('retention'):
        assert digest_state(rd.model)==old_hash
        with torch.no_grad():
            new_scores=rd.net.logic(*old_inputs)[0][:,0].reshape_as(old_scores)
        delta=float((new_scores-old_scores).abs().max())
        assert delta==0
        rows=next(iter(episodes.values()))
        first,_=pack_test(rows[:4],'k8'); later,_=pack_test(rows[4:8],'k8')
        p1=predict(model,first); predict(model,later); p2=predict(model,first)
        context_delta=float(abs(p2-p1).max()); assert context_delta==0
    with phase('save'):
        RUNS.mkdir(parents=True,exist_ok=True)
        ck=RUNS/f'{arm}-seed{seed}.pt'; torch.save(model.state_dict(),ck)
    res={'arm':arm,'seed':seed,'bank':describe_bank(bank),'source':source,'log':log,'checkpoints':ckpts,
         'final':final,'ridge':baseline,'retention':{'reader_max_score_change':delta,'reader_hash_unchanged':True,
                                                   'stored_context_max_prediction_change':context_delta},
         'reader_sha256':rd.ck_sha,'checkpoint':{'file':str(ck.relative_to(ROOT)),'sha256':nc.sha256(ck)}}
    params=sum(p.numel() for p in model.parameters())
    rt={'phases':phases,'optimizer_steps':STEPS,'fits':1,'tuning_fits':0,'exposures':exposures,
        'distinct_training_episodes':len(audit),'training_episode_digest':hashlib.sha256(b''.join(sorted(audit))).hexdigest(),
        'checkpoint_evaluation_counts':checkpoint_counts,'final_evaluation_counts':counts,'ridge_counts':ridge_counts,
        'parameters':params,'parameter_bytes':params*4,'gradient_bytes':params*4,'adam_moment_bytes':params*8,
        'adam_scalar_step_bytes':len(list(model.parameters()))*4,'profile_bytes':X.numel()*X.element_size(),
        'all_source_profile_bytes':sum(x.numel()*x.element_size() for x in profiles.values()),
        'reader_parameter_bytes':sum(p.numel()*p.element_size() for p in rd.model.parameters()),
        'task_label_bytes':bank['labels'].nbytes,'checkpoint_bytes':ck.stat().st_size,
        'context_memory_bytes_at_k8':8*(8+4),'context_memory_note':'8 int64 entity IDs and 8 float32 labels; shared profiles separate',
        'audit_payload_bytes':sum(len(x) for x in audit),'python_container_bytes':None,
        'python_container_bytes_reason':'Included in RSS, not separately profiled',
        'inference_latency':None,'inference_latency_reason':'Whole evaluation phases only; not a latency benchmark'}
    return res,rt


def task(t):
    w,c=time.perf_counter(),time.process_time()
    res,rt=run(*t)
    rt.update(wall_seconds=time.perf_counter()-w,cpu_seconds=time.process_time()-c,
              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(f'Finished {t[0]} seed{t[1]}',flush=True)
    return f'{t[0]}/seed{t[1]}',res,rt


def summary(results):
    out={}
    for arm in (*ARMS,'ridge'):
        rows=[r for r in results.values() if r['arm']==('learned' if arm=='ridge' else arm)]
        key='ridge' if arm=='ridge' else 'final'
        def stat(vals):
            return {'mean':float(np.mean(vals)),'min':min(vals),'max':max(vals),'seeds':vals}
        out[arm]={}
        for group in rows[0][key]:
            o={}
            for cond in CONDITIONS:
                o[cond]={m:stat([r[key][group][cond][m] for r in rows])
                         for m in ('balanced_accuracy','brier','confident_error_rate')}
            o['evidence_updates']={m:stat([r[key][group]['evidence_updates'][m] for r in rows])
                                    for m in rows[0][key][group]['evidence_updates']}
            o['passes_acquisition_gate']=(o['k8']['balanced_accuracy']['mean']>=.75 and
                o['k8']['balanced_accuracy']['mean']-o['k0']['balanced_accuracy']['mean']>=.1 and
                o['k8']['balanced_accuracy']['mean']-o['shuffled']['balanced_accuracy']['mean']>=.1)
            o['passes_exception_gate']=(o['evidence_updates']['exception_accuracy_after']['mean']>=.8 and
                                        o['evidence_updates']['ordinary_ba_drop']['mean']<=.05)
            out[arm][group]=o
    return out


def main():
    w=time.perf_counter(); workers=int(os.environ.get('EV_WORKERS',3))
    from experiments import amortized_introduction as AI, relations_as_objects as RO
    sources=[Path(__file__),Path(G.__file__),Path(AI.__file__),Path(RO.__file__),Path(W.__file__)]
    hashes={str(p.relative_to(ROOT)):nc.sha256(p) for p in sources}
    with __import__('multiprocessing').get_context('fork').Pool(workers,maxtasksperchild=1) as pool:
        out=pool.map(DUR.durable(task,RUNS/'done','{0}-seed{1}'),[(a,s) for a in ARMS for s in SEEDS],chunksize=1)
    results=DUR.roundtrip({k:r for k,r,_ in out})
    with io.StringIO() as buf,contextlib.redirect_stdout(buf):
        np.show_config(); blas=buf.getvalue()
    rt=G.V2.runtime({'contract_sha256':nc.sha256(CONTRACT),'runner_sha256':nc.sha256(__file__),
        'source_hashes':hashes,'workers':workers,'wall_seconds':time.perf_counter()-w,
        'aggregate_worker_cpu_seconds':sum(t['cpu_seconds'] for _,_,t in out),'tasks':{k:t for k,_,t in out},
        'blas_configuration':blas,'torch_configuration':torch.__config__.show(),
        'seed_policy':'450–453; concept split2041; paired NumPy training/evaluation streams',
        'dirty_including_untracked':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()),
        'cpu_model':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None),
        'address_space_rlimit_bytes':resource.getrlimit(resource.RLIMIT_AS),'whole_job_peak_memory_bytes':None,
        'whole_job_peak_memory_reason':'Only per-process RSS measured; no aggregate peak profiler',
        'selection_policy':'All arms/seeds/final checkpoints reported, no tuning'})
    summ=summary(results)
    G.V2.save_json(f'{REPORT}.json',{'status':NAME,'results':results})
    G.V2.save_json(f'{REPORT}-runtime.json',rt)
    G.V2.save_json(f'{REPORT}-summary.json',summ)
    print(json.dumps({a:{g:{'BA':x['k8']['balanced_accuracy'],'acquisition':x['passes_acquisition_gate'],
                            'exceptions':x['passes_exception_gate']} for g,x in groups.items()} for a,groups in summ.items()},indent=2))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='verify':
        w,c=time.perf_counter(),time.process_time(); verify()
        G.V2.save_json(ROOT/'reports/everyday-evidence-transfer-v1-verification-runtime.json',{
            'wall_seconds':time.perf_counter()-w,'cpu_seconds':time.process_time()-c,
            'scope':'Mechanical checks; no training or model selection','runner_sha256':nc.sha256(__file__),
            'contract_sha256':nc.sha256(CONTRACT)})
    else:
        main()
