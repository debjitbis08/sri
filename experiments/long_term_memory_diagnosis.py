"""Long-term memory v1, post hoc (docs/long-term-memory-v1-pilot.md): what limits memory? Pilot seeds only; nothing here is evidence for any contrast.
capacity: memory alone, Adam or SGD, jointly or in sequence, by width and by key and value source, read on the block 6 cache.
replayed: the contract's update rule and combined answer, with frozen keys and frozen values, which gate 2's rule had excluded.
"""
import json
import multiprocessing as mp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.argv = [sys.argv[0], 'diagnosis'] + sys.argv[1:]
from experiments import long_term_memory as E
from src import long_term_memory as ltm


WIDTH = ltm.width


def data(seed,b,key,value):
    tr=E.load_cache(seed)['train'][b]; n=tr['n']; 
    tg=tr['target'].astype(np.int64); q=tr[f'q_{key}']; t=tr[f't_{value}']; c=tr[f'c_{value}']
    us=np.repeat(np.arange(len(c)),n[n>0])
    return dict(q=q,t=t,c=c,us=us,tg=tg)
def batch(d,idx):
    C=d['c'][d['us'][idx]].copy(); C[np.arange(len(idx)),d['tg'][idx]]=d['t'][idx]
    return torch.from_numpy(d['q'][idx]),torch.from_numpy(C),torch.from_numpy(d['tg'][idx])
class Net(torch.nn.Module):
    def __init__(s,dk,dv,H,p,form,seed):
        super().__init__(); torch.manual_seed(seed)
        s.k=torch.nn.Linear(dk,32,bias=False); s.v=torch.nn.Linear(dv,32,bias=False); s.form=form; s.p=p
        s.a=torch.nn.Linear(32,H); s.b=torch.nn.Linear(H,32); s.s=torch.nn.Parameter(torch.tensor(1.))
    def term(s,q,C):
        h=torch.tanh(s.a(s.k(q)))
        if s.form=='k-sparse':
            keep=max(1,int(round(s.p*h.shape[-1]))); top=h.abs().topk(keep,-1).indices; h=h*torch.zeros_like(h).scatter(-1,top,1.)
        return s.s*(s.b(h)[:,None]*s.v(C)).sum(-1)
def read(net,ev,key,value,sets=('A','B')):
    out={}
    for t in sets:
        nm=f'recall/{t}'; q=torch.from_numpy(ev[E.eval_name(nm,f'q_{key}')]); c=torch.from_numpy(ev[E.eval_name(nm,f'c_{value}')])
        counts=torch.from_numpy(ev[E.eval_name(nm,'counts')]).long(); truth=torch.from_numpy(ev[E.eval_name(nm,'truth')]).long()
        with torch.no_grad(): out[t]=round(ltm.accuracy(net.term(q,c),counts,truth),2)
    return out
def run(t):
    H,form,p,opt,lr,mode,seed,key,value=t
    torch.set_num_threads(1)
    d4,d5=data(seed,4,key,value),data(seed,5,key,value)
    ev=E.load_cache(seed)['eval'][6]
    net=Net(E.KEYS[key],E.KEYS[value],H,p,form,seed)
    o=torch.optim.Adam(net.parameters(),lr) if opt=='adam' else torch.optim.SGD(net.parameters(),lr,momentum=.9)
    rng=np.random.default_rng(seed); log={}
    def steps(ds,n):
        for _ in range(n):
            d=ds[rng.integers(len(ds))]; idx=rng.integers(len(d['q']),size=32)
            q,C,y=batch(d,idx); loss=F.cross_entropy(net.term(q,C),y); o.zero_grad(); loss.backward(); o.step()
    if mode=='joint':
        steps([d4,d5],4000); log['joint']=read(net,ev,key,value)
    else:
        steps([d4],2000); log['afterA']=read(net,ev,key,value); steps([d5],2000); log['afterB']=read(net,ev,key,value)
    return t,log,sum(x.numel() for x in net.parameters())
def replayed(t):
    form,H,p,eta,beta,C,seed=t
    torch.set_num_threads(1)
    ltm.width = (lambda f: H) if H else WIDTH     # reset per task: workers are reused
    m=ltm.Memory(form,32,32,'learned',eta,beta,C,p,seed=seed)
    cache=E.load_cache(seed); cfg={'key':'frozen','value':'frozen'}; out={}
    for b in E.BLOCKS:
        ltm.train_block(m,cache['train'][b],'frozen','frozen')
        r=E.readings(cache['eval'][b],cfg,m)
        out[b]=dict(A=r['A']['combined'],B=r['B']['combined'],mA=r['A']['memory'],mB=r['B']['memory'],T=r['answer_transfer'],Tr=r['answer_transfer_readers'],unk=r['unknown']['changed'],lost=np.mean([r[k]['lost_from_correct'] for k in r if k.startswith('match/')]),rA=r['A']['readers'],rB=r['B']['readers'],paired=float(np.mean([r[k]['combined'] for k in r if k.startswith('match/') and 'paired' in k])),paired_readers=float(np.mean([r[k]['readers'] for k in r if k.startswith('match/') and 'paired' in k])))
    return t,out,float(m.s.detach())

def main():
    t0 = time.perf_counter()
    cap = [(H, form, p, opt, lr, mode, s, 'frozen', 'slow') for s in E.PILOT_SEEDS for H in (52, 256, 1024) for form, p in (('mlp', 0.), ('k-sparse', .05), ('k-sparse', .02)) for opt, lr in (('adam', 1e-3), ('sgd', .03)) for mode in ('joint', 'seq')]
    cap += [(H, 'k-sparse', .05, 'adam', 1e-3, m, s, k, v) for s in E.PILOT_SEEDS for H in (52, 1024) for m in ('joint', 'seq') for k, v in (('live', 'live'), ('slow', 'slow'), ('frozen', 'frozen'))]
    rep = [(f, H, p, eta, .9, C, s) for s in E.PILOT_SEEDS for f, H, p in (('linear', None, 0), ('mlp', None, 0), ('k-sparse', None, .05), ('k-sparse', None, .1), ('k-sparse', 256, .05), ('k-sparse', 1024, .05)) for eta in (.003, .01, .03) for C in (1, 16)]
    with mp.get_context('fork').Pool(E.WORKERS) as p:
        a = p.map(run, cap)
        b = p.map(replayed, rep)
    E.save_json(f'{E.REPORT}-diagnosis.json', {'capacity': [{'task': t, 'read': l, 'floats': n} for t, l, n in a], 'replayed': [{'task': t, 'read': o, 's': s} for t, o, s in b]})
    E.save_json(f'{E.REPORT}-diagnosis-runtime.json', E.runtime({'wall_s': time.perf_counter() - t0, 'tasks': len(cap) + len(rep)}))
    rows = {}
    for t, o, s in b:
        rows.setdefault(t[:-1], []).append(o)
    g = {}
    for t, l, n in a:
        g.setdefault((t[0], t[1], t[2], t[3], t[5], t[7], t[8]), []).append(l)
    for k, v in g.items():
        if k[0] in (52, 1024) and k[1] == 'k-sparse' and k[2] == .05 and k[3] == 'adam':
            print(k, v)
    best = sorted(rows.items(), key=lambda kv: -np.mean([x[6]['A'] + x[6]['B'] for x in kv[1]]))[:3]
    for k, v in best:
        print(k, {f: round(float(np.mean([x[6][f] for x in v])), 2) for f in v[0][6]})


if __name__ == '__main__':
    main()
