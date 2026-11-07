"""Post hoc, no-fit check: did explicit exception uptake work on fresh practice-distribution episodes?

Chosen after the main run. 1024 episodes per frozen checkpoint; same sampler/noise as training, fresh RNG.
O(12 * episodes * [(K+Q)dh + (K+Q)h² + KQh]) work; one model and one batch in memory.
This is diagnostic, not a selected checkpoint, extra training arm, or a replacement for held-out evaluation.
"""
import hashlib
import json
import resource
import time
import numpy as np
import torch
from experiments import everyday_evidence_transfer as E


def main():
    w,c=time.perf_counter(),time.process_time()
    torch.set_num_threads(1)
    bank=E.make_bank(); rows={}
    for arm in E.ARMS:
        for seed in (450,451,452,453):
            path=E.ROOT/'runs/everyday-evidence-transfer-v1'/f'{arm}-seed{seed}.pt'
            state=torch.load(path,weights_only=True)
            model=E.EvidenceLearner(state['X']); model.load_state_dict(state); model.eval()
            rng=np.random.default_rng([seed,909]); audit=set()
            ordinary=[]; exceptions=[]; probability=[]
            by_context={k:{'ordinary':[],'exception':[]} for k in (0,4,8)}
            with torch.no_grad():
                for _ in range(32):
                    ctx,labels,mask,query,target,qm=E.train_batch(bank,rng,audit)
                    p=model(ctx,labels,mask,query).sigmoid()
                    ordinary.extend(((p[:,:4]>=.5)==target[:,:4]).flatten().tolist())
                    take=qm[:,4]
                    exceptions.extend(((p[take,4]>=.5)==target[take,4]).tolist())
                    probability.extend(torch.where(target[take,4]>0,p[take,4],1-p[take,4]).tolist())
                    sizes=mask.sum(1)-qm[:,4].long()
                    for k,values in by_context.items():
                        sub=sizes==k
                        values['ordinary'].extend(((p[sub,:4]>=.5)==target[sub,:4]).flatten().tolist())
                        sub=sub & take
                        values['exception'].extend(((p[sub,4]>=.5)==target[sub,4]).tolist())
            rows[f'{arm}/seed{seed}']={'ordinary_accuracy':float(np.mean(ordinary)),
                'exception_accuracy':float(np.mean(exceptions)), 'exception_correct_probability':float(np.mean(probability)),
                'ordinary_queries':len(ordinary),'exception_queries':len(exceptions),
                'checkpoint_sha256':E.nc.sha256(path),
                'by_context':{k:{'ordinary_accuracy':float(np.mean(v['ordinary'])),
                                 'exception_accuracy':float(np.mean(v['exception'])),
                                 'exception_queries':len(v['exception'])} for k,v in by_context.items()}}
    E.G.V2.save_json(E.ROOT/'reports/everyday-evidence-transfer-v1-practice-audit.json',{
        'status':'post hoc read-only practice-distribution diagnostic; no fits', 'episodes_per_checkpoint':1024,'results':rows})
    E.G.V2.save_json(E.ROOT/'reports/everyday-evidence-transfer-v1-practice-audit-runtime.json',{
        'cpu_seconds':time.process_time()-c,'wall_seconds':time.perf_counter()-w,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'All 12 final checkpoints, 1024 fresh training-distribution episodes each; no training or selection',
        'runner_sha256':E.nc.sha256(__file__),'dependency_sha256':E.nc.sha256(E.__file__),
        'contract_sha256':E.nc.sha256(E.CONTRACT),'fits':0,'episodes':12*1024,'threads':1})
    for arm in E.ARMS:
        rs=[x for k,x in rows.items() if k.startswith(arm+'/')]
        print(arm,{k:float(np.mean([x[k] for x in rs])) for k in ('ordinary_accuracy','exception_accuracy','exception_correct_probability')})
        print('at k8', {k:float(np.mean([x['by_context'][8][k] for x in rs]))
                        for k in ('ordinary_accuracy','exception_accuracy')})


if __name__=='__main__':
    main()
