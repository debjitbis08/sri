import json
import random
from pathlib import Path
from src.model import example, settle, dream_seed

m=example()
results={'specification':'finite monotone reachability baseline v0.1',
         'node_count':len(m.nodes),'edge_count':len(m.links),'thresholds':{},
         'dream_seeds':[]}
for threshold in [0,50,100,200,230,255]:
    r=settle(m,{0},threshold)
    results['thresholds'][str(threshold)]={'active':sorted(r.active),
        'rounds':r.rounds,'traversals':r.traversals}
rng=random.Random(42)
results['dream_seeds']=[dream_seed(m,rng) for _ in range(12)]
out=Path(__file__).resolve().parents[1]/'reports'/'baseline.json'
out.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print(out.read_text())
