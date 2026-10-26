"""Artifact checks: historical reproduction, finite outputs and fixed provenance."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    wall,cpu=time.perf_counter(),time.process_time()
    output=ROOT/'reports/graded-three-module-v1-verification.json'
    runtime_output=ROOT/'reports/graded-three-module-v1-verification-runtime.json'
    if output.exists() or runtime_output.exists():raise RuntimeError('Refusing overwrite')
    source=ROOT/'reports/graded-three-module-v1-results.json'
    old_path=ROOT/'reports/three-module-ambiguity-v1-results.json'
    runtime_path=ROOT/'reports/graded-three-module-v1-runtime.json'
    new=json.loads(source.read_text());old=json.loads(old_path.read_text());runtime=json.loads(runtime_path.read_text())
    for name,digest in runtime['hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    paired=0;worst=0.
    for new_arm,old_arm in (('inner_only_b0','no_output_to_middle'),('inner_only_b1','reciprocal'),('inner_only_forward','forward_only')):
        for a,b in zip(new['configurations'][new_arm]['episodes'],old['arms'][old_arm]):
            for row,previous in zip(a['trajectory'],b['trajectory']):
                error=float(np.max(np.abs(np.array(row['output'])-previous['output'])))
                worst=max(worst,error);assert error<1e-14;paired+=1
    checked=0
    for config in new['configurations'].values():
        for episode in config['episodes']:
            for row in episode['trajectory']:
                assert row['clamp_error']==0
                assert np.all(np.isfinite(row['output'])) and np.isfinite(row['update_residual'])
                checked+=1
    assert runtime['counts_total']['updates']==33*28*4096+33*4*16*(256+1024+4096)
    result={'historical_trajectory_rows_compared':paired,'max_output_difference':worst,
            'finite_and_clamped_rows_checked':checked,'source_hashes_match':True,
            'update_accounting_matches':True,'additional_candidates':0,'additional_trajectories':0}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    timing={'wall_seconds_before_runtime_serialization':time.perf_counter()-wall,
            'cpu_seconds_before_runtime_serialization':time.process_time()-cpu,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'rss_scope':'Linux single-process lifetime maximum, including JSON parsing',
            'complexity':'O(R) traversal and storage for parsed records, R=7392 new trajectory rows',
            'command':' '.join(sys.argv),'environment_reference':'graded-three-module-v1-runtime.json',
            'environment_note':'same machine and interpreter; no intervening configuration change',
            'new_fits':0,'new_inference_calls':0,'seed_policy':'no randomness',
            'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),source,old_path,runtime_path)}}
    runtime_output.write_text(json.dumps(timing,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
