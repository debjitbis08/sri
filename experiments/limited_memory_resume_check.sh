#!/bin/bash
# Preflight: a limited-memory main-stage run killed with SIGKILL and rerun gives the results and checkpoint hashes of an
# uninterrupted run. Smoke scale; one worker so tasks run in sequence. PY = a python with the needed torch (GPU or CPU).
# Usage: NFI_DEVICE=cuda PY=/data/venvs/sri-cuda/bin/python experiments/limited_memory_resume_check.sh
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export NSY_SMOKE=1 NSY_WORKERS=1
ONLY=dense:300,sparse64:300,mixed:300,mixed-h500:300
D=runs/limited-memory-v1-main-smoke-rclk/done
rm -rf reports/limited-memory-v1-main-smoke-rc* runs/limited-memory-v1-main-smoke-rc*
NFI_TAG=-rcln NFI_ONLY=$ONLY $PY -m experiments.limited_memory pilot main > /dev/null 2>&1
NFI_TAG=-rclk NFI_ONLY=$ONLY $PY -m experiments.limited_memory pilot main > /dev/null 2>&1 &
PID=$!
until [ "$(ls $D 2>/dev/null | grep -c 'json.gz$')" -ge 1 ]; do sleep 0.2; done
kill -9 $PID; pkill -9 -P $PID; wait $PID 2>/dev/null
echo "records after the kill: $(ls $D | grep -c 'json.gz$') of 4"
NFI_TAG=-rclk NFI_ONLY=$ONLY $PY -m experiments.limited_memory pilot main > /dev/null 2>&1
echo "records after the resume: $(ls $D | grep -c 'json.gz$') of 4"
$PY -I - <<'P'
import gzip, json
L = lambda t: json.load(gzip.open(f'reports/limited-memory-v1-main-smoke-{t}-results.json.gz', 'rt'))['results']
a, b = L('rcln'), L('rclk')
strip = lambda d: {k: {x: y for x, y in v.items() if x != 'checkpoint'} for k, v in d.items()}
print('same tasks:', sorted(a) == sorted(b), '| results identical:', strip(a) == strip(b),
      '| checkpoint hashes identical:', all(a[k]['checkpoint']['sha256'] == b[k]['checkpoint']['sha256'] for k in a))
P
