#!/bin/bash
# Runs the whole CPU-vs-GPU sweep. Start only when the fan-in run has finished (it would perturb both runs).
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}  # a CUDA build of torch, e.g. torch 2.14.1+cu130
OUT=${OUT:-results.jsonl}
: > $OUT
if pgrep -f "experiments.fan_in_pilot" > /dev/null; then echo "fan-in run still active; refusing to start" >&2; exit 1; fi
run() { echo "$@" >&2; $PY bench.py "$@" >> $OUT; }

# 1. one process, batch sweep
for B in 64 270 1024 4096 16384; do
  run single --device cpu  --threads 1  --batch $B --steps 200
  run single --device cpu  --threads 6  --batch $B --steps 200
  run single --device cuda --threads 1  --batch $B --steps 200
done

# 2. realistic: W concurrent processes (CPU: 1 thread each; GPU: all share the card)
for B in 270 1024; do
  for W in 1 3 6; do
    run workers --device cpu  --threads 1 --batch $B --workers $W --steps 200
    run workers --device cuda --threads 1 --batch $B --workers $W --steps 200
  done
done
echo done >&2
