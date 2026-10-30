"""Crash-safe runs: every finished task is written to disk at once, and a rerun of the same command skips finished tasks.

Why: the runners hold all results in the parent process until the last task ends, so a power cut late in a run loses
everything. With `durable`, a cut loses at most the tasks in flight (a few minutes each on the GPU).

A task's record is `runs/<NAME>/done/<task>.json.gz`, written after the task's checkpoint file is on disk and only
accepted on rerun if that checkpoint still exists with the recorded sha256. Writes are atomic: temporary file, fsync,
rename, fsync of the directory. A corrupt or unmatched record is ignored and the task is recomputed.
"""
import gzip
import json
import os
from pathlib import Path

from experiments import number_systems as X

nc, ROOT = X.nc, X.ROOT


def fsync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_path(path.parent)


def roundtrip(obj):
    """What the final results file will contain for obj (tuples as lists, numpy scalars as floats)."""
    return json.loads(json.dumps(obj, sort_keys=True, default=float))


def load(path):
    key, res, rt = json.loads(gzip.decompress(Path(path).read_bytes()))
    ck = res.get('checkpoint')
    if ck is not None:
        f = ROOT / ck['file']
        if not f.exists() or nc.sha256(f) != ck['sha256']:
            raise ValueError('checkpoint missing or changed')
    rt['resumed'] = True
    return key, res, rt


class durable:
    """Wrap run(task) -> (key, result, runtime); `fmt` names the record from the task tuple, e.g. '{0}-seed{1}'.
    A class, not a closure, so that a multiprocessing pool can pickle it."""

    def __init__(self, run, done_dir, fmt):
        self.run, self.done_dir, self.fmt = run, Path(done_dir), fmt

    def __call__(self, task):
        path = self.done_dir / (self.fmt.format(*task) + '.json.gz')
        if path.exists():
            try:
                return load(path)
            except (OSError, ValueError, json.JSONDecodeError, EOFError):
                pass  # unreadable or unmatched record: recompute
        key, res, rt = self.run(task)
        ck = res.get('checkpoint')
        if ck is not None:
            fsync_path(ROOT / ck['file'])
        out = roundtrip((key, res, rt))
        atomic_write(path, gzip.compress(json.dumps(out, sort_keys=True).encode()))
        return out[0], out[1], out[2]
