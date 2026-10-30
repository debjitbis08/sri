# Crash-safe runs

Added 2026-10-30 after a power cut. Applies to `experiments/fan_in_pilot.py` and any runner that follows its pattern.

**Problem.** The runners held every result in the parent process until the last task finished and wrote the results file once, at
the end. A cut late in a 79-minute run lost all of it; only the checkpoint files survived, with nothing linking them to
results.

**What now happens.** `experiments/durable.py` wraps the task function:
- a finished task writes `runs/<NAME>/done/<config>-seed<seed>.json.gz` at once (checkpoint fsynced first, then the
  record written to a temporary file, fsynced, renamed, directory fsynced), from inside the worker, so a dead parent loses
  nothing;
- rerunning the same command skips tasks with a valid record. A record is valid if it reads and its checkpoint exists with the
  recorded sha256; otherwise the task is recomputed;
- resumed tasks are flagged `resumed` in the runtime JSON, which keeps their original timing. The runtime file's total wall
  time then covers only the last session.

**Loss bound.** At most the tasks in flight, about 4 minutes each on the GPU at six processes (about 6 for an 8,200-update
run), plus the data preparation (about 30 s) to restart. Progress inside a task is not checkpointed.

**Checked (smoke scale, GPU):** a run killed with SIGKILL after its first record, then rerun, gave results and checkpoint hashes
identical to an uninterrupted run; a record truncated by half was recomputed and the final results were again identical.
The checks used the GPU path's determinism; on the CPU path the same holds for the same reason.

**To resume after a cut:** run the same command again.

**Not covered.** Files not yet committed can be lost if the operating system had not written them; commit finished work and
documents often. Experiments that fork from a parent checkpoint (the memory study) must save the parent atomically with
`durable.atomic_write` or a task record.
