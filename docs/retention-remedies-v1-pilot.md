# Retention remedies v1: seed-299 pilot

Pilot for the [draft contract](retention-remedies-v1-contract.md): arm `mixed`, seed 299 (never a scored seed), N = 45, L = 2,000, GPU (GTX 1650). Code: `experiments/retention_remedies_pilot.py`; results
`reports/retention-remedies-v1-pilot.json`, timing `reports/retention-remedies-v1-pilot-runtime.json`. One seed and 90 evaluation items per marked set: **nothing below is evidence about the
remedies; it fixes working parameters and checks feasibility.** Selection rules were written before running (see the script); the first run picked grid corners, so both grids were widened and
the pilot rerun with the rules unchanged (amendment, recorded in the script). A first run on the GPU also crashed in the static branches (a CPU random generator under a CUDA default device) and was rerun after the fix.

## Selected parameters
- **Nine-way head:** width 64, 1,000 steps, weight decay 0, learning rate 0.01 (validation agreement with the teacher 0.835; the grid ran 0.74–0.84; width and steps are interior, weight decay 0 is the grid's lower bound).
- **Anchored head:** width 32, 3,000 steps, weight decay 0.001 (agreement 0.61; steps are at the grid's upper edge, but 1,000 steps gave 0.594).
- **DER weight:** the rule picks 0.01, the smallest tried (marked-pair recall after 2,000 updates: 0.389 at 0.01, 0.356 at 0.03, 0.378 at 0.1, 0.244 at 1, 0.256 at 10; plain `rehearse-M` 0.478). The sweep is not monotone and 0.01 is nearly no penalty, so one seed cannot resolve the weight: the study runs `der-M` at 0.01 (rule) and `der-M-strong` at 0.1 (penalty of the same order as the other losses), both labeled.
- Fixed by design: K = 4, replay every 10th update with 16 items, 72 random or other-symbol cues.

## Feasibility and observations (seed 299, `mixed`; teacher accuracy on M at update 0: 0.378 original matching, 0.422 on the canonical candidates)
- Destination accuracy on M (unmarked U in brackets): cue cache 0.400 (0.311); one-cue cache 0.411 (0.319); **other-symbol cues 0.389 (0.326)**; random cues 0.211 (0.170); anchored cache 0.322 (0.252). Agreement with the teacher on M: 0.80, 0.72, 0.84, 0.27, 0.50.
- Host recall on M after 2,000 updates: `ctx` 0.200 (U 0.141), `ctx-m0` 0.144, `hold-M` 0.611, `rehearse-M` 0.478, `rehearse-M-m0` 0.433, `stable-path-replay` 0.522 (U 0.33); accuracy on set B after 250 updates: 0.22 (`ctx`), 0.17 (`stable-path-replay`).
- Derived relations (which of two marked symbols has the larger count, 288 pairs of 18 symbols, chance 0.5): comparison head on canonical collections 0.96; teacher's own symbol codes 0.68; nine-way destination 0.74; **anchored destination 0.83**. The reading is feasible.
- Persistent bytes: model 1,489,192; frozen image-to-code path 859,712; frozen snapshot 920,616; destination (head 6,692 + path) 866,404; rendering buffer (K = 4, 18 pairs) 225,792; stored codes 4,608 (d = 16, 64 bytes per cue); DER extra 30,816. Temporary: 7,776 (cache), 2,376 (one cue).

## Consequences for the design (hypotheses to test, not findings)
1. **Cache-free cues may work.** Unrelated real symbols nearly match the stored cache; random codes do not. The destination may be learning the teacher's code-to-count function, which generalizes across symbols, rather than facts carried by the cues. The cache-versus-other contrast is therefore a real test, and a cache-free result would remove the per-fact cue storage.
2. **The destination is not selective** (0.31 on unmarked symbols vs 0.40 on marked), for the same reason; reading 4 matters.
3. **Compression against the snapshot is small by construction:** the frozen image-to-code path (859,712 bytes) dominates both; the destination is 94% of the snapshot's bytes. The primary contrast's compression half is expected to show a few percent saved, not a large one.
4. **Replay results exceed the pre-delay level** on marked pairs (`rehearse-M` 0.48 vs 0.38 at update 0), as `hold` did in the forgetting study; preservation is judged against the branch's own start, so this counts as preserved.

## Cost of the pilot
Total 601 s wall (data and blocks 1–4 161 s; distillation grids 89 s; about 31 s per host branch including the 250-update teaching of B; DER sweep five weights). Selection touched seed 299 only.
