# Arbitrary pairings v1: contract (draft, not frozen)

Status: **draft for the researchers' decisions; revised after an external review
(2026-10-30).** Nothing here has been run. Follows the [limited-memory results](limited-memory-v1-results.md),
which showed a trade between cross-sense transfer (best in `sparse48`) and taught matching
(higher in arms with dense units, inconclusively on the confirmation run), and left
memorization itself untested.

## Question

Does a dense, decaying memory part let the system learn specifics, and does decay preserve
or sacrifice that? Specifics are **arbitrary pairings**: symbols whose partner count
cannot be inferred from any structure the network has learned. Dense superiority is a
**hypothesis** to be tested, not an expectation from the literature (see "Prior work").

**Primary outcomes:** acquisition of the pairs and retained recall of set A. Absolute
structured-task performance is reported beside them.

## Design

**Arbitrary-pairing task.** Matching, as in the existing evaluation: given a symbol image,
pick the collection image with the matching count among nine candidates, in randomized
positions. A **symbol** is a random Unicode character with no numeric meaning. A pair is
one symbol and one count. Pair set **A** has N symbols, a multiple of nine, set by the
pilot (decision 6); a disjoint set **B** has the same size.

**Symbol selection (frozen with the contract).**
1. Candidates: codepoints in Geometric Shapes, Arrows, Dingbats and Miscellaneous Symbols.
2. Excluded: Braille patterns, dice, tally marks, digits, numerals, fractions, any
   character of the taught or transfer scripts (including CJK), and emoji.
3. Keep codepoints that every font in the renderer's list covers, checked from each
   font's character map, so no missing-glyph box stands for several symbols.
4. Render each survivor with the usual variation and drop any whose images lie within a
   distance threshold of another symbol's (threshold fixed by the pilot, before the study).
5. Draw the symbols of A and B from the survivors with a seeded stream, disjoint, and
   assign counts balanced over the nine counts.

**Same material across arms.** Within a seed, the symbol sets, count assignments, batch
schedules and renderings are identical across arms: they are seeded by seed and symbol,
never by arm. Training and evaluation use **independent renderings** of each symbol
(separate named streams; no rendering is shared), so recall is recognition of the symbol
across font, rotation and size, not of a stored bitmap.

**Schedule.** Blocks 1–3 are the existing curriculum, unchanged, without rehearsal. The
end-of-block-3 state of each arm and seed is the common **parent**. From it:

| Branch | Updates after the parent | What it trains |
|---|---|---|
| `P-AB` | block 4 (2,000): A; block 5 (2,000): B | pairs, then interference |
| `P-A` | block 4: A (shared with `P-AB`); block 5: continuation with no B | decay without interference |
| `C` | blocks 4 and 5 (4,000): continuation with no pairs | matched control for the damage of pairing |

The **continuation** trains on exactly what the delay of the limited-memory study trains
(context episodes, no pairs), with the same batch size, optimizer, decay or reset
behavior and random streams as the paired branches wherever the material coincides. Block
length is 2,000 unless the pilot sets another (decision 6). The delay and relearning
phases of the limited-memory study are not part of this contract.

Sequential A→B is kept. The alternative, A taught inside block 3, confounds pairings with
the block whose transfer is measured.

**Arms:** `sparse48`, `dense`, `mixed`, `mixed-h125`, `mixed-h500`, `mixed-reset`.
- **Rewiring and reset schedules (decision 5, decided by the researchers: Option B is the
  experiment; Option A is the debugging baseline).**
  - **Option B, the study (seeds 320–329).** RigL rewiring of the structure rows and the
    `mixed-reset` resets continue through the end of block 5 (update 10,200), at the
    inherited rate and rule, in every branch including the matched controls `C` and `P-A`.
    Decay in `mixed-h125` and `mixed-h500` is active throughout. Rewiring therefore also
    runs through the end of block 3, so the parents are not the limited-memory parents
    and blocks 1–3 are rerun on the new seeds. Pairs are learned while the structure
    part is still changing; "active decay against active reset" is the contrast.
  - **Option A, debugging only (seed 299 pilot).** The inherited schedules stop at update
    4,500, before block 4 starts at 6,200: connectivity is fixed during pairing and
    `mixed-reset` is a past intervention. It exists to debug the runner and the controls
    in a cleaner setting and to set N and block length. It yields no reading.
  - **Reading B with care.** Continuing resets and rewiring can erase or move the
    connections that hold pairs, so a deficit in a rewiring arm is damage from disruption
    as much as a limit of the architecture. `sparse48` (RigL) and `mixed-reset` carry that
    cost by design; `dense` and the decay arms without reset differ in it.
- The structure part's rewiring rule is RigL; the random-rewiring follow-up reuses the task (decision 3).

**Seeds:** 320–329, one fixed cohort of 10 (Option B); seed 299 for the Option A pilot. No second cohort and no stopping rule:
inconclusive readings stay inconclusive, and any later cohort is a new, labeled run.

## Controls

- **Rescoring against shuffled labels (evaluation check).** At the end of block 4, score
  the trained network's choices on A against labels permuted independently with a seeded
  stream. Accuracy should be chance (1/9 = 0.111) in every arm; a higher score means the
  evaluation (candidate position, rendering order or a count prior) has a shortcut.
- **Retraining on shuffled labels (learning check).** For `sparse48` and `mixed`, a branch
  `P-A'` trains block 4 on a second independent assignment of counts to the same symbols.
  Recall on A and on A' should be equivalent (margin 0.05): if learning depends on which
  counts the symbols got, the assignment leaks.
- **Chance level and the lookup reference.** Recall is reported above chance. A
  nearest-neighbor lookup on the item embedding is an optional reference (decision 4).
- Random labels remove predictable count meaning. They do not show **where** an association
  is stored; silencing shows dependence only (see Measures).

## Measures

Recall means accuracy over the N trained pairs of the named set, on the evaluation
renderings. Evaluated every 250 updates in blocks 4 and 5.

- **Acquisition (primary).**
  - Normalized area: the trapezoid mean of (recall on A − chance) over the block-4 evaluation
    points, divided by block duration, so it is the average recall above chance; the same for B in block 5.
  - Updates to reach recall 0.5, censored at the block length. Descriptive only: report the
    number censored, and compare arms by the proportion reaching 0.5 and the median
    with censoring (no inferential reading).
- **Retained recall (primary).** Recall on A at the end of block 4 (`P-AB`), and at the end
  of block 5 in `P-AB` and in `P-A`.
  - **Interference** = recall drop on A from block 4 to block 5 in `P-AB` minus the same drop
    in `P-A`. The `P-A` drop is ongoing decay or drift with no B; the difference isolates B.
- **Structured performance (reported with the primary outcomes).** The limited-memory
  study's eight transfer measures and average-coordinate transfer (a), **absolute**, at the
  end of block 3, block 4 and block 5.
  - **Damage of pairing** = `P-AB` minus `C` at the end of block 5, within seed and arm, on each measure. This is
    the matched control; the earlier delay runs have different seeds, durations and
    histories and are not used.
  - Also report the change from the arm's own end of block 3, descriptively.
- **Where the pairs live.** Four-way silencing of recall on A at the end of blocks 4 and
  5 with the interaction residual. It shows dependence under intervention, not a separate store.
- Structured Chinese matching at the end of block 5 is one of the eight measures; it is not
  a separate control.

## Readings (prespecified; margin 0.05 for accuracies)

Rule, as in the earlier pilots: **better/worse** if |mean difference| > margin and the paired
95% t-interval excludes 0; **equivalent** if the interval lies within ±margin; otherwise
inconclusive. **Noninferior** if the lower bound of the interval for (arm − reference)
exceeds −margin. Failing to show noninferiority is inconclusive, not harm; harm is the
"worse" outcome only.

Contrasts: `mixed` against `sparse48`; `mixed-h125` against `mixed`; `mixed-h125` against
`mixed-reset`. Everything else, including `mixed-h500` and `dense`, is **descriptive**
(dose and reference).

- **Memory adds specifics:** normalized acquisition area and recall on A at the end of
  block 4, `mixed` against `sparse48`: better.
- **Retained arbitrary recall:** recall on A at the end of block 5 in `P-AB`, `mixed`
  against `sparse48`; and interference as defined above.
- **Structure preserved:** absolute (a) and the eight transfer measures at the end of block 5 in
  `P-AB`, `mixed` noninferior to `sparse48`; together with the damage of pairing
  (`P-AB` − `C`) for each arm. If `mixed` is worse on absolute transfer, memory buys
  specifics at the price of structure; if only inconclusive, nothing is established.
- **Decay preserves the benefit:** `mixed-h125` against `mixed`: recall on A at the end of
  block 5 noninferior, and absolute transfer at the end of block 5 better or noninferior.
- **Decay sacrifices it:** `mixed-h125` worse than `mixed` on recall on A at the end of
  block 5 or on normalized acquisition area.
- **Active decay against active reset:** `mixed-h125` against `mixed-reset` on the same
  measures, named as such.
- **Inconclusive is not equivalent.**

## Prior work

The shared survey is in [prior work](prior-work.md#memorization-interference-and-sparse-representations-arbitrary-pairings-design).
The task is the random-label memorization test of [Zhang et al. 2017](https://arxiv.org/abs/1611.03530)
**[abstract]** applied to a matching reader. They establish that large networks fit random
labels; they do not rank dense against sparse networks, so dense superiority here is a
hypothesis. [Feldman 2020](https://arxiv.org/abs/1906.05271) **[abstract]** argues that
memorizing rare examples is needed for accuracy on long-tailed data, and
[Hooker et al. 2019](https://arxiv.org/abs/1911.05248) **[abstract]** finds that compression
hurts atypical exemplars; both support asking the question, not a ranking of these arms.
[Ahmad and Scheinkman 2019](https://arxiv.org/abs/1903.11257) **[abstract]** study sparse
high-dimensional representations' robustness to noise and interference in image and speech
classifiers; that does not establish reduced sequential interference for these readers,
so the direction of the interference reading is open. Catastrophic interference
(McCloskey and Cohen 1989, French 1999) and fast and slow weights (Hinton and Plaut 1987
**[abstract]**) and complementary learning systems are **[familiar]** context. A
nearest-neighbor lookup is the cheaper reference (decision 4). The design is an internal
control for the limited-memory study, not a new method.

## Computation

Notation: P parameters, E evaluation renderings per symbol, K the evaluation points
(candidate collections, 9), S surviving symbols, C candidate codepoints, F fonts.

- **Symbol rendering and filtering (once, per cohort).** Coverage check O(C·F) character-map lookups.
  Rendering R variants per survivor O(S·R) image draws at 28×28. The distance filter is
  pairwise: O(S²·784) for R = 1, or O((S·R)²·784) if every variant is compared; a seeded
  subsample bounds it. Storage: O(S·R·784) floats.
- **Training per branch:** time linear in updates and in P (as the limited-memory run).
  Per arm and seed beyond the parent: 2,000 (block 4, shared) + 2,000 (B) + 2,000
  (`P-A` block 5) + 4,000 (`C`) = 10,000 updates, plus 2,000 for `P-A'` in two arms. With
  6,200 for blocks 1–3 the total is 16,200 updates, about 2.0 times a limited-memory run.
- **Evaluation:** each point scores N symbols × E renderings × 9 candidate collections:
  O(N·E·9) forward passes; twice (A and B in block 5), plus the structured measures at
  three checkpoints and four-way silencing at two. Evaluations run every 250 updates in
  blocks 4 and 5; the limited-memory run's evaluation took about 27 s.
- **Extrapolated wall time** (to be replaced by measured costs): the limited-memory stages
  ran at 55 s effective per run with six GPU processes for 8,200 updates, so about 110 s per
  arm-seed here, about 110 minutes for 6 arms × 10 seeds, before the added evaluations.
- **Pilot (seed 299 only, all candidates recorded):** N in {18, 45, 90} × block length in
  {1,000, 2,000} × the six arms = 36 arm-runs of the schedule above (the branch set needed to read
  floor and ceiling), about 66 minutes at the same effective rate, plus the filter threshold
  search. All candidates and their costs are reported, not only the chosen one.
- **Persistent state:** model and Adam moments (about 3P floats) per checkpoint; checkpoints at the
  parent, end of block 4, end of block 5 for each branch. In the limited-memory run the
  checkpoints took 511 MB for 80 runs (about 1.6 MB each); here about 6 per arm-seed, so about 0.6 GB for the
  cohort. There is no replay buffer in this design.
- Nondeterministic timing is kept in the runtime files, apart from the scientific results.

## Risks

- **Floor or ceiling:** N pairs and 2,000 updates could leave every arm at chance or at
  1.0. The pilot sets N and block length before freezing, using seed 299 only.
- **Leakage and shortcuts:** addressed by the exclusions, independent renderings,
  randomized candidate positions and the two shuffled-label checks.
- **Look-alike symbols:** at 28×28 many symbols are close. The filter removes near-duplicates;
  residual confusion lowers recall for every arm alike.
- **Disruption during pairing:** under Option B the rewiring and resets run while pairs are
  learned (see Arms). Option A, seed 299 only, shows what happens without them; the two are
  not combined in one reading.
- **Shared code:** both tasks use the 16-dimensional code and the readers. A drop in
  structure can come from added updates alone, so the damage of pairing is read against
  the matched continuation `C`, not against zero or the earlier delay.

## Decisions for the researchers

1. Symbol source: random Unicode symbols with the exclusions and filters above (default), or a
   different set.
2. Schedule: sequential A→B with the matched controls `C` and `P-A` (recommended), or A taught
   together with block 3.
3. Whether the random-rewiring arms run in this task from the start. The reviewer's
   recommendation is to add the matched continuation before wiring variants, as done here.
4. Whether to add a nearest-neighbor lookup as a recall reference.
5. Decided: Option B (rewiring and resets continue through block 5) is the study; Option A
   (inherited schedules, stopping at update 4,500) is the debugging baseline on seed 299.
6. Pair-set size N (a multiple of nine, tried at 18, 45 and 90) and block length (1,000 or
   2,000), set by the seed-299 pilot unless you want fixed values.
