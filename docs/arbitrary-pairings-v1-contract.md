# Arbitrary pairings v1: contract (draft, not frozen)

Status: **draft for the researchers' decisions.** Nothing here has been run. Follows the
[limited-memory results](limited-memory-v1-results.md), which showed a trade between
cross-sense transfer (best in `sparse48`) and taught matching (higher in arms with dense
units, inconclusively on the confirmation run), and left memorization itself untested.

## Question

Does a dense, decaying memory part let the system learn specifics without undermining
structure, and does decay help or only remove useful information? Specifics here are
**arbitrary pairings**: items whose partner cannot be inferred from any structure the
network has learned, so only memory can supply it.

## Design

**Arbitrary-pairing task.** Matching, as in the existing evaluation: given an item, pick
the collection image with the matching count among the nine candidates. The item is a
specific image from a script outside the three taught scripts (English, Chinese,
Devanagari) and the held-out Thai transfer script; the default is **Roman numerals**
(decision 1). Each specific image is assigned a count by a seeded random permutation
that ignores its true numeral, with each count assigned equally often. A pair is one image
and one count. Pair set **A** has 90 images (10 per count); a disjoint set **B** of the same
size is built the same way.

**Why this isolates memory.** An image's assigned count carries no information about other
images, so a network can answer on A only by storing the image–count association. The
same images with new random assignments (a **shuffled control**) should score at chance,
which is 1/9 = 0.111, if the network is not exploiting a leak.

**Schedule (decision 2).** The first three blocks are the existing curriculum, unchanged,
without rehearsal. Then:

1. **Block 4:** teach set A, 2,000 updates, with the structured curriculum's batch size
   and no rehearsal.
2. **Block 5:** teach set B, 2,000 updates (interference with A).
3. **Delay and relearning** as in the limited-memory study, with the rehearsal buffer
   consulted only in relearning; relearning includes A (secondary).

Recommendation: sequential, as above. The alternative is to teach A inside block 3 beside
Devanagari, which tests learning "together", but it confounds the pairings with the block
whose transfer is measured.

**Arms:** `sparse48`, `dense`, `mixed`, `mixed-h125`, `mixed-h500`, `mixed-reset`. Decay and
reset schedules are as in the limited-memory study; decay stays on in blocks 4 and 5.
Wiring is RigL; the random-rewiring follow-up reuses this task (decision 3).

**Seeds:** 320–329, fresh. One cohort with the readings below fixed before it runs; a
second fresh cohort (330–339) only if a reading is borderline, decided before looking.

## Measures

All scored per seed and arm; recall means accuracy on the trained pairs of the named set.

- **Acquisition speed:** recall on A (and, in block 5, on B) every 250 updates; the area
  under the recall curve above chance over block 4, and updates to reach recall 0.5
  (censored at 2,000).
- **Retention under interference:** recall on A at the end of block 4, at the end of
  block 5, and after the delay; the drop from block 4 to block 5.
- **Transfer cost:** the eight transfer measures of the limited-memory study at the end of
  block 3 (before pairings), block 4, block 5 and after the delay; average-coordinate
  transfer (a) as before. The cost of pairing is each arm's change from its own end of
  block 3.
- **Where pairs live:** four-way silencing (memory, structure, both) of recall on A at the
  end of block 4 and block 5, with the interaction residual.
- **Controls:** the shuffled control on A at the end of block 4; accuracy on the
  structured Chinese task at the end of block 5 (does pairing damage taught structure).

## Readings (prespecified; margin 0.05 for accuracies, paired 95% t-interval)

- **Memory adds specifics:** recall on A at the end of block 4 is better in `mixed` than in
  `sparse48`. Also read the area under the curve.
- **No cost to structure:** `mixed` is equivalent to or better than `sparse48` in its
  change of (a) and of the eight transfer measures from the end of block 3. If not, memory
  buys specifics at the price of structure.
- **Interference:** the drop in recall on A from block 4 to block 5, `sparse48` against
  `mixed`; the direction is not predicted (sparse overlaps less, dense stores more).
- **Decay helps:** `mixed-h125` better than `mixed` on the transfer change after block 5
  with recall on A not lower by more than the margin.
- **Decay removes information:** `mixed-h125` lower than `mixed` on recall on A at the end
  of block 5 or after the delay by more than the margin.
- **Decay beyond reset:** `mixed-h125` against `mixed-reset` on recall and on the transfer
  change.
- **Inconclusive is not equivalent.**

## Prior work

The shared survey is in [prior work](prior-work.md#memorization-interference-and-sparse-representations-arbitrary-pairings-design).
The task is the standard random-label memorization test (Zhang et al. 2017 **[abstract]**)
applied to a matching reader; the arms correspond to a sparse network, a dense network, and
fast-and-slow components (Hinton and Plaut 1987 **[abstract]**, complementary learning
systems **[familiar]**). Literature predictions are that memory-bearing arms acquire
arbitrary pairs better (Zhang, Feldman 2020 **[abstract]**), sparse representations
interfere less (Ahmad and Scheinkman 2019 **[abstract]**; French 1999 **[familiar]**),
and compression-like sparsity loses atypical exemplars first (Hooker et al. 2019
**[abstract]**). The cheaper alternative, a lookup table or kNN on the item embedding, is
a reference point for recall, not an arm (decision 4). The design adds an internal control
for the limited-memory study, not a new method.

## Computation

- **Time:** per run, the limited-memory run's 8,200 updates plus 4,000 for blocks 4 and 5,
  plus evaluations every 250 updates; linear in updates. The estimate before measuring is
  about 1.5 times a limited-memory run, so about 85 s effective per run on the GPU with six
  processes, and about 85 minutes for 6 arms × 10 seeds. This is an extrapolation, to be
  replaced by measured costs.
- **Storage:** two sets of 90 images, held in memory; checkpoints at the end of each block
  as before (about 12 MB per run), so about 0.7 GB for the cohort.
- **Tuning cost:** the pair-set size and block length are set from a pilot on seed 299
  only; its cost is recorded. No seed in the study is used for it.
- Nondeterministic timing is kept in the runtime files, apart from the scientific results.

## Risks

- **Floor or ceiling:** 90 pairs and 2,000 updates could leave every arm at chance or at
  1.0, which would make the contrasts uninformative. The seed-299 pilot sets the size
  before freezing.
- **Resets destroy pairings:** `mixed-reset` zeroes code-input weights every 100 updates
  until update 4,500 of the earlier schedule; whether it runs through blocks 4 and 5 is
  decision 5.
- **Leakage:** the script's true numerals could help if pairs were assigned by value; the
  shuffled control and the seeded permutation guard against this.
- **Confounds with the number code:** both tasks share the 16-dimensional code and the
  readers, which is the point, but a drop in structure can come from the added updates
  alone. The cost of pairing is therefore read against the arm's own change in the
  limited-memory study's delay, not against zero.

## Decisions for the researchers

1. Script for the pairs: Roman numerals (default), or another script outside the three
   taught and the Thai transfer scripts.
2. Schedule: sequential blocks 4 and 5 (recommended), or A taught together with block 3.
3. Whether the random-rewiring arms run in this task from the start.
4. Whether to add a nearest-neighbor lookup as a recall reference.
5. Whether the reset schedule runs through blocks 4 and 5.
6. Pair-set size and block length are set by the seed-299 pilot unless you want fixed values.
