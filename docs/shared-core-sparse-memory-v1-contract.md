# Shared-core sparse memory v1: draft contract

Status: **draft, revised 2026-11-02 after the [stage 0 pilot](shared-core-sparse-memory-v1-stage0-pilot.md). Not frozen. No further
implementation or run is authorized by this document. The pilot's lock: `conjunctive` route, host `sparse48`, sensory source the
frozen encoder, stage 0 candidate `split-recon-bg` at λ 0.1.** Numerical choices are proposed pilot settings. The previous premise, a core pretrained only on relations, failed on
existing codes and is withdrawn; see *Why the first design changed*. Open items
are listed under *Before freezing*.

## Question and commitment

The study now asks two questions, in order.

**Stage 0: can one core hold both relations and enough input identity to address
memory, or do the two trade off?**

**Stage 1: given an address that identifies symbols, does locally writable sparse
memory keep old associations under load, and does the core add anything to that
address beyond a sensory code?**

Stage 1 starts on the core's own code if a stage 0 arm passes, and otherwise on the
conjunctive address, which the user approved on 2026-11-01. Sense-only addressing runs on
either route, since a sensory code should be enough to trigger memory within one sense, and
the core stays in the design for what a sense cannot do alone: reaching a fact across senses
(user, 2026-11-01). A trade-off in stage 0 is
a result, not a failed run: it would mean the two jobs need separate codes.

Associative retrieval, the former second question, moves to its own contract; see
*Deferred: associative retrieval*.

The architecture:

1. Input supplies structured cues to a shared core.
2. The core's response, alone or combined with a sensory code, activates a small
   set of memory units.
3. Units are selected by address vectors and return content vectors.
4. Learning a particular association changes local memory contents. Learning a
   general relation is a separate operation; this study does not consolidate
   knowledge into the core.

There is no answer-label classifier before retrieval and no task-ID router.

## Why the first design changed

The first draft used a pretrained `sparse48` host. A core trained for count and
relations has no reason to keep what distinguishes one new symbol from another:
its job is to place different forms of a count together. Asking it to address
symbols asks for what that training removes.

**[project evidence, exploratory]** Read on the long-term pilot caches, block 3,
seeds 392 and 393. Symbols are the pair sets from the 512-symbol pool, not yet
taught at block 3. Each has 5 renderings there, so these use 4 to fit and 1 to test;
no block-3 checkpoint was kept to regenerate all 64. Chance for 90 symbols is
0.011. These were ad hoc reads, not committed runners, and are not evidence for
any contrast.

| Reading, 90 symbols | host's 16-d code | frozen encoder, 32-d |
|---|---|---|
| Variance on the top direction | 0.99, 0.99 | |
| Identity by cosine prototype | 0.11, 0.14 | 1.00, 0.99 |
| Identity, all 16 dimensions whitened | 0.22, 0.26 | |
| Identity, linear classifier using symbol labels (LDA) | 0.28, 0.35 | |
| Explicit store, perfect one-hot contents | 0.21, 0.27 | 1.00, 0.99 |

On collections, the top direction correlates −0.96 and −0.98 with count, and it
is the same direction the symbols fall on (cosine 1.00). So new symbols are placed
on a number line. Whitening does not recover them, so the loss is not only one of
scale; a nonlinear readout was not tried. Pairings are arbitrary, so a symbol's
position says nothing about its paired count, and neighbors need different
contents. Local memory on these codes read 0.11–0.13 of A after A at every grid
setting, chance, and mean-centring did not help. This reproduces the weak identity
of the [arbitrary pairings pilot](arbitrary-pairings-v1-pilot.md) (0.10–0.35 by
nearest neighbor, seed 299).

**This is a result about that training, not about shared cores in general.** Hence
stage 0.

**Local writes work when given good keys.** Fixed random addresses, frozen-encoder
keys, the real 64-rendering A and B streams from the cache, one-hot contents, one
address bank:

| H, k | A after A | A after A+B | B after A+B | load loss |
|---|---|---|---|---|
| 128, 4 | 0.85, 0.82 | 0.71, 0.69 | 0.65, 0.60 | 0.14, 0.13 |
| 256, 4 | 0.89, 0.91 | 0.80, 0.83 | 0.73, 0.74 | 0.09, 0.08 |
| 512, 4 | 0.95, 0.92 | 0.88, 0.87 | 0.78, 0.83 | 0.07, 0.05 |
| 512, 16 | 0.98, 0.95 | 0.92, 0.88 | 0.85, 0.89 | 0.06, 0.06 |
| 4096, 16 | 1.00, 1.00 | 0.98, 0.97 | 0.98, 0.94 | 0.02, 0.03 |

One-hot contents remove the value and decoder limits, which the design forbids in
the candidate. They show the keys' and collisions' share alone.

**Learned addresses do better at the same size.** The rule under *Mechanism*, same
streams, keys and one-hot contents. Unlike the table above, unknown coverage falls well
below 1. These settings are picked from 18 after seeing them, on two seeds, so the best
row is optimistic. Learned addresses have not been tried on any host's code.

| H, k, τ | A after A | A after A+B | B after A+B | A loss | unknown symbols answered | units used |
|---|---|---|---|---|---|---|
| 256, 1, 0.85 | 0.98, 0.97 | 0.98, 0.94 | 0.92, 0.96 | 0.00, 0.04 | 0.16, 0.13 | 108, 149 |
| 256, 1, 0.70 | 0.84, 0.91 | 0.82, 0.91 | 0.71, 0.73 | 0.02, 0.00 | 0.49, 0.46 | 64, 72 |
| 256, 4, 0.85 | 0.95, 0.97 | 0.87, 0.89 | 0.88, 0.90 | 0.08, 0.08 | 0.13, 0.14 | 166, 222 |
| 256, 1, 0.50 | 0.63, 0.67 | 0.51, 0.64 | 0.51, 0.53 | 0.12, 0.03 | 0.90, 0.88 | 38, 39 |

τ matters more than any other setting here. At H = 128 with τ = 0.85, units can run out: seed 393
has 219 forced writes at k = 1 and 5,897 at k = 4, the latter reading 0.60 for B, and seed 392 has
247 at k = 4. In the
reverse order, B then A, at H = 256, k = 1, τ = 0.85, A reads 0.96, 0.97 and B 0.95, 0.91.

**The earlier bars were not reachable with cosine decoding.** Even the right count's
mean code reads 0.61, 0.63 against candidates. Count prototypes read 0.64, 0.66 on
the host's code and 0.78, 0.69 on the frozen encoder. An explicit store with frozen
keys and live values reads 0.54–0.60. The earlier 0.80 gate looked like `flat-store`'s
0.99, but `flat-store` compares a stored count label with the truth, without
candidates, and is not comparable.

## Stage 0: a core for both jobs

The code is a linear projection from each sense's encoder output (`StandardNet.proj`),
16-dimensional through the fixed `CODE` constant. Widening it and adding a
reconstruction objective are implementation changes to be declared with the runner.

| Host arm | Code | Readers read | Added objective |
|---|---|---|---|
| `sparse48` | 16 | all 16 | none; reference |
| `wide32` | 32 | all 32 | none |
| `wide32-recon` | 32 | all 32 | reconstruct the input from the code |
| `wide32-recon-bg` | 32 | all 32 | as above, also on unlabelled background symbols |
| `split-recon-bg` | 16 + 16 | the first 16 only | as above, from all 32 |

**Constraints that hold across arms.** Reader units keep `sparse48`'s fan-in per unit
and count, one code coordinate per operand and 48 units, so a wider code gets no wider readers: removed
freedom is the pressure.
No identity labels and no relation-specific head or loss. Reconstruction is a
generic sensory objective: mean squared error on images, through a decoder used only
in training, with weight λ in {0.1, 1} on pilot seeds. Its parameters and time are
charged. **Images only** in this version: the readings are on images, and taps are
left out at the cost of an asymmetry between senses.

**Implementation, declared for the stage 0 runner.** `experiments/shared_core_stage0.py`.

- **Width.** `wide32` and its reconstruction arms have a 32-d projection, and `g` and `P` built on 32
  coordinates. Their reader masks draw one coordinate per operand from all 32 and RigL moves within them.
- **`split-recon-bg`.** `g`, `P` and their masks are `sparse48`'s on the same seed, reading the first 16
  coordinates, and the prediction loss compares only those. The projection's first 16 rows start as
  `sparse48`'s, and its other 16 rows are drawn from their own stream. Those rows receive gradient only from
  reconstruction.
- **Decoder.** Mirror of the image encoder: code to 128, to 32 × 7 × 7, two transposed convolutions to 28 × 28,
  ReLU between layers and a sigmoid on the output, 211,265 parameters at width 32, about 845 KB with no optimizer
  state and about 2.5 MB with Adam's. MSE is the mean per pixel over the images in the term.
- **Optimizer.** The decoder joins the host's Adam at the same learning rate and the same gradient clip, 5.0 over all
  parameters.
- **Reconstruction material.** `wide32-recon` uses every image item in the update's batch. `-bg` arms add 64
  background images per update.

**Why the background stream.** Curriculum images are few forms and collections, while
addressing needs generalization to new symbols. Reconstruction on digits alone may
not provide it, and the frozen encoder, which does generalize, trained on 1,987 background symbols,
with their labels. The stage 0 background set is those symbols: 32 renderings each, per seed,
already thinned against the 512-symbol test pool by glyph distance over shared fonts, used
here without labels and drawn uniformly by their own stream. The stream adds examples, reported
separately from curriculum exposures.

**Host training.** Blocks 1–3, 2,000 updates each, otherwise as arbitrary pairings
Option A. RigL stops at update 4,500. Every arm saves its block-3 checkpoint, so all
64 training renderings can be read through it.

### Stage 0 readings, at block 3

On symbols no host arm has seen, 64 training and 5 held-out renderings per symbol,
loads of 90 and 180 symbols.

- **Identity:** nearest cosine prototype, from the code and from the encoder output
  before projection, which is read on each arm and trains nothing.
- **Explicit store with perfect contents:** nine-way count, the address ceiling.
- **Answer-level transfer:** the inherited `transfer8` set, read through the host's
  readers, with `sparse48` on the same seed beside it. These eight decide preservation.
- **Coordinate organization, descriptive only:** average-coordinate transfer (a), on all
  coordinates and on the ones readers read. It averages a separate count probe over every
  coordinate, so appended identity coordinates lower it with relational answers unchanged:
  16 at 0.80 plus 16 at 0.50 average 0.65. It decides nothing.
- **Value ceiling:** the right count's mean code through each declared decoder, and
  count prototypes on collections.
- **Shape:** participation ratio, top-direction share and its correlation with count.
  Are the readers actually using the extra coordinates?
- **Reference:** the frozen encoder, identity and store, 839,040 bytes.

### Stage 0 criteria

Provisional, set on pilot seeds, then frozen. Margin 0.05 is the audit margin of earlier
studies.

An arm **addresses** if identity and the store each reach 0.80 at 90 symbols.

An arm **holds relations** by the seed-wise mean of `transfer8` against `sparse48`, margin
0.05, with a guard on each withheld script.

**Why this form [project evidence].** `sparse48` at block 3 on arbitrary pairings seeds
320–329: Chinese withheld matching has mean 0.09, below chance, so it can barely show a
loss, and Devanagari withheld matching has mean 0.35. Observed paired differences from other
arms on those seeds give 95% half-widths at ten seeds of 0.11–0.15 for Devanagari and 0.04–0.06
for the mean of eight. So a per-script pass at margin 0.05 would mostly read inconclusive even for
an unchanged arm, and a pass on the mean at ten seeds is marginal. Hence the mean as the decision,
**20 seeds** for this stage, about 0.03–0.04 half-width, and each script as a guard only. The Chinese
guard is close to vacuous at this base, and that is stated rather than hidden.

### Locking the design on pilot seeds

**Everything is locked before any confirmation material exists,** in this order: stage 0 pilot,
route, stage 1 pilot and selection. Two seeds cannot support intervals, so these steps use pilot
means, and they are noisy: a difference on the mean of eight has a standard error of about 0.04–0.06.

| Pilot reading | Route | Host | Address source |
|---|---|---|---|
| an arm addresses, and its transfer mean is at least −0.05 against `sparse48` | `core` | that arm, with each λ a separate configuration; among several, the smallest code, then least added cost (nothing, then reconstruction, then reconstruction with background), then the higher transfer mean, then the smaller λ | its code |
| otherwise | `conjunctive` | `sparse48`, the established relational host | its encoder output before projection if that reaches 0.80 on identity and the store, otherwise the frozen encoder, charged |

**The stage 0 candidate** is the `core` route's arm. On the `conjunctive` route it is the added arm with
the highest pilot identity, so a trade-off can still be read.

### Stage 0 confirmation outcomes

The candidate and `sparse48` on seeds 404–423. Paired seed bootstrap, 10,000 resamples, 95%
two-sided. **Holds** if the lower bound of the `transfer8` mean is at least −0.05 and neither withheld script
is lost. **Lost,** for the mean or a script, means a mean of at most −0.05 with an upper bound below 0.

| Outcome, first that applies | Condition | Meaning |
|---|---|---|
| `candidate-holds-both` | candidate addresses and holds | this core did both, on fresh seeds |
| `candidate-trade-off` | candidate addresses and transfer is lost | in this candidate the two jobs compete |
| `candidate-identity-not-reached` | candidate does not address | this candidate did not keep identity on fresh seeds |
| `inconclusive` | otherwise | intervals reported; no claim |

**Addresses** here uses the means, with intervals reported. **These outcomes concern the one candidate.**
Other arms have only exploratory pilot readings, so no result establishes that no shared-core design works.

**Confirmation never switches architecture.** Stage 1 confirmation runs the locked host, address and memory
setting whatever stage 0 confirmation reads. If a `core` route's host fails there, stage 1 is reported with that
label and no conjunctive result is substituted on those seeds.

The pre-projection reading is a screen for the sensory source only. A failure on `sparse48` says little, since
that encoder was not asked to keep identity.

**Expectations, recorded before any run.** `wide32` alone does not reach 0.80.
Background reconstruction helps identity most. `split-recon-bg` holds relations best
among the reconstruction arms. `candidate-trade-off` is the most likely outcome.

**Failure modes.** Reconstruction may use the coordinates the readers need, reducing
transfer; `split-recon-bg` tests that directly. Pixel reconstruction can favour
low-level detail over symbol identity, and a rendering-invariant objective would then
need its own revision. Readers may never touch the added coordinates, which is the
point of `split` and a matter to read in `wide32`.

## Stage 1: local memory

### Address

| Address | z | Role |
|---|---|---|
| `core` | the locked host's code | candidate on the `core` route |
| `perception` | the code of a matched host with no relational training | control for `core` |
| `conjunctive` | concatenation of the core code and `W_K` of the sensory code, each normalized | candidate on the `conjunctive` route |
| `sensory` | `W_K` of the sensory code alone, normalized the same way | control for `conjunctive` |
| `sense-only` | the sense's own code directly, with no core and no `W_K` | candidate on either route, with its own outcome |
| `random-features` | a fixed seeded random projection of pixels, same width | cheap baseline; no attribution |
| `frozen-encoder` | the frozen key encoder | reference for what good keys give; charged 839,040 bytes |

**`conjunctive` replaces the first draft's no-bypass rule when it is used** (user,
2026-11-01).

**`W_K` is PCA whitening, fit on unlabelled material.** The sampled streams have no temporal order worth
predicting, so next-input prediction was dropped. The fit is the mean and top 32 principal directions of the
image sensory code over the 1,987 background symbols, one rendering each, closed form, with no labels. Taps have no
background set, so the tap fit uses the host's tap training stream, on the hosts whose sensory source has taps. Storage:
32 × source width + source width floats per sense. Time: O(nW² + W³) for n fit items. Whether whitening helps
or hurts cosine reads is a stage 1 pilot reading.

**Controls remove one thing only.** `W_K` trains once per seed, on the sensory code alone, and then
stays frozen. `conjunctive` and `sensory` use that same matrix, encoder and normalization, so they
differ only by the core component. `perception` uses the same architecture, projection width,
updates, reconstruction objective, λ and background stream as the locked arm, with readers and
relational losses removed. If that arm has no reconstruction, it uses `wide32-recon-bg`'s, since a host
with no objective learns nothing. One extra host fit per seed, charged. τ and the grid apply separately to
each address, by the same rule, on pilot seeds.

**`sense-only` and `sensory` differ.** `sense-only` asks whether a sense can address memory by itself.
Its source is the same as `conjunctive`'s: the locked host's encoder output before projection if that
reaches 0.80 on identity and the store, otherwise the frozen encoder, charged. `sensory` adds `W_K`, only
to match `conjunctive`. So `sense-only` does not wait on `W_K`'s specification.

**The core may have no observable job on these pairs.** A symbol's arbitrary count
does not depend on relational context, and a symbol appears in one sense, so `sense-only` and
`conjunctive` may tie, and a tie is not evidence against the core. Its job is tested below.

### Cross-sense fixture

**Why the core stays in the address.** A sense's code reaches only what that sense wrote; the core is what
senses share. Teach an association from one sense, and query it from another, with no write from the second.

Cues are counts, since these are what exist in several senses. Teach each written form of a count, in English,
an arbitrary target from a seeded **derangement** of the nine counts, so that answering with the query's own count
scores zero rather than chance. Write 64 training renderings per count, one pass in a seeded order, 576 writes, each
with a fresh training collection of the target count as v. Query with `D.TAPS[('eval', n)]` and, separately, with
English evaluation renderings; candidates as in the primary reading. English is the script all hosts train on with every count
paired, so a failure here is not a failure of transfer to withheld counts. This holds only nine facts, so it tests
reach and not load. Writes for this fixture use a separate empty memory.

Read on every route, with these addresses: the host's `core` code alone, `conjunctive`, `sense-only` and `sensory`.

**The fixture has its own setting for each address,** since `core` on the `conjunctive` route may have no eligible setting on the
pairs. Grid k in {1, 4}, τ in {0.5, 0.7, 0.85}, H = 128, which nine facts cannot limit much. On pilot seeds take the largest
mean of the tap and same-sense reads, ties to the higher τ and then the smaller k.

**A sensory source with no taps.** The frozen encoder reads only images. If it is the sensory source, a tap query has no sensory code:
`sense-only` and `sensory` abstain, which counts as wrong, and `conjunctive` reads the core part alone: dot(core(z), core part of a_i, normalized) ≥ τ. A zero sensory part would
hold the similarity below every τ in the grid [derived], so this fallback is what lets the address be tested at all, and the locked route uses this source. If the source is the host's encoder, tap
codes come through the host's sequence encoder and are compared directly.

**Predictions [derived].**

- `sense-only` and `sensory` cannot reach across senses: the query's code comes from another encoder,
  so they should abstain or read at chance.
- `core` can, if forms and taps of a count land together. Count is what `sparse48`'s code holds, so this may
  work on the same code that fails symbol identity.
- **`conjunctive` without its fallback fails by construction.** With its two parts normalized and joined, a query from another
  sense matches at about half the core's similarity, so at most about 0.5, at or below every τ. Lower τ is not a fix, since it
  merges symbols. So the read falls back to the core part when a query has no sensory code, as defined above. A
  query with a sensory code from a different encoder has no such guard, and the fixture reads that on a host-encoder source only.

### Mechanism

**Addresses are learned; fixed addresses are not a candidate** (user, 2026-11-01). Let z
in R^d be the normalized cue and H the maximum unit count. Memory starts with no units.
An allocated unit i has a cue sum c_i in R^d, whose normalized direction is its address
a_i, a content sum u_i in R^m and a write count n_i. m is the code width of the selected
host. k is the number of winners and τ the vigilance.

`I(z) = top_k allocated i with dot(z, a_i) ≥ τ`

Ties break by unit index. A zero response abstains. **Read:**

`v_i = u_i / n_i`, and `read(z) = mean(v_i for i in I(z))`.

An empty I(z) abstains. There is no global softmax over units, learned global scale or
output bias.

**Write,** for an association (cue, target), with target vector v:

1. If I(z) is empty and a free unit remains, allocate one unit at z.
2. If I(z) is empty and none remains, use the nearest allocated unit, and count this as a
   **forced write**. Nothing is erased or replaced; replacement is a later choice.
3. On each selected i: `c_i <- c_i + z`, `u_i <- u_i + v`, `n_i <- n_i + 1`. All other state
   stays bitwise unchanged.

So an address is the running mean direction of the cues it took, and it moves less as its
count grows. Cue and target are the only inputs: symbol IDs, count IDs and task labels do
not route, and are not stored. Apply writes in presentation order. No momentum, weight
decay, replay buffer, optimizer state or global gradient update.

Dead units cannot occur, since units exist only once written. Utilization is the allocated
fraction, reported. τ sets how far a unit generalizes, and it depends on the address code's
similarity distribution, so it is selected per address, on pilot seeds.

**Decoder.** `decode(content, candidates)` is cosine against the candidates' codes, with the host's matching head as a
diagnostic. This was revised after the stage 0 pilot, where on `sparse48` the head's value ceiling read 0.52, 0.61, below the
decoder floor, and cosine read 0.60, 0.62. Cosine clears the floor by 0.01 on two seeds, and the study stops at `decoder-limited` if the
stage 1 pilot's value ceiling falls below it.

Learning addresses mixes content interference with rerouting, so both are read separately
under *Mechanisms*. `fixed-random` stays as a reference for what learning buys.

### What follows without running [derived]

- **Protection covers every competing address.** With a frozen host and decoder, a read depends only
  on I(q) and on the contents of its units. So writing a new set cannot change the read of q if no write
  selected a unit in I(q), and every other unit that was **moved or allocated** ends below τ for q, or below
  q's k-th winner. An untouched old winner is not enough: an existing nonwinning unit can move into the
  winning set. With q at 0°, units at −40° and 60°, k = 1 and τ = 0.7, two writes at 25° update only the
  second unit, whose similarity to q rises from 0.50 to 0.804, past the first's 0.766. No allocation or
  forced write occurs. This is for the read, not for correct answers.
- **Contents can mix before memory is full.** A new cue within τ of an old unit writes to it, so conflicting
  contents can combine at any occupancy. Forced writes add to this; they do not start it.
- **Order matters.** Allocation depends on what came first, so writes do not commute, unlike
  the fixed rule, whose states were identical in either order. Old loss therefore measures
  sequential effects as well as load, and both orders are primary.
- **It is close to a store without the oracle.** At high τ and k = 1, this approaches one
  prototype per symbol, found without symbol grouping. Expect it to approach the explicit store
  from below, on the same keys, and not to beat it.
- **τ cannot rescue a poor address.** It needs same-symbol similarity above different-symbol
  similarity. On the frozen encoder these are 0.91–0.93 and 0.05. On `sparse48`'s code, 0.67–0.85
  and 0.17–0.42, with 90-way identity of 0.11–0.14, so stage 0 still comes first.
- **Counts do not forget.** With `u/n`, a new write's weight falls as a unit's count
  rises, so a new or corrected pair is diluted by what was written before. The equal
  schedule hides this. This memory is also not the decaying, recency-biased one in the
  target architecture.

### Schedule and material

Arbitrary-pairings symbol generator, nine candidate counts. Initial load: 45 symbols in
A and 45 disjoint symbols in B, balanced over the nine counts, 64 training and 5
held-out renderings per symbol. A third disjoint set of 45 is never written. Renderings
and candidate order are paired across arms. The known label constructs the target
collection during teaching; no answer or set membership is supplied at retrieval.

Freeze the host, encoders and decoder at the block-3 checkpoint in the **stationary
primary condition**. Memory starts empty.

| Phase | Host updates | Memory teaching |
|---|---|---|
| A | none | 2,000 presentation steps, 16 A items each |
| B | none | 2,000 presentation steps, 16 B items each |
| Delay | none | 2,000 empty steps; no writes |

Use the inherited seeded pair sampler, with replacement; report unique examples and
exposures separately. Read A and B before teaching and every 250 steps, including phase
boundaries. Reading never trains. The empty delay must leave state and predictions
unchanged, as an audit.

**Both orders.** Every setting and comparator runs A then B and B then A, on the same seed. In each order
the **old** set is the one taught first and the **new** set the one taught second.

**Unequal exposure probe.** B taught at 1/10 of A's exposures, descriptively, for the
dilution above.

**Changed-pair probe.** Teach 18 A symbols a second count at equal and at 1/10 exposure,
and read which answer wins.

**Moving-core stress condition.** From the same checkpoint, continue the inherited
context-plus-pairs host schedule and context-only delay, with memory detached. Addresses
use live responses for every part the host computes: the core code, and the sensory code when the source is the host's
encoder. A frozen-encoder sensory source stays frozen, as the locked address defines it, and is charged and reported as
such, so on that source only the core part drifts. Values and candidates run twice: live codes, and codes from a block-3 copy of the encoder and
projection, charged. The long-term pilot found values the main limiter, so both are
needed. **Prediction:** routes break early. Addresses move only when written, so old ones do not
follow a drifting code, and old cues should fall below τ and abstain rather than answer wrongly,
while re-taught symbols allocate new units. As context, on existing `sparse48` codes with fixed
addresses at H = 256, 0.43–0.58 of a rendering's selected units from block 3 remain selected at
blocks 4–6. Report separately; stationary success does not establish success under core learning.

**Seeds.** Pilot 392 and 393, whose `sparse48` codes are already inspected, so they stay
exploratory. Proposed confirmation: stage 0 on 404–423, stage 1 on 404–413, the first ten of the same hosts. Load probes,
the unequal-exposure and changed-pair probes and the moving-core stress condition run on 404–406, descriptively.
No reservation or use was found in the contracts, runners or reports on 2026-11-01; recheck before generating
material. No confirmation data may influence the route, host, address source, objective or memory setting.

### Comparators

1. **Local sparse memory**, as above, on each address in the table.
2. **Trainable memory:** the long-term pilot's forms `mlp` and `k-sparse` from R^d to R^m,
   cross-entropy over nine candidates of `⟨M(z), v(c)⟩`, with no reader logits, since the
   host is frozen here. Same streams and both orders. This scores candidates directly, so its
   decoder differs and the comparison is of complete mechanisms. Width matched to the
   selected local memory's bytes, charging weights and momentum. Grid: two forms × η in
   {0.003, 0.01, 0.03}, β 0.9, C 1, six configurations. Its frozen-frozen result in that
   pilot, 0.52–0.59 falling to 0.16–0.23, is the prediction.
3. **Explicit store:** one normalized cue prototype and target vector per symbol. Symbol
   grouping is an oracle privilege unavailable to the candidate. A diagnostic, with actual
   bytes reported.
4. **`fixed-random`:** the same contents and reads on seeded Gaussian addresses that never move,
   with no τ, at the same H and k. A reference for what learning addresses buys, not a candidate.
5. **Disjoint, overlapping and moved-competitor fixtures:** unit tests with constructed inputs, including the
   counterexample above, not empirical results.

**No dense-write variant.** k = H has no meaning here. With learned addresses and τ = −1, the first write
allocates one unit and every later cue matches it. With fixed addresses, all units receive every write and
stay identical. Either way it is one global running average, not dense storage at comparable capacity. The
dense comparison is `mlp`, at matched bytes.

Host alone is reported for context, since the frozen host has no pair answer to lose. Presentation
order and renderings are paired across local-memory comparisons.

### Readings and decision rules

**Primary, in each order.** Old after old; old and new after both. On independent renderings,
abstentions count as wrong. Chance for a forced nine-way answer is 1/9; report coverage
separately.

**Old loss** = old after old − old after both, which contains both load and order effects.

**Bars,** provisional and frozen after the pilot, since stage 0 decides the decoder's ceiling. The
paired reference is the explicit store on the same address and decoder, holding what memory has been
taught at that point, so only the old set for old after old, and both sets after both.

- **Decoder floor,** on pilot means: the value ceiling reaches at least 0.60, against chance 0.11. Otherwise
  the study stops at `decoder-limited` and revises the decoder, before any memory setting is selected. Existing
  codes read 0.61–0.63 with cosine, so this bar is close to what is known to be reachable.
- `acquires`, for each order and each of old after old and new after both: the lower 95% bound of candidate
  minus store is at least −0.15, the near-storage margin of the long-term contract, **and** the lower 95% bound
  of the candidate's own score is at least 0.50. So a weak store and a weak candidate cannot pass together.
- `holds under load`, for each order: the upper 95% bound of old loss is at most 0.05.

Paired seed bootstrap, 10,000 resamples, 95% two-sided. **A resample takes whole seeds, with both orders
kept together.** Readings within a seed are not independent units.

| Outcome, first that applies | Condition |
|---|---|
| `retains-under-load` | acquires and holds under load, in both orders |
| `load-unresolved` | acquires in both orders; an old loss bound above 0.05 |
| `acquisition-limited` | otherwise |

Failing to establish a bound is inconclusive, not proof of destruction. Claim improvement
over trainable memory only if, on the per-seed mean over orders, the paired interval for lower old
loss excludes zero and new acquisition is noninferior by margin 0.05. Each order is reported beside it.

**A claim for the address.** One contrast, on the per-seed mean over orders of old and new after both:
`core` against `perception`, or `conjunctive` against `sensory`, credited only if the paired interval
excludes zero in its favour. `random-features` and `frozen-encoder` show whole-pipeline differences and earn
no attribution. A label alone earns no claim for the shared core.

**`sense-only`.** It gets its own outcome from the table above, and is reported beside the locked address.
The paired contrast with the locked address, on the same per-seed mean, reads better, equivalent or inconclusive at margin
0.05. If `sense-only` is equivalent or better, then within one sense the core is not needed for these pairs.

**Cross-sense reach.** Primary for the core's reason to exist: taps queries on the fixture. **Reach** is shown for an
address if the lower 95% bound of its score is above chance, 0.11. Beating `sense-only` is not part of the test on the locked route: it has no tap
encoder and abstains, so that comparison is vacuous. `core` and `conjunctive` are read side by side, on English and on taps. On the locked route a tap query reads only the core part,
so success shows access through the shared relational code, and does not show that concatenation produces cross-sense matching. A claim that the core earns its place in the
address needs reach, on `core` or `conjunctive`.

**Unknown symbols.** On the unwritten third set: coverage, the share of queries answered,
and the same for known symbols, so a high τ cannot look good by refusing everything.
On the frozen-key probe coverage was 0.13–0.16 at τ = 0.85 and 0.88–0.90 at τ = 0.5, against
at least 0.97 with fixed addresses at k ≥ 4. Vigilance is therefore a candidate novelty signal,
reported here and not used in selection.

**Mechanisms.** Units allocated and forced writes, by phase. Of the old set's units, the share the new
set's writes changed, and how far their addresses moved. **Rerouting:** of old queries, the share whose
I(q) changed through a newly allocated unit, through an existing unit moving in, and through a winner moving
out, against the share whose winners stayed and whose contents changed in place. Within-symbol and between-symbol support overlap; symbols per
unit and units per symbol; conflicting targets per unit; changed inactive parameters;
exact-training-cue against held-out-cue recall. Aggregate by seed before inference. Large overlap
is a candidate sign of shared structure, not evidence of a relation.

**Relations.** With the host frozen, host-only invariance is an audit. Report combined
behaviour only after a combined answering rule is specified; do not reuse the earlier
unstable additive residual.

### Gates

Before selection, write a provenance-bound gate report. Failed or absent gates prevent
selection; report and revise rather than silently relax.

- A stage 0 pilot report exists, with its route, host and address source recorded.
- No target, symbol identity, phase label or evaluation answer enters routing.
- Writes satisfying the protection condition leave old outputs and inactive state unchanged, and the
  moved-competitor fixture changes the winner as derived.
- Reads preserve all persistent state, counters and RNG policy.
- Empty-step invariance holds for the local arm.
- The explicit store reaches the decoder's value ceiling minus 0.05 on this address. If not,
  the address, not memory, is the limit.
- Cache reconstruction, checkpoint restore and fresh-run results agree.
- Label permutation audits use repeated permutations and their finite-sample distribution.

### Selection

Local grid: H in {128, 256, 512}, k in {1, 4}, τ in {0.5, 0.7, 0.85}, 18 configurations,
each in both orders. H is a limit, not an allocation, so settings that never fill it are
identical and need one run. **There is no byte cap for eligibility.** The earlier 64 KiB
came from a different mechanism and would have excluded H = 512 by 2 KiB, deciding the outcome
by itself. Report every setting's bytes, by limit and by units used.

**On pilot means, by the worst order.** A setting is eligible if, in both orders, its old after old and new
after both are at least the store's minus 0.15 and at least 0.50. Among eligible settings take the smallest
bytes with worst-order old loss at most 0.05; if none, the lowest worst-order old loss, marked. Ties within 0.01
go to smaller measured read work. If none is eligible, report acquisition failure and select no winner. Trainable
memory is tuned by the same rule. Include rejected candidates in cost accounting. If a selected τ sits at the end
of its range, record that before confirmation; the grid does not expand on confirmation seeds.

`sense-only` is selected by the same rule, separately, so each address has its own locked setting.

Confirmation uses the locked setting, its address control, `sense-only`, the cross-sense fixture and the applicable
comparators, in both orders where order applies, on seeds 404–413. No adaptive grid expansion. Load probes N = 18 and 90, the unequal-exposure and
changed-pair probes and the moving-core stress condition use seeds 404–406, with no retuning, and are descriptive.

## Relation to the target architecture

This draft departs from recorded decisions on purpose. **The user accepted the departures below
on 2026-11-01 and rejected one: fixed addresses, which the draft has dropped from the candidate.**
It is a step towards the system and not the system.

| Target | Here | Why, and what would close it |
|---|---|---|
| `sparse48` is the shared core | `sparse48` is only the reference; arms widen the code and add reconstruction | its 16-d code does not identify new symbols; stage 0 reads what this costs in transfer |
| Recent memory with a context window beside long-term memory | long-term role only | a later step, as in long-term memory v1 |
| Everything is a network; explicit stores are diagnostics | learned-address units with summed contents, close to slots | the user's unit-vector proposal, in local form; compare with trainable memory |
| Learned key and value projections | addresses learned by a local rule; no learned value projection, and `W_K` only under `conjunctive` | keeps rerouting readable; next study |
| Sensory-only memory, nothing marks cue or outcome | marked pairs | material from the curriculum; next study |
| Memory combined in the forward pass | memory read alone | the additive residual proved unstable; a rule must be designed |
| Sparse host, not frozen stand-ins | primary condition frozen, stress condition moving | isolates load; stress reads the real case |
| Decaying, recency-biased memory | none | deliberately absent; changed-pair probe reads the cost |
| Memory trains the sparse network | not tested | a separate role |

The user also approved `conjunctive` on 2026-11-01, for use when no core passes stage 0.

## Deferred: associative retrieval

Requirements carried to its own contract. A recurrent architecture claim needs that
contract's specification and its result.

**Mechanism must exist first.** A read returns a count code, and nothing was written at
that code as an address, so feedback has nothing to complete. The contract needs
autoassociative writes or a defined mapping from content to cue. Recurrence on a code
without identity cannot restore identity [derived].

Retrieval is read-only. First evaluate one read, then a bounded feedback search using the
same memory. Proposed budgets: 1, 4 and 16 memory probes per query; every restart or branch
counts as a probe. No confidence-based extra unreported probes and no oracle selection of
the best intermediate answer. The final-answer rule, feedback equation, cue anchoring, state
initialization, cycle handling and any perturbation distribution are fixed with the core's
operations `respond(cue, workspace)` and `feedback(original_cue, retrieved_content, workspace)`.
Retrieval has its own temporary state, does not overwrite the state interpreting the input,
and uses the same relational parameters. Randomness is query-seeded independently of targets.
Record cycles and budget exhaustion apart from convergence.

Compare feedback with the first read and with independent cue perturbations using the same
probes, state and answer-selection rule, plus a read-count-matched explicit store. Do not use
raw random perturbation as evidence of relational feedback.

Direct symbol-to-count recall may offer no useful intermediate cue. Before claiming recovery
across associations, add a cue-chain fixture: teach x -> y and y -> z, with distinct symbol
identities and no x -> z write; query x with the same two-hop instruction for every example;
answers are z among nine held-out candidate renderings. Every edge, representation and relation
cue is charged. It must not hand the reader the intermediate symbol.

Report misses recovered, correct answers spoiled, net gain, probe count and latency. Failure
within a budget is `not recovered`, never `erased`. An evaluator-only diagnostic may test whether
contents at saved original supports still decode; these traces are unavailable to the reader.

**Earlier evidence is unfavourable.** Recurrence added little in [translation memory](translation-memory-v1-pilot.md),
and [streaming retrieval](streaming-retrieval-v1-results.md) read `does-not-beat` for input-driven dynamics.

## Prior work and what this tests

Shared reading: [sparse local memory notes](sparse-local-memory-notes.md),
[prior work](prior-work.md#interference-protection-after-the-long-term-memory-pilot) and
[relational-memory survey](memory-relational-prior-work.md). The sparse-memory search was on
2026-11-01; queries and evidence levels are in the notes.

**Established names.**

- **[familiar; read before use]** Kanerva's sparse distributed memory is the closest established
  method for the `fixed-random` reference: fixed hard locations, activation near the address and counter writes.
  Its capacity analysis and its known weakness on correlated data bear on that reference, and on how far learned
  addresses are needed. The candidate's own closest method is the next item.
- **[familiar; read before use]** The learned-address rule is closest to adaptive resonance theory
  (vigilance, allocation of a new unit on mismatch) and leader clustering, with online k-means for the
  moving address. Growing neural gas is a relative. Known failure modes: sensitivity to vigilance, to
  presentation order and to category proliferation under noise. All three appear in the probe above.
- **[familiar; read before use]** FlyHash for sparse random expansion as a similarity hash, and
  Neural Episodic Control for an appended key-value memory with nearest-neighbor reads.
- **[read]** [SDMLP](https://arxiv.org/pdf/2303.11934) combines top-k with normalization,
  non-negativity and no biases; it discusses dead units, optimizer effects and a static-input-manifold
  limitation. It also learns addresses, by gradient; this draft uses a simpler local rule and does
  not claim to reproduce it.
- **[read]** [FlyModel](https://arxiv.org/pdf/2107.07617) couples sparse expansion to restricted
  associative writes. Its output-specific update differs from averaging vector contents here.
- **[read]** [Elephant](https://arxiv.org/pdf/2310.01365) connects interference to gradient overlap.
  This draft tests explicit local writes, not Elephant activations.
- **[read, earlier excerpts]** Tolman–Eichenbaum separates transferable structure and particular
  sensory bindings, the closest precedent for `conjunctive` and `split-recon-bg`. Context, not
  evidence that either works here.
- **[familiar; read before use]** Reconstruction as an auxiliary objective against invariance:
  autoencoders and the information-bottleneck view of compression against retention.

**Project evidence.**

- [Translation memory](translation-memory-v1-pilot.md) reached 0.907 old-word recall at 1,024 units
  and 41 active, **on a separate frozen key encoder**, not the host's code. Explicit storage was better
  and cheaper. Adding an identity objective to the host then failed the perception gate at 0.59 and 0.21,
  with its effect on transfer unmeasured; stage 0 reads both.
- The [long-term pilot](long-term-memory-v1-pilot.md) showed strong A-to-B interference in trainable
  memories, and values as a limiter. It did not test protected local writes.
- [Limited memory](limited-memory-v1-results.md) and [arbitrary pairings](arbitrary-pairings-v1-results.md)
  give `sparse48` the highest transfer, and dense units inside the host cost transfer. So a wider code
  may cost transfer too.

**What this adds.** Whether transfer and identity on new symbols can share one code, measured together,
is the question not already answered here. For stage 1, disjoint writes needing only an audit and storage
on good keys are not discoveries. The empirical questions are collisions at load, dilution and the core's
contribution. No novelty claim is made.

**If a shared core is tried again.** Pixel reconstruction rewards recovering appearance, while addressing needs symbols kept apart across renderings. Contrastive learning on identity-preserving
augmentations, as in [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) **[external review; not read]**, is closer to that. Pairing separate renderings by known symbol would be supervision and must be declared. That is a separate experiment,
not another λ sweep. The review also pointed to the [Tolman–Eichenbaum Machine](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106/) as direction, not prediction, since its learned binding differs from concatenation.

**Cheaper alternatives.** The explicit store, frozen encoder and `random-features` are included. Replay
and protected expert expansion are outside this comparison; neither is claimed inferior.

## Scaling

F facts, H units, W code width, Q candidates, P weights of a comparator.

| Part | Storage | Per read or write | Limit |
|---|---|---|---|
| Wider code | linear in W | readers fixed; projection linear in W | whether identity scales with symbols, not just W |
| Reconstruction decoder | training only, at its own size | one forward and backward per image | model size; absent at deployment |
| Local memory | at most 4H(d + m + 1) bytes, only for allocated units | O(hd + h log k + km) with h units allocated, exhaustive | h grows about linearly with distinct symbols, by 1.2–1.7 units per symbol on the probe; search grows with h |
| Explicit store | linear in F | linear in F | the same in the worst case, and has fewer parts |
| Trainable memory | P + optimizer state | O(QP) | capacity is measured, not guaranteed |

- **Facts.** Storage follows what is taught. Similar cues can mix contents at any occupancy, and once H is
  reached forced writes add collisions quickly, with no forgetting to free units. The N = 18, 45, 90 load
  probes measure this; on the frozen-key probe, H = 128 already fills at τ = 0.85 on some settings. τ should also depend on
  load, since similar symbols grow more common as the pool grows.
- **Read work.** Exhaustive scoring gives no advantage over a slot store. An approximate index would, but
  would add its own structure and recall error. Not used here.
- **Answer space.** Scoring Q candidates does not survive a large answer space, and a mean-of-contents read
  has no way to hold several answers.
- **CPU.** [CPU viability](cpu-efficiency-design-constraint.md) must constrain the design, and this study does not establish it. Report
  inference, continued learning and initial construction separately. Inference includes the sense, the core and the memory search; continued
  learning is the write path alone. Local writes need no backward pass, and that is the part with a chance of being cheap on a CPU. Host
  fits run on the GPU and are construction cost. The host's `MaskedLinear` multiplies a dense tensor by its mask, so its executed work is
  dense, and its forward cost is reported as executed rather than by active connections.
- **What would not survive scale-up.** Nine counts, 90 symbols, marked pairs, curriculum-chosen material, a
  fixed frozen stationary host and pixel reconstruction of 28 × 28 images.

## Computation, storage and reproducibility

*Estimates, not measurements, except where stated.*

**Stage 0.** Two arms without λ and three with two values each, two pilot seeds: 16 host fits, 6,000 updates each. Time per
update comes from the long-term host, 318 s for 12,000 updates measured on the GTX 1650, so about 160 s per
fit before reconstruction's cost, which must be measured. Theoretical: host O(updates × batch × active
weights), plus O(images × decoder weights) per reconstruction update. Readings: O(symbols × renderings) forward
passes through a checkpoint, then O(renderings × symbols) cosine scores per identity read. Confirmation: the
candidate and `sparse48`, 20 seeds, 40 fits, plus one `perception` host or `W_K` fit per seed used by stage 1. Record
every fit, including rejected arms and λ, as selection cost.

Let F = 2N associations, E evaluation renderings per symbol, T = 4,000 teaching steps, B = 16 items per step,
H units, d address width, m content width, Q = 9 candidates and J the address cost.

- **Stage 1 construction:** no address generation for the candidate; plus measured host and representation
  construction. `fixed-random` adds O(Hd + Hm).
- **Addressing:** O(J + hd + h log k) per cue with h units allocated, h ≤ H, exhaustive scoring with a bounded
  heap. Name and cost any other selection algorithm.
- **Local read and write:** read O(km); write O(k(d + m)), with its address update; decoding adds its Q-dependent
  work. Training total at most O(TB(J + Hd + H log k + k(d + m))), with no backward pass. Trainable comparator costs
  include forward and backward operations, not just P.
- **Evaluation:** O(FE) queries per checkpoint, plus relational and unknown queries.
- **Persistent memory:** at most 4H(d + m + 1) bytes for the candidate, storing c_i and deriving a_i, plus all host, encoder and decoder parameters. Separate
  shared costs from incremental memory bytes. No retained raw examples in the candidate. The explicit store's 90 symbols
  at d = m = 32 take about 23 KB.
- **Temporary workspace:** at least O(BH + B(d + m) + Bk), plus core activations. Evaluation traces and trajectory
  caches are study workspace, priced separately.
- **Selection:** 18 local configurations and six trainable ones, each in two orders, times two pilot seeds and each address used, including the address control.
  Confirmation and load probes are separate line items. Account for any reuse.

**Stage 1 time, estimated.** One local run has 64,000 writes at about 512 × 32 + 4 × 48 operations each, under 10⁹ in all, plus 17 reads of about 1,300 queries each,
so seconds per run on a CPU once codes are cached. Selection is at most 18 × 2 orders × 2 seeds × 4 addresses = 288 local runs, minutes on a CPU. Trainable comparators are the larger cost,
per the long-term pilot's rates.

Follow [computation reporting](computation-reporting.md) in full. Keep scientific JSON deterministic and all time and
hardware observations in separate runtime files. Record source, contract, data and checkpoint hashes, command, dirty
revision, versions, dtype, CPU, GPU, BLAS, threads and workers, wall and aggregate worker CPU time by phase, RSS and
accelerator peak memory. Record examples, exposures, writes and candidates scored. Use null with a reason for unavailable
historical costs; do not retrofit frozen reports.

Benchmark read and write batches on the CPU, and on the GPU where it is used, after 20 warmups over 100 repetitions; report
median and interquartile range, with allocation, transfer and synchronization scope stated. Report quality against bytes, total construction, selection and
training cost and read latency, plus amortized cost for repeated queries.

Use independent RNG streams for host, background material, addressing, presentation and evaluation. Atomic checkpoints
include content, counts, cursors and RNG state; resumption must equal uninterrupted science. Reuse requires matching
provenance. Proposed output prefix: `reports/shared-core-sparse-memory-v1-*`; source and runner names will be declared with
implementation, not represented as existing.

## Before freezing

1. Run the value diagnostic: an oracle address returning the right symbol's accumulated live value through the actual decoder, beside perfect contents, to separate address failure from value and decoding failure.
1. Run the stage 1 and fixture pilots on the locked route, then select settings and set the stage 1 bars. On this route `sense-only` and `frozen-encoder` are one run, and `perception` is not used.
2. Record the cross-sense fixture's pilot reading of `conjunctive`, with its fallback.
3. Read adaptive resonance theory and Kanerva's sparse distributed memory, and record them in [prior work](prior-work.md), with the other [familiar] items.
4. Recheck seeds and lock host checkpoints, data and code hashes.

## Changes from external review

Review of this draft, 2026-11-01, all checked and fixed above:

- The protection condition was false for moving addresses: an existing nonwinning unit can move into the winning
  set. It now covers all competing addresses, and contents can mix before memory is full.
- Average-coordinate transfer could manufacture a trade-off, since appended coordinates lower its average. Preservation now
  uses answer-level `transfer8`, and coordinates are descriptive.
- Stage 0 confirmation appeared to choose stage 1's architecture on shared seeds. Route, host, address source and memory setting
  now lock on pilot seeds, a failed confirmation is a result, and conclusions concern the one candidate.
- Address comparisons did not isolate the core. Controls now remove only the core component or relational training.
- The dense-write control collapsed to one global average under either address rule. It is removed.
- Acquisition compared bounds with a measured S and had no floor. It now uses paired differences, a decoder floor and an
  absolute floor.
- Reverse order had no defined role. Old and new follow presentation order, both orders are primary, selection uses the
  worst order, and seeds resample whole.

Found while fixing these: per-script gates at ten seeds cannot pass for an unchanged arm, so stage 0 uses the mean of eight at
20 seeds, with script guards.

## Changes from pilot

Before the stage 0 pilot, 2026-11-02, with no pilot result seen:

- `W_K` is PCA whitening on unlabelled material, not next-input prediction.
- The fixture uses English, a derangement, its own setting per address and a defined rule for a sensory source with no taps.
- The lock rule now resolves λ and the tie between `split-recon-bg` and `wide32-recon-bg`.
- The stress condition defines a frozen sensory source, and it, the probes and the load probes are assigned to seeds 404–406.
- CPU reporting is stated by resource target. The Kanerva reference was corrected and the background set is 1,987 symbols.

After the stage 0 pilot, 2026-11-02:

- No arm addressed, so the route is `conjunctive`. An exploratory check at λ 10 and 100 did not change this: identity 0.18–0.32. The bar of 0.80 stays as provisional, since no arm came near enough to test it.
- The decoder is now cosine, with the head as a diagnostic.
- `conjunctive` reads the core part alone on a query with no sensory code.

The figures under *Why the first design changed* come from existing caches.
