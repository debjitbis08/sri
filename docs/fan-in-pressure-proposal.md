# Fan-in limits as pressure on the representation

Date: 2026-10-29. Status: **draft proposal; not a frozen contract**.
No implementation or experiments have been run for this proposal.
Replaces the bounded learnable locality proposal of the same day; its
ordered-input design is kept as a [deferred variant](#deferred-variant-locality-on-ordered-inputs).

## Question

Does limiting how many units each unit may combine arrange the shared code,
so that never-paired forms are placed and untaught relations transfer?

Each unit in the next layer combines input from the previous layer. In a
dense layer every unit may combine every previous unit. A weight *can* go to
zero, but training gives no reason for it to; the network keeps the freedom
and uses it. This proposal removes part of that freedom: each unit reads at
most k units of the previous layer, a hard limit held for the whole of
training. Which units it reads can be fixed or learned. The first study
fixes them, to test the limit itself.

## Why this is the next lever

The last three number-line pilots showed one pattern
([thinking pilot](think-pilot-v1-results.md#interpretation)): every change
that gave the answerer more freedom (generic comparator for heads, recurrent
loop for one pass) reduced placement and untaught transfer; objectives that
put pressure on the code (next-item prediction) recovered part of it.
Transfer comes from pressure on the representation, not from the power of
the reasoner.

Those pilots varied the *objective* and the *answerer's form*. This one
varies *how much freedom each layer has to combine its inputs*, in the
layers that build the code and in the layers that read it. It adds no
relation-specific heads or losses, and answers are still read from the
comparator, consistent with the
[architecture audit decisions](architecture-audit.md#researchers-decisions-2026-10-29).

Expressive power is not the question. A fan-in-limited network is a special
case of the dense one [derived]; the hypothesis is about what training finds
when the freedom is removed, not about what can be represented. The
[two-module expressivity analysis](two-module-graded-expressivity-analysis.md)
reaches the same conclusion for SRI: an advantage has to be a learning or
resource claim.

## Base network

The [sequence pilot](sequence-pilot-v1-results.md) network
(`src/generic_net.GenericSequenceNet`): the most recent heads-free network
without action tokens. It is the standard network's perception, a shared
16-dim code, the generic comparator `g(a, b, relation)` and the next-item
predictor `P(c_prev, c_cur)`. Data, losses, curriculum, rehearsal and
batches are unchanged.

Its dense layers, as (input → output):

| Part | Layer | Shape |
| --- | --- | --- |
| Front end, per sense | `fc` after the convolutions | 1568 or 1024 → 128 |
| Front end, per sense | `out` | 128 → 64 |
| Projection, per sense | `proj` (into the shared code) | 64 → 16 |
| Comparator `g` | first layer | 16 + 16 code + 8 token → 64 |
| Comparator `g` | second, third layer | 64 → 64 → 1 |
| Predictor `P` | first layer | 16 + 16 → 64 |
| Predictor `P` | second, third layer | 64 → 64 → 16 |

## Where the limit goes

Two sites, tested separately and together:

- **Encoder site: `out` and `proj`.** The last two dense layers before the
  code, `128 → 64 → 16`, are the progressively smaller layers of the
  original idea. Each unit reads `k = φ · n_in` units of the layer below.
  The convolutional layers and `fc` are perception and stay as they are.
- **Reader site: the first layers of `g` and `P`.** Each hidden unit reads k
  of the 16 code dimensions, and the *same* k dimensions of both operands,
  so that it compares like with like. The relation token stays fully
  connected: it names the question, it is not part of the representation.
  Later layers of `g` and `P` stay dense, so features of few code
  dimensions are combined freely afterwards.

Using the same subset for both operands is a declared design choice. It
biases each hidden unit toward comparing matching dimensions; independent
subsets per operand are not tested in the first study.

**Masks.** Each limited unit gets exactly k inputs. Subsets are drawn at
random but balanced, so that every input unit is read by `⌊m k / n⌋` or
`⌈m k / n⌉` units of the next layer and none is left unread. Masks come
from a generator seeded separately from the model and data, so batches and
draws stay paired by seed with the dense network. Masks are fixed for the
whole run.

**Initialization.** Standard initialization scales weights by the full
fan-in. Surviving weights are rescaled by `1/√φ` so each unit's input
variance matches the dense network's. The masked network's parameters
therefore do not start equal to the dense network's; only data and draws
are paired.

## Stage 1: does the limit help?

D-varied+rehearsal only, seeds 100–102, paired by seed with the sequence
pilot. Two fan-in fractions: φ = 1/4 and φ = 1/16.

| Configuration | Encoder `out` (of 128) | Encoder `proj` (of 64) | Readers (of 16) |
| --- | --- | --- | --- |
| Dense (rerun) | 128 | 64 | 16 |
| Encoder, φ = 1/4 | 32 | 16 | 16 |
| Encoder, φ = 1/16 | 8 | 4 | 16 |
| Readers, φ = 1/4 | 128 | 64 | 4 |
| Readers, φ = 1/16 | 128 | 64 | 1 |
| Both, φ = 1/4 | 32 | 16 | 4 |
| Both, φ = 1/16 | 8 | 4 | 1 |

7 configurations × 3 seeds = 21 runs. The dense rerun goes through the new
masked code with full masks and must reproduce the sequence pilot's
D-varied+rehearsal results exactly. That checks the implementation; if it
does not reproduce, stop and find out why before reading any other arm.

At readers φ = 1/16, each first-layer unit of `g` sees one code dimension of
each operand. `g` then becomes a learned combination of per-dimension
comparisons, which is the strongest form of the limit on the reading side.

## Stage 2: only if a limit helps

Run only if stage 1 meets the advance rule below, at the best site and φ:

| Arm | Question |
| --- | --- |
| Dense + L1 on the same layers, matched to the same edge count | Is soft pressure (which the model can pay to escape) enough? |
| Dense, pruned to k after training, then fine-tuned | Does pressure need to act *during* learning? |
| Learned subsets, budget k held throughout (RigL-style) | Does choosing which units combine add to the limit itself? |
| DropConnect at rate 1 − φ | Is random, changing sparsity enough, or must it be fixed? |
| All three curriculum arms, more seeds | Does the effect hold beyond the pilot setting? |

Fixed subsets in stage 1 and learned subsets in stage 2 separate the two
questions: does the limit help, and does learning *which* units combine add
anything.

## Confirmation after stage 1 (written before running)

Added 2026-10-29 after [stage 1](fan-in-pilot-v1-results.md). No configuration met the
advance rule. Readers at φ = 1/16 improved untaught comparison on every script (Chinese
0.58 → 0.81 in every seed), but the result confounds the limit with tying the same code
dimension across both operands. The researchers also asked whether learning *which*
dimensions each unit reads helps, compared with random choices. This confirmation tests
the one observed configuration and does not search further φ values or sites.

**Seeds 200–209 (fresh), D-varied+rehearsal, readers only (first layers of `g` and `P`),
one code dimension per operand per hidden unit.**

| Arm | Selection | Question |
| --- | --- | --- |
| `dense` | none | No-pressure reference on fresh seeds |
| `read-q16` | random, tied (same dimension for both operands) | Does the stage-1 effect replicate? |
| `read-q16-untied` | random, independent dimension per operand | Does the limit help without tying? |
| `learn-q16-tied` | learned, tied | Does learning which dimension beat random, with tying? |
| `learn-q16-untied` | learned, independent per operand | Does learning beat random without tying? Does it learn to tie? |

**Learned selection.** Each hidden unit keeps a scalar weight per operand and a vector of
16 selection logits per operand (one shared vector when tied). The forward pass always
reads exactly one dimension, the argmax, so the limit holds throughout training. Gradients
reach the logits through a straight-through softmax (temperature 1): the backward pass
treats the selection as `softmax(logits)`. Logits start at 0 with +1 on the random arm's
dimension for the same seed, and the scalar weights start at the random arm's rescaled
weights, so each learned arm starts from exactly the same function as its random
counterpart and differs only by learning. Logits train with the same Adam optimizer and
learning rate (1e-3) as everything else.

**Primary contrasts and rule.** With ten seeds, a contrast is **better** (or **worse**) if
the mean paired difference exceeds the margin (0.05 for accuracies) and the 95% paired
t-interval excludes zero; **equivalent** if the interval lies within ±margin; otherwise
inconclusive. Primary measure: untaught comparison on Chinese forms. Secondary: the other
seven primary measures of stage 1, the fit check and the probe.

1. **Replication:** `read-q16` against `dense`.
2. **Limit without tying:** `read-q16-untied` against `dense`, and against `read-q16`.
3. **Learned grouping:** `learn-q16-tied` against `read-q16`; `learn-q16-untied`
   against `read-q16-untied`.

**Mechanism diagnostics for learned arms:** the fraction of units whose selected dimension
changed from initialization, and, for `learn-q16-untied`, the fraction of units reading
the same dimension of both operands (chance about 1/16 under random selection).

**Predictions:**
- Replication: `read-q16` better than dense on Chinese comparison, smaller than in stage 1.
- Untied: weaker than tied; much of the stage-1 gain comes from tying.
- Learned against random: equivalent or inconclusive. Straight-through selection is
  noisy, and the code's variance lies mostly along one direction (participation ratio
  about 1.1–1.4), so many dimension choices may be similarly useful. If tying is what
  helps, `learn-q16-untied` should move above chance toward reading the same dimension of
  both operands.

**Prior work for learned selection [familiar; not read today]:** straight-through
estimators (Bengio, Léonard and Courville 2013); Gumbel-softmax and concrete relaxations
(Jang et al. 2017; Maddison et al. 2017); concrete autoencoders, which learn to select k
input features (Abid et al. 2019). Known failure modes: biased straight-through gradients,
selections that rarely switch, or that oscillate. A cheaper alternative for choosing
connections is magnitude-based prune-and-regrow (RigL).

**Cost:** 50 runs, about 900–980 s of worker CPU each at 11 concurrent workers (stage 1),
so about 13 hours of CPU and about 80–90 minutes of wall time in five waves. Learned
arms add 16 logits per operand per hidden unit (2,048 per layer untied, 1,024 tied).

Commands: `NSY_WORKERS=11 .venv/bin/python -m experiments.fan_in_pilot pilot confirm`,
then `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot confirm`.

## Learned grouping (written before running)

Added 2026-10-30 after the [confirmation](fan-in-pilot-v1-confirm-results.md). Its learned
selection hardly moved (1–5% of units switched), so it did not test whether learning which
units combine helps. This experiment uses an established method that is built to change
connections, in two settings where the choice should matter.

**Method: prune and regrow (RigL; Evci et al. 2020, [abstract](https://arxiv.org/abs/1911.11134)).**
Each limited unit keeps exactly k incoming connections throughout training. Every 100
updates, until 75% of training (update 4,500 of 6,000), each unit drops its
`max(1, round(α_t k))` active connections with the smallest weight magnitude and grows the
same number of inactive connections with the largest magnitude of the loss gradient at
that update. `α_t` decays from 0.3 to 0 by a cosine schedule. Grown weights start at zero
with zeroed Adam moments. If a dropped connection is regrown at the same update, its
weight and moments are restored (no change). Readers apply this per operand (k per
operand). The interval, fraction, schedule and end point are the RigL paper's defaults
**[familiar; the abstract does not state them]**, fixed here, not tuned. The Thai transfer
phase keeps the final masks.

**Settings where the choice should matter.**
- **Encoder at φ = 1/16** (`out` 8 of 128, `proj` 4 of 64). Random limits here harmed
  placement and comparison in stage 1. If learned grouping matters, choosing which
  front-end features to combine should remove some of that harm.
- **Readers at φ = 1/4, untied** (4 of 16 code dimensions per operand per unit). RigL
  needs more than one connection per unit to swap among, so the one-dimension setting
  cannot be used.

**Arms, seeds 210–219 (fresh), D-varied+rehearsal:**

| Arm | Question |
| --- | --- |
| `dense` | Reference |
| `enc-q16` | Random encoder limit (repeats stage 1's harmful setting on fresh seeds) |
| `enc-q16-rigl` | Same limit, learned grouping |
| `read-q4-untied` | Random reader limit, 4 per operand |
| `read-q4-untied-rigl` | Same limit, learned grouping |

The random and RigL arms start from the same masks and weights for each seed.

**Runtime settings, changed for speed:** 6 workers × 2 threads, and channels-last
activations in the image encoder. On short runs this gave 1.5× the throughput of 11 × 1
in the earlier layout, and repeated runs were identical. It changes floating-point
rounding, so no arm is compared with the earlier experiments' checkpoints; every arm here
is rerun.

**Primary contrasts** (paired 95% t-interval rule of the confirmation, margins 0.05 for
accuracies and 0.03 for AUROCs):
1. `enc-q16-rigl` against `enc-q16`, and `read-q4-untied-rigl` against `read-q4-untied`,
   on the eight transfer measures of stage 1 and on the four taught measures.
2. Each limited arm against `dense`, as context.

No single primary measure is declared: the question is whether learned grouping changes
transfer at all, so all eight transfer measures are reported with their intervals, and
the conclusion counts how many are better, worse, equivalent or inconclusive.

**Mechanism diagnostics:** the fraction of final connections that differ from the
initial mask; the number of drop/grow swaps that changed a connection; for readers, the
mean overlap between the two operands' dimension sets (random expectation 1 of 4).

**Predictions:**
- RigL matches or beats random sparsity on taught fit; this is the established result for
  accuracy at fixed density.
- Encoder: RigL reduces the random limit's harm to transfer, because a random choice of 4
  of 64 front-end features can miss what is needed. Moderately likely.
- Readers: little difference. The code's variance lies mostly along one direction spread
  over its 16 dimensions, so most subsets carry count. If tying helps after all, reader
  overlap rises above 1.
- Either way, a large share of connections should change (more than 30%), unlike the
  straight-through selection.

**Cost:** 50 runs. At about 330 s of wall time per run with 6 concurrent, about 50
minutes. RigL adds one dense gradient per limited layer per update (the masked layer's
gradient is computed before masking) and a top-k per unit every 100 updates.

Commands: `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot rigl`,
then `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot rigl`.

## Learned grouping at one dimension per operand (written before running)

Added 2026-10-30 after the [learned-grouping results](fan-in-pilot-v1-rigl-results.md), at
the researchers' request: RigL where the reader limit helped in the
[confirmation](fan-in-pilot-v1-confirm-results.md), one code dimension per operand.

**Method.** RigL as above, with k = 1 per operand. At each connectivity update every unit
drops its one connection per operand and regrows the dimension with the largest gradient
magnitude, the dropped one included; if the dropped dimension wins, nothing changes. A
switched connection starts at weight zero. Interval, schedule and end point as above.

**Arms, seeds 220–229 (fresh), D-varied+rehearsal, runtime settings as above:**

| Arm | Question |
| --- | --- |
| `dense` | Reference |
| `read-q16-untied` | Random reader limit, one dimension per operand (the confirmation's untied arm, fresh seeds) |
| `read-q16-untied-rigl` | Same limit, learned grouping, same starting masks |

**Contrasts** (paired 95% t-interval rule): RigL against random on the eight transfer
and four taught measures; each limited arm against dense as context, which also
re-tests the confirmation's untied gains on new seeds.

**Diagnostics:** the fraction of final connections not in the initial mask, connections
grown, and the fraction of units reading the same dimension of both operands (random
about 1/16).

**Predictions:**
- RigL against random: no difference, as in the other two settings.
- Random untied against dense: the confirmation's gains recur in direction (Chinese and
  English comparison, Devanagari placement, isolated judgment), with some measures not
  passing the rule.
- Switching is frequent: with one connection per operand, each update compares it
  against all 15 alternatives.

**Cost:** 30 runs, 6 workers × 2 threads, about 350 s each: about 30 minutes.

Commands: `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot rigl1`,
then `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot rigl1`.

## Replication with a fixed coordinate (written before running)

Added 2026-10-30 after the [one-dimension RigL results](fan-in-pilot-v1-rigl1-results.md),
at the researchers' request. There, RigL's live reader units converged on one code
coordinate per seed, and untaught transfer rose sharply. Two questions:

1. **Does it replicate** on fresh seeds?
2. **Is learning needed, or is concentration enough?** If every reader unit is wired to
   one coordinate chosen in advance, with no learning, does the same transfer appear? The
   fixed arms also have no zero-initialized connections, so dead units from RigL's
   regrowth cannot occur; this separates concentration from RigL's loss of reader
   capacity.

**Arms, seeds 230–239 (fresh), D-varied+rehearsal, runtime settings as above:**

| Arm | Question |
| --- | --- |
| `dense` | Reference |
| `read-q16-untied` | Random reader limit, one dimension per operand |
| `read-q16-untied-rigl` | RigL from the same masks (the replication) |
| `read-q16-fixed` | Every first-layer unit of `g` and `P` reads coordinate d of both operands |
| `read-q16-fixed75` | 48 of 64 units per reader read d of both operands; the other 16 keep the random arm's masks |

- **The coordinate d** is drawn once per seed (`fanin_mask` stream, `fixed-coordinate`),
  uniform over the 16 code dimensions, and is the same for `g` and `P`, as RigL found.
  Nothing chooses it from data.
- **Fixed arms** start from the random arm's masks, with the chosen units' code columns
  replaced. The kept fraction per unit (2 of 32) and therefore the rescaling are
  unchanged. The masks are held for the whole run.
- **75%** approximates RigL's final state in `g` (65–79% of live units' code connections
  on the shared coordinate in the three seeds checked). It was not tuned.
- The `fixed` arm turns each reader's first layer into a function of one scalar per item
  (plus the relation token in `g`). Later layers cannot see the other 15 coordinates.
  The code's other coordinates are still trained, through `P`'s prediction target (the
  squared distance to collection codes uses all 16).

**Contrasts** (paired 95% t-interval rule, eight transfer and four taught measures):
RigL − random and RigL − dense (replication); fixed − RigL and fixed75 − RigL (is
learning needed); fixed − dense, fixed75 − dense, fixed − fixed75.

**Pre-declared readings:**
- **Replicated** if RigL − random is better on taps → withheld Chinese forms and on English
  comparison, the two largest effects in rigl1.
- **Concentration is enough** if `fixed` or `fixed75` is equivalent to or better than RigL
  on both of those measures. **The learned choice matters** if both fixed arms are worse
  than RigL on either. Otherwise unresolved.

**Diagnostics:**
- Reader units never active (dead), by arm.
- Among live units only: the fraction reading the same dimension of both operands, and
  the share of code connections on the most-read dimension. This replaces rigl1's
  overlap measure, which counted dead units.
- Code structure: the share of collection-code variance on the first principal
  direction and on the readers' most-read coordinate; the cosine between the two; and
  the offset of each form set's mean from the collections' mean outside that coordinate,
  in collection standard deviations along it. These were measured ad hoc on three rigl1
  seeds and are now computed for every checkpoint.

**Predictions:**
- RigL replicates in direction on both measures. Expected to pass on taps → withheld
  Chinese; English comparison may be smaller on fresh seeds.
- `fixed` loses more taught matching than RigL. A single scalar per item has to carry
  every taught distinction, including Chinese paired matching, which fell under RigL.
- `fixed75` comes closest to RigL on transfer. Weak expectation: transfer with fixed
  concentration is lower than RigL, because RigL's coordinate is chosen where count
  already lies, and a random d must be built up from scratch.
- No dead units to speak of in the fixed arms (under 10%).

**Prior work.**
- A single scalar read by every relation is the classic accumulator or
  mental-number-line model of magnitude (Gallistel and Gelman; Dehaene) **[familiar]**,
  and the linear `larger` head this project removed. The fixed arm imposes the axis but
  not its content; whether that counts as hand-given structure is the researchers' call.
- In machine learning it is a one-dimensional bottleneck between encoder and readers
  (information bottleneck; ordinal regression on one latent score) **[familiar]**. These
  predict that a scalar bottleneck aligns the code on one axis, at a cost to any task
  that needs more than one axis. That matches rigl1's costs.
- RigL as above. No source read today predicts whether a learned or a fixed coordinate
  transfers better.

**Cost:** 50 runs, 6 workers × 2 threads, about 345 s each: about 50 minutes. Analysis
about 1 minute.

Commands: `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot fixed`,
then `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot fixed`.

## Timing or choice of the coordinate (written before running)

Added 2026-10-30 after the [fixed-coordinate results](fan-in-pilot-v1-fixed-results.md), at
the researchers' request. A coordinate fixed from the start matched RigL on Chinese forms
but not on English or Devanagari. The fixed arm differs from RigL in two ways:

- **Timing.** RigL starts from spread-out random readers and concentrates gradually. The
  fixed arm is concentrated from the first update, before English (the first block) is
  learned.
- **Choice.** RigL's coordinate is found from gradients, probably where count already
  lies. The fixed arm's is drawn without data.

**Design: a 2 × 2 on seeds 230–239, paired with the fixed-coordinate run.**

| | Coordinate drawn without data | RigL's coordinate for the seed |
| --- | --- | --- |
| **Concentrated from the start** | `read-q16-fixed` (rerun) | `read-q16-fixed-rigld` |
| **Concentrated gradually** | `read-q16-gradual` | `read-q16-gradual-rigld` |

Also `dense` and `read-q16-untied-rigl`, rerun. The three rerun arms must reproduce the
fixed-coordinate run exactly; the analysis checks this.

- **RigL's coordinate** is the live `g` units' most-read dimension of
  `read-q16-untied-rigl` for the same seed in the fixed-coordinate run (5, 1, 15, 7, 13, 12,
  7, 2, 9, 2). It is taken from a finished run, so it uses information the network does
  not have at the start: an oracle for "where count settles in this seed". The drawn
  coordinates are 13, 1, 6, 13, 3, 6, 10, 8, 2, 2. **In seeds 231 and 239 the two are
  the same**, so the start arms are identical there, and so are the gradual arms. Eight
  seeds carry the choice contrast.
- **Gradual schedule.** The gradual arms start from the random arm's masks. At each of
  RigL's connectivity updates (every 100 updates until update 4,500), units move onto the
  coordinate, in a fixed random order per reader, so that the moved share follows the
  integral of RigL's cosine drop fraction: share(x) = x + sin(πx)/π at fraction x of the
  update period. That is 27 of 64 units by update 1,000, 49 by 2,000 (end of the English
  block), 60 by 3,000, and all 64 by 4,500. A moved unit keeps its two code weights,
  carried to the new columns; their Adam moments are reset. No weight starts at zero, so
  RigL's dead units are not reproduced.
- The share follows RigL's change budget, not its measured concentration curve, which was
  not recorded.

**Contrasts** (paired 95% t-interval rule):
- timing: gradual − fixed (drawn coordinate), gradual-rigld − fixed-rigld;
- choice: fixed-rigld − fixed, gradual-rigld − gradual;
- each new arm − RigL.

**Pre-declared readings,** on taps → withheld Chinese forms and English comparison (the
two measures of the earlier rules) and Devanagari comparison (where the fixed arm also
fell short):
- **Timing matters** if gradual is better than start for at least one coordinate on
  English or Devanagari comparison.
- **Choice matters** if RigL's coordinate is better than the drawn one, at either timing,
  on English or Devanagari comparison.
- **RigL is explained** if `gradual-rigld` is equivalent to or better than RigL on all
  three measures. If no new arm reaches RigL, something else in RigL matters (the
  partial concentration, its dead units, or the order in which units move).

**Predictions:**
- Timing matters: gradual arms improve English comparison over the start arms, because
  English is placed while readers still see several dimensions. Moderate confidence.
- Choice matters less: with a drawn coordinate, collections already put 47% of their
  variance on it, as under RigL.
- `gradual-rigld` comes close to RigL but stays below it on English comparison.
- Chinese gains of the fixed arm remain in all four cells.

**Prior work.** Gradually increasing a constraint during training is curriculum or
continuation (Bengio et al. 2009) and gradual pruning schedules (Zhu and Gupta 2017)
**[familiar]**. These predict that late or gradual constraints keep more of what was learned
early than constraints imposed from the start. No source read today predicts the effect on
transfer.

**Cost:** 60 runs, 6 workers × 2 threads, about 345 s each: about 60 minutes on CPU.
Analysis about 1 minute.

Commands: `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot timing`,
then `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot timing`.

## Measures

The sequence pilot's measures, scored identically
(`experiments.transition_pilot_analysis`):

- **Primary (untaught transfer):**
  - withheld Chinese and Devanagari forms placed at the end of their own
    blocks;
  - comparison on Chinese, Devanagari and English forms;
  - taps → withheld Chinese and Devanagari forms; form → taps, Chinese
    withheld.
- **Fit check (taught):** taps → collection, Chinese paired at its own
  block, comparison and succession on collections.
- **Secondary:** retention, judgment revision, next-judgment coherence,
  4-shot Thai, composition through `P`.
- **Arrangement probe (diagnostic, not an answer):** a ridge regression of
  count from collection codes, fitted on collections only and applied
  unchanged to form and taps codes. It measures whether forms sit where
  collections of the same count sit, independently of `g`. Probe labels
  never update the network.
- **Mechanism diagnostics:** effective edge count per limited layer,
  dead units (no gradient for the last 10% of training), and code
  rank/variance.

## Predicted outcomes

Written before any run.

- **Most likely overall: no gain.** Random sparse networks at moderate
  sparsity typically match dense accuracy; nothing known to us shows fan-in
  limits improving systematic transfer [familiar]. A null result is
  informative here.
- **Readers more than encoder.** A reader that can only compare a few
  dimensions at a time cannot answer taught questions by arbitrary mixtures
  of the code, so the code must put count on consistent dimensions.
  Expected: readers φ = 1/4 better than dense on untaught comparison and
  placement, if any arm is.
- **Encoder at φ = 1/4: equivalent to dense.** Two layers of 32 and 16
  inputs per unit still let information reach the code from anywhere.
- **φ = 1/16: taught fit drops.** At 8, 4 and 1 inputs per unit the
  network is expected to underfit somewhere; transfer then falls with it.
  If transfer rises while fit stays, the limit is acting as pressure.
- **Arrangement probe tracks placement.** If placement improves without the
  probe improving, the gain is in how `g` reads, not in the code.

## Decision rules

Margins as in the audit: 0.05 for accuracies, 0.03 for AUROCs, paired
differences against the dense rerun.

- **Fit first.** A configuration whose taught measures fall by more than
  the margin is labelled underfit. Its transfer results are reported but
  do not count for or against the limit.
- **Advance to stage 2** if a configuration that is not underfit is better
  on at least two primary measures and worse on none.
- If only the arrangement probe improves, report that and do not advance.
- If every configuration is equivalent or underfit, the fan-in limit does
  not supply arranging pressure on this network. Report it; do not search
  further φ values to find a winner.

This is a pilot, three seeds, not headline evidence.

## Prior work and expected limits

Search/read date: 2026-10-29. Read today: abstracts of Evci et al. (2020)
and Khalfaoui-Hassani et al. (2023), recorded in
[prior work](prior-work.md#fan-in-pressure). Everything else is from memory,
marked **[familiar]**, and must be read before a contract is frozen.

- **Pruning and sparse training.** Magnitude pruning, lottery tickets and
  dynamic sparse training (RigL, [abstract](https://arxiv.org/abs/1911.11134);
  sparse evolutionary training, Mocanu et al. 2018 [familiar]) show sparse
  networks matching dense accuracy at moderate sparsity. They target
  efficiency and in-distribution accuracy, not transfer. Predicts: fit is
  preserved at φ = 1/4; says nothing about placement.
- **Random sparsity at initialization.** Fixed random sparse networks
  trained from scratch can match dense ones when wide enough
  (Liu et al. 2022, *The unreasonable effectiveness of random pruning*
  [familiar]). Supports fixed random masks as a fair first arm.
- **DropConnect and dropout** (Wan et al. 2013; Srivastava et al. 2014
  [familiar]). Random, changing masks regularize but tend to spread
  information redundantly across units, the opposite of arranging it.
  A stage-2 control.
- **L1 and weight decay [familiar].** Soft sparsity pressure; the model can
  pay the penalty to keep a connection. A stage-2 control.
- **Sparse coding and bottlenecks [familiar].** Limits on activity or width,
  not on connections. Our limit is on connections; width is unchanged.
- **Additive and factorized readouts [familiar].** At readers φ = 1/16, the
  first layer of `g` resembles a model built from per-dimension
  comparisons. Such readouts generalize systematically when the relevant
  quantity lies on fixed dimensions, and fail when it is spread across
  them. This is the mechanism by which reader limits could arrange the
  code.
- **Expected limits [derived].** The limited network is a special case of
  the dense one; any effect is on what training finds. Later dense layers
  can recombine everything, so the limit is partial by design. Fixed
  random masks can cut a needed path; balanced masks reduce but do not
  remove that risk.

Known failure modes: underfitting at small φ, dead units, the rescaled
initialization changing optimization rather than the limit itself (the
dense rerun keeps standard initialization; a φ = 1 run with rescaling is
identical to it, so this cannot be separated without a further control),
and seed variance larger than the effect with three seeds.

## Computation plan

Per limited layer with n inputs, m outputs and fan-in k:

- **Implementation:** dense weights times a fixed binary mask. Forward and
  backward cost O(B n m) per batch, the same as the dense layer; a sparse
  kernel would cost O(B m k). Report that the masked-dense form is used.
- **Storage:** weights, gradients and Adam moments as for the dense layer,
  plus one n × m mask per limited layer. Effective parameters
  (unmasked weights):

| Layer (per sense where applicable) | Dense | φ = 1/4 | φ = 1/16 |
| --- | --- | --- | --- |
| `out`, 128 → 64 | 8,192 | 2,048 | 512 |
| `proj`, 64 → 16 | 1,024 | 256 | 64 |
| `g` first layer, code part | 2,048 | 512 | 128 |
| `P` first layer | 2,048 | 512 | 128 |

- **Mask construction:** O(n m) per layer, once per run.
- **Runs:** 21, about 612 s of worker CPU each (the sequence pilot's
  per-run figure), so about 3.6 hours of CPU and two waves on 11 workers,
  about 25 minutes of wall time. Analysis about 40 s.
- **Arrangement probe:** one ridge fit per checkpoint on 16-dim codes;
  negligible.
- **Selection and tuning:** none. φ values, sites and the rescaling rule
  are fixed here, not searched.

Follow [computation reporting](computation-reporting.md): record worker
CPU, wall time, peak RSS, versions, hashes and dirty status in runtime JSON,
separate from the deterministic results. No measurements exist yet.
Proposal preparation time was not instrumented and is unavailable.

## Deferred variant: locality on ordered inputs

A second form of the limit fixes the subsets by position: each unit reads
only nearby units. It needs inputs with a real order (time, space), so it
does not apply to the number network's code, whose 64 and 16 coordinates
have no meaningful order.

The earlier design, kept for later: an ordered synthetic fixture with local
motifs, paired views with independent nuisance, a `64 → 32 → 16` local
encoder with hard bound R and k learned interpolated sampling positions per
unit. Its review found that the edge budget 2k must be well below the
window (for example R = 8, k = 2: 4 edges of 16 sites), that linear
interpolation gives position gradients reaching only adjacent sites (DCLS,
[abstract](https://arxiv.org/abs/2306.00817), moved to Gaussian
interpolation for this reason), that relations probed linearly must be
linearly readable from a witness code, and that a CNN is expected to win
when motifs shift per example. Its first comparison would be learned
positions against fixed local subsets and a CNN, all with the limit held
throughout training.

Take this up only if fan-in limits help on the number network, or if a
task with ordered inputs becomes the focus.

## Work required before execution

1. Implement masked linear layers, balanced mask construction and the
   rescaled initialization; check that full masks reproduce the sequence
   pilot exactly.
2. Implement the arrangement probe and the mechanism diagnostics.
3. Read the [familiar] sources above and record them in prior work.
4. Run stage 1; write results; decide on stage 2 by the rules above.
