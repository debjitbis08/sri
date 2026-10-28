# Number systems meta v1: pilot results

Date: 2026-10-28. Specification: [number systems meta v1](number-systems-meta-v1-spec.md)
(draft; not frozen). Command: `NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta pilot`.
Artifacts: `reports/number-systems-meta-v1-pilot-results.json.gz`, `-runtime.json`;
checkpoints in gitignored `runs/`. One seed (100). **Pilot: descriptive, not headline
evidence.** No setting was tuned; all are the spec's defaults.

## Source training at equal budget

| Arm | Source steps | Episodes | Training encodings | Training CPU (s) | Labelled accuracy | Complementary | Cross-representation probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary | 3,000 | 0 | 5.376 M | 1,143 | 0.93 | 0.11 | 0.27 |
| ordinary + dev | 3,000 | 0 | 5.376 M | 1,184 | 0.87 | 0.11 | 0.24 |
| meta | 1,657 | 413 | 5.375 M | 1,182 | 0.90 | 0.06 | 0.23 |
| meta-I | 1,657 | 413 | 5.375 M | 1,150 | 0.93 | 0.09 | 0.18 |

- Budgets match in encodings, and training CPU time agrees within 4%.
- The episodes cost meta about 45% of its source steps. Source label accuracy held up;
  the cross-representation probe dropped slightly (0.23 against 0.27).
- **Episode losses barely fell.** Over training, the query loss went from 3.04 to 2.58,
  and the preservation loss (two sources after adaptation) from 6.06 to 5.76. Ten
  satellite updates at learning rate 1e-2 still wreck the sources at the end of training.

## Few-shot adaptation (interface mode, learning rate 1e-2, up to 200 updates)

Best accuracy within the budget, means over tree, german and thai_digits:

| Examples per number | No source | ordinary | ordinary + dev | meta | meta-I |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.17 | 0.23 | 0.22 | 0.24 | 0.25 |
| 8 | 0.22 | 0.33 | 0.34 | 0.33 | 0.38 |
| 32 | 0.31 | 0.55 | 0.55 | 0.61 | 0.61 |

At the episode budget (10 updates), means over image held-outs and k:

| | ordinary | ordinary + dev | meta | meta-I |
| --- | --- | --- | --- | --- |
| held-out accuracy | 0.13 | 0.15 | 0.19 | 0.22 |
| source retention (after / before) | 0.25 | 0.28 | 0.24 | 0.40 |

- **Meta adapts somewhat faster in the regime it was trained for.** At 10 updates it
  reaches 0.19–0.22, against 0.13–0.15 for the ordinary arms. At 32 examples per number
  its best accuracy is 0.61 against 0.55 for both ordinary arms, so the gain is not just
  the extra development data. The clearest single case: thai_digits at k = 32 reaches
  0.73 with meta, against 0.58–0.59 for the ordinary arms.
- **The gain does not depend on the shared order.** Meta-I matches or exceeds meta.
- **Retention fails everywhere.** Interface adaptation at learning rate 1e-2 cuts the
  sources' labelled accuracy to 0.17–0.34 of its starting value after 200 updates in every
  arm. No image run meets the retention requirement. The preservation term did not
  protect the sources for meta. Meta-I keeps more at 10 updates (0.40); with one seed I
  can't tell whether that is real.
- Hand (sequence medium, no episodes) reaches the accuracy criterion at k = 8 and 32 in
  every arm. **It fails retention badly.** *Correction (external review):* an earlier
  version gave a retention of about 0.93. That is the mean over 11 sources, of which only
  Morse is affected. In interface mode the 10 image sources are exactly unchanged, so
  the change in the mean is Morse's alone:
  - Morse's accuracy on its labelled numbers falls from 1.00 to about 0.33 in every arm
    and every k (a drop of 0.66–0.68);
  - a drop of exactly two thirds suggests Morse collapses onto a single label.
  
  The image held-out retention figures above are means over 11 sources, 10 of them
  affected, so they are nearly the affected-medium values. Later runs report retention
  per source.

## Cost to criterion

The criterion was fixed as the no-source accuracy at k = 32 after 200 updates:
- tree 0.37, thai_digits 0.39, hand 0.74;
- **german 0.16, which is degenerate.** No source barely learns German, so every arm
  meets it before any update. German cost to criterion is uninformative.

| Held out, k | No source | ordinary | ordinary + dev | meta | meta-I |
| --- | --- | --- | --- | --- | --- |
| tree, 8 | > 200 | > 200 | 50 | 50 | 30 |
| tree, 32 | 200 | 40 | 20 | 50 | 30 |
| thai_digits, 8 | > 200 | > 200 | > 200 | > 200 | 50 |
| thai_digits, 32 | 200 | 75 | 75 | 40 | 30 |
| hand, 8 | > 200 | 75 | 100 | 100 | 50 |
| hand, 32 | 200 | 75 | 100 | 50 | 30 |

Updates to criterion, accuracy only. With the retention requirement, every image cell is
> 200.

Meta-I is fastest in every row except tree at k = 32, where ordinary + dev is faster (20
against 30). It is also fastest on Hand, the medium without
episodes, so its advantage is not specific to the trained image satellite. One seed and
coarse scoring points do not support a ranking.

## Reading

- Within this pilot, the learning-efficiency objective did speed up adaptation a little at
  an equal budget. The data-matched control shows it is not just the development data. It
  did not help retention, and its benefit did not depend on compatible number order.
- The preservation term failed because the inner adaptation is too destructive. The
  episodes' preservation loss stayed near its starting value. The shared-satellite
  setting may need a smaller inner learning rate, a larger λ, or more episodes before the
  term can work. Those are the tuning choices the spec leaves to development
  representations.
- Shared number structure is still not doing measurable work in learning cost: meta-I
  ≥ meta here, and I14 ≥ C14 in Stage A.
- Design faults to fix before more runs:
  1. A no-source criterion breaks when no-source does not learn (German). Use a fixed
     accuracy level, set per representation from the development representations, or
     report updates to several levels.
  2. The scoring grid is coarse at low update counts.
  3. One seed.

## Gradient diagnostic (development only)

The external review raised a question: does the episode's gradient carry the intended
signal at all? The pilot's episode is first-order. Write θ for the parameters outside the
satellite and φ' = U(φ, θ) for the adapted satellite. The full gradient is

> dJ/dθ = ∂J/∂θ + (∂φ'/∂θ)ᵀ ∂J/∂φ'

and the pilot keeps only the first term. It changes the core so that the
already-adapted satellite does better, not so that the satellite adapts differently.

- **Command:** `NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta_gradient pilot`.
- **Artifacts:** `reports/number-systems-meta-v1-pilot-gradient-results.json.gz`, `-runtime.json`.

### Setup

- Checkpoints: meta and ordinary.
- Development representations only, 2 episodes each.
- Float64.
- **The inner loop is 5 steps of plain SGD at α ∈ {0.003, 0.01},** so the unrolled update
  can be differentiated. This differs from the pilot's 10 Adam steps at 1e-2.

For each episode:
- the exact gradient through the inner steps, and the first-order estimate;
- central finite differences along 3 random directions and along both gradients;
- outer steps of length η ∈ {0.01, 0.03, 0.1} along each unit-normalized gradient,
  scored on 6 fresh development episodes. Scores: query loss and accuracy after
  adaptation, preservation loss, and the image sources' labelled accuracy before and
  after adaptation.

### Findings

**1. The exact gradient is correct, but only locally.**
- At ε = 1e-7, finite differences match the exact directional derivatives with a median
  relative error of 2–5 × 10⁻⁶.
- At ε = 1e-5 the median error is 0.7–1.0.
- With zero inner steps the objective is smooth along the same line. After one inner
  step it shows small jumps; after five, jumps of about 0.08 in J over 2 × 10⁻⁵ along the
  gradient.
- One candidate mechanism is the ReLU front end: each unit's gradient changes abruptly
  at its on/off point, and each inner step moves the satellite by α times that gradient.
  The test does not separate this from high curvature or unstable adaptation, which
  could produce similar finite-difference behaviour.
- At the step sizes training uses, any gradient of this objective, exact or first-order,
  may therefore be a weak guide. The original MAML experiments nonetheless used ReLU
  networks successfully (Finn et al. 2017, sections 5.1–5.2), so ReLU does not by itself
  rule out useful second-order credit.

**2. The first-order estimate is not a small perturbation of the exact gradient.**

| Checkpoint, α | Cosine, satellite | Cosine, shared parameters | Norm ratio exact / first-order (shared) |
| --- | --- | --- | --- |
| meta, 0.003 | 0.00 | 0.21 | 1.6 |
| meta, 0.01 | 0.17 | 0.21 | 1.5 |
| ordinary, 0.003 | 0.21 | 0.64 | 2.7 |
| ordinary, 0.01 | 0.04 | 0.01 | 13 |

- Along random directions, the first-order estimate misses the true directional
  derivative by a median relative error of 1.1–1.7.
- So the omitted term dominates. That includes the part that reaches the shared core.

**3. Small outer steps: the exact direction reduces forgetting a little from the meta
checkpoint; neither direction helps from the ordinary one.**

Means over 6 episodes × 6 fresh episodes, as changes from no step:

| Checkpoint, α | Direction, η | Preservation loss | Source accuracy after adaptation |
| --- | --- | --- | --- |
| meta, 0.003 | exact, 0.03 | −0.021 | +0.021 |
| | first-order, 0.03 | +0.104 | −0.036 |
| meta, 0.01 | exact, 0.03 | −0.141 | +0.035 |
| | first-order, 0.03 | +0.076 | −0.015 |
| ordinary, 0.003 | exact, 0.03 | +0.094 | −0.034 |
| | first-order, 0.03 | +0.107 | −0.045 |

- Source accuracy after adaptation mixes two effects: the outer step's change to source
  accuracy before adaptation, and forgetting during adaptation. Forgetting alone, measured
  as before minus after (changes from no step, η = 0.01–0.03):

  | Checkpoint, α | exact | first-order |
  | --- | --- | --- |
  | meta, 0.003 | −0.014, −0.019 | −0.009, +0.001 |
  | meta, 0.01 | −0.005, −0.047 | −0.001, −0.008 |
  | ordinary, 0.003 | +0.036, +0.027 | +0.077, +0.012 |
  | ordinary, 0.01 | +0.067, +0.057 | +0.040, +0.059 |

  From the meta checkpoint, the exact direction reduces forgetting by 0.005–0.047, and
  the first-order direction by at most 0.009. From the ordinary checkpoint, both increase
  it.
- At η = 0.1, both directions damage the sources even before adaptation (−0.17 to
  −0.45), the first-order one usually more.
- From the ordinary checkpoint at α = 0.01, inner adaptation is unstable. Same-episode
  objectives explode (above 10¹⁰), and 3 fresh adaptations diverge.
- **Query accuracy after the inner steps is at chance** (0.09–0.14), while the image
  sources fall from 0.88–0.90 to 0.52–0.81. Support performance was not measured, so this
  does not separate ineffective adaptation from fitting the support set without
  generalizing. The query loss barely changes under any outer step. Either way, these
  episodes mostly measure the consequences of damaging updates, not the cost of acquiring
  a capability.

### Reading

- The reviewer's concern holds within the tested setting (5 SGD steps; the pilot used
  10 Adam steps, so this does not diagnose the pilot's update rule directly):
  - the first-order signal points in a substantially different direction from the
    signal through adaptation;
  - from the meta checkpoint the exact direction reduced forgetting slightly more than
    the first-order one;
  - the pilot's preservation failure cannot be attributed to the inner learning rate
    alone.
- It has not been shown that correcting the gradient produces useful learning: query
  accuracy stays at chance.
- Two obstacles come before any further pilot:
  1. **A functioning episode.** The inner adaptation must improve support *and* query
     performance stably before its learning-efficiency gradient can be judged. Otherwise
     the preservation term can be satisfied by adapting less, which is not learning more
     efficiently.
  2. **The post-adaptation objective is rough at practical step sizes.** Smooth
     activations (for example softplus or GELU) are a candidate remedy, to be tested by a
     controlled comparison, not a demonstrated requirement. Estimators that average over
     the roughness (smoothing, evolution-strategies perturbations) are alternatives.
- Cost: the exact gradient took 2.4× the CPU time of the first-order one (152 s against
  63 s over 24 episodes, B = 5). That ratio grows with B.

### Cost of the diagnostic

- Wall 896 s on 6 workers; worker CPU 5,142 s; largest worker peak RSS 1.3 GB.
- Selection cost:
  - in-process test runs of about 2 minutes in total;
  - one launch that failed on dtype;
  - one launch that crashed after 18 of 24 jobs (about 11 minutes) on a diverging
    adaptation. Divergences are now recorded instead of crashing.

## Episode audit (development only)

Design: [spec, "Episode audit"](number-systems-meta-v1-spec.md#episode-audit-development-only-added-2026-10-28-after-the-gradient-diagnostic),
fixed before running.

- **Command:** `NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta_episode pilot`.
- **Artifacts:** `reports/number-systems-meta-v1-pilot-episode-results.json.gz`, `-runtime.json`.
- **Runs:** 288 adaptation runs: 2 checkpoints × 3 development representations ×
  2 episodes × 8 optimizer settings × 3 parameter sets.

**No procedure met the functioning rule** (query gain ≥ 0.15 with image-source
retention ≥ 0.9) on either checkpoint. Every procedure that learned also forgot.

Means over 6 episodes (support gain / query gain / retention):

| Procedure | ordinary, 25 updates | ordinary, 50 updates | meta, 50 updates |
| --- | --- | --- | --- |
| Adam 3e-4, whole satellite | +0.23 / +0.07 / 0.91 | +0.42 / +0.11 / 0.87 | +0.54 / +0.18 / 0.87 |
| Adam 3e-4, front end only | +0.21 / +0.07 / 0.91 | +0.39 / +0.10 / 0.88 | +0.43 / +0.14 / 0.90 |
| Adam 1e-3, whole satellite | +0.54 / +0.16 / 0.73 | +0.78 / +0.21 / 0.69 | +0.88 / +0.32 / 0.62 |
| Adam 1e-3, SRI part only | +0.04 / +0.04 / 0.99 | +0.11 / +0.06 / 0.94 | +0.18 / +0.13 / 0.78 |
| Adam 1e-2, whole satellite (the pilot's inner rule) | +0.21 / +0.09 / 0.16 | +0.53 / +0.14 / 0.16 | +0.74 / +0.24 / 0.18 |
| SGD 1e-2, whole satellite (as in the gradient diagnostic) | +0.08 / +0.05 / 0.35 | +0.16 / +0.08 / 0.42 | +0.09 / +0.04 / 0.48 |

Reading against the review's table:

- **The gradient diagnostic's episodes were ineffective, not overfitting.** Plain SGD at
  1e-2 barely improves the support set (+0.02 to +0.16) while retention falls to
  0.35–0.52. So those episodes damaged the sources without adapting.
- **Adam adapts usefully.** Support and query both improve: query gains of 0.1–0.3 within
  25–50 updates.
- **The pilot's inner rule (Adam 1e-2, 10 updates) learns, but keeps only 0.17 of source
  accuracy.**
- **There is a trade-off frontier, and no procedure gets past it.**
  - Smaller learning rates and adapting only the SRI part keep retention near 1 but learn
    little.
  - Larger ones learn more and forget more.
  - The closest points to the rule are Adam 3e-4 at 50 updates: meta, whole satellite,
    +0.18 at retention 0.87; meta, front end only, +0.14 at 0.90.
- **Meta's frontier is somewhat better than ordinary's** (for example +0.18 against +0.11
  query gain at the same retention). But meta's outer loss used these development
  representations, so this is not a clean comparison.
- **Diverged runs:** 16 in total, all plain SGD at 1e-2 or above (13 of them at 1e-1).

What this means for the next step:
- With one shared image satellite, learning a new representation in this architecture
  currently costs the sources accuracy at every learning rate tried.
- The preset rule asked for an episode that already solves the trade-off. Improving that
  trade-off is what the meta objective was meant to learn, so the rule may be
  circular. Changing the rule after seeing the results is a decision for the
  researchers, not something done here.
- Two non-circular options:
  1. Define "functioning" as stable support and query improvement (for example Adam 3e-4
     or 1e-3, 25–50 updates). Audit the gradient through that procedure, and treat
     retention as the quantity the outer loop must improve.
  2. Change what adapts for a new representation, so that retention is not traded away
     by construction. That is an architecture change, which the review asked to defer.

Cost: wall 441 s on 6 workers; worker CPU 2,463 s; largest worker peak RSS 1.0 GB.
Selection cost: one in-process test job (3 s).

## Gradient audit through the selected procedure (development only)

Option 1 ([spec](number-systems-meta-v1-spec.md#decision-after-the-episode-audit-2026-10-28)):
the inner procedure is Adam at 1e-3 on the whole image satellite for 25 updates, unrolled
differentiably.

- **Command:** `NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta_audit pilot`.
- **Artifacts:** `reports/number-systems-meta-v1-pilot-audit-results.json.gz`, `-runtime.json`.
- **Setup:** float64; ordinary and meta checkpoints; 6 episodes each, with 6 fresh
  episodes for every outer step.

### The episode now learns

No outer step, means over fresh episodes:

| Checkpoint | Support before → after | Query before → after | Image sources before → after |
| --- | --- | --- | --- |
| ordinary | 0.10 → 0.55 | 0.09 → 0.21 | 0.90 → 0.67 |
| meta | 0.08 → 0.68 | 0.11 → 0.35 | 0.88 → 0.60 |

Support and query both improve, so this is useful adaptation by the review's table. It
forgets 0.24–0.28 of source accuracy.

### Exact versus first-order gradient

- **Verified locally, rough at practical scales.** At ε = 1e-7–1e-8, finite differences
  match the exact directional derivatives with a median relative error of 3 × 10⁻⁶ to
  8 × 10⁻⁵. Along a random direction, one test episode still matched at 1e-5. But the
  maximum errors are about 1–6, and the median error at 1e-5 is 1.0.
  - Along the negative exact gradient, the true slope at 1e-7 is *positive* (uphill) in
    6 of 12 episodes, at values up to about 5,300.
  - So the gradient's local validity often breaks within 10⁻⁷.
  - Candidate causes, not separated here: ReLU on/off changes, and Adam's normalization.
    Adam's m̂/√v̂ behaves like the sign of tiny gradients, so parameters with near-zero
    gradients become extremely sensitive.
- **The two directions are close to unrelated.**

  | Checkpoint | Cosine, satellite (mean, range) | Cosine, shared parameters (mean, range) |
  | --- | --- | --- |
  | ordinary | 0.07 (−0.36 to 0.57) | 0.04 (−0.35 to 0.47) |
  | meta | −0.29 (−0.61 to 0.03) | −0.03 (−0.44 to 0.27) |

  The first-order estimate misses the true slope along random directions by a median of
  160–190%.
- **CPU cost:** exact 339 s against first-order 113 s over 12 episodes, a ratio of 3.0.

### Small outer steps on fresh episodes

Changes from no step:
- query gain is (query after − query before) with the step, minus the same without it;
- forgetting is source accuracy before adaptation minus after.

| Checkpoint | Direction, η | Query gain | Forgetting | Source accuracy before adaptation |
| --- | --- | --- | --- | --- |
| ordinary | exact, 0.01 | −0.007 | −0.006 | +0.002 |
| | exact, 0.03 | −0.013 | −0.010 | −0.002 |
| | first-order, 0.01 | +0.000 | −0.010 | +0.003 |
| | first-order, 0.03 | −0.013 | −0.035 | −0.019 |
| meta | exact, 0.01 | −0.004 | −0.013 | −0.000 |
| | exact, 0.03 | −0.009 | −0.029 | −0.013 |
| | first-order, 0.01 | −0.006 | −0.016 | −0.005 |
| | first-order, 0.03 | −0.015 | −0.023 | −0.014 |

- **Neither direction improves learning on fresh episodes.** Query gain is unchanged or
  slightly worse.
- **Both reduce forgetting slightly** (0.006–0.035).
- **The exact direction is no better than the first-order one** here. On the ordinary
  checkpoint the first-order direction reduces forgetting more.
- These are single steps, so they test the local direction, not training over many
  episodes.

### Reading

- With a working inner procedure:
  - the exact learning-efficiency gradient can be computed and verified locally;
  - it differs almost completely from the first-order estimate;
  - neither direction, in a single small step, makes new learning cheaper;
  - both trade a little learning for a little less forgetting.
- The objective is rough along the gradient directions even through Adam, so
  single-step tests are weak evidence either way.
- Whether many outer steps of either kind improve the support / query / retention
  frontier is the open question. That is the equal-compute training comparison, at a
  measured exact / first-order cost ratio of 3.0.

Cost: wall 1,607 s on 6 workers; worker CPU 9,458 s; largest worker peak RSS 2.4 GB.
Selection cost: one in-process test of about 30 s.

## Cost

Wall 1,561 s:
- data 28 s;
- source phase 1,210 s, with 4 arms in parallel and about 1,200 s CPU each (episodes
  about 510 s of that in the meta arms);
- adaptation 322 s, 60 runs on 6 workers.

Worker CPU 6,670 s; largest worker peak RSS 1.9 GB. Selection cost: two smoke runs
(about 30 s each, one failing on a result-key bug). Using 6 workers instead of 11 kept
the adaptation phase short (about 5 min for 60 runs).
