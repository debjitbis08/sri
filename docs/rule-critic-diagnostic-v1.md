# Rule critic diagnostic v1: can the told-rules critic learn a composing step check?

**Contract, frozen 2026-11-07 before the run** (user: "continue" after the draft). It follows [rule-guided search v1](rule-guided-search-v1.md), where the
told-rules critic accepted valid programs only at length 1, and an external suggestion pasted by the user: a small
diagnostic of the same rule encoder and MLP critic, before another full search experiment. Its three possible causes
(spurious witnesses on the confirmed graph, the harder fresh-rule task, unconverged training) are separated here. The
components are unchanged: the frozen readers and their graphs, and the rule encoder and MLP critic of
`experiments/rule_guided_search.py`. Only the rule fragment, the training signal and the reads change. Planned runner
`experiments/rule_critic_diagnostic.py`; reports `reports/rule-critic-diagnostic-v1{,-summary,-runtime}.json`.

## Fragment: ancestor-style recursion only

Every rule set defines one derived relation H from one base relation R with two rules:

- seed: R → H, i.e. R(x, y) → H(x, y), or with R's arguments reversed;
- recursion: R ∘ H → H (`lin-l`: R(x, y) ∧ H(y, z) → H(x, z)), or H ∘ R → H (`lin-r`).

Within a rule set, R has the same orientation in both rules. The valid programs are exactly R^k (forward) or (R⁻¹)^k
(reversed), for every k ≥ 1. A pairwise check represents this exactly:

- START → R is allowed;
- R → R is allowed;
- R → END is allowed;
- nothing else is.

Counting, conjunction and combinations of relations are deferred.

- **Variation:**
  - R is any hop relation with ≥ 5 true facts;
  - orientation is forward or reversed;
  - recursion side is `lin-l` or `lin-r`;
  - the derived symbol is fresh in every rule set.
- **Held out:**
  - 20% of hop relations never appear as R in training (relation assignments). They always include `parent-of` and
    `just-bigger-than`, so the taught `ancestor-of` and `bigger-than` are held-out instances;
  - for training relations, one of the four (orientation × side) combinations per relation, chosen once (rule
    instances).

## Three training conditions (same encoder, same critic, same fragment)

| Condition | Signal | What it tests |
|---|---|---|
| `confirmed-mil` | question labels, max over candidate programs (as in rule-guided search v1), confirmed graph | reproduces the difficulty |
| `oracle-mil` | the same on the oracle graph | the contribution of the readers' wrong facts |
| `direct` | valid / invalid program labels, binary cross-entropy on ∏ c_t | whether the encoder and critic can learn the intended check at all |

`direct` is a **diagnostic control with extra supervision**, stated as such. Its labels come from the symbolic language
of the rule set (offline, data only). The deployed component stays neural and is never given proofs. Like the script-told arms of the drift studies, it is
a comparison (an upper bound), not a candidate design.

- **Questions (MIL conditions):** for each training rule set, up to 20 T questions of proof length ≤ 2 and 20 F
  questions, half of them connected negatives. Candidates are every program of ≤ 2 hops in the condition's graph, as
  before. Labels are the oracle closure. F questions that the condition's graph makes derivable are excluded, as
  before.
- **Programs (`direct`):** for each training rule set, the valid programs of length 1–2 and an equal number of
  negatives of length 1–2, half random and half hard (below).
- **Budget:** 300 training rule sets per seed, 3,000 steps (double v1's 1,500, a fixed extension), 16 rule sets per
  step, Adam lr 3e-3. Checkpoints at 500, 1,000, 2,000 and 3,000 steps, with the program reads below at each.
- **Seeds:** 450–453 (the reader checkpoints). 3 conditions × 4 seeds = 12 runs.

## Reads (program level, entity-free; on held-out relations and held-out instances separately; four seeds)

1. **Valid acceptance by length** 1–12: the share of valid programs R^k accepted (every check > 0.5).
2. **Hard negatives by length,** 2–12:
   - one step's direction reversed;
   - one step's relation replaced by another hop relation;
   - a splice of R^i with S^j (S another relation);
   - the right relation in the wrong orientation throughout.

   Plus random programs. Each is reported as an acceptance rate.
3. **Where errors occur:** for rejected valid programs and accepted negatives, which check decided: START (first step),
   internal transition, or END. Shares.
4. **Invariance:** the change in accepted/rejected verdicts on valid and hard-negative programs under fresh-symbol
   renaming, consistent variable renaming (x ↔ z in every atom of both rules), and rule order (seed and recursion
   swapped). Each should leave every verdict unchanged.
5. **Checkpoint curves:** reads 1–2 at each checkpoint. Longer training counts as justified only if held-out valid
   acceptance rises while hard-negative acceptance stays low.
6. **Answers (secondary, cheap):** the rule-guided search v1 beam with the trained critic on taught `ancestor-of` and
   `bigger-than` (held-out relations), on oracle and confirmed graphs, by length.
7. **MIL witness quality:** for T training questions, the share whose highest-scoring candidate at the end of training
   is a valid program.

## Interpretation (fixed before the run)

- `direct` succeeds and `oracle-mil` fails: the problem is witness selection and credit assignment in MIL training,
  even on true facts (a path of true facts can still be an invalid proof).
- `oracle-mil` succeeds and `confirmed-mil` fails: the problem is the readers' wrong facts and training robustness.
- `direct` fails on this fragment: inspect the rule encoding and optimization before touching the walker.

"Succeeds" means held-out valid acceptance ≥ 0.9 at lengths 3–12 with hard-negative acceptance ≤ 0.1, on held-out
relations.

## Predictions

1. **[hypothesis]** `direct` succeeds on held-out relations and instances. The fragment is a bigram pattern, which the
   critic represents.
2. **[hypothesis]** `oracle-mil` falls short of `direct`. Even on true facts, true questions often have shorter invalid
   witnesses through other relations, and the maximum picks them. Witness quality (read 7) < 0.8.
3. **[hypothesis]** `confirmed-mil` is worst. Witness quality < 0.5.
4. **[hypothesis]** Invariance: `direct` changes ≤ 2% of verdicts under each renaming.
5. **[hypothesis]** Most `confirmed-mil` errors on valid programs fall at internal transitions (as in v1, which
   accepted only length 1).

## Prior work

- **Spurious programs in weakly supervised semantic parsing** **[familiar]**: training on answers rewards programs that
  happen to give the right answer (Guu et al. 2017; Liang et al. 2018, MAPO). The user's pasted review points to
  [ACL 2021 short paper 22](https://aclanthology.org/2021.acl-short.22/) **[pointer, not read]**. This is the MIL
  witness problem here; the `direct` condition removes it.
- **Grammatical inference:** a bigram pattern is identifiable from short strings and extends to any length (the
  composition critic check's basis) **[familiar]**.
- **What this adds:** which of the three causes stops the told-rules critic, on a fragment it can represent, before
  more search work. "Nothing new beyond a needed internal control" is the honest answer for the method itself.

## Scale

The critic and encoder are unchanged: size independent of graph size and length. `direct` needs no graph and no
candidate enumeration: O(programs) per rule set. The MIL conditions cost as in v1, with one base relation per rule set.

## Computation (declared, estimates)

- **Graphs:** about 1 CPU min per seed.
- **Candidates:** `confirmed-mil` about 2,400 programs per question, as in v1; `oracle-mil` far fewer.
- **Training, 3,000 steps:**
  - `confirmed-mil`: about 0.9 s per step, 45 min;
  - `oracle-mil`: about 10 min;
  - `direct`: about 5 min.
- **Reads:** a few minutes per checkpoint.
- **Totals:** about 1 h wall on 5 workers. No tuning.
- **Runtime JSON:** by phase and checkpoint.

## Implementation (recorded before the main run)

- **Runner:** `experiments/rule_critic_diagnostic.py`. It reuses rule-guided search v1's readers, graphs, rule encoder,
  MLP critic, candidate enumeration and beam.
- **Relation split:** seed 2029, the same for every run seed. `parent-of` and `just-bigger-than` are forced into the
  held-out relations.
- **Held-out instance per training relation:** seed 2030.
- **`direct` batches:** per rule set, both valid programs (R and R·R, oriented) and two fresh negatives each step: one
  random and one hard of a random kind, each of length 1–2.
- **"Where errors occur"** uses the first check that fails, for rejected valid programs. For accepted negatives it uses
  the first check that the perfect pairwise table (START→R*, R*→R*, R*→END) would reject.
- **Smoke speeds:** after vectorizing program scoring, 0.83 CPU s per step (`confirmed-mil`); the smoke reads were
  unchanged by the vectorization.
- **Bug fixed during the main run:** a `direct` splice negative needs length 2, and the retry loop could draw length 1
  repeatedly and crash. Splices now always have length 2. The MIL runs had finished and were kept; only `direct` was
  rerun.

## Commands

    EV_WORKERS=5 .venv/bin/python -m experiments.rule_critic_diagnostic       # rerun to resume
    .venv/bin/python -m experiments.rule_critic_diagnostic summary

## Results (2026-11-07)

The MIL runs used revision `14864c6`. The `direct` runs used `e53a722`, after the splice fix, which touches only
`direct` batches. 12 runs, 5 workers. Means over seeds 450–453; error locations summed.

### Valid-program acceptance (every check > 0.5), by length

| Condition | held-out relations, length 1 / 2–12 | held-out instances, length 1 / 2–12 | MIL witness quality |
|---|---|---|---|
| `confirmed-mil` | 0.38 / 0.015 | 0.61 / 0.02 | 0.34 |
| `oracle-mil` | 0.69 / 0.018 | 0.84 / 0.09 | 0.60 |
| `direct` | 0.55 / 0.54 (flat) | **0.88 / 0.88 (flat)** | — |

### Hard negatives accepted (held-out relations, mean over lengths 2–12)

| Condition | reversed step | replaced step | splice | wrong orientation | random |
|---|---|---|---|---|---|
| `confirmed-mil` | 0.012 | 0.000 | 0.000 | 0.007 | 0.000 |
| `oracle-mil` | 0.003 | 0.001 | 0.002 | 0.015 | 0.000 |
| `direct` | 0.106 | 0.174 | 0.088 | 0.000 | 0.003 |

### Where errors occur (held-out relations)

| Condition | valid rejected at START / internal / END | negatives accepted at START / internal / END |
|---|---|---|
| `confirmed-mil` | 1,392 / 1,672 / 53 | 836 / 92 / 0 |
| `oracle-mil` | 840 / 2,167 / 15 | 1,796 / 43 / 0 |
| `direct` | 1,152 / 110 / 236 | 380 / 3,143 / 0 |

### Invariance (share of verdicts changed; held-out relations)

| Condition | fresh symbols | variable renaming (x ↔ z) | rule order |
|---|---|---|---|
| `confirmed-mil` | 0.009 | 0.014 | 0.000 |
| `oracle-mil` | 0.006 | 0.013 | 0.000 |
| `direct` | 0.005 | **0.154** | 0.000 |

### Checkpoints: longer training did not help

Held-out relations, valid acceptance at lengths 3–12, at steps 500 / 1,000 / 2,000 / 3,000:

| Condition | valid acceptance | witness quality |
|---|---|---|
| `confirmed-mil` | 0.015 / 0.007 / 0.015 / 0.015 | 0.27 → 0.34 |
| `oracle-mil` | 0.037 / 0.029 / 0.037 / 0.018 | 0.58 → 0.60 |
| `direct` | 0.71 / 0.59 / 0.54 / 0.54 (hard negatives 0.19 → 0.09) | — |

### Answers with the beam (taught relations, held out as relations)

| Condition | `ancestor-of`, oracle graph, length 1 / 2–6 | `bigger-than`, oracle graph, length 1 / 2–12 |
|---|---|---|
| `confirmed-mil` | 0.86 / 0.48 | 0.75 / 0.51–0.64 |
| `oracle-mil` | 0.87 / 0.49 | 0.66 / 0.38–0.48 |
| `direct` | 0.61 flat | 0.50–0.57 |

### Against the predictions

1. **Failed.** `direct` on held-out relations: 0.54 at lengths 3–12, hard negatives 0.09–0.17. On held-out instances
   of training relations: 0.88 at every length.
2. **Held.** `oracle-mil` witness quality 0.60. Valid programs of length ≥ 2: 0.02 against `direct`'s 0.54.
3. **Held.** `confirmed-mil` witness quality 0.34, the worst.
4. **Partly.** `direct` is invariant to fresh symbols (0.005) and rule order (0.000), not to variable renaming (0.154).
5. **Held, narrowly.** 54% of `confirmed-mil`'s rejections of valid programs fall at internal transitions.

### What this means (by the interpretation fixed before the run)

**`direct` fails on this fragment, so the rule encoding comes before the walker.** The reads also say which parts work
and which do not:

- **The pairwise critic can compose.** Under direct supervision, acceptance is flat in length: 0.88 at every length
  1–12 on held-out instances of training relations, after training only on lengths 1–2. A critic that accepts the
  first step and the R → R transition carries to any length. Composition itself is not the problem.
- **What fails with direct supervision is binding the rule's relation to the step's relation for a relation never seen
  in a rule.** On held-out relations, rejections fall at START (1,152 of 1,498). Accepted negatives fall at internal
  transitions with a replaced or spliced relation (3,143). The critic has learned "a transition from the right kind of
  step to a step of the same kind", not "the relation named in the rule". It is also not invariant to renaming
  variables (0.15): it reads the variable embeddings as content.
- **Question-label training never learns the internal transition, even on true facts.** Both MIL conditions accept
  almost no valid program of length ≥ 2 (0.015–0.09), on held-out relations and on held-out instances of training
  relations. Their rejections fall at internal transitions. The maximum over candidates picks other witnesses: the
  best-scoring program is a valid proof for 0.60 of true questions on true facts and 0.34 on the readers' facts. Wrong
  facts make it worse, but witness selection fails even without them.
- **Longer training is not the answer.** Checkpoint curves are flat or falling from step 500 on.
- **Two separate problems, then:**
  - (1) *credit assignment*: question labels do not teach which program proved the answer, even on clean facts (the
    spurious-program problem of weakly supervised parsing);
  - (2) *relation binding*: even with program labels, the encoder does not tie a rule's relation code to the matching
    step for relations outside training.
- **Candidates for the next step**, not chosen:
  - for (2): a matching structure in the critic's input, such as the similarity of each step's relation code to each
    rule atom's code, instead of free attention, and variable roles given as positions instead of learned
    embeddings. Both are small established pieces (pointer and copy mechanisms, relative and role encodings);
  - for (1): a witness constraint during training, such as weighting candidates by how well they fit the rule
    (marginal likelihood instead of the maximum), or a curriculum from the program labels to question labels.

### Computation (measured)

`reports/rule-critic-diagnostic-v1-runtime.json`. Per run, CPU s:

| Phase | `confirmed-mil` | `oracle-mil` | `direct` |
|---|---|---|---|
| training, 3,000 steps | 3,591–3,805 | 152–176 | 65–68 |
| candidates | 14–15 | 4–6 | — |
| graph construction | 49–60 | 49–60 | 49–60 |
| checkpoint reads | 10–28 | 10–28 | 10–28 |
| final reads | 10–27 | 10–27 | 10–27 |

- **Peak RSS:** 0.51–0.91 GB.
- **Total:** 16,906 CPU s (4.7 CPU h). Wall time is the sum of the first launch, about 65 min to the crash, and the
  resumed `direct` runs (150 s).
- **No tuning.**
