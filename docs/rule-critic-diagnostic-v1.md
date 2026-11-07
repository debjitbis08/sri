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
