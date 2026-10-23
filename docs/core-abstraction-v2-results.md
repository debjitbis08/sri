# Core abstraction v2 — connected evidence selects the teacher core and transfers; donor dependence holds for S1 only, so the minimal target fails on C2

**Completion:** `complete`. Every correctness gate passed. All 4 teachers × 5
splits × 2 assignments (40 runs) were run through the three arms (I, IR, C), all
surviving private branches, all nine cores, the reference branches and the
fixed-core controls.

**Headline:**

- **Minimal target not met.** The contract requires at least one executable
  witness for each of S1 and C2. S1 has 4 of 10 runs; C2 has 0 of 10. The
  result is the same under the core-dependent criterion and the stricter v1
  route criterion.
- **Connected evidence solves the v1 problem for both primaries.** Under arm C
  the declared branch rule picks the teacher's private blocks in all 10 runs of
  every teacher. The installed core is the teacher's sign class (`(-3,3)`, the
  first of `±(3,-3)` in minimal-change order) for S1, C2 and T1 in every run.
  These installed models get every mixed-evidence query jointly exact (36/36),
  lose no coupled training behaviour and regress no mixed or within query.
- **C2 fails only the donor control.** Each C2 satellite's own connected
  observations already restrict its core mask to `±(3,-3)` (`|C_A| = |C_B| = 2` in
  every run). Recipient-only consolidation therefore installs the same core,
  and the partner's information adds nothing. This is condition 2 of section
  7.1 working as declared, not a transfer or preservation failure.
- **S1 passes exactly when one satellite's evidence is uninformative.** In the
  four passing runs, satellite A's own mask stays full (all nine cores) and B's
  pins `±(3,-3)`. All 32 witnesses have recipient A. In the other six runs both
  masks pin the core and donor control fails as for C2. T1 shows the same
  pattern in 2 of 10 runs.
- **Isolated information still selects nothing.** Arms I and IR install `c0` in
  every run and pass none, as disclosed before computation.
- **S2 shows no hidden complementary-core prior.** Connected evidence leaves 7
  cores including `c0`, so arm C installs `c0` and gains nothing. A fixed
  `±(3,-3)` prior fails on S2's true branch, while `±(3,3)` would pass.

[Contract](core-abstraction-v2-contract.md) (sha256 `e724d8fc…`) ·
[manifest](../reports/core-abstraction-v2-manifest.json) (written before
computation) · [feasibility](../reports/core-abstraction-v2-feasibility.json)
(Stage A) · [selector](../reports/core-abstraction-v2-selector.json) (Stage B)
· [runtime](../reports/core-abstraction-v2-runtime.json).

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Prior-artifact hashes (v1 code, contract, results and reports; splits; shared-input substrate and oracle) and contract hash | yes |
| v1 private masks reproduced exactly (40 runs) | yes |
| Private and core masks identical over the five frozen orders | yes |
| IR state equals I | yes |
| Teacher core in `C_AB` on the true branch | yes, all runs |
| Every response set used (11,854) against the independent oracle | 0 mismatches |
| Private blocks byte-identical before and after installation; isolated retention | yes |
| Serialization: installed vector alone, reloaded into a fresh simulator, reproduces all 81 response sets (120 installs) | yes |
| Donor clause: whenever recipient-only consolidation fails, its mask holds an alternative prediction | yes |
| Byte-identical regeneration (manifest, feasibility, selector) in a temporary directory | yes |
| Tests | 10 new; full suite 257, all passing |

**Provenance.** The review edits, contract rename and manifest were completed
before any v2 quantity was computed. The manifest was not committed to git
before the scored run, so the order is attested by its hash and file times. Code
hashes are in each report's `provenance`.

## Stage A — context-consistent feasibility (evaluator-only)

| Teacher | True branch: runs with a feasible core | Feasible sign class | Wrong branches feasible without / with connected acquisition |
| --- | --- | --- | --- |
| S1 | 10/10 | `±(3,-3)` | 0 / 0 (of 1) |
| C2 | 10/10 | `±(3,-3)` | 70 / 0 (of 80) |
| S2 | 10/10 | `±(3,3)` | 0 / 0 (of 1) |
| T1 | 10/10 | `±(3,-3)` | 0 / 0 (of 2) |

Under the revised preservation target, the teacher core on the true branch is
feasible in every run, under both route criteria. This was the disclosed
analytical expectation. The only feasible core on any true branch is the
teacher's own sign class.

For C2, 70 wrong-block branches have a core that is feasible against held-out
targets: the compensation that section 2 forbids rewarding. Connected
acquisition removes all of them, because no core reproduces the connected
training marginals on those branches (`C_AB` empty on 80 of 90).

## Stage B — selector results

### B.1 Arms

| Teacher | Arm | Installed core (runs) | On true branch | Selector passes | Executable success | Compensating |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | I, IR | `c0` (10) | 9 | 0 | 0 | 0 |
| S1 | **C** | `(-3,3)` (10) | 10 | **4** | **4** | 0 |
| C2 | I, IR | `c0` (10) | 0 | 0 | 0 | 0 |
| C2 | **C** | `(-3,3)` (10) | 10 | **0** | **0** | 0 |
| S2 | I, IR | `c0` (10) | 9 | 0 | 0 | 0 |
| S2 | C | `c0` (10) | 10 | 0 | 0 | 0 |
| T1 | I, IR | `c0` (10) | 9 | 0 | 0 | 0 |
| T1 | C | `(-3,3)` (10) | 10 | 2 | 2 | 0 |

The v1-route counts equal the core-dependent counts in every cell. Passing
runs: S1 split 0, 1 and 2 swapped, and 3 base; T1 splits 3 and 4 base.
Connected evidence changes the installed core in every S1, C2 and T1 run, and
in no S2 run.

### B.2 What the installed models do (arm C)

| Teacher | Mixed net gain (A, B) | Joint mixed exact | Coupled training pairs at `c0` (`|P0|` A, B) | Lost | Within gains / losses | Isolated items changed in coupled context |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | (8, 8) | 36/36 | 12–14, 9–10 | 0 | 202 / 0 | 3–6 |
| C2 | (16, 16) | 36/36 | 9–10, 7–10 | 0 | 278 / 0 | 5–8 |
| S2 | (0, 0) | 28/36 | 15, 15 | 0 | 0 / 0 | 0–3 |
| T1 | (8, 8) | 36/36 | 12–13, 11–13 | 0 | 106 / 0 | 1–5 |

The last column is v1's failed criterion, now reported only as context
sensitivity. It is unchanged from v1: the same cores that v1 counted as
forgetting are the ones that reproduce the coupled teacher here.

### B.3 Witness controls

| Teacher | Core-dependent gains (arm C) | Donor control passes | Comparator control passes | Witnesses |
| --- | --- | --- | --- | --- |
| S1 | 160 | 32 (4 runs, recipient A) | 160 | 32 |
| C2 | 320 | 0 | 320 | 0 |
| T1 | 160 | 16 (2 runs, recipient A) | 160 | 16 |

Both single-link cuts remove every gain counted here. Arms I and IR (both `c0`)
and memorization never predict a gained mixed response. All failures are on
condition 2:

| Teacher | Own-mask sizes on true branch `(|C_A|, |C_B|)` | Runs |
| --- | --- | --- |
| S1 | (9, 2) | 4: all pass |
| S1 | (2, 2) | 6: donor control fails |
| C2 | (2, 2) | 10: donor control fails |
| T1 | (7, 2) | 2: both pass |
| T1 | (2, 2) | 8: donor control fails |
| S2 | (7, 7) or (9, 7) | 10: `C_AB` holds 7 cores including `c0`; no gain |

When a recipient's own connected marginals already narrow its mask to the
teacher's sign class, the donor contributes nothing, and the declared control
correctly withholds credit.

### B.4 Fixed-core controls

On both the executable and true branches, `±(3,-3)` passes `CP`, `TR` and core
dependence in all 10 runs of S1, C2 and T1, with and without connected
acquisition. `±(3,3)` does the same for S2 only. No other fixed core passes for
any teacher. For S1 and T1, a fixed `±(3,-3)` prior would explain every arm-C
pass. The connected evidence's contribution is to select that core without
teacher identity, and to leave S2 on `c0` instead of imposing it.

## Readings

| Failure label (arm C) | S1 | C2 | S2 | T1 |
| --- | --- | --- | --- | --- |
| `inconsistent_private_assembly` | 0 | 0 | 0 | 0 |
| `wrong_private_branch` | 0 | 0 | 0 | 0 |
| `core_ambiguity` | 0 | 0 | 10 | 0 |
| `selected_c0` / `no_transfer` | 0 | 0 | 10 | 0 |
| `preservation_lost` / `mixed_regression` | 0 | 0 | 0 | 0 |
| `donor_control_failed` | 6 | 10 | 0 | 8 |

1. **The v1 obstruction was the preservation target, not the substrate.**
   Judged in context, the teacher core preserves everything and transfers
   fully.
2. **Fifteen axis-only connected marginals identify both the private branch and
   the core sign class** for S1, C2 and T1 in every run, without held-out data.
   The remaining limits are a property of the criterion, not missing
   information.
3. **Donor-dependent selection is the binding condition.** Where a satellite's
   own connected observations suffice, the abstraction is selected without the
   partner. That is real acquired coupling, but not the donor-mediated kind the
   contract asks for.
4. **S2's connected evidence cannot separate `c0` from its own core.** Its
   coupled training behaviour equals the `c0` behaviour, so the minimal-change
   rule correctly declines to install an unsupported core.

## What this does not establish

- Endogenous or local acquisition. The masks are an exact global hypothesis
  reference using full-system simulation (contract section 5), holding up to
  180 bits of mask state before installation.
- Abstraction beyond recovering the teacher's coupling. Every successful core
  is the teacher's sign class on the teacher's blocks.
- Preservation under later interfering learning. Nothing is learned after
  installation.
- Reliability. The minimal target is a small-witness existence criterion.

## Accounting

| Item | Value |
| --- | --- |
| Isolated filter evaluations (5 orders + repetition pass, 40 runs) | 15,475 |
| Connected core-filter evaluations (all branches, 5 orders) | 36,184 |
| Simulation requests / unique response sets | 1,565,432 / 11,854 |
| External observations per run | I 15; IR 30; C 30 |
| Maximum mask state before installation | 180 bits (C2: 9 branches × 2 × 9 + 18) |
| Persisted after installation | 7 coefficients |
| Wall time | ≈5 s |

## Next obligations (not run)

1. The stopping rule (section 9) applies. Do not relax condition 2 or change the
   assignment to recover a C2 witness inside v2.
2. If donor-mediated transfer remains the goal, a new contract must create a
   task in which the recipient's own permitted observations cannot pin the core.
   It must declare that design before computation and not choose it from these
   results alone.
3. The exact-mask reference now serves as the upper comparator for a later
   endogenous or local rule. Such a study must test preservation under
   subsequent interfering learning.
