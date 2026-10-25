# Local operators, learned grouping, and exact completion: first result

Completed 2026-10-25. Positive within a supplied four-unit scaffold, with a
negative small-case speed comparison. No unsupervised discovery, local plasticity,
graded recurrent settling, or new witness on the frozen relational task.

[Prospective contract](local-relation-circuit-v1-contract.md) ·
[implementation](../experiments/local_relation_circuit.py) ·
[scientific results](../reports/local-relation-circuit-v1-results.json) ·
[runtime and selection costs](../reports/local-relation-circuit-v1-runtime.json)

## 1. A compact local grouping circuit exists

Each edge applies two binary functions to its neighboring bits. Their outputs
are summed separately around the ring; a small decoder maps the two counts to
an inner class. The supplied XOR/AND control implements all four relation classes
exactly on all 16 outer patterns. This forward grouping does not enumerate outer
configurations. Sharing and summation make it invariant to cyclic rotation.

For outward interpretation, a chain dynamic program carries costs indexed by
the current bit and two counters. It merges partial interpretations with the same
conditional state, restores the closing edge, selects the inner class, and
traces back an outer completion. It does not carry a list of complete patterns.
The supplied circuit matched exact enumeration on all 5,184 preference/evidence
cases, preserving observations and selecting members of the full optimum sets.

This is local factor evaluation with a scheduled conditional-cost algorithm.
Counters and profiles are not single scalar neurons. No arbitrary asynchronous
unit update rule or recurrent stability result is implied.

## 2. Local detectors and a decoder can be identified from labels

Training used exactly eight declared labeled patterns. Selection used training
consistency alone, before held-out evaluation. The learned objects are local
binary truth tables and a count-to-class decoder; the entire grouping table over
16 complete patterns was not supplied to the deployed model.

| Candidate family | Candidates checked | Training-consistent | Selected held-out result |
| --- | --- | --- | --- |
| One shared binary detector, one count | 16 | 0 | No fit |
| Two shared binary detectors, two counts | 256 | 136 | 8/8 correct, no unknowns |

The lexicographically first consistent pair has codes (1,2):

```
g1(a,b) = 1 exactly when (a,b)=(0,0)
g2(a,b) = 1 exactly when (a,b)=(0,1).
```

Its six learned decoder entries are:

| Count of 00 edges | Count of 01 edges | Inner class |
| --- | --- | --- |
| 0 | 0 | Uniform |
| 0 | 1 | One different |
| 0 | 2 | Alternating |
| 1 | 1 | Adjacent pairs |
| 2 | 1 | One different |
| 4 | 0 | Uniform |

Every consistent two-detector candidate correctly classified all 16 patterns,
including all 8 held-out patterns. The learned operator itself is not uniquely
identified: 136 different ordered pairs fit. The chosen pair is a declared
convention, not the best or uniquely correct algebra. The one-detector negative
excludes only the specified summed-binary-detector family, not all scalar units.

**What was supplied:** adjacency, two detector channels, weight sharing, summation,
supervised class labels, and a sparse decoder family. All held-out patterns are
rotations of training patterns. Consequently, any consistent invariant circuit
must generalize to them; this is a structural guarantee under this split, not
discovery of rotation invariance. Complement behavior is supported by labeled
complement examples. The finite exhaustive parameter search is not a local
synaptic learning law. No claim about unsupervised or unrelated data follows.

## 3. Learned contextual completion also matches the exact reference

The selected decoder is total over the four-bit domain. Its return solver matched
both its own exhaustive reference and the target grouping's exact reference in
all 5,184 cases. Each validation compared optimum energy and membership in the
optimal set, rather than requiring identical choices among ties. Both models
had 384 tied cases. There were no observation violations or solver failures.

With stored pattern 0011 and preferred inner class adjacent pairs, both circuits
return 0011 with no evidence or observed prefix 00, return 1100 with observed
prefix 11, and return observed 0101 with the inner class revised to alternating.
Thus the same compact inner class can drive different compatible reconstructions.

## 4. Computation: enumeration is faster at this size

Matched benchmark: all 81 observation patterns, stored outer default 0011,
inner default adjacent pairs; one warmup batch and seven measured batches.
Allocation, counters, profiles, gate lookup and traceback are included. Values
below are amortized per-query wall times, obtained from complete batches, not
individually timed-query distributions.

| Method | Median microseconds/query | Batch-derived min–max microseconds/query |
| --- | --- | --- |
| Direct 16-pattern enumeration | 15.15 | 15.10–15.32 |
| Supplied local circuit return | 26.74 | 26.57–27.72 |
| Learned local circuit return | 26.45 | 26.25–26.79 |

The learned local solver is about 1.75 times slower here. Avoiding exponential
pattern enumeration does not guarantee a small-case speed win. The direct method
uses an analytic class function and no precomputed completion cache. Both methods
were benchmarked through their actual validation call paths. CPU timing medians
and all seven samples are in the runtime receipt. No scaling sweep was performed.

With fixed two-detector count features, the scheduled return has O(n^3) work and
traceback-entry storage, compared with O(n*2^n) direct enumeration. Current profiles
need O(n^2) entries. This bound does not mean the learned four-bit decoder extends
to longer rings: it is undefined on many new possible counts. An appropriate
total decoder and useful grouping at those lengths remain unproved.

The learned detector tables contain eight logical bits; six explicit decoder
entries need up to 48 further bits for two 3-bit counts and a 2-bit class, excluding
container overhead. For comparison, a fixed four-bit lookup table can store its
16 two-bit labels in 32 bits. This example therefore does not establish a raw
storage saving either. The selected Python model occupies 1,162 recursively
counted object bytes; its descriptive JSON is 280 bytes. Inference needs no raw
training patterns, optimizer state, or replay.

## 5. Selection cost, correctness, and reproducibility

Executed `python3 -m experiments.local_relation_circuit` once after fixing the
contract and implementation. All correctness assertions passed on the first run.
Training scored all 272 candidates: 2,176 labeled-pattern exposures from only
8 distinct patterns, including rejected candidates. This used 16,896 local gate
evaluations. One-detector identification took about 0.00053 s wall; two-detector
identification took about 0.00908 s. Full version-space evaluation used another
17,408 gate evaluations and about 0.01476 s. Test outcomes did not select the model.

Each circuit's 5,184-case return validation considered 56,832 prefix extensions,
created 44,288 profile entries, and checked 15,872 terminal candidates. Maximum
live two-layer frontier size was 11 entries and maximum stored traceback layers
14 entries per conditioned root. Learned validation additionally checked a direct
reference using its own learned class function, with that cost included.

Whole-run wall/CPU time was about 0.9490/0.9484 s, including validation, selection,
version-space diagnostics, benchmarks, and scientific serialization. Peak RSS
was 60,688 KiB, including interpreter, all audit/results objects and serialization;
this is not deployed circuit memory. Separate allocator workspace is unavailable.
All phase counts, phase timing, environmental metadata, and source/contract/result
hashes are recorded. Timings remain separate from deterministic scientific JSON.
There were no optimizer steps, random seeds, accelerator calls, or empirical data
collection. Manual research/development time is unmeasured.

## 6. What has advanced and what has not

We now have a constructive local factorization and exact return without enumerating
complete outer configurations, plus supervised identification of local detectors
within that scaffold. We do not yet have a broadly learned algebra or a cheap
graded reciprocal unit rule. At the checked size, direct enumeration is faster
and a flat label table can be smaller.

The next informative test would vary ring size or the desired equivalence, define
a compact decoder family that covers those changes, and separate architectures
whose supplied invariances match the task from those whose invariances do not.
A new prospective contract is needed before that learning/scaling experiment.
