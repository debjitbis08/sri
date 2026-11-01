# Test-time neural memory: Titans, MIRAS and ATLAS

Reading date: 2026-10-31. User supplied the Google Research overview and
the [ATLAS proceedings page](https://proceedings.mlr.press/v306/behrouz26b.html).
Followed the overview's primary-paper links; searched for the ATLAS title
when the proceedings PDF could not be rendered. No experiment is authorized
or frozen by these notes.

## Evidence and scope

- [Titans](https://arxiv.org/html/2501.00663v1), **[read: memory mechanism,
  equations 8–14 and parallelization discussion; not a full empirical audit]**.
  A neural memory learns projected key–value associations at inference using
  gradient updates, momentum and input-dependent decay. Outer-loop training
  learns the surrounding projections and update controls. Difference: this is
  not our clipped binary memory and does not demonstrate lifelong adaptation
  of sensory encoders. **[hypothesis]** A small trainable neural memory is a
  more relevant comparator for continuous sensory association than a fixed
  Hebbian write rule alone.
- [MIRAS](https://arxiv.org/pdf/2504.13173), **[read: framework, model table,
  and Huber/retention sections; not full proofs or experiments]**.
  Separates memory architecture, internal objective, retention regularizer and
  update algorithm. Proposes alternatives including robust objectives.
  Difference: mathematical correspondence is not evidence of equivalence to
  our complete system. **[hypothesis]** Separate those four choices in future
  ablations; outlier robustness may help sensory noise without assuming that
  surprise always means useful information.
- [ATLAS](https://arxiv.org/pdf/2505.23735), **[read: arXiv v1 introduction,
  Omega-rule formulation and architecture overview; proceedings abstract;
  final proceedings PDF unavailable through the reader]**. Omega updates
  optimize over a recent window. ATLAS also uses expressive features and
  optimizer changes. The preprint explicitly distinguishes within-context
  adaptation from learning that survives clearing the memory. Difference:
  this does not establish lifelong sensory learning. **[hypothesis]** A bounded
  recent-experience window could support association learning across adjacent
  sensory events; its retained state and compute must count.

## Implications, not established results

**[derived]** A fixed neural memory can have bounded persistent size and
per-event work independent of elapsed stream length when architecture,
window and update count are fixed. Neither statement guarantees fixed recall
as independent associations accumulate. Count parameters, fast memory weights,
momentum, retained windows and temporary training workspace separately.

**[derived]** A memory read is not automatically reinforcement. With a squared
association loss, an exactly reconstructed target contributes zero error
gradient; momentum and decay can still act. Copying a read as its own detached
target and immediately fitting the same prediction also supplies zero error.
Use-dependent strengthening needs its own mechanism and causal experiment;
do not identify it with surprise-driven writes or recent-window rehearsal.

**[hypothesis]** Compare an LSTM, a gated delta-rule matrix, a small neural
memory with learned updates, and a bounded explicit store on the same paired
sensory stream. Establish a simple baseline before adding ATLAS's feature
expansions or optimizer. Provide no hidden target IDs to the operational memory.
Test rare useful events, repeated retrieval, sensory confirmation, corrupted
events and encoder changes. Retaining one global context indefinitely is a new
evaluation setting, not a consequence of reported long-context benchmarks.
