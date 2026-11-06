# Sparse local memory: addressing and interference

Reading date: 2026-11-01. Discussion notes, not a frozen contract or authorization
to run an experiment. User proposes classifying an input, activating k
connections, and storing vectors on units for the classifier to choose among.

## Sources checked

Searches: `SDMLP sparse distributed memory continual learning`; `Elephant sparse
neural network continual learning fly circuit`. Primary PDFs subsequently read:

- [SDMLP, Bricken et al.](https://arxiv.org/pdf/2303.11934)
  **[read: introduction, mechanism summary, ablation table, limitations and
  optimizer appendix excerpts]**: top-k, normalization, non-negativity and no
  biases; addresses dead units and stale momentum. Reports continual learning
  without replay or task labels. Its static-manifold limitation is explicit;
  joint training with a moving encoder was unsuccessful in the reported initial
  experiments. **[hypothesis]** Learned unit addresses are a close operational
  match to the user's proposal, provided the full locality mechanism is kept.
- [FlyModel, Shen et al.](https://arxiv.org/pdf/2107.07617)
  **[read: abstract, introduction and circuit mechanism]**: sparse random
  expansion followed by winner-take-all; learning changes only synapses from
  active units to the associated output. Other synapses stay unchanged.
  **[difference]** Its associative output update is not ordinary all-output
  cross-entropy training and does not use a learned address classifier.
- [Elephant networks, Lan and Mahmood](https://arxiv.org/pdf/2310.01365)
  **[read: abstract and training-dynamics analysis, sections 2–3 excerpts]**:
  emphasizes sparse gradients as well as sparse representations. A first-order
  expression relates prediction interference to inner products of output
  gradients on old and new inputs. **[difference]** This motivates local
  influence, not an exact k-address routing algorithm or a general guarantee of
  zero forgetting under finite nonlinear updates.

## Interpretation of the proposed module

**[proposal]** Each unit i has an address vector a_i and content vector v_i.
Input q selects the k most similar addresses, and reads a weighted combination
of their contents. The selection is classification into memory neighborhoods,
not prediction of the answer label. Content updates affect only selected units.
Choosing k units generally activates more than k scalar connections.

**[derived]** With fixed addresses, input representation and read weights, and
no shared changing output parameters, disjoint selected unit sets give exact
protection of an old read from a new content-only update. Overlapping sets can
still interfere. Learned addresses can also change old routing, even when only
a few addresses move. Global projections, biases, normalization across units,
scales, momentum and weight decay must be audited for nonlocal changes.

**[hypothesis]** Stable addressing with local content writes is a useful first
implementation; adaptive addresses can follow with explicit checks for old
routing changes. Track old/new recall, unit-set overlap, active-unit coverage,
and changes to inactive units. Same-symbol renderings should share addresses;
unrelated arbitrary associations should overlap little. Fixed addressing is a
mechanism control, not a commitment to permanently frozen perception.

**[derived cost]** For H dense addresses of dimension d and contents of dimension
m, exhaustive similarity scoring costs O(Hd), top-k selection adds its selection
cost, and local read/write costs O(km). Persistent address/content storage is
O(H(d+m)), plus optimizer and utilization state. Sparse writes do not by
themselves make address search O(k).
