# Forget-and-relearn, iterated learning, and the fan-in reset result

2026-10-30. User-supplied literature follow-up. This corrects an omission in the
earlier mechanism discussion: forget-and-relearn is an established framework
directly relevant to the result, not a newly proposed general principle.

## Sources and reading scope

Searches: "Zhou 2022 Fortuitous forgetting connectionist networks ICLR forget
relearn"; "Ren 2020 neural networks iterated learning Kirby"; "Kirby Cornish Smith
2008 cumulative cultural evolution laboratory linguistic structure iterated
learning PNAS". Follow-up: "Li Bowling 2019 Ease of Teaching compositional
language receiver reset". Primary sources below were opened. The OpenReview
PDF challenged access; Zhou's arXiv conference-paper PDF was read instead.

- **Zhou, Vani, Larochelle and Courville, ICLR 2022:**
  [Fortuitous Forgetting in Connectionist Networks](https://arxiv.org/pdf/2202.00155).
  **[read: §§2–3, 4.2, 5.1–5.2, Appendix A5's continual-learning qualification]**.
  Partial perturbations disproportionately damage non-compositional mappings
  in their controlled setting. Their layer-freezing experiments implicate
  adaptation of retained early features. They challenge a purely ease-of-teaching
  explanation and emphasize relearning, including improvements during interaction.
  Their setting permits iterative re-exposure; it does not establish retention
  after evidence permanently disappears. Our sparse moving edges and number task
  are different interventions and outcomes.
- **Kirby, Cornish and Smith, PNAS 2008:**
  [Cumulative cultural evolution in the laboratory](https://doi.org/10.1073/pnas.0707835105).
  **[read: abstract, iterated-learning description, discussion and conclusions]**.
  Human transmission chains become more learnable and structured. Without a
  constraint against ambiguity, simplification can lose distinctions; structured
  transmission need not preserve expressivity. Our repeated parameter resets
  are not a human transmission chain. The analogy is selection for recoverable
  structure, with task performance needed to distinguish useful simplification
  from information loss.
- **Ren, Guo, Labeau, Cohen and Kirby, ICLR 2020:**
  [Compositional Languages Emerge in a Neural Iterated Learning Model](https://arxiv.org/pdf/2002.01365).
  **[read: §§3.1–3.2, Algorithm 1, §§4.1, 4.3 and discussion; Appendix C's
  qualification of Eq. 11]**. Their reset agents undergo limited imitation,
  interaction and transmission. Structured languages have an observed learning
  advantage, and the pretraining budget matters. Their generational-improvement
  argument assumes the favorable learning bias; Appendix C explicitly qualifies
  its central inequality rather than providing an unconditional theorem. Our
  network has no generational imitation/transmission stage and no corresponding
  guarantee of increasing compositionality.
- **Li and Bowling, NeurIPS 2019:**
  [Ease-of-Teaching and Language Structure from Emergent Communication](https://proceedings.neurips.cc/paper_files/paper/2019/file/b0cf188d74589db9b23d5d277238a929-Paper.pdf).
  **[read: abstract and introduction]**. Replacing listeners pressures a retained
  speaker toward easier-to-teach communication. This is the closest architectural
  analogy to retaining an encoder while disrupting readers. It is not the same
  algorithm, and the easier-readout interpretation must be tested against the
  selective-forgetting alternative emphasized by Zhou.

## Updated interpretation of this project's result

**[Analogy/hypothesis]** The random-rewire arm is plausibly a forget-and-relearn
intervention. It retains encoders and substantial reader state, disrupts selected
connections, and continues supervised learning. Meaningful training signals,
not the random choices themselves, can favor useful recovered features.

Three explanations must remain distinct:

1. Useful structure is immediately more resistant to the reset.
2. Useful structure is reconstructed faster after the reset.
3. Repeated recovery changes the retained encoder representation, making it
   more reusable under different readers.

They can coexist. Final transfer and rank do not identify which occurs. Calling
all three "simplicity pressure" hides the experimentally important distinctions.
The literature shifts priority from finding a special coordinate toward the
survival-and-recovery trajectory. It does not establish that RigL and random
rewiring are equivalent, nor that all erased knowledge is undesirable.

The project's task is shared quantity across modalities; an approximately scalar
code is not itself compositional language. Transfer scores, exact matching,
within-script order and retention must remain separate outcomes. The taught-fit
cost could include lost distinctions rather than solely removal of shortcuts.

## A conditional mathematical model of the learning-speed filter

**[Derived here; not attributed as a theorem of the papers.]** Suppose a scalar
learning mode has fixed target a*, recovery rate λ>0, and gradient-flow dynamics
da/dt=λ(a*−a). Each cycle first attenuates its previous end-of-cycle amplitude
by r∈[0,1), then relearns for T time units on the same target. Hence

    a_(k+1) = (1−exp(−λT))a* + r exp(−λT) a_k.

Solving the linear recurrence gives the limiting retained fraction

    a_∞/a* = (1−exp(−λT)) / (1−r exp(−λT)),  for a*≠0.

It increases with λT and with r. For equal r, quickly learned modes are retained
more strongly; for equal λ, reset-resistant modes are retained more strongly.
If λT is large, even slow distinctions relative to other modes can recover and
the filter disappears. If λT is tiny, all modes with appreciable forgetting are
suppressed. This supplies a conditional reason for an intermediate reset interval.

Assumptions: independent fixed-target modes, linear recovery, fixed attenuation,
and continued evidence. Random coordinate changes, feature learning, Adam and
ReLU deaths violate these assumptions. Fast modes are not automatically useful
or compositional: that association must be measured. This is a symbolic
derivation only, not a fitted model or numerical experiment.

## What to measure before another mechanism claim

- At a fixed checkpoint, on copies, measure immediately before/after a declared
  reset and after matched recovery budgets. Separate retained performance,
  recovery slope and eventual transfer; avoid comparisons distorted by floor
  or ceiling effects.
- Freeze encoder and projection during recovery on one copy; allow both to
  update on another. This distinguishes immediate readout recovery from a
  change in representation during that recovery window. It differs from the
  earlier diagnostic of training fresh readers on already frozen final codes.
- Compare reset-in-place with rewire-and-reset, and vary recovery interval at
  matched exposure and total reset budget. Do not assume more forgetting is
  monotonically better. Selection of intervals requires a new contract and
  accounting for all candidates.
- Use fresh-reader learning curves on frozen final representations to test the
  teachability consequence. It can support improved code accessibility without
  proving that teachability caused the training benefit.

For the decaying-memory proposal, continuous attenuation may instantiate related
pressure but is not equivalent to discrete resets. Its form-free delay lacks
the old targets needed for repeated old-task relearning. Acquisition and rehearsal
are where a forget-and-relearn mechanism can consolidate those mappings.

Cheaper established baselines before inventing more wiring rules: partial
parameter reset without topology changes and later-reader reset with retained
encoders. Their rates, layer scope, initialization and optimizer-state policy
must be specified, not imported as equivalent to the published algorithms.

No new training or computation claims. References provide prior mechanisms and
test designs; they do not prove the cause of this pilot's transfer.
