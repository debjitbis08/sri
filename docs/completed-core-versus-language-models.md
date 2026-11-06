# A completed core compared with modern language models

2026-11-01. Exploratory comparison, not a finalized design, novelty claim or
capability prediction. Follows [core generality, context and perception](core-generality-context-and-perception.md).

## Scope

"Completed" is conditional: perception, contextual state, memory, outputs and
continued learning are not all implemented and validated in one system. Compare
the current code/sparse-reader line and its proposed extensions with documented
model families, not undisclosed proprietary internals or every system called an
LLM. An LLM is a model category, not one uniquely specified architecture.

## Similarities and possible differences

**[derived architectural comparison]** Both can employ learned vector
representations, nonlinear neural transformations, gradient-based training,
predictive objectives, contextual attention and sensory encoders. Calling a
vector a "code" rather than an "embedding" establishes no functional difference.
Multimodality and memory alone are not distinguishing inventions.

Our current distinctive intervention is restricted coordinate fan-in in the
first reader layer, combined in some arms with connection disruption or RigL.
Dense later layers still combine the features. This differs from selecting a
subset of large expert networks per token: feature-level connectivity and
expert-level routing are different granularities, though they can coexist.

The proposed separation between reusable relational structure and locally
writable memories of specific associations could produce a different adaptation
behavior from a fixed-parameter base language model. That separation is not yet
shown to preserve both transfer and particulars under continued learning.
Session memory writes, retained memory across sessions and consolidation into
core parameters must be assessed separately; none implies the others.

Two possible endpoints **[hypotheses]**:

- If attention and a conventional predictive backbone do most computation and
  our readers are a small auxiliary branch, the result may be a language-model
  variant with a sparse relational component and memory.
- If sensory modules, a reusable relational core and independently writable
  memory have causally demonstrated distinct functions, with stable ongoing
  adaptation, the result would differ more substantially in organization and
  learning behavior. A text generator could then be one output interface.

No present evidence establishes that this system understands while LLMs merely
memorize, that it learns on fewer examples, or that it will be cheaper. Sparse
masks in the current implementation use dense operations; sparse connectivity
is not a measured speedup. A complete system can still be a language model if
trained to model language, regardless of its internal organization.

## What would make the difference substantive

Use matched controls to test cross-task reuse, acquisition of new associations,
retention and interference, memory-to-core consolidation, and quality versus
total learning/inference cost. Ablate the core while retaining the same sensory,
context and memory components: otherwise those components may explain the gain.
Compare against dense readers and relevant memory-augmented sequence models.
This is a list of discriminators, not a new experiment contract or authorized run.

## Primary-source record

Search date 2026-11-01. Queries covered Llama 3 architecture, DeepSeek-V3 MLA/MoE,
Titans test-time memory, ATLAS context memorization and Transformer attention.
Evidence is **[abstract]** from primary abstracts/search extracts; no exhaustive
2026 architecture survey or full-paper audit was undertaken.

- [Llama 3 (2024)](https://arxiv.org/abs/2407.21783): documents a dense Transformer
  model family and experiments adding image/video/speech capabilities. Different
  scale and training from ours. Implication **[analogy]**: learned shared vectors
  and multiple senses alone do not establish architectural distinctness.
- [DeepSeek-V3 (2024/2025)](https://arxiv.org/abs/2412.19437): uses mixture of
  experts and multi-head latent attention. Its routed experts are different
  from our coordinate-sparse first layers. Implication **[analogy]**: modern
  language modeling already includes selective computation.
- [Titans (2025)](https://arxiv.org/abs/2501.00663): combines attention with a
  neural memory that memorizes context at test time. Not our memory contract or
  evidence of lifelong consolidation. Implication **[analogy]**: writable neural
  memory is an existing sequence-model research direction.
- [ATLAS (2025)](https://arxiv.org/abs/2505.23735): studies learning to memorize
  context at test time. If this is the Atlas meant earlier by the user, adopting
  it would explicitly bring the project closer to that line; the identity was
  not confirmed. No performance or implementation equivalence is asserted.

No numerical computation was performed. Future comparisons must count encoder
and memory construction, selection, adaptation and inference under
[computation reporting](computation-reporting.md), with separate deterministic
results and runtime receipts.
