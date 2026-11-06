# From sparse relational readers to grammar learning

2026-11-01. Research discussion, not a contract or experiment. Follows
[hypothesis-class properties](sparse-reader-hypothesis-properties.md).

## What scaling does and does not supply

**[derived from source]** The current GenericSequenceNet P sees two independently
encoded items, not a recurrent summary of an entire observed prefix. The CNN
sequence encoder in `src/number_core.py::MorseEncoder` represents one sensory
item using fixed-shaped convolutions and flattening; it is not an across-item
language model. P's imagined rollouts are not persistent state updates while
reading real observations. The current g is a pairwise relation scorer.

**[derived]** Under this interface, histories with the same last two encoded
items get identical next-item predictions, regardless of width or training data.
For bracket tokens, prefixes `([]` and `[[]` both end in `[]`, but completing
the remaining outer bracket requires `)` versus `]`. Restrict attention to
closing the remaining structure, since other valid extensions can also exist.
Their sets of legal continuations differ. This is an information obstruction,
not merely a difficult optimization problem. A contextual encoder would change
the premise, and must be counted as an architectural change.

The code-reader class can express non-counting relations, but no result yet
shows the sparse-disruption training advantage transfers to grammar. A finite
larger network may learn bounded patterns; increasing width alone cannot supply
history missing at its input.

## Memory requirements that can be proved

**[derived]** A deterministic streaming recognizer whose entire persistent state
has B bits has at most 2^B states (assuming no other growing store or external
position counter). If two prefixes require different legal suffixes, exact
recognition must distinguish them. This is the distinguishable-prefix argument
underlying finite-state language theory.

For balanced brackets with k>=2 types and maximum nesting depth D, there are
k^D possible all-opening prefixes of depth D. Each has its own unique closing
suffix; appending that suffix to a different prefix fails. Thus at least k^D
states, or D log2(k) bits, are required just to distinguish these prefixes.
Unbounded exact recognition requires unbounded distinguishable memory states.
One bracket type has a different lower bound: depths 0,...,D require at least
D+1 states, hence log2(D+1) bits. A counter alone does not retain bracket types.

These are worst-case exact statements, not predictions of practical natural
language performance. Infinite-precision real-number encodings evade the finite
state premise mathematically; fixed finite precision and robustness cannot be
silently replaced by that idealization. An explicit stack is a sufficient
mechanism for bracket languages, not a requirement for all useful NLP or all
possible recognition architectures. A single deterministic stack also does not
recognize every context-free language; general parsing may require ambiguity
management. Arbitrary formal languages have no one uniformly learnable solution.

## Changes worth testing

1. **Prefix context.** Replace raw two-item prediction by a shared recurrent
   update h_t=F(h_(t-1), e(x_t)), or contextual token representations with causal
   attention and order information. Keep a dense version as a baseline. The
   sparse reader can be tested inside F or as a scorer of contextual states;
   these are different experiments. Gates/residual paths are candidate ways to
   retain state; memory retention is required, that implementation is not.
2. **Preserve distinctions needed by syntax.** Identity, role, agreement and
   pending dependencies can vary independently. Give the representation enough
   capacity and test whether it retains them. Named grammar slots and parse
   labels are additional supervision, not necessary assumptions. In particular,
   do not assume every coordinate should redundantly encode one scalar as in
   the count fixture. Sparse groups or larger fan-in are alternatives if one
   coordinate per operand obstructs composition; no such failure is established.
3. **Appropriate prediction and use.** Replace the nine-count answer interface
   with a vocabulary distribution, including end-of-sequence, or the outputs of
   a specified parsing/transduction task. The existing candidate-distance loss
   already produces a categorical distribution, but is tied to collection
   candidates. Natural language has multiple valid continuations; a single
   next-code target is not a complete specification of that distribution.
   Parsing, execution or interpretation tests whether learned patterns are used
   beyond grammaticality scoring. Count all supervised structure supplied.
4. **Scalable working memory if needed.** For a declared nesting/generalization
   target, test a stack, dynamic slots, attention over history or a parser.
   Memory for the current prefix differs from long-term memorized facts. Adding
   an arbitrary-pair store does not by itself implement pending grammatical
   dependencies. A stack provides structural bias that must be acknowledged.
5. **Check disruption, rather than assume it helps.** Preserve dense and static
   sparse controls, and match disruption severity. Resetting a state-update
   network may destroy useful long-range state dynamics. The existing result
   concerns readers of count codes, not recurrent transition learning.

## Small discriminating evaluation before scaling

Use symbolic tokens first to separate grammar learning from perception. Start
with a finite-state grammar requiring memory beyond the last two symbols;
then balanced brackets with two types (depth and length holdouts separately);
then role/composition-sensitive transduction and controlled language agreement.
Use generated negatives matched on superficial length/token-count cues; include
wrong bracket types and illegal order, not only unmatched counts. Report complete
string validity, legal-next-token performance and task execution/parse accuracy
as appropriate. Test new combinations and new lexical assignments, not only new
renderings. Reversal and copying can diagnose memory organization, but are not
the same task as language membership.

Compare a dense recurrent baseline, a sparse recurrent variant, and, for nested
tasks, an explicit-memory reference. Cheap finite-state or stack oracles establish
the task's algorithmic solution without claiming a learned result. Separate
architecture-selected development grammars from held-out grammar instances.
BLiMP-style minimal pairs assess grammatical preferences; COGS-style semantic
parsing assesses compositional use. Neither alone establishes general language
competence, and benchmark scores do not prove an unbounded grammar was learned.

## Prior work record

Search date 2026-11-01. Queries covered Hahn self-attention formal-language
limits, Weiss/Goldberg/Yahav finite-precision RNNs, Grefenstette stack/queue
transduction, BLiMP and COGS. Evidence below is **[abstract]** from primary
landing pages or search summaries, not a full theorem review.

- [Weiss et al. (2018)](https://aclanthology.org/P18-2117/) analyze practical
  computational power of recurrent architectures with stated precision/time
  assumptions. Their recurrent models differ from our two-item feedforward
  predictor. Implication **[analogy]**: state dynamics and numerical assumptions
  matter; finite-precision terminology alone is not a transferable theorem.
- [Hahn (2020)](https://aclanthology.org/2020.tacl-1.11/) gives formal-language
  limitations under specified self-attention assumptions. This is neither our
  architecture nor an impossibility result for every Transformer variant or
  practical NLP. Implication **[analogy]**: increasing capacity and proving
  unbounded recognition are distinct claims.
- [Grefenstette et al. (2015)](https://arxiv.org/abs/1506.02516) study recurrent
  networks augmented by differentiable stacks, queues and deques on transduction
  tasks. Their mechanism and tasks differ from sparse-reader training.
  Implication **[analogy]**: explicit structured memory is an established control,
  not a novel proposal or a guarantee of learning arbitrary grammar.
- [Warstadt et al., BLiMP](https://arxiv.org/abs/1912.00582) introduce linguistic
  minimal-pair evaluation; [Kim and Lin, COGS](https://aclanthology.org/2020.emnlp-main.731/)
  introduce compositional semantic-interpretation evaluation. Different from
  counting and from formal recognition proofs. Implication **[analogy]**: separate
  grammatical preference from compositional interpretation in our evaluation.

## Computation for a future contract

No run or numerical analysis accompanies this note. Declare length T, hidden
width H, vocabulary V, fan-in, stack depth, batch size and all tuning candidates.
A dense recurrent transition has typical O(T H^2) work and O(H) streaming state;
full BPTT typically retains O(T H) activations plus parameters and optimizer.
A dense vocabulary head adds O(T H V) work. Standard full attention includes
O(T^2 H+T H^2) work per layer and a straightforward implementation materializes
O(T^2) attention scores; implementation-specific memory improvements must be
reported separately. A discrete bracket stack stores O(D log k) bits and uses
constant-time push/pop; a differentiable implementation may scan its memory,
so these oracle costs do not automatically describe it. The current masked dense
reader does not automatically realize sparse-operation savings. Record generation,
training, selection, evaluation and peak memory, with deterministic scientific
results separate from timings, under [computation reporting](computation-reporting.md).
