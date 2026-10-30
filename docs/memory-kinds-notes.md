# Kinds of memory and learned routing (design note, not a contract)

Written 2026-10-31 from a research discussion after the [limited-memory results](limited-memory-v1-results.md)
and during the [arbitrary-pairings](arbitrary-pairings-v1-contract.md) preparation. Nothing here has been run.
Prior-work tags as in [prior work](prior-work.md); every citation below is **[familiar]** and must be read before a
contract is frozen.

## The four kinds, and what they are for

| Kind | Holds | Time scale | Measured today by |
|---|---|---|---|
| Relational system | structure that supports transfer across senses | long; shapes the whole network | the eight transfer measures, (a) |
| Recent associative memory | a pair learned a moment ago | tens to hundreds of updates | pairing acquisition curves |
| Durable storage | pairs kept through interference and delay | thousands of updates | retained recall; the fork's handoff |
| Working memory | items held within an episode, across distractors | steps within an episode | nothing yet |

The 16 dense units with decay already span the middle two by half-life (125, 500, 2,000), but as one pool.

## What a label is not

Calling a part "working memory" or "durable storage" because of its design is not the mechanism. A part earns
the label only by a **dissociation**: silencing it, loading it with interference, or delaying it changes the task that
needs that kind and not the others (a double dissociation where possible).

## The learning problem (researchers' point, 2026-10-31)

The system has to **learn to use the right memory for the right task**, during learning and at inference. Having
several memories is not enough. Training must also teach the routing. This asks, in order:

1. **What selects the memory?** A task cue, a context signal, or an inferred property of the input. If the cue is
   given, routing may collapse into reading the cue. If it must be inferred, the system needs something to infer from.
2. **How is routing learned?** Candidates: a learned gate over memory pools (mixture-of-experts style), a learned
   write and read policy (memory-augmented networks), or no explicit gate with routing emerging from pressure
   (limited capacity and decay). The limited-memory work so far is the third kind, with the two parts' roles left to
   emerge.
3. **What pressure makes routing worth learning?** A mix of tasks where one kind of memory is necessary and another is
   harmful or wasteful: arbitrary pairings (memory), structured relations (structure), within-episode holding (working
   memory). Decay and capacity limits create the pressure; the curriculum must keep all tasks present so no pool takes
   everything.
4. **How is selectivity measured?** Routing selectivity (does gate or usage differ by task), and causal tests: silence
   the right memory for a task and the wrong one, and compare. A system that uses the wrong memory should fail in a
   specific way.

## Failure modes to design against

- **Collapse:** one pool takes all tasks, so the labels mean nothing.
- **Shortcut routing:** the gate learns task identity from a leaked cue and the memories do not specialize.
- **Credit assignment through decay:** a decaying memory's contribution is hard to attribute, so the gate sees a noisy signal.
- **Handoff versus routing:** the fork showed knowledge moving from memory to structure; a routed system must choose
  when to keep it in memory and when to hand it off.

## Prior work to read before a contract

All **[familiar]**: mixture of experts and learned gating (Jacobs et al. 1991; Shazeer et al. 2017); neural Turing
machines and differentiable neural computers (Graves et al. 2014, 2016); memory-augmented meta-learning
(Santoro et al. 2016); fast weights (Hinton and Plaut 1987; Ba et al. 2016); complementary learning systems
(McClelland et al. 1995; Kumaran et al. 2016); working-memory models in networks (gated recurrent units and
LSTMs hold items within an episode). The established parts are the mechanisms; what a routing study would add is
the learned choice among memories of different time scales under decay pressure, on this program's number tasks.

## What this means for current work

- The pairings study stays as designed. It tests recent against retained recall in one pool and does not train a router.
- Working memory needs a new task (a within-episode holding task) and its own contract.
- A routing study would add pools of different half-lives, a curriculum interleaving the three task families, and the
  dissociation and selectivity measures above. It would be written as a new contract with the literature review done first.
