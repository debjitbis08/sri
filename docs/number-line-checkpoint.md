# Number-line checkpoint

2026-10-29. Checkpoint for the number line, from the number concept network to multi-step
teaching. Written to take stock before choosing between consolidation and further
capabilities. It summarizes; each claim's evidence and caveats are in the linked results.
Draft for the researchers. Everything here is **pilot evidence**: no number-line contract has been frozen and confirmed
on fresh seeds since the number concept network.

## Setting

Number representations (collections of dots, dice, MNIST digits, tallies, Morse, scripts and
number words; later heard taps) feed small SRI networks. Each input kind enters through a
front end and a linear module coupled by learned translations to a shared core, and to an
output module, under a quadratic energy solved exactly. The core carries learned relations:
a successor map S (*next*), a *same* judgment and, from Part D, a *larger* (comparison) head.
From number construction on, the answer is collection matching (which of 9 collections has
this count), never a label space. Counts are 1–9; inputs are synthetic; from number systems
v1 on, module layers are 64, 32 and 16 wide.

## What the line set out to answer

From the researchers' framing (2026-10-28 and 2026-10-29):
1. **Grouping the representations** (problem 1): expected to have standard solutions.
2. **Pressure to use the core relations** (problem 2): to be invented, and tied to
   confidence: answer by instinct, reach for the relations when unsure.
3. **Teaching like children:** counts in order, one language at a time, grounded in
   collections.
4. **Conceptual instinct:** sense that something is wrong without calculating, because the
   concepts are assimilated; say "I don't know", then ask and learn.
5. **Assimilation through varied contexts**, a few representations acting as senses, and
   rehearsal.
6. **Curiosity (Part C):** detect structured gaps ("something lies between 3 and 5") and ask.
7. **Imagery:** what the system "dreams" when a signal starts in the core.
8. **Belief from sources** and their reliability (later).

## Established in pilots

1. **Relations give meaning to items that were never grounded.**
   - Cross-relation pilot: with a shared successor and one code space, unpaired dice 2, 4
     and 6 align exactly in 3/3 seeds; without succession, 0.00
     ([cross-relation pilot](number-cross-relation-pilot.md)).
   - Part A: forms never paired with collections are matched at 0.31–0.46 through counting
     transitions alone, against 0.40–0.54 for paired forms (chance 0.11)
     ([Part A](number-construction-v1-part-a-results.md)).
2. **Relational inference recovers answers, but leaves no trace by itself.** Revision through
   the learned successor with grounded neighbours lifts matching from 0.21–0.32 to 0.62–0.72;
   a swapped item shows the answer comes from the context
   ([B1](number-construction-v1-b1-results.md)). Training withheld forms on the core's own
   inferences added nothing (Part A pilot 4).
3. **Rehearsal handles forgetting** across languages taught in blocks (Chinese at the end
   0.20 → 0.53), and the conceptual results survive with it
   ([Part D](number-construction-v1-part-d-results.md)).
4. **Varied contexts through a second sense beat repeated contexts** at matched exposure
   (taps paired with collections plus a comparison loss; +0.08–0.17 on acquisition and
   retention). Which of the two components matters is not separated. Repeated contexts were
   worse than none.
5. **Concepts compose where they were never taught** (Part D). Without any taps–form pairing,
   taps match forms at 0.76–0.91, withheld forms included. *Larger*, trained only on
   collections and taps, compares forms at 0.77–0.92. This is the strongest evidence of
   assimilation, as opposed to information a fitted probe can recover.
6. **Learned relations flag violations where concepts are held.** Summed successor residuals
   detect a swapped step in a counting run at AUROC 0.94–0.97 for retained languages and near
   chance for forgotten ones, with no separate anomaly training (Part D). A cheap relational
   consistency check predicts instinct–inference agreement (AUROC 0.66–0.96); recognition
   signals barely do, and the SRI energy runs the wrong way
   ([confidence analysis](number-construction-v1-part-a-results.md)).
7. **Core states project back toward count-specific sensory features** (imagery probe, Part D).
   Prototypes, the inferred successor and the state inferred between two neighbours, solved
   outward, have nearest real inputs of the right count (0.57–0.93; random control 0.04),
   across senses. Decoded "dreams" are blurred and not shown to preserve count.
8. **Relations compose once repeated application is taught.** Successor-then-compare judges
   hidden results better than ignoring the action
   ([composition](number-construction-v1-composition-results.md)). Forward crossings first
   failed because one-step training allows short steps (a learned step covered about a fifth
   of a real step in two of three seeds). An explicit multi-step objective restores full
   steps (0.38 → 0.95) and brings forward crossings to the shown-result level in every seed
   (0.68–0.69) ([multi-step](number-construction-v1-multistep-results.md)).

## Negative results worth keeping

- **Labels do not propagate.** Under sparse labels, unlabelled counts stay at chance
  ([core network](number-core-network-v1-pilot-results.md),
  [number systems v1](number-systems-v1-pilot-results.md)); one-anchor transfer fails
  ([concept network](number-concept-network-v1-results.md)). This led to collection matching.
- **Meta-learning traded learning against forgetting.** First-order MAML adapts slightly
  faster but protects nothing; every inner procedure that learns also forgets
  ([meta v1](number-systems-meta-v1-pilot-results.md)).
- **Episode order gives no advantage** over the same evidence shuffled (Part A).
- **Persistent state between counting steps never trained stably** (Part A; a runaway in
  state magnitude, unresolved).
- **Gating revision on predicted agreement never beat random gating.** Disagreement is
  detectable; which side is right is not (confidence analysis).
- **Equality through the comparison head** teaches ties but, alone, flattens the successor
  along the comparison direction and breaks composition (multi-step).
- **A new script is not learned.** Thai stays at chance after 4 shots per count in every
  setup.

## Not established

- **Instinct as a learned, fast capability.** Every positive result in 6 and 8 is an explicit
  calculation over learned relations. No single-pass detector has been trained or tested on
  held-out combinations.
- **Detecting a conflict between relations** (successor and comparison together) by any means
  other than explicit composition.
- **Knowing which of instinct and inference to trust**, and a revision controller that helps.
- **Asking questions (Part C)** and **belief from sources:** not started.
- **Acquiring a new interface** (script) quickly; English words stay slow (about 0.3).
- **Equality**, taught without harming order.
- **Count-preserving generation** in imagery (only feature-level retrieval).

## Caveats on all of the above

1. **Seeds reused.** Parts A and D, the composition test and multi-step teaching all use seeds
   100–102, and many design choices (including review amendments) were made after seeing
   them. Headline claims need a frozen contract and fresh seeds.
2. **No non-SRI comparison network.** The same losses, curriculum, contexts and rehearsal in
   an ordinary network (shared encoder, the same relation heads, no energy coupling) have not
   been run. Until they are, the composition and assimilation results cannot be attributed to
   SRI rather than to the training design. The cross-relation pilot's ablations (free judge,
   no succession) are the only architecture-level controls, and they predate Part D.
3. **Scale.** Nine counts, synthetic inputs, one small architecture, one image and one
   sequence sense.
4. **Readouts.** Matching among 9 candidates and comparison verdicts are the system's own
   mechanisms, not fitted probes, which strengthens 5; but taps → form chooses among one
   language's forms, an easier task than isolated collection matching.
5. **Costs of multi-step teaching:** paired forms, English and form comparisons lose
   0.04–0.07, about the seed spread.

## Open decisions

- Keep the multi-step objective as the default?
- Teach equality through the *same* judgment instead of the comparison head?
- Split taps from the comparison loss in the varied contexts?
- Consolidate first (comparison network, frozen contract, fresh seeds) or extend first
  (single-pass instinct test, then Part C)?

## Methods to carry forward

- Prefer the system's own answer mechanisms (matching, judgments) over fitted probes, and
  report untaught compositions separately from taught pairings.
- Stratify relational tests so shortcuts fail: critical items, equality strata, and an
  action-and-claim rule beside comparison-only baselines.
- Diagnose before remedying, then confirm the diagnosis by intervention (short steps →
  multi-step objective).
- Make control arms reproduce earlier arms exactly (shared random streams, separate streams
  for added losses), and check it.
- Record protocol amendments with their order of events when results were seen first.
- Keep computation sections to measured costs.

## Bearing on what comes next

Two directions, not yet chosen:
- **Consolidate:** run the non-SRI comparison network on the Part D and multi-step designs,
  then freeze a contract for the main claims (untaught compositions, rehearsal, multi-step
  composition) and confirm on fresh seeds. This decides whether the line's results are about
  SRI at all.
- **Extend:** the instinct test proper (a single-pass detector against explicit composition on
  held-out combinations, including relational conflicts with hidden outcomes), then Part C
  with a concrete trigger (a violation in a retained concept, or a gap state).

The [conceptual instinct notes](conceptual-instinct-notes.md) hold the design discussion for
the second direction.
