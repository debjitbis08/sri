# SRI: Learning Through Relational Abstraction

*Research direction — October 2026*

## 1. Motivation

SRI currently emphasizes error reduction and energy minimization. While useful for imitation, prediction, and adaptation, these mechanisms do not necessarily produce understanding.

A positive-definite quadratic energy has a unique equilibrium for each fixed inference problem. More importantly, minimizing compatibility penalties does not explain how reusable abstractions emerge.

**Hypothesis:** Learning involves discovering relational structures that remain meaningful across different experiences and domains.

## 2. Example: Driving to Painting

A person drives 1,000 km, experiencing movement, anticipation, fatigue, uncertainty, and eventual release.

Later, they produce an abstract painting expressing similar feelings without depicting the journey.

The transferable knowledge is not the sensory content but an underlying relational pattern:

**Tension → Persistence → Transformation → Release**

This pattern could also appear in music, mathematics, storytelling, or other activities.

The research challenge is to discover such patterns without explicitly programming their meaning.

## 3. Two Complementary Learning Phases

### Wake: Experiential Learning

- Interact with the environment.
- Learn through imitation, prediction, and error correction.
- Acquire domain-specific relational structures.
- Adapt rapidly to new evidence.

**Primary purpose:** Acquire experience.

### Sleep: Structural Abstraction

- Revisit and recombine acquired experiences.
- Discover similarities in their relational organization.
- Identify invariants and compositional patterns.
- Consolidate reusable structures into the core.
- Explore counterfactual configurations through dreams.
- Preserve useful knowledge through deeper abstraction rather than raw replay alone.

**Primary purpose:** Reorganize experience into transferable understanding.

These are dominant objectives, not necessarily exclusive biological or computational phases.

## 4. Connection to Algebraically Constrained Relational Webs

The relational-web research provides candidate mathematical foundations:

- **Algebraic compatibility:** Relationships need not represent similarity or equality.
- **Path composition:** Different experiential sequences may share underlying algebraic relationships.
- **Local-to-global constraints:** Local rules can restrict possible collective structures.
- **Structural invariants:** Useful relationships may survive transformations and changes of representation.
- **Quotients and equivalence:** Abstraction may remove irrelevant distinctions while preserving meaningful relational behavior.

A possible abstraction map is:

\[
\phi_i:W_i\rightarrow A
\]

where \(W_i\) represents an experience and \(A\) a reusable abstract relational structure.

An important requirement is approximate preservation of composition:

\[
\phi(g\circ f(x))
\approx \bar g\circ\bar f(\phi(x)).
\]

The relational-web studies do not yet provide an abstraction-learning algorithm; that bridge remains to be developed.

## 5. Rethinking Optimization

Rather than optimizing one universal energy function, investigate distinct requirements:

- **Coherence:** Internal relationships remain mutually compatible.
- **Compression:** Repeated structure can be represented economically.
- **Structural preservation:** Meaningful compositional relationships survive abstraction.
- **Stability and plasticity:** Established knowledge persists while useful changes remain possible.
- **Transfer:** Learned fundamentals work in unfamiliar domains.

Reconstruction error remains useful for wake learning but should not define understanding.

Sleep optimization might involve constrained structural transformation rather than ordinary gradient descent on prediction error.

## 6. Proposed Minimal Experiment

Construct two artificial domains with different observations and actions but shared hidden relational patterns.

Compare:

1. Wake-only prediction learning.
2. Wake learning with ordinary replay.
3. Wake learning with compression.
4. Wake learning followed by relational abstraction.

Evaluate whether the fourth system can transfer acquired relational knowledge to unseen tasks in the second domain.

Include controls for representational capacity, abstraction validity, and leakage of the intended shared structure.

## 7. Central Research Question

**Can an adaptive relational system discover, preserve, and reuse structural invariants across experiences without being explicitly taught what those invariants mean?**

The immediate goal is not to construct a larger SRI module, but to establish a minimal mechanism for relational abstraction and cross-domain transfer.

A successful result would demonstrate a capability qualitatively different from reducing reconstruction error.
