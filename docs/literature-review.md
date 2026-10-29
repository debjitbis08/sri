# Literature review for contracts

Effective 2026-10-26. Applies to new experiment contracts and substantive
extensions. Frozen historical contracts remain unchanged; later reviews of them
go in [prior work](prior-work.md), not into the frozen files.

The aim is practical: avoid rebuilding a known method under a new name, avoid
directions already known to fail, and find cheaper established alternatives or
baselines before spending a run.

Keep recording useful findings during research discussions, even before a
contract is proposed. Put topic-specific reading notes in docs/ and link them
from docs/prior-work.md, separating source evidence from program implications
and untested hypotheses. Start from the program's problem: understand the best
relevant solutions and why they work, then identify useful improvements.

## Before freezing a contract

Each contract gets a **Prior work** section, written before freezing, that:

- names the established method each component corresponds to, using the
  standard term where one exists (for example "predictive coding learning
  rule" rather than only "clamped-energy rule");
- states which results the literature already predicts or explains, and
  whether any part of the outcome follows from theory without running it;
- lists known failure modes of the chosen approach and how the design avoids
  them, or why it accepts them;
- names established baselines or cheaper alternatives, and either includes
  them or says why not;
- states what, if anything, the study adds beyond prior work. "Nothing new;
  this is a needed internal control" is an acceptable answer.

Keep the section short and point to [prior work](prior-work.md) for the
shared survey. Add new findings to that survey rather than repeating them in
each contract.

## Search record and honesty

- Record the search date and the queries or sources used.
- Tag the evidence level of **each** claim that informs a design, not the
  survey as a whole. Use the tags defined in [prior work](prior-work.md):
  read, abstract, familiar, derived, analogy.
- For each source, keep three things separate:
  - what it actually demonstrated;
  - how its setting differs from this architecture;
  - the hypothesis the project draws from it.

  A related result is context unless the design shows why it transfers.
- Do not equate methods because their update equations look alike.
  Distinguish:
  - similar local update rules;
  - equivalent inference, such as the same conditional mean;
  - equivalent statistical estimators, such as the same likelihood.
- Name implemented variants accurately. A mechanism inspired by a published
  method is described as such and is not reported as a test of that method.
- "No match found" means only that these searches found none. It is not a
  novelty claim. Novelty claims need a targeted search and remain provisional.
- Cite primary sources with links. Secondary summaries (press releases,
  blog posts, search-engine digests) are pointers, not evidence.
- Prior results from different fixtures or scales are context, not
  predictions this study must reproduce.
