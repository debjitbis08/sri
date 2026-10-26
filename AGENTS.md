# Research workflow

Keep README.md focused on describing the project. Put experiment contracts,
results and research-progress updates in docs/ rather than adding them to the
README.

For new experiments and substantive extensions, follow
`docs/computation-reporting.md`: specify theoretical time/storage complexity
and collect measured computation data, including selection and tuning costs.
Keep nondeterministic timing separate from deterministic scientific results.
Do not rewrite frozen historical artifacts to retrofit this requirement;
identify missing historical cost data explicitly in comparisons.

Before freezing a new contract, review prior work as described in
`docs/literature-review.md`: name the established methods the design
corresponds to, what theory or earlier results already predict, known failure
modes and cheaper alternatives. Record shared findings in `docs/prior-work.md`.
