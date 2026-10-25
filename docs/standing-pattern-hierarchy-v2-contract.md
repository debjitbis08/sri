# Stored-pattern hierarchy v2: graded inter-module coupling

Prospective amendment, 2026-10-26, requested by the user while v1 was running:
all corresponding layers couple, weakly outside and strongly inside. Retain v1
as the innermost-only control; do not change its source or artifacts. All v1
contract provisions apply except the following explicit changes.

Coupling is sum_l gamma_l/2 * ||x_input,l-x_output,l||^2, with per-coordinate
strengths gamma=(0.05,0.2,1,4) at widths (8,4,2,1). There is no normalization by
width: total coefficients per layer are (0.4,0.8,2,4), so both individual and
aggregate strength increase inward. Identity alignment between corresponding
units is supplied, not learned. These strengths were chosen before reading v1
scores, not tuned against those scores. v1 scores were subsequently inspected
before writing this amendment; this is not an independent blind confirmation.

Quadratic frozen/plastic, affine and zero-detail arms now use graded coupling.
Keep the uncoupled quadratic arm and add an explicit innermost-only quadratic
control (strength 4). Six arms x eight queries x 2048 updates = 98,304 main
updates. No inner target or true latent variable is supplied during inference.
Early output can now respond directly to weak outer coupling; success alone
cannot establish that information traveled inward then outward. The innermost-
only control is necessary for that distinction. Coupling can also bias output
toward input details even when the modules ought to represent different signals.

With no evidence, separately preferred module patterns are generally incompatible
with identity coupling. Run 2048 unclamped activity steps starting at those patterns,
record the resulting joint default and free gradient; do not assert the separate
patterns remain equilibrium. This is a diagnostic, not a proof of convergence or
uniqueness. Preserve the local zero-evidence equilibrium check with coupling zero.
The core prior still selects zero for independent modules. Graded coupling may
shift the joint preferred cores. No claim of a global convex energy is made.

Benchmarks now use graded quadratic coupling. Direct reconstruction is a task
comparator, not the minimizer of the coupled energy. Same 28 fits, examples,
fixed horizons and reporting requirements. Gradients additionally check every
cross-layer coupling term. Additional coupling costs O(N) time/storage. Store
15 float64 coupling coefficients (120 bytes). Record no-evidence work and main,
benchmark and direct inference work. No parameter/strength search, retuning or
selection. Run once:
`.venv/bin/python -m experiments.standing_pattern_hierarchy_graded`.
