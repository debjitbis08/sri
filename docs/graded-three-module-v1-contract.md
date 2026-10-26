# Graded connections in three 8→4→2 modules

Prospective contract, 2026-10-26. Preserve all historical artifacts. Reuse the
three-module ambiguity v1 local maps, 28 fixtures, evidence clamps, zero free
initialization, outer-only output, no STM/learning, step size 0.1, and horizons
0,1,4,16,64,256,1024,4096. This is a supplied linear rank-two task. Hidden codes
are used only to construct the supplied maps and evaluate complete fixtures;
no answer or inner clamp enters inference.

## Connections and declared grid

Couple corresponding layers after fixed orthogonal alignment, not raw numerical
identity. If h_m=P_m R c and x_m=O_m U P_m R c, use A_h=P_mᵀ,
A_x=(U P_mᵀ Uᵀ + I-UUᵀ)O_mᵀ and A_c=I. P_middle swaps its first two
coordinates; other P are identity. O_output reverses rows and alternates signs;
other O are identity. Every A is square orthogonal, so coupling constrains the
full layer, not just a projection onto the two latent coordinates. At the shared
code manifold all aligned layers agree. Alignment is supplied, not learned.

For adjacent modules and each layer l, compatibility is
0.5*w_l*||A_left,l z_left,l - A_right,l z_right,l||².
Forward influence always has gain 1. Intermediate→input feedback has gain 1
except forward-only controls, where it is 0. Output→intermediate gain is
b in {0,0.25,1}. A derivative block is multiplied by the receiving direction's
gain. Unequal direction gains need not descend a common scalar energy.

Six relative profiles (outer,middle,inner): inner-only (0,0,1), inward-graded
(0.05,0.2,1), uniform (1,1,1), outward-graded (1,0.2,0.05), outer-only (1,0,0),
middle-only (0,1,0). Normalize per-unit weights so 8*w_outer+4*w_middle+
2*w_inner=2, matching the v1 inner-only coupling budget on each edge. This fixes
aggregate reciprocal coupling curvature, not every directional gain or FLOP.
Cross all six profiles with three output feedback gains: 18 configurations.
Add inward-graded total-strength gains 0.25 and 4, each crossed with the same
three output feedback gains (6). Add the original unnormalized inward weights
(0.05,0.2,1), crossed with the three feedback gains (3). Finally add six
forward-only normalized profile controls with both feedback gains zero (6).
Total 33 configurations, all reported; no post-result grid expansion or selection.
This is an exploratory sensitivity comparison, not held-out model selection or
learned coupling. No claim of a general optimum follows from ranking the grid.

## Evaluation and verification

Complete fixtures: output RMSE, paired changes under either cue, first declared
horizon reaching RMSE<=1e-4 and all-case update residual<=1e-5. Missing evidence:
distance to the compatible affine output set, with rank-one ambiguity disclosed.
Contradictory evidence: no correctness target; report common-code infeasibility,
clamp error, local incompatibility and each configuration's weighted reciprocal
energy. Different profiles define different energies: do not rank their raw
energies as a common accuracy measure. Record stationary diagnostics and distance
to the dynamically selected stationary output. Compute this using the free
operator's left-nullspace invariants, not an arbitrary minimum-norm solution.

Check orthogonality and common-manifold alignment, normalized budget, finite-
difference gradients for each profile/strength reciprocal energy, local versus
assembled connection gradients, zero input→output direct edge, feedback gain
blocks, clamp preservation, frozen maps, and descent only for truly symmetric
arms. Reproduce v1 inner-only operators exactly, and trajectories within numerical
tolerance. Check constrained update spectral radius for each of four clamp masks;
report any instability without changing step size or extending budgets. Stationary
solve uses SVD tolerance 1e-10 and checks conserved projections/solution residual.
Ill-conditioned/unsolved diagnostics must be reported, not treated as convergence.

## Computation

Executed dense state dimension S=42. With P=33 configurations, M=28 cases,
T=4096: O(PMTS²) update work, O(S²) deployed operator and O(S) state. Reference
local sparse work is O(K(Nh+hd)+E(N²+h²+d²)) per aligned update, not executed.
Constructing dense residual Hessians costs O(LS²); SVD/eigen diagnostics O(S³)
each. Full-layer alignment construction O(N³) upper bound; fixed small maps
here. No fits, replay, optimizer or learned statistics. Count candidate evaluations
as 33, including all unsuccessful profiles, and all science/check/benchmark calls.
All profile construction, scoring and diagnostic cost belongs to exploratory
comparison cost even though no model is selected. Report its phases explicitly.

Benchmark each configuration at 256,1024,4096 and the direct joint-input SVD
reference: one warmup, three measured batches of 16 sequential queries, wall/CPU
median and range. Include allocations and updates, exclude scoring and checks.
Compare quality versus measured latency and matrix/state bytes; no scaling claim
from these dimensions. Deployment is construction + Q*inference. Record hashes
of contract, runner and imported source, revision/dirty status, environment, BLAS,
threads/workers, hardware limits, dtype, counts, whole/phase timing and peak RSS.
Scientific JSON is deterministic; runtime JSON separate. Manual design time is
unavailable. Benchmark arm traversal follows declared order, not randomized;
small timing differences are not causal speed claims. Historical v1 timings are
not contemporaneous measurements, although inner-only controls are rerun here.
