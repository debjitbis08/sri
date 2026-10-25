# Three-module graded hierarchy v3

Prospective contract, 2026-10-26. User requests an intermediate module. Preserve
v1/v2 artifacts. All v2 and inherited v1 specifications apply except these changes.

Topology is input (0) ↔ intermediate (1) ↔ output (2); there is no direct 0–2
edge. Every module has continuous layers 8→4→2→1 and the same local energy and
quadratic preferred-detail operator. Each of the two edges couples corresponding
layers with per-unit strengths (0.05,0.2,1,4), weak outside and strong inside.
Only module 0 outer activity is clamped; only module 2 outer activity is output.
There is no inner target or intermediate clamp during inference.

Keep the original input/output generator, nine training pairs, eight held-out
pairs, fit procedure and rates exactly. The intermediate module's 21 stored
coefficients start at zero: it prefers equal children, with no preferred detail.
Do not supply or fit intermediate targets. This is an explicit neutral-detail
initialization choice, not evidence that all possible intermediate preferences
have been tested. It still has the same state dimensions, local factors and weak
core prior as the other modules. In the plastic arm all three modules' coefficients
adapt; each query starts from the same pretrained endpoints and neutral middle.
This is not learning the middle from a continuous training stream.

Keep six arms: quadratic frozen/plastic, affine frozen, zero-detail frozen,
uncoupled quadratic, innermost-only quadratic chain. The latter also has two edges
and a middle module; do not confuse it with v2's two-module control. Use the same
horizons through 2048, including early outputs without waiting for equilibrium.
No extension based on results. Compare all curves with frozen v2 results, with
both matched update count and measured inference cost. Direct task reconstruction
still uses input mean and stored output expansion, bypassing the middle and energy.

Check finite-difference gradients for every activity and parameter (including
middle parameters), fit of endpoint coefficients, zero independent defaults,
pair-averaging identity, descent and clamping. Additionally verify zero direct
input–output derivative coupling and nonzero nearest-neighbor coupling by
perturbing one output coordinate and inspecting activity gradients. Verify the
input and desired output fixtures reproduce v2 exactly. No-evidence run: 2048
unclamped steps with frozen coefficients; record joint defaults/residual without
assuming convergence. Main queries: 6*8*2048=98,304 updates; no-evidence 2,048;
benchmarks 67,584, totaling 167,936, plus checks and 28 endpoint fits.

For K modules of outer width N, local factor count K*(N-1) and edge count K-1.
Activity/gradient and parameter costs are O(KN) and O(KNp), p<=3. Update cost
O(KNp); T-step inference with J trials O(TJKNp). Endpoint fitting remains
O(N*(M*p²+p³)) with M=9, since the middle is not fitted. Direct output reconstruction
is O(Np) in principle; this runner expands all modules and pays O(KNp). No grouping
search, tuning, selection or replay. Model coefficients: 63 float64 =504 bytes;
activity:45 float64 =360 bytes; fixed coupling/index arrays:120+168 bytes; total
retained arrays:1152 bytes, excluding Python overhead and temporary workspace.

Use the same timing protocol, environment and separate deterministic science and
nondeterministic runtime reports. Record whole/phase times, benchmark spread,
counts, provenance and peak RSS. No hardware-efficiency claim from these small
sizes. Historical v2 timing is available but not a contemporaneous randomized
benchmark. Run once:
`.venv/bin/python -m experiments.standing_pattern_hierarchy_chain`.
