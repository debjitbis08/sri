# Ambiguous distributed evidence v1

Prospective contract, 2026-10-26, written before execution. This is a bounded
linear mechanism test, not a replacement for the frozen 8→4→2→1 studies.
Two continuous modules have 4 outer and 2 inner activities each. Fixed local
maps reconstruct distinct outer patterns from a shared two-dimensional code.
The maps and alignment are supplied, not learned. This removes the previous
pressure to equate incompatible outer representations. It does not establish
that the original scalar-centre architecture solves this task.

A observes only its first outer coordinate (a); B observes only its first (b).
All other outer coordinates are free and are read directly as outputs. Each
module alone leaves the other latent coordinate unidentified. Use all 16 pairs
from {-1,-0.5,0.5,1}², with zero initialization except cue clamps, no STM, no
parameter updates, no tuning and no random sampling. These are exhaustive
fixture evaluations, not held-out evidence of learned generalization. Changing
either cue changes missing outputs in both modules.

Local energy is 0.5||x_m-D_m c_m||². Reciprocal coupling adds
0.5||c_A-c_B||². D_A rows are (1,0),(0,1),(1,1)/sqrt(2),
(1,-1)/sqrt(2); D_B rows are (0,1),(-1,0),(1,-1)/sqrt(2),
(1,1)/sqrt(2). Only aligned inner coordinates are coupled. Simultaneous gradient
steps use rate 0.1. The forward A→B arm applies the coupling derivative only
to B; B→A is its mirror. Source activities still evolve under their local
energy. Such directed dynamics need not minimize the reciprocal energy.
Isolated modules omit coupling. A direct joint-input baseline extracts a,b
from the two observed coordinates and applies the same fixed maps. It receives
exactly the same cues and maps as recurrence.

Read outer outputs at steps 0,1,4,16,64,256,1024, regardless of settling. Report
missing-coordinate RMSE overall and per module, and update residual. A diagnostic
convergence threshold is free-gradient norm <=1e-5; mark failures at the cap.
No extension or selection after results. Feedback contribution requires lower
missing-output RMSE than both directed controls at 1024 on the full fixture;
practical superiority additionally requires a quality–cost advantage over the
direct baseline. No universal necessity claim follows from edge deletions.

Verify gradients numerically, cue preservation, frozen maps, reciprocal energy
descent, exact direct reconstruction, isolated ambiguity, and sensitivity to
both cues. Record all four trajectories and direct outputs. Benchmark each arm
at each nonzero budget using one warmup and five measured batches of 16 cases;
include allocations and updates, exclude scoring, report wall/CPU median and
range. Benchmark direct inference identically. Run once; no fitting, selection,
or tuning. All timing is separate from deterministic scientific JSON.

For K=2 modules, N=4 outer units, d=2 inner units and E=1 coupling edge:
construction/storage of supplied maps O(KNd); inference per update O(KNd+Ed),
T steps O(T(KNd+Ed)); direct O(KNd). Activity state O(K(N+d)), temporary workspace
O(K(N+d)); parameters KNd float64 values (128 bytes), activity 12 float64 (96
bytes). No optimizer, replay, statistics, or learned state. Count fixture storage
separately. Full batch benchmark workspace scales by sample count only for
retained output arrays; cases execute sequentially. Report source/contract hashes,
revision/dirty status, environment/BLAS/thread settings, phase costs, inference
calls and updates including checks/warmups/benchmarks, and process peak RSS.
Historical studies are not equal-task comparators; their timing cannot establish
relative efficiency here. Manual design cost is unavailable.
