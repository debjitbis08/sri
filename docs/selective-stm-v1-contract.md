# Selective STM v1: frozen long-term weights

Prospective contract, 2026-10-26. User separates inference from major learning:
only selected modules have immediate memory, controlled by a parameter; major
learning is intended for sleep/dream. This study tests selective immediate memory
only. It does not implement, simulate or validate sleep/dream learning.

Use the frozen three-module v3 quadratic weights, eight held-out signal pairs,
and graded chain. No weights change, no fitting or optimization of weights occurs.
The historical nine-pair least-squares pretraining is initialization, not a sleep
algorithm. Preserve all earlier artifacts.

Only the middle module receives a 15-coordinate STM trace m. Add
lambda/2 * ||x_middle-m||² to the existing energy, with predeclared lambda values
0 (control), 0.1 and 1. These are experimental conditions, not selected candidates.
During externally signaled input-present intervals, update m by
m <- (1-alpha)*m + alpha*x_middle after each activity update, alpha=0.2. During
input-absent intervals hold m fixed. Alpha is zero for the endpoint modules: they
have no STM trace. The write gate is explicitly supplied by the protocol, not
learned. There is no decay in this first test, so persistence is partly imposed.
All arms maintain/update the middle trace equally; lambda=0 prevents its readback.

Start each episode from the v3 independent default with m=x_middle. Run four
consecutive phases without resetting activity or memory: observe signal s for
1024 updates; remove all clamps and recall s for 1024; observe -s for 256; remove
all clamps and recall -s for 256. The replacement is another existing held-out
signal, not a new independent example. Record outer output at phase steps
0,1,16,64,256 and (where present)1024. No inner target is ever supplied. At new
input onset only the input module's outer activities are clamped; at removal no
input is replaced with zeros. Ground-truth output is used exclusively for scoring.
There are 3*8*2560=61,440 activity updates and 30,720 trace writes.

Keep state rate 0.1, shared gradient descent/backtracking (up to 20 halvings),
float64, one process/thread. Check every activity and trace derivative by central
differences (1e-6, tolerance 1e-7), frozen weights byte-for-byte, descent for each
fixed-trace activity update and the subsequent trace write, input clamping, and
exact lambda-zero equivalence to one existing v3 frozen update. Weights need no
gradient update. No convergence stopping; read residuals diagnostically.

Report per-phase RMSE against desired outer output, output change from the end
of first encoding during recall, old-target error after replacement, every outer
readout and mean trace motion. Stored weights and STM must be distinguished.
A direct comparator retains the input scalar mean in a one-value trace with the
same write gate and alpha, then reconstructs the output using frozen functions.
It receives no output target. Report its errors and memory: this task may remain
solvable without recurrent dynamics. Do not claim a recurrence advantage.

Cost: with K=3,N=8,p=3, activity updates cost O(KNp) plus O(N) STM operations;
T updates and J energy trials cost O(TJKNp). New fitting/selection/tuning cost is
zero; loading frozen weights is O(KNp). Original v3 pretraining and full-study
costs remain separately available. Persistent arrays are the v3 1152 bytes plus
120 STM bytes =1272 bytes. Comparator trace costs 8 bytes plus stored output
parameters. No optimizer state, raw input replay or newly fitted sufficient
statistics. m is retained history and is charged as such. Full-run/phase timing
is sufficient for this small mechanism script; no inference speed claim or timing
sweep. Record counts, hashes (including imported source and loaded artifacts),
revision/dirty status, environment, peak RSS and limitations separately from
deterministic scientific results. Manual development time and allocator peak are
unavailable. No search, outcome-based reruns or extension. Execute once:
`.venv/bin/python -m experiments.selective_stm`.
