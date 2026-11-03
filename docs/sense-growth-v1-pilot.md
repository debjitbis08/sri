# When should sight open a centre? Growth rules on stress streams

Status: **pilot, 2026-11-04, seed 392, both kind orders. Not a contract.** Code: `experiments/sense_growth.py`. Report: `reports/sense-growth-v1-pilot.json.gz`.

## Question

The kept-centre, frozen-gate memory ([stability pilot](sense-stable-v1-pilot.md)) keeps what it holds whenever a centre opens, so the remaining question is when to open one. The shift rule of the [adapt pilot](sense-adapt-v1-pilot.md) worked on its streams, but those streams let each kind arrive for a while at a time. This pilot asks how three rules behave on streams that break that assumption.

Only the decision to open centres is tested, with no memory reads. It runs on DINOv2-S codes (the routing model) of 9,000 images per kind, and labels only score.

## Rules

- **`shift`.** The adapt pilot's rule. A batch of 256 joins the nearest centre if its mean code reaches that centre's threshold; otherwise it goes to a forming pool that never expires. 512 pooled inputs open a centre.
- **`shift-window`.** The same, except that stale rejections expire: the pool keeps only batches from the last 4.
- **`pile-up`.** Each input goes to its nearest centre. A centre is flagged when the inputs it receives in a batch, centred on it, bunch more than its own members do: the length of their mean unit vector exceeds the 0.999 quantile of bootstrapped member samples.
  - The bootstrap sample size is rounded down to a power of two.
  - At least 32 inputs are needed for a test.
  - A flagged centre's inputs are split by two-means on raw codes, with the old centre held fixed. Those nearer the new side go to a pool with the same 4-batch window.
  - 512 pooled inputs open a centre.

Two choices were fixed on synthetic Gaussian clusters, before any real code was used: the two-means split, which replaced taking every input that leaned the new way, and rounding the sample size down rather than up. On those clusters, `pile-up` then opened exactly one pure centre per kind on runs, gradual and returning streams, with no false centres. On interleaved kinds it opened 5 of 6.

## Streams

Kinds come in the seed's order and its reverse.

| Stream | Before the tail | Tail |
|---|---|---|
| `runs` | each kind for 4 batches, as in the pilots | 200 batches of known kinds in runs of 1–8 (51,200 inputs) |
| `interleaved` | a new kind every 8 batches; each batch half the newest kind, half the earlier ones | 100 batches of even mixtures |
| `gradual` | each kind fades into the next over 16 batches | 50 batches of even mixtures |
| `returning` | runs of 4 batches in kind positions 0 1 2 0 1 3 4 0 5 2 | as `runs` |

## Results

Each cell is forward/reverse. There are 6 kinds. A pure centre is one where at least 0.9 of the inputs that formed it are one kind. Extra centres are those beyond one per majority kind. Routing purity is the share of inputs over the last 50 batches whose nearest centre's majority kind is their own.

| Rule | Stream | Centres | Kinds with a pure centre | Extra centres | Opened in the tail | Routing purity |
|---|---|---|---|---|---|---|
| `shift` | runs | 7/6 | 6/6 | 1/0 | 1/0 | 0.94/0.95 |
| | interleaved | 9/8 | 1/1 | 4/3 | 3/2 | 0.79/0.76 |
| | gradual | 33/32 | 6/6 | 27/26 | 1/1 | 0.94/0.94 |
| | returning | 8/7 | 6/6 | 2/1 | 2/1 | 0.99/0.96 |
| `shift-window` | runs | **6/6** | 6/6 | **0/0** | **0/0** | 0.96/0.95 |
| | interleaved | 8/7 | 1/1 | 3/2 | 2/1 | 0.78/0.76 |
| | gradual | 34/33 | 6/6 | 28/27 | 2/1 | 0.94/0.93 |
| | returning | 7/6 | 6/6 | 1/0 | 1/0 | 0.99/0.97 |
| `pile-up` | runs | 20/15 | 6/6 | 14/9 | 14/9 | 0.96/0.95 |
| | interleaved | 4/5 | 2/4 | 0/0 | 0/0 | 0.62/0.82 |
| | gradual | **6/7** | 5/6 | **0/1** | 0/0 | 0.90/0.95 |
| | returning | 8/7 | 6/6 | 2/1 | 2/1 | 0.98/0.96 |

**Why `pile-up` opens false centres on familiar runs** (diagnosed in `runs`, forward). All 14 extra centres are 100% banana: the rule splits one kind, it does not mix kinds. Banana batches send about 40 inputs per batch to the tree centre, where they bunch strongly (pile-up 0.83–0.87 against a threshold of 0.39). The roughly 210 banana inputs left at banana's own centre are then a truncated sample, a little above that centre's threshold too (0.28–0.41 against 0.27–0.33). Routing inputs before testing them biases the test whenever two kinds' codes overlap. Synthetic clusters don't overlap, so the smoke test missed it.

## What this shows

- **No rule is safe yet on all four kinds of stream.**
- **`shift-window` is the best rule when kinds come in runs.** It opened no false centre in 51,200 familiar inputs in either order, and its expiring pool removed the extra centres `shift` made.
- **Batch-mean rules fail on blended streams.**
  - On interleaved kinds, `shift` and `shift-window` find a pure centre for only 1 of 6 kinds and open centres for blends instead.
  - On gradual change they open a centre at almost every step of a fade, 32–34 in all.
  - This is the expected failure: a batch's mean describes the batch, not its inputs.
- **`pile-up` handles gradual change.**
  - It opens 6–7 centres for 6 kinds, with no false centre in the tail.
  - On interleaved kinds it finds 2–4 of 6, and misses rather than mixes.
- **`pile-up` over-splits overlapping kinds on familiar streams.** It opened 9–14 extra centres, each pure. In the kept-centre memory, extra centres of one kind cost state and read time, and split that kind's facts across partitions. They don't mix coordinate systems, and reads search every centre.

## Next

These ideas were not run.

- **Test pile-up before routing.** Compare the inputs a centre receives with bootstrapped member samples that have been routed the same way, against the same centres. Or test a whole batch against a mixture of the existing centres. Either removes the truncation bias diagnosed above.
- **Combine the two rules.** Use `shift-window` for batches that look like one kind, and the pile-up test for batches that look mixed.
- **Merge centres.** Merge two centres whose members cannot be told apart, so over-splitting can be undone.

## What this does not establish

- **Read results.** No memory reads ran, so the cost of extra or missing centres to recall is not measured.
- **Other seeds, or kinds beyond these six.**
- **When to grow a gate.** Gate growth was not tested: each centre still refits and freezes its own gate.

## Prior work

These are familiar methods, not re-read for this pilot.

- **Online clustering with a novelty threshold:** leader clustering and DP-means (Kulis and Jordan 2012). `shift` is a batch-level form.
- **Change detection:** CUSUM and Page–Hinkley. `shift-window`'s expiring pool is a crude finite-memory detector.
- **Novelty detection on mixtures:** for example out-of-distribution detection by Mahalanobis distance to class-conditional Gaussians (Lee et al. 2018). Testing an input against all centres jointly, rather than its nearest one alone, is the standard way to avoid the routing bias seen here.

## Scaling

Per batch of b inputs and k centres:
- `shift` costs O(k d) for the batch mean.
- `pile-up` costs O(b k d) to route, O(b d) per tested centre, and a bootstrap of O(B s d) per centre and sample size, computed once and cached (B = 300 draws, s ≤ 256).

Neither rule stores inputs beyond one window of the forming pool, at most 4 × 256 codes.

## Computation

- **Codes:** GTX 1650, DINOv2-S on 54,000 images, 1,158 s wall. It shared the GPU with another job.
- **Rules:** CPU, 40 s wall, 46 s CPU for all 24 rule × stream × order runs.

No model was trained.
