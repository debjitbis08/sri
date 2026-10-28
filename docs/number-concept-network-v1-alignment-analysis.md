# Number concept network v1: cross-modal latent alignment analysis

Date: 2026-10-28. Post hoc, after the [confirmation results](number-concept-network-v1-results.md)
and external review; not part of the frozen [contract](number-concept-network-v1-contract.md).
No training. Reproduce with `.venv/bin/python experiments/number_concept_alignment_analysis.py`
(38 s, one thread; needs the confirmation checkpoints in `runs/`). Output:
`reports/number-concept-network-v1-alignment-analysis.json` and `-runtime.json`.

Question: the aim is strongly correlated digit and dice latent spaces. How correlated
are they after v1, and why not more? All measures use held-out test images. Each
dice test image is paired with a random MNIST test image of the same number.
Alignment maps are fitted on class centroids using labels. They are diagnostics,
not transfer methods the contract allows.

## Measures

- **Paired linear CKA** on `[Re, Im]` latents is invariant to rotations of the
  coordinates. References: two same-class digit pairings within the digit modality
  (the class-level ceiling), and shuffled cross-modal pairing.
- **Raw coordinate correlation:** the mean Pearson correlation of each of the 16
  real coordinates across pairs. It is high only if both modalities use the *same*
  coordinates.
- **Centroid RSA:** correlation of the 6×6 centroid cosine matrices.
- **Oracle alignment readout:** dice latents are mapped by a fitted transform and
  read out against the ten digit centroids. Each map is fitted on counts 1–6 and,
  more stringently, leave-one-class-out (LOCO: fit on five counts, score the
  sixth). Maps: the diagonal phase group `diag(exp(i beta))`, which commutes with
  the rotation successor and is the contract's predicted symmetry; diagonal complex
  scaling; complex unitary; and real orthogonal on the real view.
- **Class-level anchor:** the diagonal phase correction computed from the dice
  count-1 *class centroid* alone, i.e. one correspondence at class rather than
  image level.

## Results (arms U and A; five seeds per cell)

| | Rotation U | Rotation A | MLP U | MLP A |
| --- | --- | --- | --- | --- |
| Paired CKA (digit–digit ceiling 0.63–0.69) | 0.44–0.53 | 0.43–0.50 | 0.49–0.54 | 0.47–0.54 |
| Raw coordinate correlation (digit–digit 0.72–0.78) | −0.26–0.15 | 0.04–0.32 | −0.18–0.34 | −0.25–0.51 |
| Matched-centroid angle, counts 1–6 | 60–132° | 41–104° | 42–120° | 33–121° |
| Readout without alignment, counts 2–6 (mean) | 0.338 | 0.391 | 0.263 | 0.404 |
| Diagonal phase map, LOCO (mean) | **0.980** | **0.983** | 0.345 | 0.545 |
| Unitary map, LOCO (mean) | 0.71 | 0.78 | 0.31 | 0.57 |
| Class-level anchor phase correction (mean) | 0.829 | 0.832 | 0.146 | 0.328 |

Shuffled-pair CKA is 0.00 in every run. Controls without a consistent relation
(SH, AO) reach CKA 0.14–0.25, and no alignment recovers them (LOCO 0.00–0.06).
SW (donor S) reaches CKA 0.41–0.58, but its diagonal-phase LOCO readout is
0.00–0.48.

## Findings

1. **The relation creates a shared geometry.** With the correct successor relation
   (U, A, SW), cross-modal CKA is 0.43–0.58: most of the digit–digit ceiling
   (0.63–0.69) and far above the controls (0.14–0.25). Within-modality successor
   training gives the two spaces a correlated *shape* without any shared data.
2. **The coordinates are not shared.** Raw coordinate correlation is near zero
   (−0.26 to 0.51 against 0.72–0.78 within digits). The shapes match, but they are
   expressed in different coordinate frames.
3. **For rotation, the frame offset is exactly the predicted symmetry.** A
   per-coordinate phase offset, eight angles that commute with S, explains almost
   all of the misalignment. Fitted on five counts, it classifies the held-out sixth
   at 0.90–1.00 (mean 0.98), in U as well as A. Larger groups (unitary, orthogonal)
   generalize worse because they overfit six points. So the rotation dice space
   *is* the digit space up to the eight-angle ambiguity that the contract's
   prior-work section derived.
4. **The single-image anchor does not resolve that symmetry; a class-level anchor
   largely would.** In theory one anchored point fixes all eight phases. In
   practice the dice encoder matches the anchor *image* (residual 0.01–0.03), while
   the count-1 class centroid stays 41–59° from `c_1` (rotation A). That is an
   angular error, not only a magnitude one. Correcting phases from the count-1 class
   centroid alone raises rotation readout from 0.39 to 0.83 (seeds 0.77–1.00).
5. **For the MLP there is no small symmetry to fix.** The diagonal phase group does
   not commute with an MLP successor, and neither it nor the unitary group aligns
   reliably (LOCO 0.35–0.57). The MLP spaces have a similarly correlated shape (CKA
   ≈ 0.5), but the mismatch is not a simple coordinate symmetry, so one anchor has
   no small ambiguity to remove. Unlike rotation, nothing here shows which
   correspondence would be enough.
6. **A donor operator gives correlated shape, but not in the recipient's phase
   group.** SW's CKA matches A's, yet its phase alignment fails. The donor orbit
   uses different rotation frequencies, so the S-commuting correction no longer
   lines up with the recipient's frame.

## Implications

The v1 objective produces correlated latent geometry but leaves the coordinate
frames untied. For rotation the remaining gap is a known, eight-parameter symmetry.
It failed to be fixed because the anchor acts on one image rather than the class.
Two changes follow, both within the stage-2 information boundary: stage 2 already
has dice class labels and the digit-1 centroid.

- **Class-level anchoring:** apply the anchor target to count-1 dice images
  generally (e.g. to the batch count-1 mean, or to every count-1 image), not to
  one fixed image. The cross-modal correspondence stays the same (dice one ↔ digit
  one); only within-dice labels are added, which the successor loss already uses.
  The phase correction above (≈0.83 for rotation) predicts what is achievable
  without changing the representation; training could do better or worse.
- **Explicit frame tying:** measure success by the quantities above (raw coordinate
  correlation and matched angles, not only readout), and add an equality or
  orbit-target term so that dice codes are pulled onto the digit orbit
  `S^(n-1)(c_1)` rather than merely ordered.

Neither change explains the MLP's misalignment. The rotation variant has the more
tractable structure here, even though it scored no better in v1.

## Cost

38 s wall, 38 s CPU, peak RSS 427 MB, single thread, CPU only. No fits beyond
closed-form alignments (SVD of 8×8 complex and 16×16 real matrices per fit).
