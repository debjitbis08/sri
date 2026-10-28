"""Post-hoc cross-modal latent alignment analysis for number concept network v1.

Not part of the frozen contract (docs/number-concept-network-v1-contract.md); reads the
confirmation checkpoints in runs/ and the held-out test sets, trains nothing. Questions:

1. How correlated are the digit and dice latent spaces? Label-paired linear CKA,
   representational similarity of class centroids, raw per-coordinate correlation and
   matched-centroid angles, with digit-vs-digit and shuffled-pairing references.
2. Why is alignment missing? Readout after oracle alignments fitted on class centroids:
   the S-commuting diagonal phase group, diagonal complex scaling, complex unitary and
   real orthogonal maps (in-sample and leave-one-class-out), plus a class-level anchor
   (phase correction from the count-1 centroid alone). Digit-side chain S^k(c_1).

Label-based alignments are diagnostics; none is a valid transfer method under the contract.
Writes reports/number-concept-network-v1-alignment-analysis.json and a runtime JSON.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import number_concept as nc  # noqa: E402

torch.set_num_threads(1)
RUNS = ROOT / 'runs/number-concept-network-v1-confirmation'
OUT = ROOT / 'reports/number-concept-network-v1-alignment-analysis'
SEEDS, ARMS = range(5), ('U', 'A', 'SH', 'AO', 'SW')


def load(seed, variant):
    ck = torch.load(RUNS / f'stage1-{seed}-{variant}.pt', weights_only=False)
    digit, s = nc.Modality(nc.stream('load')), nc.make_successor(variant, nc.stream('load'))
    digit.load_state_dict(ck['digit'])
    s.load_state_dict(ck['successor'])
    return digit, s, ck['centroids']


def load_dice(seed, variant, arm):
    m = nc.Modality(nc.stream('load'))
    m.load_state_dict(torch.load(RUNS / f'dice-{seed}-{variant}-{arm}.pt', weights_only=False)['dice'])
    return m


def cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return float((y.T @ x).pow(2).sum() / ((x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()))


def coord_corr(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    r = (x * y).sum(0) / (x.norm(dim=0) * y.norm(dim=0)).clamp_min(1e-12)
    return float(r.mean())


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(float(nc.unit(a[None]) @ nc.unit(b[None]).T), -1, 1))))


def normed(c):
    return c / c.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)


# Alignment maps fitted from dice centroids D (n, 8) to digit centroids C (n, 8), unit-normalized.
def fit_phase(D, C):
    beta = torch.angle((normed(C) * normed(D).conj()).sum(0))
    return lambda z: z * torch.polar(torch.ones_like(beta), beta)


def fit_diag(D, C):
    d, c = normed(D), normed(C)
    q = (c * d.conj()).sum(0) / d.abs().pow(2).sum(0).clamp_min(1e-12)
    return lambda z: z * q


def fit_unitary(D, C):
    w, _, vh = torch.linalg.svd(normed(C).T @ normed(D).conj())
    u = w @ vh
    return lambda z: z @ u.T


def fit_orthogonal(D, C):
    d, c = nc.flat(normed(D)), nc.flat(normed(C))
    w, _, vh = torch.linalg.svd(c.T @ d)
    o = w @ vh
    return lambda z: (lambda f: torch.complex(f[:, :nc.D], f[:, nc.D:]))(nc.flat(z) @ o.T)


FITS = {'diagonal_phase': fit_phase, 'diagonal_complex': fit_diag, 'unitary': fit_unitary,
        'real_orthogonal': fit_orthogonal}


def macro(pred, y, classes):
    return float(np.mean([(pred[y == c] == c).mean() for c in classes]))


def analyse(seed, variant, arm, digit, s, cents, dimg, dlab, gimg, glab, pairing):
    m = load_dice(seed, variant, arm)
    with torch.no_grad():
        zd = nc.encode(m.encoder, dimg)
        zg = nc.encode(digit.encoder, gimg)
        dc = nc.centroids(zd, dlab, nc.FACES)
        gc = cents[list(nc.FACES)]
        out = {'matched_centroid_angle_deg': {str(n): angle(dc[i], gc[i]) for i, n in enumerate(nc.FACES)},
               'centroid_norm_ratio': {str(n): float(dc[i].abs().pow(2).sum().sqrt() / gc[i].abs().pow(2).sum().sqrt())
                                       for i, n in enumerate(nc.FACES)}}
        cd, cg = nc.cosine_matrix(dc, dc).numpy(), nc.cosine_matrix(gc, gc).numpy()
        iu = np.triu_indices(6, 1)
        out['centroid_rsa_pearson'] = float(np.corrcoef(cd[iu], cg[iu])[0, 1])
        out['paired_cka'] = cka(nc.flat(zd), nc.flat(zg[pairing['matched']]))
        out['paired_cka_shuffled'] = cka(nc.flat(zd), nc.flat(zg[pairing['shuffled']]))
        out['paired_coordinate_correlation'] = coord_corr(nc.flat(zd), nc.flat(zg[pairing['matched']]))
        out['paired_unit_cosine_mean'] = float((nc.unit(zd) * nc.unit(zg[pairing['matched']])).sum(1).mean())
        base = nc.predict(zd, cents)
        out['readout_none'] = macro(base, dlab, range(2, 7))
        out['aligned_in_sample'], out['aligned_leave_one_class_out'] = {}, {}
        for name, fit in FITS.items():
            out['aligned_in_sample'][name] = macro(nc.predict(fit(dc, gc)(zd), cents), dlab, range(2, 7))
            loco = []
            for i, n in enumerate(nc.FACES):
                keep = [j for j in range(6) if j != i]
                f = fit(dc[keep], gc[keep])
                mask = dlab == n
                loco.append(float((nc.predict(f(zd[torch.as_tensor(mask)]), cents) == n).mean()))
            out['aligned_leave_one_class_out'][name] = {'per_count': dict(zip(map(str, nc.FACES), loco)),
                                                        'macro_2_6': float(np.mean(loco[1:]))}
        # Class-level anchor: phase correction from the count-1 dice centroid only (S-commuting group).
        anchor_q = fit_phase(dc[:1], gc[:1])
        out['class_anchor_phase_readout_2_6'] = macro(nc.predict(anchor_q(zd), cents), dlab, range(2, 7))
        # Within-dice successor fidelity, measured against dice centroids (angles).
        sd = s(dc)
        out['dice_successor_angle_deg'] = {f'{a}->{a + 1}': angle(sd[i], dc[i + 1]) for i, a in enumerate(nc.FACES[:-1])}
    return out


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    raw, _ = nc.load_mnist(ROOT / 'data/mnist')
    gimg = torch.from_numpy(raw['test_images'].astype(np.float32) / 255)
    glab = raw['test_labels']
    pool = nc.build_dice('standard', 'test')
    dimg, dlab = torch.from_numpy(pool['images']), pool['labels']
    rng = nc.stream('alignment_pairing')
    matched = np.array([rng.choice(np.flatnonzero(glab == n)) for n in dlab])
    shuffled = rng.permutation(matched)
    pairing = {'matched': torch.as_tensor(matched), 'shuffled': torch.as_tensor(shuffled)}
    res = {'contract': 'docs/number-concept-network-v1-contract.md', 'status': 'post hoc; label-based alignments are diagnostics',
           'split': 'test', 'runs': {}, 'digit': {}}
    for variant in nc.VARIANTS:
        for seed in SEEDS:
            digit, s, cents = load(seed, variant)
            with torch.no_grad():
                zg = nc.encode(digit.encoder, gimg)
                # Digit-vs-digit reference: two independent same-class pairings within the digit modality.
                other = np.array([rng.choice(np.flatnonzero(glab == n)) for n in glab[matched]])
                chain = [cents[1]]
                for _ in range(8):
                    chain.append(s(chain[-1]))
                chain = torch.stack(chain)
                cm = nc.cosine_matrix(cents, cents).numpy()
                np.fill_diagonal(cm, -1)
                res['digit'][f'{seed}/{variant}'] = {
                    'chain_from_c1_predictions': nc.predict(chain, cents).tolist(),
                    'chain_angle_to_target_deg': [angle(chain[k], cents[k + 1]) for k in range(9)],
                    'nearest_centroid_half_angle_deg': [float(np.degrees(np.arccos(cm[n].max())) / 2) for n in range(10)],
                    'digit_digit_paired_cka': cka(nc.flat(zg[matched]), nc.flat(zg[other])),
                    'digit_digit_coordinate_correlation': coord_corr(nc.flat(zg[matched]), nc.flat(zg[other])),
                }
            for arm in ARMS:
                res['runs'][f'{seed}/{variant}/{arm}'] = analyse(seed, variant, arm, digit, s, cents, dimg, dlab,
                                                                 gimg, glab, pairing)
    OUT.parent.mkdir(exist_ok=True)
    Path(f'{OUT}.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    Path(f'{OUT}-runtime.json').write_text(json.dumps({
        'scope': 'whole script, one thread; analytical post-hoc check, not an inference benchmark',
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'runner_sha256': nc.sha256(__file__),
        'module_sha256': nc.sha256(nc.__file__), 'torch': torch.__version__, 'numpy': np.__version__,
        'checkpoints': str(RUNS.relative_to(ROOT))}, indent=1) + '\n')


if __name__ == '__main__':
    main()
