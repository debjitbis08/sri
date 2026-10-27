"""Post-hoc spectral drift check on relational linear preflight v1 results.

Not part of the frozen contract. Compares the eigenvalue moduli of each P2
binding fit's eight effective operators with those of the frozen source (the
P1 multi-horizon search), and compounds the gap to L = 128 (G1 applied 128
times, G2 64 times). Reads only the stored results file.
"""

import gzip
import json

import numpy as np

RESULTS = 'reports/relational-linear-preflight-v1-results.json.gz'


def main():
    with gzip.open(RESULTS) as f:
        d = json.load(f)
    print('condition target | max |d modulus| median/max | L=128 log-amplitude gap median/max | X128 median')
    for cond in d['P2']:
        for target in ('s2', 't', 'broken'):
            dev, amp, x = [], [], []
            for w, fit in d['P2'][cond].items():
                fit = fit[target]
                if fit.get('failed'):
                    continue
                ref = d['P1'][cond][w]['multi_horizon']['eig_moduli']
                ref = ref['G1'] + ref['G2']
                worst_dev = worst_amp = 0.0
                for i, (e, r) in enumerate(zip(fit['eig_moduli'], ref)):
                    e, r = np.sort(e), np.sort(r)
                    power = 128 if i < 4 else 64
                    worst_dev = max(worst_dev, float(np.max(np.abs(e - r))))
                    worst_amp = max(worst_amp, float(np.max(np.abs(np.log(e / r)))) * power)
                dev.append(worst_dev)
                amp.append(worst_amp)
                x.append(fit['X']['128'])
            print(f'{cond} {target} | {np.median(dev):.2e} / {max(dev):.2e} | '
                  f'{np.median(amp):.2f} / {max(amp):.2f} | {np.median(x):.3f}')


if __name__ == '__main__':
    main()
