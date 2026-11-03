"""Fetch CIFAR-100 for step 2 of docs/general-frozen-encoder.md and convert it to arrays, for `experiments.general_encoder pilot heldout`. Not run automatically.

Run from outside any download folder, with an interpreter that has pyarrow and pillow, in isolated mode:
  mkdir -p /data/external-downloads/cifar100-hf runs/external/cifar100
  B=https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100
  curl -sSL -C - -o /data/external-downloads/cifar100-hf/train.parquet $B/train-00000-of-00001.parquet
  curl -sSL -C - -o /data/external-downloads/cifar100-hf/test.parquet $B/test-00000-of-00001.parquet
  /data/venvs/cifar-convert/bin/python -I experiments/general_encoder_fetch.py /data/external-downloads/cifar100-hf runs/external/cifar100/cifar100-full.npz

Checks counts: 50,000 and 10,000 images, 100 fine and 20 coarse labels, each balanced.
"""
import io
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def main(src, out):
    d = {}
    for split in ('train', 'test'):
        t = pq.read_table(f'{src}/{split}.parquet')
        names = t.schema.names
        image = next(n for n in names if 'img' in n or 'image' in n)
        rows = t.column(image).to_pylist()
        x = np.stack([np.asarray(Image.open(io.BytesIO(r['bytes'])).convert('RGB'), np.uint8) for r in rows]).transpose(0, 3, 1, 2)
        fine = np.array(t.column('fine_label').to_pylist(), np.uint8)
        coarse = np.array(t.column('coarse_label').to_pylist(), np.uint8)
        expected = 50000 if split == 'train' else 10000
        assert len(x) == expected and x.shape[1:] == (3, 32, 32)
        assert set(np.bincount(fine)) == {expected // 100} and set(np.bincount(coarse)) == {expected // 20}
        d[f'x_{split}'], d[f'fine_{split}'], d[f'coarse_{split}'] = x, fine, coarse
        print(split, x.shape, flush=True)
    np.savez(out, **d)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
