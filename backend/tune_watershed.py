"""Find watershed_split params that actually help dense under-segmentation.

Segments a few val frames once, then sweeps min_distance / min_size locally and
scores each, so we set an honest default (and know when the lever helps).
"""

import os

import numpy as np

import arena
from arena.labels import relabel_sequential
from arena.processing import _min_size, _watershed_split
from arena.scoring import score

arena.configure(
    token=os.environ.get("ARENA_TOKEN", "devtoken-aaa"),
    backend_url="https://carnation-workshop--cell-arena-models-web.modal.run",
    leaderboard_url="https://carnation-workshop--cell-arena-leaderboard-web.modal.run",
    data_url="https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data",
)

val_imgs, val_labels = arena.load_local_val()
imgs, labs = val_imgs[:5], val_labels[:5]

print("segmenting 5 val frames with cpsam_v2 ...")
base = list(arena.segment_all(dict(enumerate(imgs)), model="cpsam_v2").values())
gt_cells = [int(l.max()) for l in labs]
base_cells = [int(b.max()) for b in base]
print(f"GT cells:    {gt_cells}")
print(f"pred cells:  {base_cells}  (under-segmenting)")
print(f"baseline mAP: {score(base, labs)['map']:.4f}\n")

print(f"{'min_dist':>8} {'min_size':>8} {'avg_cells':>10} {'mAP':>8}")
for md in [5, 8, 10, 12, 15, 20]:
    for ms in [15, 30, 60]:
        refined = []
        for m in base:
            w = _min_size(_watershed_split(m, min_distance=md), ms)
            refined.append(relabel_sequential(w)[0])
        avg = np.mean([int(r.max()) for r in refined])
        print(f"{md:>8} {ms:>8} {avg:>10.0f} {score(refined, labs)['map']:>8.4f}")
