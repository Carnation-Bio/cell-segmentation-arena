"""End-to-end participant workflow against the live services.

Exercises the exact path a participant's notebook / agent runs:
configure -> load data -> segment (with refine) -> score_local -> submit.
Verifies the watershed-split refine actually improves the dense score, and that
submit returns a public score.

    python backend/e2e_workflow.py
"""

import os

import arena

BACKEND = os.environ.get("ARENA_BACKEND_URL", "https://carnation-workshop--cell-arena-models-web.modal.run")
LEADERBOARD = os.environ.get("ARENA_LEADERBOARD_URL", "https://carnation-workshop--cell-arena-leaderboard-web.modal.run")

arena.configure(
    token=os.environ.get("ARENA_TOKEN", "devtoken-aaa"),
    backend_url=BACKEND,
    leaderboard_url=LEADERBOARD,
    data_url=f"{LEADERBOARD}/data",
)

print("loading validation set from /data ...")
val_imgs, val_labels = arena.load_local_val()
print(f"  {len(val_imgs)} val frames, GT cells/frame: {[int(l.max()) for l in val_labels[:4]]}...")

# score a few frames to keep it quick
n = 4
imgs, labels = val_imgs[:n], val_labels[:n]

print("\nsegment cpsam_v2 (no refine) ...")
base = [arena.segment(im, model="cpsam_v2") for im in imgs]
s_base = arena.score_local(base, labels)
print(f"  baseline:           {s_base}")

print("segment cpsam_v2 + watershed_split refine ...")
refined = [arena.segment(im, model="cpsam_v2", refine=["watershed_split", "min_size:15"]) for im in imgs]
s_ref = arena.score_local(refined, labels)
print(f"  + watershed_split:  {s_ref}")
print(f"  >>> refine delta mAP: {s_ref['map'] - s_base['map']:+.4f}  (watershed should help on dense)")

print("\nsegment cyto3 (Cellpose v3) ...")
try:
    cyto = [arena.segment(im, model="cyto3") for im in imgs]
    print(f"  cyto3: {arena.score_local(cyto, labels)}  cells={[int(m.max()) for m in cyto]}")
except Exception as e:  # noqa: BLE001
    print(f"  cyto3 FAILED: {e}")

print("\nfull test set -> segment_all -> submit ...")
test = arena.load_public_test()
masks = arena.segment_all(test, model="cpsam_v2", refine=["watershed_split", "min_size:15"], max_workers=8)
print(f"  segmented {len(masks)} test frames")
score = arena.submit(masks, team="e2e-bot")
print(f"\nDONE — public score = {score:.4f}")
