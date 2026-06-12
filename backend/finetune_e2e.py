"""Verify fine-tuning improves the HELD-OUT score (the acceptance criterion).

Trains on references + half the validation set, evaluates on the other half, so
any gain is real (not memorized).
"""

import os

import arena

arena.configure(
    token=os.environ.get("ARENA_TOKEN", "devtoken-aaa"),
    backend_url="https://carnation-workshop--cell-arena-models-web.modal.run",
    leaderboard_url="https://carnation-workshop--cell-arena-leaderboard-web.modal.run",
    data_url="https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data",
)

refs = arena.load_reference_frames()
val_imgs, val_labels = arena.load_local_val()

train = [(im, lab) for _id, im, lab in refs] + list(zip(val_imgs[:6], val_labels[:6]))
eval_imgs, eval_labels = val_imgs[6:], val_labels[6:]
print(f"train on {len(train)} labeled frames, eval on {len(eval_imgs)} held-out frames\n")

base = arena.segment_all(dict(enumerate(eval_imgs)), model="cpsam_v2")
base_score = arena.score_local(base, dict(enumerate(eval_labels)))
print(f"baseline cpsam_v2:  {base_score}")

lr = float(os.environ.get("FT_LR", "1e-5"))
epochs = int(os.environ.get("FT_EPOCHS", "100"))
print(f"\nfine-tuning (n_epochs={epochs}, lr={lr}) ...")
adapter = arena.finetune(train, base_model="cpsam_v2", n_epochs=epochs, learning_rate=lr)

tuned = arena.segment_all(dict(enumerate(eval_imgs)), model=adapter)
tuned_score = arena.score_local(tuned, dict(enumerate(eval_labels)))
print(f"\nfine-tuned {adapter}:  {tuned_score}")
print(f">>> held-out mAP delta: {tuned_score['map'] - base_score['map']:+.4f}")
print(f">>> cells/frame: baseline {[int(m.max()) for m in base.values()]}")
print(f">>>              tuned    {[int(m.max()) for m in tuned.values()]}")
print(f">>>              truth    {[int(l.max()) for l in eval_labels]}")
