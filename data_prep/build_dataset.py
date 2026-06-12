"""Build the dense-SHSY5Y competition dataset from public LIVECell.

Pulls SHSY5Y phase-contrast frames at the 3-3.5 day timepoints (filenames
contain ``_03d``) from the public LIVECell S3 bucket over HTTP range requests
(via ``remotezip`` — never downloads the 1.2 GB archive), decodes the COCO
polygon ground truth to integer label images, and writes a fixed-seed disjoint
split:

    reference (3, labeled)   given to teams as "what good looks like"
    val       (12, labeled)  given to teams for score_local()
    train     (30, raw)       given to teams to self-label / fine-tune
    test      (60, raw)       given to teams UNLABELED; predicted + submitted

The 60 test frames are shuffled together and never tagged in the participant
bundle. A *secret* public/private split of those 60 (30 + 30) is written
server-side only: every submission is scored against both, the public score is
shown live, the private board is revealed at the end (Kaggle convention).

Outputs
-------
``<out>/bundle/``   participant bundle (images + reference/val labels + manifest)
                    -> uploaded to the public R2 bucket, fetched by the notebook
``<out>/server/``   hidden ground truth (public_gt.npz, private_gt.npz,
                    split_secret.json) -> goes to the leaderboard's Modal Volume,
                    never exposed to clients

Run::

    python -m data_prep.build_dataset --out data/build
    python -m data_prep.build_dataset --out data/build --upload   # push bundle to R2
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from remotezip import RemoteZip
from skimage.draw import polygon as draw_polygon

from arena.wire import encode_mask

LIVECELL_S3 = "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021"
IMAGES_ZIP = f"{LIVECELL_S3}/images.zip"
ANN_URL = "{base}/annotations/LIVECell_single_cells/shsy5y/{split}.json"

DENSE_TOKEN = "_03d"  # 3.0-3.5 day band: the timepoints where every model fails
MIN_CELLS = 500       # dense floor; median in-band frame has ~760 cells
SEED = 1234

# Disjoint split sizes (sum 105, drawn from ~121 in-band frames).
N_REFERENCE = 3
N_VAL = 12
N_TRAIN = 30
N_TEST = 60           # split secretly into 30 public + 30 private


def _cache_dir() -> Path:
    d = Path("data/cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_coco(split: str) -> dict:
    """Load a LIVECell SHSY5Y COCO split, caching the JSON locally."""
    import urllib.request

    path = _cache_dir() / f"shsy5y_{split}.json"
    if not path.exists():
        url = ANN_URL.format(base=LIVECELL_S3, split=split)
        print(f"  downloading {split}.json ...")
        urllib.request.urlretrieve(url, path)
    return json.loads(path.read_text())


def build_pool() -> dict[str, dict]:
    """Combined pool of dense SHSY5Y frames across all LIVECell splits.

    Returns ``{stem: {"file_name", "h", "w", "anns": [polygon-segmentations]}}``
    for ``_03d`` frames with at least ``MIN_CELLS`` annotated cells.
    """
    pool: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        coco = _load_coco(split)
        by_img: dict[int, list] = {}
        for ann in coco["annotations"]:
            by_img.setdefault(ann["image_id"], []).append(ann)
        for im in coco["images"]:
            name = im["file_name"]
            if DENSE_TOKEN not in name:
                continue
            anns = by_img.get(im["id"], [])
            if len(anns) < MIN_CELLS:
                continue
            stem = name[:-4] if name.endswith(".tif") else name
            pool[stem] = {
                "file_name": name,
                "h": im["height"],
                "w": im["width"],
                "anns": anns,
            }
    return pool


def rasterize(anns: list[dict], h: int, w: int) -> np.ndarray:
    """COCO polygon annotations -> ``(h, w)`` int label image (0=bg, 1..N).

    Painted largest-cell-first so a small cell is never erased by a larger
    neighbor at a touching border.
    """
    label = np.zeros((h, w), dtype=np.int32)
    ordered = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)
    for idx, ann in enumerate(ordered, start=1):
        for poly in ann["segmentation"]:
            xs = np.asarray(poly[0::2], dtype=np.float64)
            ys = np.asarray(poly[1::2], dtype=np.float64)
            rr, cc = draw_polygon(ys, xs, shape=(h, w))
            label[rr, cc] = idx
    return label


def split_frames(stems: list[str]) -> dict[str, list[str]]:
    """Deterministic disjoint split. Sorted-then-shuffled for reproducibility."""
    ordered = sorted(stems)
    random.Random(SEED).shuffle(ordered)
    need = N_REFERENCE + N_VAL + N_TRAIN + N_TEST
    if len(ordered) < need:
        raise RuntimeError(f"need {need} dense frames, pool has {len(ordered)}")
    cut1 = N_REFERENCE
    cut2 = cut1 + N_VAL
    cut3 = cut2 + N_TRAIN
    cut4 = cut3 + N_TEST
    return {
        "reference": ordered[:cut1],
        "val": ordered[cut1:cut2],
        "train": ordered[cut2:cut3],
        "test": ordered[cut3:cut4],
    }


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG", optimize=True)


def build(out: Path) -> dict:
    """Build the participant bundle + hidden server ground truth on disk."""
    print("Building dense-SHSY5Y dataset from LIVECell ...")
    pool = build_pool()
    print(f"  dense pool: {len(pool)} frames ({DENSE_TOKEN}, >= {MIN_CELLS} cells)")
    sets = split_frames(list(pool))

    # secret public/private split of the 60 test frames
    test = list(sets["test"])
    random.Random(SEED + 1).shuffle(test)
    public_ids = sorted(test[: N_TEST // 2])
    private_ids = sorted(test[N_TEST // 2 :])

    bundle = out / "bundle"
    server = out / "server"
    img_cache = _cache_dir() / "images"
    img_cache.mkdir(exist_ok=True)

    # resolve every needed frame to its path inside images.zip, then pull once
    needed = {pool[s]["file_name"]: s for grp in sets.values() for s in grp}
    print(f"  fetching {len(needed)} frames via remotezip ...")
    images: dict[str, np.ndarray] = {}
    with RemoteZip(IMAGES_ZIP) as z:
        name_to_path = {
            p.split("/")[-1]: p for p in z.namelist() if p.split("/")[-1] in needed
        }
        for fname, stem in needed.items():
            local = img_cache / fname
            if local.exists():
                raw = local.read_bytes()
            else:
                raw = z.read(name_to_path[fname])
                local.write_bytes(raw)
            images[stem] = np.array(Image.open(io.BytesIO(raw)).convert("L"))

    manifest: dict = {
        "dataset": f"LIVECell SHSY5Y dense ({DENSE_TOKEN}, >= {MIN_CELLS} cells)",
        "image_size": [520, 704],
        "seed": SEED,
        "counts": {k: len(v) for k, v in sets.items()},
        "sets": {"reference": [], "val": [], "train": [], "test": []},
        "files": {},
    }

    def emit_image(stem: str, group: str) -> None:
        rel = f"{group}/{stem}.png"
        _save_png(images[stem], bundle / rel)
        manifest["files"][stem] = {"image": rel, "set": group}
        manifest["sets"][group].append(stem)

    def emit_label_for_participant(stem: str, group: str) -> None:
        info = pool[stem]
        lab = rasterize(info["anns"], info["h"], info["w"])
        rel = f"{group}/{stem}_label.png"
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle / rel).write_bytes(encode_mask(lab))
        manifest["files"][stem]["label"] = rel

    # reference + val: image AND label given to participants
    for group in ("reference", "val"):
        for stem in sets[group]:
            emit_image(stem, group)
            emit_label_for_participant(stem, group)

    # train: raw image only
    for stem in sets["train"]:
        emit_image(stem, "train")

    # test: raw image only (labels stay hidden, server-side)
    for stem in sets["test"]:
        emit_image(stem, "test")

    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # hidden server-side ground truth
    server.mkdir(parents=True, exist_ok=True)
    for bucket, ids in (("public", public_ids), ("private", private_ids)):
        gt = {stem: rasterize(pool[stem]["anns"], pool[stem]["h"], pool[stem]["w"]) for stem in ids}
        np.savez_compressed(server / f"{bucket}_gt.npz", **gt)
    (server / "split_secret.json").write_text(
        json.dumps({"public": public_ids, "private": private_ids}, indent=2)
    )

    print(f"  bundle -> {bundle}  ({sum(manifest['counts'].values())} frames)")
    print(f"  server GT -> {server}  (public {len(public_ids)} / private {len(private_ids)})")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/build"))
    ap.add_argument("--upload", action="store_true", help="push bundle to R2 after build")
    args = ap.parse_args()
    build(args.out)
    if args.upload:
        from data_prep.upload_r2 import upload_bundle

        upload_bundle(args.out / "bundle")


if __name__ == "__main__":
    main()
