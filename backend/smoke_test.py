"""Smoke-test the deployed /segment endpoint on a real dense frame.

    ARENA_BACKEND_URL=https://...modal.run python backend/smoke_test.py

Loads the densest local validation frame (image + ground truth), segments it
with each model, and prints found-cell count + AP score. Sanity check: the
models should reproduce the brief's ballpark (cyto3 ~900-1000, cpsam_v2 ~700 on
a ~1400-cell frame) and score non-trivially.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from arena.scoring import score
from arena.wire import decode_mask

URL = (os.environ.get("ARENA_BACKEND_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")).rstrip("/")
TOKEN = os.environ.get("ARENA_TOKEN", "devtoken-aaa")
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cyto3", "cpsam_v2"]

if not URL:
    sys.exit("set ARENA_BACKEND_URL or pass the URL as argv[1]")

bundle = Path("data/build/bundle")
man = json.loads((bundle / "manifest.json").read_text())


def load_img(stem: str) -> np.ndarray:
    return np.array(Image.open(bundle / man["files"][stem]["image"]).convert("L"))


def load_lab(stem: str) -> np.ndarray:
    return decode_mask((bundle / man["files"][stem]["label"]).read_bytes())


# densest validation frame
val = man["sets"]["val"]
stem = max(val, key=lambda s: int(load_lab(s).max()))
img, gt = load_img(stem), load_lab(stem)
print(f"frame {stem}: GT = {int(gt.max())} cells\nbackend = {URL}\n")


def segment(model: str) -> None:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = requests.post(
        f"{URL}/segment",
        json={"model": model, "image": b64, "params": {}},
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=600,
    )
    r.raise_for_status()
    j = r.json()
    mask = decode_mask(base64.b64decode(j["mask"]))
    s = score(mask, gt)
    print(
        f"{model:10s} found {j['n_instances']:5d} cells in {j['seconds']:5.1f}s "
        f"| mAP@[.5:.95]={s['map']:.3f}  F1@0.5={s['f1']:.3f}"
    )


if __name__ == "__main__":
    for model in MODELS:
        segment(model)
