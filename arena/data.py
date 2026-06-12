"""Load the workshop dataset (images, reference + validation labels).

Fetches from the hosted bundle (``data_url``) over HTTP, or from a local path if
``data_url`` points at a directory (handy for offline dev). The hidden test
labels are never here — only the leaderboard has those.
"""

from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import requests
from PIL import Image

from arena.config import get_config
from arena.wire import decode_mask


def _read(rel: str) -> bytes:
    base = get_config().data_url
    if not base:
        raise RuntimeError("no data_url set — call arena.configure(data_url=...)")
    if base.startswith(("http://", "https://")):
        resp = requests.get(f"{base}/{rel}", timeout=120)
        resp.raise_for_status()
        return resp.content
    with open(os.path.join(base, rel), "rb") as fh:
        return fh.read()


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(_read("manifest.json").decode())


def _image(stem: str) -> np.ndarray:
    rel = _manifest()["files"][stem]["image"]
    return np.array(Image.open(io.BytesIO(_read(rel))).convert("L"))


def _label(stem: str) -> np.ndarray:
    rel = _manifest()["files"][stem]["label"]
    return decode_mask(_read(rel))


def _map(fn, stems, workers: int = 8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, stems))


def load_public_test() -> dict[str, np.ndarray]:
    """The test images to segment + submit. ``{image_id: image}`` (no labels)."""
    stems = _manifest()["sets"]["test"]
    return dict(zip(stems, _map(_image, stems)))


def load_train_frames() -> dict[str, np.ndarray]:
    """Raw training frames (no labels) for self-labeling / fine-tuning."""
    stems = _manifest()["sets"]["train"]
    return dict(zip(stems, _map(_image, stems)))


def load_reference_frames() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The 3 fully-labeled reference frames: ``[(id, image, label), ...]``."""
    stems = _manifest()["sets"]["reference"]
    return [(s, i, l) for s, i, l in zip(stems, _map(_image, stems), _map(_label, stems))]


def load_local_val() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The labeled validation set for ``score_local``: ``(images, labels)``."""
    stems = _manifest()["sets"]["val"]
    return _map(_image, stems), _map(_label, stems)
