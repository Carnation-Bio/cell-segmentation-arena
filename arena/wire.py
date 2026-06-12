"""Compact wire format for instance masks (submission transport).

A label image is encoded as a PNG whose RGBA channels are the four little-endian
bytes of a ``uint32`` label. This is:

* **Lossless** — exact integer labels survive the round trip.
* **Compact** — label images are mostly flat regions, which PNG compresses hard.
* **Architecture-independent** — bytes are always little-endian, so a mask
  encoded on a participant's laptop decodes identically on the Linux backend.

Labels are relabeled to a contiguous 1..N before encoding (instance *identity*,
not the specific integers, is what the scorer cares about), so any prediction —
however sparsely labeled — fits the format.

A whole submission is ``{image_id: base64(png_bytes)}``; helpers for that live
here too so the participant client and the server share one codec.
"""

from __future__ import annotations

import base64
import io
from typing import Mapping

import numpy as np
from PIL import Image

from arena.labels import relabel_sequential


def encode_mask(labels: np.ndarray) -> bytes:
    """Encode a ``(H, W)`` label image to PNG bytes."""
    lab, _ = relabel_sequential(labels)
    arr = np.ascontiguousarray(lab.astype("<u4"))
    rgba = arr.view(np.uint8).reshape(arr.shape[0], arr.shape[1], 4)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def decode_mask(data: bytes) -> np.ndarray:
    """Decode PNG bytes back to a ``(H, W)`` int64 label image."""
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    rgba = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    h, w = rgba.shape[:2]
    return rgba.view("<u4").reshape(h, w).astype(np.int64)


def encode_submission(masks: Mapping[str, np.ndarray]) -> dict[str, str]:
    """Encode ``{image_id: label_image}`` to ``{image_id: base64 PNG}``."""
    return {
        image_id: base64.b64encode(encode_mask(m)).decode("ascii")
        for image_id, m in masks.items()
    }


def decode_submission(payload: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Decode ``{image_id: base64 PNG}`` back to ``{image_id: label_image}``."""
    return {
        image_id: decode_mask(base64.b64decode(b64))
        for image_id, b64 in payload.items()
    }
