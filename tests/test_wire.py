"""Tests for the mask wire format (lossless PNG round-trip)."""

from __future__ import annotations

import numpy as np

from arena.scoring import score
from arena.wire import (
    decode_mask,
    decode_submission,
    encode_mask,
    encode_submission,
)


def test_roundtrip_preserves_instances():
    rng = np.random.default_rng(1)
    lab = np.zeros((96, 128), dtype=np.int64)
    for i in range(1, 40):
        r, c = rng.integers(0, 80, size=2)
        lab[r : r + 10, c : c + 8] = i
    decoded = decode_mask(encode_mask(lab))
    # Instance identity is preserved: a perfect self-score after the round trip.
    assert score(decoded, lab)["map"] == 1.0


def test_roundtrip_handles_sparse_huge_labels():
    lab = np.zeros((32, 32), dtype=np.int64)
    lab[2:10, 2:10] = 90000
    lab[20:28, 20:28] = 7
    decoded = decode_mask(encode_mask(lab))
    # Two instances in, two distinct instances out, scoring identically.
    assert score(decoded, lab)["map"] == 1.0
    assert len(np.unique(decoded)) == 3  # background + 2 cells


def test_empty_mask_roundtrips():
    lab = np.zeros((16, 16), dtype=np.int64)
    decoded = decode_mask(encode_mask(lab))
    assert decoded.shape == (16, 16)
    assert decoded.max() == 0


def test_submission_dict_roundtrip():
    masks = {
        "img_a": np.pad(np.ones((4, 4), dtype=np.int64), 2),
        "img_b": np.zeros((8, 8), dtype=np.int64),
    }
    out = decode_submission(encode_submission(masks))
    assert set(out) == {"img_a", "img_b"}
    assert score(out["img_a"], masks["img_a"])["map"] == 1.0
