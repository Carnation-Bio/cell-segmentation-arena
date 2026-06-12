"""Tests for the client-side process/refine pipeline steps."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk

from arena.processing import apply_process, apply_refine


def _dumbbell() -> np.ndarray:
    """Two overlapping circles labeled as ONE instance (a merged blob)."""
    lab = np.zeros((60, 100), dtype=np.int32)
    for cx in (35, 65):
        rr, cc = disk((30, cx), 18, shape=lab.shape)
        lab[rr, cc] = 1
    return lab


def test_watershed_split_separates_merged_cells():
    merged = _dumbbell()
    assert len(np.unique(merged)) - 1 == 1  # one merged blob going in
    split = apply_refine(merged, ["watershed_split"])
    assert len(np.unique(split)) - 1 == 2  # two cells coming out


def test_watershed_split_leaves_isolated_cell_alone():
    lab = np.zeros((60, 60), dtype=np.int32)
    rr, cc = disk((30, 30), 15, shape=lab.shape)
    lab[rr, cc] = 1
    split = apply_refine(lab, ["watershed_split"])
    assert len(np.unique(split)) - 1 == 1  # single peak -> unchanged


def test_min_size_prunes_small_instances():
    lab = np.zeros((40, 40), dtype=np.int32)
    lab[2:4, 2:4] = 1  # 4 px speck
    rr, cc = disk((25, 25), 8, shape=lab.shape)
    lab[rr, cc] = 2  # big cell
    out = apply_refine(lab, ["min_size:30"])
    assert len(np.unique(out)) - 1 == 1


def test_remove_edge_drops_border_touching():
    lab = np.zeros((40, 40), dtype=np.int32)
    lab[0:10, 0:10] = 1  # touches the corner
    rr, cc = disk((25, 25), 8, shape=lab.shape)
    lab[rr, cc] = 2
    out = apply_refine(lab, ["remove_edge"])
    assert len(np.unique(out)) - 1 == 1


def test_process_returns_uint8_same_shape():
    img = (np.random.default_rng(0).random((50, 70)) * 255).astype(np.uint8)
    out = apply_process(img, ["gaussian:1", "clahe", "normalize"])
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_process_passthrough_when_no_steps():
    img = (np.random.default_rng(1).random((20, 20)) * 255).astype(np.uint8)
    assert np.array_equal(apply_process(img, None), img)


def test_unknown_step_raises():
    with pytest.raises(ValueError, match="unknown"):
        apply_process(np.zeros((10, 10), np.uint8), ["frobnicate:3"])
    with pytest.raises(ValueError, match="unknown"):
        apply_refine(np.zeros((10, 10), np.int32), ["zap"])


def test_byo_callable_steps():
    # a participant's own function can be a step, mixed with string shortcuts
    img = (np.random.default_rng(0).random((40, 40)) * 255).astype(np.uint8)
    out = apply_process(img, ["gaussian:1", lambda x: x * 0 + 7])
    assert out.dtype == np.uint8  # custom step ran, result still normalized to uint8

    lab = np.zeros((40, 40), dtype=np.int32)
    lab[5:15, 5:15] = 1
    refined = apply_refine(lab, [lambda m: (m > 0).astype(np.int32) * 9])
    assert set(np.unique(refined)) == {0, 1}  # relabeled to contiguous after the custom step
