"""Tests for the pure-numpy instance scorer."""

from __future__ import annotations

import numpy as np
import pytest

from arena.scoring import THRESHOLDS, average_precision, score, score_local


def _blank(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.int64)


def _rect(lab: np.ndarray, label: int, r0: int, c0: int, r1: int, c1: int) -> None:
    lab[r0:r1, c0:c1] = label


def test_perfect_match_scores_one():
    gt = _blank()
    _rect(gt, 1, 2, 2, 10, 10)
    _rect(gt, 2, 20, 20, 35, 35)
    pred = gt.copy()
    result = score_local(pred, gt)
    assert result["map"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["n_images"] == 1


def test_completely_disjoint_scores_zero():
    gt = _blank()
    _rect(gt, 1, 2, 2, 10, 10)
    pred = _blank()
    _rect(pred, 1, 40, 40, 50, 50)  # nowhere near the GT cell
    result = score(pred, gt)
    assert result["map"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_one_of_two_found_is_half_map():
    gt = _blank()
    _rect(gt, 1, 2, 2, 12, 12)
    _rect(gt, 2, 30, 30, 45, 45)
    pred = _blank()
    _rect(pred, 1, 2, 2, 12, 12)  # exactly matches GT #1, misses GT #2
    result = score(pred, gt)
    # TP=1, FP=0, FN=1 at every threshold -> AP = 1/2 everywhere -> mAP = 0.5
    assert result["map"] == pytest.approx(0.5)
    # micro F1@0.5 = 2*1 / (2*1 + 0 + 1) = 2/3
    assert result["f1"] == pytest.approx(2 / 3)


def test_under_segmentation_is_penalized():
    # Two touching GT cells; one prediction blob covering both. This is the
    # dense-cell failure mode the workshop is about — it must score poorly.
    gt = _blank()
    _rect(gt, 1, 10, 10, 30, 20)  # left cell  (200 px)
    _rect(gt, 2, 10, 20, 30, 30)  # right cell (200 px), touching
    pred = _blank()
    _rect(pred, 1, 10, 10, 30, 30)  # one blob over both (400 px)
    iou = average_precision(gt, pred)
    # Each GT has IoU 200/400 = 0.5 with the merged pred. Greedy matches ONE of
    # them at t=0.5; the other GT is a FN and the merged pred is consumed.
    # At t=0.5: TP=1, FP=0, FN=1 -> AP=0.5. Above 0.5: no match -> AP=0.
    result = score(pred, gt)
    assert result["ap_per_threshold"][0.5] == pytest.approx(0.5)
    assert result["ap_per_threshold"][0.55] == pytest.approx(0.0)
    assert result["map"] < 0.1  # only the 0.5 bin contributes -> 0.05


def test_over_segmentation_is_penalized():
    # One GT cell, two predicted halves. One half can match (IoU 0.5), the other
    # is a false positive.
    gt = _blank()
    _rect(gt, 1, 10, 10, 30, 30)  # 400 px
    pred = _blank()
    _rect(pred, 1, 10, 10, 30, 20)  # left half (200 px) -> IoU 0.5
    _rect(pred, 2, 10, 20, 30, 30)  # right half (200 px) -> IoU 0.5
    result = score(pred, gt)
    # t=0.5: greedy matches one half (TP=1), other half FP=1, FN=0 -> AP=1/2.
    assert result["ap_per_threshold"][0.5] == pytest.approx(0.5)


def test_empty_prediction_scores_zero():
    gt = _blank()
    _rect(gt, 1, 2, 2, 10, 10)
    pred = _blank()
    result = score(pred, gt)
    assert result["map"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_both_empty_scores_one():
    result = score(_blank(), _blank())
    assert result["map"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_noncontiguous_labels_match_contiguous():
    gt = _blank()
    _rect(gt, 1, 2, 2, 10, 10)
    _rect(gt, 2, 20, 20, 30, 30)
    pred = gt.copy()
    # Relabel pred with sparse, out-of-order ids — score must be unchanged.
    pred_sparse = np.zeros_like(pred)
    pred_sparse[pred == 1] = 500
    pred_sparse[pred == 2] = 7
    assert score(pred_sparse, gt)["map"] == pytest.approx(score(pred, gt)["map"])
    assert score(pred_sparse, gt)["map"] == pytest.approx(1.0)


def test_deterministic_and_order_independent():
    rng = np.random.default_rng(0)
    gt = _blank(128, 128)
    pred = _blank(128, 128)
    for i in range(1, 12):
        r, c = rng.integers(0, 110, size=2)
        _rect(gt, i, r, c, r + 12, c + 12)
        # jitter the prediction a little so IoUs vary
        _rect(pred, i, r + 1, c, r + 12, c + 11)
    a = score(pred, gt)["map"]
    b = score(pred, gt)["map"]
    assert a == b  # exact, not approx — reproducibility is a hard requirement


def test_multi_image_averaging():
    gt1 = _blank()
    _rect(gt1, 1, 2, 2, 12, 12)
    perfect = gt1.copy()
    miss = _blank()  # empty prediction -> AP 0 on image 2
    result = score([perfect, miss], [gt1, gt1])
    # image1 mAP=1.0, image2 mAP=0.0 -> dataset mAP=0.5
    assert result["map"] == pytest.approx(0.5)
    assert result["n_images"] == 2


def test_thresholds_are_ten_from_50_to_95():
    assert len(THRESHOLDS) == 10
    assert THRESHOLDS[0] == pytest.approx(0.5)
    assert THRESHOLDS[-1] == pytest.approx(0.95)
