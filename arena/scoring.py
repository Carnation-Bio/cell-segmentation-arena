"""Instance-segmentation scoring — pure numpy, no pycocotools, no scipy.

This is the single source of truth for the metric. The participant's
``score_local`` and the leaderboard server both call ``score`` here, so a local
score and a leaderboard score are computed by identical code.

Metric (the Cellpose / LIVECell convention):

* Match predicted instances to ground-truth instances by IoU.
* Greedy one-to-one matching: highest-IoU pairs first, each instance used once.
* At an IoU threshold ``t``: a matched pair with IoU >= t is a true positive
  (TP); an unmatched prediction is a false positive (FP); an unmatched GT is a
  false negative (FN).
* ``AP@t = TP / (TP + FP + FN)``  (this is the Cellpose AP, *not* the integrated
  COCO precision-recall AP).
* ``mAP@[.5:.95]`` averages AP over thresholds 0.50, 0.55, ..., 0.95.
* ``F1@0.5`` is reported as a friendly headline (micro-averaged over images).

Label arrays are ``(H, W)`` integer images: 0 = background, 1..N = instances.
Matching is greedy with deterministic tie-breaking, so the score is fully
reproducible: same inputs always give the same number.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from arena.labels import relabel_sequential

# IoU thresholds 0.50, 0.55, ..., 0.95 (ten of them).
THRESHOLDS: np.ndarray = np.round(np.linspace(0.5, 0.95, 10), 2)

LabelArray = np.ndarray


class Score(dict):
    """A scoring result. A plain dict (JSON-friendly) with a readable repr."""

    def __repr__(self) -> str:
        return (
            f"Score(mAP@[.5:.95]={self['map']:.4f}, "
            f"F1@0.5={self['f1']:.4f}, n_images={self['n_images']})"
        )


def _iou_matrix(gt: LabelArray, pred: LabelArray) -> tuple[np.ndarray, int, int]:
    """IoU between every GT instance and every predicted instance.

    Returns ``(iou[n_gt, n_pred], n_gt, n_pred)``. Background row/column dropped.
    """
    gt, n_gt = relabel_sequential(gt)
    pred, n_pred = relabel_sequential(pred)
    if gt.shape != pred.shape:
        raise ValueError(
            f"prediction shape {pred.shape} != ground-truth shape {gt.shape}"
        )
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64), n_gt, n_pred

    # Pixel-overlap histogram including the background row/col (index 0).
    overlap = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64)
    np.add.at(overlap, (gt.ravel(), pred.ravel()), 1)

    area_gt = overlap.sum(axis=1, keepdims=True)
    area_pred = overlap.sum(axis=0, keepdims=True)
    union = area_gt + area_pred - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, overlap / union, 0.0)
    return iou[1:, 1:], n_gt, n_pred


def _greedy_tp(iou: np.ndarray, t: float) -> int:
    """True positives at threshold ``t`` via greedy one-to-one matching.

    Highest-IoU eligible pairs are matched first; each GT and each prediction is
    consumed at most once. Ties break deterministically by (gt index, pred
    index), so the result never depends on input ordering.
    """
    if iou.size == 0:
        return 0
    gt_idx, pred_idx = np.where(iou >= t)
    if gt_idx.size == 0:
        return 0
    vals = iou[gt_idx, pred_idx]
    # Primary key (last) = -vals -> descending IoU; then gt index, then pred index.
    order = np.lexsort((pred_idx, gt_idx, -vals))
    gt_used = np.zeros(iou.shape[0], dtype=bool)
    pred_used = np.zeros(iou.shape[1], dtype=bool)
    tp = 0
    for k in order:
        g = gt_idx[k]
        p = pred_idx[k]
        if gt_used[g] or pred_used[p]:
            continue
        gt_used[g] = True
        pred_used[p] = True
        tp += 1
    return tp


def _counts(
    gt: LabelArray, pred: LabelArray, thresholds: np.ndarray
) -> np.ndarray:
    """Per-threshold ``[tp, fp, fn]`` for one image. Shape ``(len(thresholds), 3)``."""
    iou, n_gt, n_pred = _iou_matrix(gt, pred)
    out = np.zeros((len(thresholds), 3), dtype=np.int64)
    for i, t in enumerate(thresholds):
        tp = _greedy_tp(iou, float(t))
        out[i] = (tp, n_pred - tp, n_gt - tp)
    return out


def average_precision(
    gt: LabelArray, pred: LabelArray, thresholds: np.ndarray = THRESHOLDS
) -> np.ndarray:
    """Per-threshold AP for a single image (convenience wrapper).

    ``AP@t = TP / (TP + FP + FN)``. An image with no GT and no prediction scores
    1.0 (nothing to find, nothing found); no GT but some predictions scores 0.0.
    """
    counts = _counts(gt, pred, thresholds)
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    denom = tp + fp + fn
    ap = np.where(denom > 0, tp / np.maximum(denom, 1), 1.0)
    return ap


def score(
    preds: Sequence[LabelArray] | LabelArray,
    gts: Sequence[LabelArray] | LabelArray,
    thresholds: np.ndarray = THRESHOLDS,
) -> Score:
    """Score predictions against ground truth over one or many images.

    ``preds``/``gts`` are aligned: ``preds[i]`` is scored against ``gts[i]``.
    A single 2D array is accepted as a one-image shorthand.

    Returns a :class:`Score` with ``map`` (mAP@[.5:.95], the ranking metric),
    ``f1`` (F1@0.5, micro-averaged), ``ap_per_threshold``, and ``n_images``.
    """
    if isinstance(preds, np.ndarray) and preds.ndim == 2:
        preds = [preds]
    if isinstance(gts, np.ndarray) and gts.ndim == 2:
        gts = [gts]
    if len(preds) != len(gts):
        raise ValueError(
            f"got {len(preds)} predictions but {len(gts)} ground-truth images"
        )
    if not preds:
        raise ValueError("nothing to score: empty prediction list")

    # (n_images, n_thresholds, 3) tensor of [tp, fp, fn].
    per_image = np.stack([_counts(g, p, thresholds) for p, g in zip(preds, gts)])

    tp = per_image[:, :, 0]
    fp = per_image[:, :, 1]
    fn = per_image[:, :, 2]
    denom = tp + fp + fn
    ap = np.where(denom > 0, tp / np.maximum(denom, 1), 1.0)  # (n_images, n_thr)

    ap_per_threshold = ap.mean(axis=0)  # mean over images, per threshold
    map_score = float(ap_per_threshold.mean())

    # F1@0.5, micro-averaged: pool TP/FP/FN across images at the 0.5 threshold.
    i50 = int(np.argmin(np.abs(thresholds - 0.5)))
    tp50, fp50, fn50 = int(tp[:, i50].sum()), int(fp[:, i50].sum()), int(fn[:, i50].sum())
    f1_denom = 2 * tp50 + fp50 + fn50
    f1 = (2 * tp50 / f1_denom) if f1_denom > 0 else 1.0

    return Score(
        map=map_score,
        f1=float(f1),
        ap_per_threshold={
            float(t): float(a) for t, a in zip(thresholds, ap_per_threshold)
        },
        n_images=len(preds),
    )


def score_local(pred_masks, val_labels) -> Score:
    """Score your predictions against the local validation labels.

    Accepts aligned lists, single arrays, or dicts keyed by image id (the shape
    ``segment_all`` returns) — in the dict case, only shared ids are scored.
    This is exactly the metric the leaderboard uses, so iterate against it
    locally (it's free and instant) before spending a submission.
    """
    from collections.abc import Mapping

    if isinstance(pred_masks, Mapping) and isinstance(val_labels, Mapping):
        keys = [k for k in val_labels if k in pred_masks]
        if not keys:
            raise ValueError("no overlapping image ids between predictions and labels")
        return score([pred_masks[k] for k in keys], [val_labels[k] for k in keys])
    return score(pred_masks, val_labels)
