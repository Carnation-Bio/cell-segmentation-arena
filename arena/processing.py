"""Composable scikit-image pipeline steps — the participant's main lever.

Steps are plain strings, ``"name"`` or ``"name:arg"`` / ``"name:a,b"``, applied
left to right. Two stages:

* ``process`` runs on the *image* before the model (denoise, contrast, ...).
* ``refine`` runs on the *label mask* after the model (split, prune, clean).

Everything here is ordinary local scikit-image — no GPU, no network. Read it,
copy it, change it.

The string specs are just shortcuts. The pipeline is **open**: a step can also be
**any function** (image->image for ``process``, labels->labels for ``refine``),
and since ``segment`` is plain array-in / array-out you can wrap it with whatever
code you like. Don't feel limited to this menu.

    def my_step(img):
        return some_scikit_image_thing(img)

    segment(img, model="cpsam_v2",
            process=["clahe", my_step],            # mix shortcuts + your own
            refine=["watershed_split", "min_size:30"])
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.measure import regionprops

# -----------------------------------------------------------------------------
# process: image -> image  (uint8 grayscale in, uint8 grayscale out)
# -----------------------------------------------------------------------------


def _as_float(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img.astype(np.float64)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def _gaussian(img, sigma: float = 1.0):
    return filters.gaussian(_as_float(img), sigma=float(sigma))


def _median(img, size: float = 3):
    return filters.median(_to_uint8(_as_float(img)), morphology.disk(int(size)))


def _clahe(img, clip: float = 0.01):
    return exposure.equalize_adapthist(_as_float(img), clip_limit=float(clip))


def _invert(img):
    return 1.0 - _as_float(img)


def _normalize(img, low: float = 1, high: float = 99):
    f = _as_float(img)
    lo, hi = np.percentile(f, [float(low), float(high)])
    return exposure.rescale_intensity(f, in_range=(lo, hi))


def _tophat(img, size: float = 15):
    return morphology.white_tophat(_to_uint8(_as_float(img)), morphology.disk(int(size)))


def _unsharp(img, radius: float = 2, amount: float = 1.0):
    return filters.unsharp_mask(_as_float(img), radius=float(radius), amount=float(amount))


PROCESS: dict[str, Callable] = {
    "gaussian": _gaussian,
    "median": _median,
    "clahe": _clahe,
    "invert": _invert,
    "normalize": _normalize,
    "tophat": _tophat,
    "unsharp": _unsharp,
}


# -----------------------------------------------------------------------------
# refine: labels -> labels
# -----------------------------------------------------------------------------


def _watershed_split(labels, min_distance: float = 7):
    """Split each instance into sub-cells by distance-transform watershed.

    Isolated cells (a single distance peak) pass through unchanged; blobs the
    model merged (multiple peaks) are split. This is the fix for models that
    *merge* touching cells (e.g. a coarse text-prompt mask). It can't recover
    cells the model never detected — score it on validation before trusting it.
    Larger ``min_distance`` splits more conservatively.
    """
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for region in regionprops(labels):
        sl = region.slice
        local = labels[sl] == region.label
        dist = ndi.distance_transform_edt(local)
        coords = peak_local_max(
            dist, min_distance=int(min_distance), labels=local, exclude_border=False
        )
        if len(coords) <= 1:
            out[sl][local] = next_id
            next_id += 1
            continue
        markers = np.zeros(local.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        split = segmentation.watershed(-dist, markers, mask=local)
        for sub in range(1, int(split.max()) + 1):
            piece = split == sub
            if piece.any():
                out[sl][piece] = next_id
                next_id += 1
    return out


def _min_size(labels, size: float = 30):
    # Done by hand (not remove_small_objects) to be stable across skimage versions.
    labels = np.asarray(labels)
    out = labels.copy()
    for region in regionprops(labels):
        if region.area < int(size):
            sl = region.slice
            out[sl][labels[sl] == region.label] = 0
    return out


def _remove_edge(labels):
    return segmentation.clear_border(np.asarray(labels))


def _fill_holes(labels):
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    for region in regionprops(labels):
        sl = region.slice
        filled = ndi.binary_fill_holes(labels[sl] == region.label)
        out[sl][filled] = region.label
    return out


def _expand(labels, distance: float = 2):
    return segmentation.expand_labels(np.asarray(labels), distance=int(distance))


def _open(labels, size: float = 1):
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    for region in regionprops(labels):
        sl = region.slice
        opened = morphology.binary_opening(labels[sl] == region.label, morphology.disk(int(size)))
        out[sl][opened] = region.label
    return out


REFINE: dict[str, Callable] = {
    "watershed_split": _watershed_split,
    "min_size": _min_size,
    "remove_edge": _remove_edge,
    "fill_holes": _fill_holes,
    "expand": _expand,
    "open": _open,
}


# -----------------------------------------------------------------------------
# step parsing + application
# -----------------------------------------------------------------------------


def _parse(step: str) -> tuple[str, list[float]]:
    name, _, rest = step.partition(":")
    args = [float(a) for a in rest.split(",")] if rest else []
    return name.strip(), args


def _apply(steps, registry: dict[str, Callable], x: np.ndarray) -> np.ndarray:
    for step in steps or []:
        if callable(step):
            # bring-your-own step: any function image->image (process) or
            # labels->labels (refine). The pipeline is open, not a fixed menu.
            x = step(x)
            continue
        name, args = _parse(step)
        fn = registry.get(name)
        if fn is None:
            raise ValueError(
                f"unknown {('process' if registry is PROCESS else 'refine')} step {name!r}; "
                f"available: {sorted(registry)}, or pass your own function"
            )
        x = fn(x, *args)
    return x


def apply_process(image: np.ndarray, steps: Sequence[str] | None) -> np.ndarray:
    """Run image-space steps; always returns a uint8 grayscale image."""
    if not steps:
        return _to_uint8(_as_float(image)) if np.asarray(image).dtype != np.uint8 else np.asarray(image)
    return _to_uint8(_apply(steps, PROCESS, image))


def apply_refine(labels: np.ndarray, steps: Sequence[str] | None) -> np.ndarray:
    """Run label-space steps; returns a relabeled int32 label image."""
    from arena.labels import relabel_sequential

    out = _apply(steps, REFINE, np.asarray(labels))
    return relabel_sequential(out)[0].astype(np.int32)
