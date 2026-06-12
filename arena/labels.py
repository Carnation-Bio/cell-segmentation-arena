"""Label-image utilities shared across the toolkit.

A label image is ``(H, W)`` integer: 0 = background, 1..N = instances.
"""

from __future__ import annotations

import numpy as np


def relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Relabel foreground instances to a contiguous 1..N, keeping 0 as background.

    Returns ``(relabeled, n_instances)``. Robust to sparse or very large label
    values — uses ``searchsorted`` rather than a dense lookup table, so a
    prediction labeled e.g. ``{7, 500, 90000}`` is handled without allocating a
    90000-element table.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2D label image, got shape {labels.shape}")
    out = np.zeros(labels.shape, dtype=np.int64)
    fg = labels != 0
    ids = np.unique(labels[fg])
    if ids.size:
        out[fg] = np.searchsorted(ids, labels[fg]) + 1
    return out, int(ids.size)
