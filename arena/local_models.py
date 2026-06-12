"""Optional local inference — run the small/medium models on your own GPU.

Reached via ``segment(..., backend="local")``. Needs the heavier, opt-in deps
from ``requirements-local.txt`` (cellpose etc.) — they're deliberately out of the
default install so the golden path stays tiny. SAM-3 and fine-tuning always stay
remote (they need more than a laptop has).
"""

from __future__ import annotations

import numpy as np

_cache: dict = {}


def _get(key: str, build):
    if key not in _cache:
        _cache[key] = build()
    return _cache[key]


def segment_local(image, model="cpsam_v2", text=None, boxes=None, points=None, params=None):
    """Segment locally with cellpose. Returns an int label array."""
    params = dict(params or {})
    if model in {"sam3"}:
        raise RuntimeError("SAM-3 is too large for local; use backend='modal' (the default).")

    try:
        from cellpose import models as cp
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "backend='local' needs the optional deps: pip install -r requirements-local.txt"
        ) from exc

    arr = np.asarray(image)
    if model == "cyto3":
        diameter = params.pop("diameter", None)
        m = _get("cyto3", lambda: cp.Cellpose(gpu=True, model_type="cyto3"))
        masks = m.eval(arr, diameter=diameter, channels=[0, 0], **params)[0]
    elif model in {"cpsam_v2", "cpsam"} or model.startswith("adapter-"):
        m = _get(model, lambda: cp.CellposeModel(gpu=True, pretrained_model=model))
        masks = m.eval(arr, **params)[0]
    else:
        raise ValueError(f"unknown local model {model!r}")
    return np.ascontiguousarray(masks.astype(np.int32))
