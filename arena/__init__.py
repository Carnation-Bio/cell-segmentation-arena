"""cell-arena — participant toolkit for the dense-cell segmentation workshop.

Public API (all importable straight from ``arena``)::

    from arena import segment, finetune, submit          # backend calls
    from arena import score_local                         # local self-scoring
    from arena import show, compare, zoom                 # local visualization
    from arena import load_public_test, load_reference_frames, load_local_val

Exports are loaded lazily (PEP 562) so importing one symbol never drags in the
others' heavy deps — e.g. the leaderboard server imports ``arena.scoring`` and
``arena.wire`` without ever touching matplotlib.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.3.1"

# public name -> submodule that defines it
_EXPORTS = {
    "segment": "arena.client",
    "segment_all": "arena.client",
    "run_pipeline": "arena.client",
    "finetune": "arena.client",
    "submit": "arena.client",
    "score_local": "arena.scoring",
    "show": "arena.viz",
    "gallery": "arena.viz",
    "browse": "arena.viz",
    "compare": "arena.viz",
    "zoom": "arena.viz",
    "configure": "arena.config",
    "load_public_test": "arena.data",
    "load_train_frames": "arena.data",
    "load_reference_frames": "arena.data",
    "load_local_val": "arena.data",
}


def __getattr__(name: str):  # noqa: D401 — PEP 562 lazy attribute loader
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'arena' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(_EXPORTS)


if TYPE_CHECKING:  # static help only; never executed
    from arena.client import finetune, segment, segment_all, submit
    from arena.config import configure
    from arena.data import (
        load_local_val,
        load_public_test,
        load_reference_frames,
        load_train_frames,
    )
    from arena.scoring import score_local
    from arena.viz import compare, show, zoom
