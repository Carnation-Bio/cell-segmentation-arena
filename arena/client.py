"""Backend calls: segment, finetune, submit.

These talk to the model backend (segment/finetune) and the leaderboard (submit)
over plain HTTP with your team token. ``segment`` also runs the local
``process``/``refine`` scikit-image steps around the model call, so the whole
pipeline is one composable function.
"""

from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Mapping, Sequence

import numpy as np
import requests
from PIL import Image

from arena.config import get_config
from arena.processing import apply_process, apply_refine
from arena.wire import decode_mask, encode_mask, encode_submission


def _auth() -> dict[str, str]:
    cfg = get_config()
    if not cfg.token:
        raise RuntimeError("no team token set — call arena.configure(token=...) first")
    return {"Authorization": f"Bearer {cfg.token}"}


def _png_b64(image: np.ndarray) -> str:
    img = np.asarray(image)
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = np.zeros(img.shape, np.uint8) if hi <= lo else ((img - lo) / (hi - lo) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _raise(resp: requests.Response) -> None:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise RuntimeError(f"backend {resp.status_code}: {detail}")


def _listify(x):
    return np.asarray(x).tolist() if x is not None else None


def segment(
    image: np.ndarray | None = None,
    model: str = "sam3",
    text: str | None = None,
    boxes=None,
    points=None,
    process: Sequence[str] | None = None,
    refine: Sequence[str] | None = None,
    backend: str = "modal",
    image_id: str | None = None,
    params: dict | None = None,
    timeout: float = 600,
) -> np.ndarray:
    """Segment one image into an instance-label array.

    ``process`` (scikit-image steps) runs locally before the model; ``refine``
    (e.g. ``["watershed_split", "min_size:30"]``) runs locally after. With
    ``backend="local"`` the small models run on your own GPU instead of Modal.
    """
    if backend == "local":
        from arena.local_models import segment_local

        proc = apply_process(image, process) if image is not None else image
        mask = segment_local(proc, model=model, text=text, boxes=boxes, points=points, params=params)
        return apply_refine(mask, refine)

    cfg = get_config()
    payload: dict = {"model": model}
    if text is not None:
        payload["text"] = text
    if boxes is not None:
        payload["boxes"] = _listify(boxes)
    if points is not None:
        payload["points"] = _listify(points)
    if params:
        payload["params"] = params

    if image is not None:
        payload["image"] = _png_b64(apply_process(image, process))
    elif image_id is not None:
        if process:
            raise ValueError("process= needs the image array; pass image=, not image_id=")
        payload["image_id"] = image_id
    else:
        raise ValueError("segment() needs image= or image_id=")

    resp = requests.post(f"{cfg.backend_url}/segment", json=payload, headers=_auth(), timeout=timeout)
    _raise(resp)
    mask = decode_mask(base64.b64decode(resp.json()["mask"]))
    return apply_refine(mask, refine)


def segment_all(
    images: Mapping[str, np.ndarray], model: str = "sam3", max_workers: int = 8, **kwargs
) -> dict[str, np.ndarray]:
    """Segment many images concurrently. ``{id: image}`` -> ``{id: mask}``.

    This is how the notebook segments the whole test set quickly against the
    warm backend pool.
    """

    def one(item):
        key, img = item
        return key, segment(img, model=model, **kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(one, images.items()))


def run_pipeline(pipeline, images: Mapping[str, np.ndarray], max_workers: int = 8, progress: bool = True) -> dict:
    """Apply your own ``pipeline(image) -> mask`` to many images, concurrently.

    ``{id: image}`` -> ``{id: mask}``. This is how the workstation runs *your*
    approach over the validation or test set without you writing a loop. Prints a
    live progress counter (the GPU backend isn't instant — give it a few seconds).
    """
    from concurrent.futures import as_completed

    keys = list(images)
    n = len(keys)
    if progress:
        print(f"running your pipeline on {n} frames on the GPU backend "
              f"(~{max(3, n // 2)}s, not instant — hang tight)...", flush=True)
    out: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(pipeline, images[k]): k for k in keys}
        for done, fut in enumerate(as_completed(futures), start=1):
            out[futures[fut]] = fut.result()
            if progress and (done % max(1, n // 5) == 0 or done == n):
                print(f"  {done}/{n} frames done", flush=True)
    return out


def _encode_labels(labels: Iterable) -> list[dict]:
    """Encode an annotation set to wire form. Accepts ``[(image, label), ...]``
    or ``{id: (image, label)}``."""
    pairs = labels.values() if isinstance(labels, Mapping) else labels
    out = []
    for image, label in pairs:
        out.append({"image": _png_b64(image), "label": base64.b64encode(encode_mask(label)).decode("ascii")})
    return out


def finetune(labels: Iterable, base_model: str = "cpsam_v2", timeout: float = 1200, **hyperparams) -> str:
    """Fine-tune ``base_model`` on a handful of labeled frames; returns an
    ``adapter_id`` usable as ``segment(model=adapter_id)``."""
    cfg = get_config()
    payload = {"base_model": base_model, "hyperparams": hyperparams, "labels": _encode_labels(labels)}
    resp = requests.post(f"{cfg.backend_url}/finetune", json=payload, headers=_auth(), timeout=timeout)
    _raise(resp)
    adapter_id = resp.json()["adapter_id"]
    print(f"fine-tuned {base_model} -> adapter {adapter_id!r}; use segment(model={adapter_id!r})")
    return adapter_id


def submit(pred_masks: Mapping[str, np.ndarray], team: str, timeout: float = 300) -> float:
    """Submit predictions for the test set; returns your live public score."""
    cfg = get_config()
    payload = {"team": team, "masks": encode_submission(pred_masks)}
    resp = requests.post(f"{cfg.leaderboard_url}/submit", json=payload, headers=_auth(), timeout=timeout)
    _raise(resp)
    body = resp.json()
    score = body["public_score"]
    print(
        f"submitted {len(pred_masks)} masks — you're on the board.\n"
        f"  public score (mAP@[.5:.95]) = {score:.4f}   |   F1@0.5 = {body.get('public_f1', float('nan')):.4f}"
    )
    return score
