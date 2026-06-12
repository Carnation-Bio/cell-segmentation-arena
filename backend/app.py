"""Model backend — the generic inference API for the workshop.

A single Modal app exposing a token-authed HTTP endpoint:

    POST /segment   {model, image|image_id, text, boxes, points, params} -> {mask}
    POST /finetune  {labels, base_model, hyperparams}                     -> {adapter_id}
    GET  /health

The backend is a *pure model server*: it runs the model and returns instance
masks. All scikit-image pre/post-processing (blur, CLAHE, watershed-split, ...)
lives client-side in the ``arena`` package, so the participant's pipeline stays
ordinary, inspectable Python.

Models are served by warm GPU classes (one per model family, isolated images so
their deps never collide). The CPU web endpoint authenticates and dispatches.

Deploy::

    modal deploy -e workshop backend/app.py
    ARENA_WARM_CELLPOSE=2 ARENA_WARM_SAM3=2 modal deploy -e workshop backend/app.py  # session warm pool

Deploy with Python >=3.10 (FastAPI introspects real annotations here — a
`from __future__ import annotations` would stringize the closure-local request
models and break body parsing).
"""

import os

import modal

app = modal.App("cell-arena-models")

CUDA = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
TORCH_CU126 = "https://download.pytorch.org/whl/cu126"

# Persisted caches so model weights download once across cold starts.
weights_vol = modal.Volume.from_name("arena-weights", create_if_missing=True)
images_vol = modal.Volume.from_name("arena-images", create_if_missing=True)
adapters_vol = modal.Volume.from_name("arena-adapters", create_if_missing=True)

tokens_secret = modal.Secret.from_name("arena-tokens")
# HF secret name is configurable so the gated-SAM3 token isn't hard-coded here.
HF_SECRET_NAME = os.environ.get("ARENA_HF_SECRET", "carnation-hf")

# How many containers to keep warm (0 in dev; bumped at deploy for the session).
WARM_CELLPOSE = int(os.environ.get("ARENA_WARM_CELLPOSE", "0"))
WARM_SAM3 = int(os.environ.get("ARENA_WARM_SAM3", "0"))
WARM_OMNIPOSE = int(os.environ.get("ARENA_WARM_OMNIPOSE", "0"))

# -----------------------------------------------------------------------------
# Images
# -----------------------------------------------------------------------------
cellpose_image = (
    modal.Image.from_registry(CUDA, add_python="3.12")
    .pip_install("torch", "torchvision", index_url=TORCH_CU126)
    .pip_install("cellpose", "packaging", "pillow", "scikit-image", "numpy")
    .env({"CELLPOSE_LOCAL_MODELS_PATH": "/weights/cellpose"})
    .add_local_python_source("arena")
)

sam3_image = (
    modal.Image.from_registry(CUDA, add_python="3.12")
    .pip_install("torch>=2.7", "torchvision", index_url=TORCH_CU126)
    .apt_install("git")
    .pip_install("git+https://github.com/facebookresearch/sam3.git")
    # sam3's image processor transitively imports its tracker + training-data
    # modules, so these "optional" deps are actually required. sam3 pins numpy<2.
    .pip_install(
        "einops", "pycocotools", "opencv-python-headless", "numba",
        "python-rapidjson", "pandas", "scikit-image", "scikit-learn", "scipy",
        "psutil", "decord", "pillow", "numpy<2",
    )
    .env({"HF_HOME": "/weights/hf"})
    .add_local_python_source("arena")
)

# cyto3 is a Cellpose v3 model — v4's name list doesn't include it (it silently
# aliases to cpsam_v2), so it gets its own pinned-v3 image.
cyto3_image = (
    modal.Image.from_registry(CUDA, add_python="3.12")
    .pip_install("torch", "torchvision", index_url=TORCH_CU126)
    .pip_install("cellpose>=3.1,<4", "packaging", "pillow", "numpy<2")
    .env({"CELLPOSE_LOCAL_MODELS_PATH": "/weights/cellpose3"})
    .add_local_python_source("arena")
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "pillow", "numpy")
    .add_local_python_source("arena")
)

CELLPOSE_MODELS = {"cpsam_v2", "cpsam"}  # real, distinct v4 weights
CYTO3_MODELS = {"cyto3"}  # served by a separate Cellpose-v3 runner
SAM3_MODELS = {"sam3"}
OMNIPOSE_MODELS: set[str] = set()  # added in a later, isolated deploy
ALL_MODELS = CELLPOSE_MODELS | CYTO3_MODELS | SAM3_MODELS | OMNIPOSE_MODELS


def _hf_secret() -> modal.Secret:
    """Gated SAM-3 token. Name/env configurable so no product name lives here."""
    name = os.environ.get("ARENA_HF_SECRET", "workshop-hf")
    env = os.environ.get("ARENA_HF_SECRET_ENV")
    return modal.Secret.from_name(name, environment_name=env) if env else modal.Secret.from_name(name)


# -----------------------------------------------------------------------------
# Cellpose runner (cpsam_v2 / cyto3 / cpsam)
# -----------------------------------------------------------------------------
@app.cls(
    image=cellpose_image,
    gpu="L4",
    volumes={"/weights": weights_vol, "/images": images_vol, "/adapters": adapters_vol},
    timeout=600,
    scaledown_window=300,
    min_containers=WARM_CELLPOSE,
)
class CellposeRunner:
    @modal.enter()
    def setup(self) -> None:
        import os as _os

        _os.makedirs("/weights/cellpose", exist_ok=True)
        self._models: dict = {}

    def _model(self, name: str):
        from cellpose import models

        if name not in self._models:
            if name.startswith("adapter-"):
                # a fine-tuned adapter saved by FinetuneRunner (cellpose train_seg
                # writes to {save_path}/models/{model_name})
                adapters_vol.reload()
                pretrained = f"/adapters/models/{name}"
            else:
                # cpsam_v2 / cpsam are native v4 names; cyto3 loads legacy weights
                # by name. Always pretrained_model= (model_type= mis-loads).
                pretrained = name
            self._models[name] = models.CellposeModel(gpu=True, pretrained_model=pretrained)
        return self._models[name]

    @modal.method()
    def segment(self, model_name: str, image, params: dict | None) -> dict:
        import time

        import numpy as np

        params = dict(params or {})
        model = self._model(model_name)
        arr = np.asarray(image)
        t0 = time.time()
        out = model.eval(arr, **params)
        masks = out[0]
        return {
            "mask": np.ascontiguousarray(masks.astype(np.int32)),
            "n_instances": int(masks.max()),
            "seconds": round(time.time() - t0, 2),
        }


# -----------------------------------------------------------------------------
# SAM-3 runner (text + box/point prompts)
# -----------------------------------------------------------------------------
@app.cls(
    image=sam3_image,
    gpu="L4",
    volumes={"/weights": weights_vol, "/images": images_vol},
    secrets=[_hf_secret()],
    timeout=900,
    scaledown_window=300,
    min_containers=WARM_SAM3,
)
class Sam3Runner:
    @modal.enter()
    def setup(self) -> None:
        import os as _os

        from huggingface_hub import login

        token = _os.environ.get("HF_TOKEN")
        if token:
            login(token=token)  # gated SAM-3 weights need an authorized HF account
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.processor = Sam3Processor(build_sam3_image_model())

    @modal.method()
    def segment(self, model_name: str, image, text, boxes, points, params: dict | None) -> dict:
        import time

        import numpy as np
        import torch
        from PIL import Image

        params = dict(params or {})
        arr = np.asarray(image)
        rgb = Image.fromarray(arr).convert("RGB")
        label = np.zeros(arr.shape[:2], dtype=np.int32)

        def to_np(x):
            return np.asarray(x.detach().float().cpu()) if hasattr(x, "detach") else np.asarray(x)

        def paint(out, start: int) -> int:
            masks = to_np(out["masks"])
            scores = to_np(out["scores"]) if "scores" in out else np.ones(len(masks))
            if masks.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
                masks = masks[:, 0]
            nid = start
            for i in np.argsort(scores):  # low score first -> high-conf wins overlaps
                nid += 1
                label[masks[i] > 0.5] = nid
            return nid

        t0 = time.time()
        # SAM-3 weights are bfloat16 — run under autocast so inputs match.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(rgb)
            # 'confidence' is the real lever: lower -> SAM-3 returns more detections.
            if "confidence" in params:
                self.processor.set_confidence_threshold(float(params["confidence"]), state=state)
            if boxes:
                nid = 0
                for box in boxes:
                    self.processor.reset_all_prompts(state)
                    out = self.processor.add_geometric_prompt(
                        box=[float(x) for x in box], label=True, state=state
                    )
                    nid = paint(out, nid)
            else:
                out = self.processor.set_text_prompt(prompt=text or "cell", state=state)
                paint(out, 0)
        return {
            "mask": np.ascontiguousarray(label),
            "n_instances": int(label.max()),
            "seconds": round(time.time() - t0, 2),
        }


# -----------------------------------------------------------------------------
# Cyto3 runner (Cellpose v3 — the classic, genuinely different model)
# -----------------------------------------------------------------------------
@app.cls(
    image=cyto3_image,
    gpu="L4",
    volumes={"/weights": weights_vol, "/images": images_vol},
    timeout=600,
    scaledown_window=300,
    min_containers=WARM_CELLPOSE,
)
class Cyto3Runner:
    @modal.enter()
    def setup(self) -> None:
        import os as _os

        _os.makedirs("/weights/cellpose3", exist_ok=True)
        from cellpose import models

        self.model = models.Cellpose(gpu=True, model_type="cyto3")

    @modal.method()
    def segment(self, model_name: str, image, params: dict | None) -> dict:
        import time

        import numpy as np

        params = dict(params or {})
        diameter = params.pop("diameter", None)  # None -> auto-estimate
        arr = np.asarray(image)
        t0 = time.time()
        masks = self.model.eval(arr, diameter=diameter, channels=[0, 0], **params)[0]
        return {
            "mask": np.ascontiguousarray(masks.astype(np.int32)),
            "n_instances": int(masks.max()),
            "seconds": round(time.time() - t0, 2),
        }


# -----------------------------------------------------------------------------
# Fine-tune runner (A100) — implemented in the finetune task
# -----------------------------------------------------------------------------
@app.cls(
    image=cellpose_image,
    gpu="A100-40GB",
    volumes={"/weights": weights_vol, "/adapters": adapters_vol},
    timeout=1800,
    scaledown_window=120,
)
class FinetuneRunner:
    @modal.method()
    def train(self, team: str, base_model: str, frames: list, hyperparams: dict) -> str:
        import os
        import uuid

        import numpy as np
        from cellpose import models, train

        hp = dict(hyperparams or {})
        imgs = [np.asarray(im) for im, _ in frames]
        labels = [np.asarray(lab).astype(np.int32) for _, lab in frames]

        base = base_model if base_model in CELLPOSE_MODELS else "cpsam_v2"
        model = models.CellposeModel(gpu=True, pretrained_model=base)

        slug = "".join(c if c.isalnum() else "_" for c in team)[:16]
        adapter_id = f"adapter-{slug}-{uuid.uuid4().hex[:8]}"
        os.makedirs("/adapters/models", exist_ok=True)
        train.train_seg(
            model.net,
            train_data=imgs,
            train_labels=labels,
            n_epochs=int(hp.get("n_epochs", 100)),
            learning_rate=float(hp.get("learning_rate", 1e-5)),
            weight_decay=float(hp.get("weight_decay", 0.1)),
            batch_size=int(hp.get("batch_size", 1)),
            save_path="/adapters",
            model_name=adapter_id,
        )
        adapters_vol.commit()
        return adapter_id


# -----------------------------------------------------------------------------
# Auth (runs inside the web container)
# -----------------------------------------------------------------------------
def _allowed_tokens() -> set[str]:
    raw = os.environ.get("ARENA_TOKENS", "")
    return {t.strip() for t in raw.replace(",", "\n").split() if t.strip()}


def _team_for_token(token: str | None) -> str:
    """Return the team id for a valid token, else raise 401.

    A token is ``<team>:<secret>`` or just ``<secret>`` (team = secret prefix).
    The allowed set is the full token strings.
    """
    from fastapi import HTTPException

    if not token or token not in _allowed_tokens():
        raise HTTPException(status_code=401, detail="invalid or missing team token")
    return token.split(":", 1)[0]


# -----------------------------------------------------------------------------
# Web endpoint
# -----------------------------------------------------------------------------
@app.function(
    image=web_image,
    secrets=[tokens_secret],
    timeout=600,
    min_containers=int(os.environ.get("ARENA_WARM_WEB", "0")),
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import base64
    from io import BytesIO

    import numpy as np
    from fastapi import FastAPI, Header, HTTPException
    from PIL import Image
    from pydantic import BaseModel

    from arena.wire import decode_mask, encode_mask

    api = FastAPI(title="cell-arena model backend")

    def _team(authorization: str | None) -> str:
        token = authorization.removeprefix("Bearer ").strip() if authorization else None
        return _team_for_token(token)

    def _load_image(image_b64: str | None, image_id: str | None) -> np.ndarray:
        if image_b64:
            return np.array(Image.open(BytesIO(base64.b64decode(image_b64))).convert("L"))
        if image_id:
            images_vol.reload()
            return np.array(Image.open(f"/images/{image_id}.png").convert("L"))
        raise HTTPException(status_code=400, detail="provide 'image' (b64 PNG) or 'image_id'")

    class SegmentRequest(BaseModel):
        model: str = "cpsam_v2"
        image: str | None = None
        image_id: str | None = None
        text: str | None = None
        boxes: list | None = None
        points: list | None = None
        params: dict | None = None

    class FinetuneRequest(BaseModel):
        base_model: str = "cpsam_v2"
        labels: list[dict]
        hyperparams: dict = {}

    @api.get("/health")
    def health():
        return {"ok": True, "models": sorted(ALL_MODELS)}

    @api.post("/segment")
    def segment(req: SegmentRequest, authorization: str | None = Header(default=None)):
        team = _team(authorization)
        image = _load_image(req.image, req.image_id)
        params = req.params or {}

        if req.model in CELLPOSE_MODELS:
            result = CellposeRunner().segment.remote(req.model, image, params)
        elif req.model in CYTO3_MODELS:
            result = Cyto3Runner().segment.remote(req.model, image, params)
        elif req.model in SAM3_MODELS:
            result = Sam3Runner().segment.remote(
                req.model, image, req.text or "cell", req.boxes, req.points, params
            )
        elif req.model.startswith("adapter-"):
            result = CellposeRunner().segment.remote(req.model, image, params)
        else:
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")

        return {
            "team": team,
            "model": req.model,
            "mask": base64.b64encode(encode_mask(result["mask"])).decode("ascii"),
            "n_instances": result["n_instances"],
            "seconds": result["seconds"],
        }

    @api.post("/finetune")
    def finetune(req: FinetuneRequest, authorization: str | None = Header(default=None)):
        team = _team(authorization)
        frames = []
        for item in req.labels:
            img = np.array(Image.open(BytesIO(base64.b64decode(item["image"]))).convert("L"))
            lab = decode_mask(base64.b64decode(item["label"]))
            frames.append((img, lab))
        adapter_id = FinetuneRunner().train.remote(team, req.base_model, frames, req.hyperparams)
        return {"team": team, "adapter_id": adapter_id, "base_model": req.base_model}

    return api
