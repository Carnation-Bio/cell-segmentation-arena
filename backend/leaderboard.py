"""Leaderboard service — receives masks, scores vs hidden GT, serves the board.

A separate Modal app (CPU). The hidden public + private ground truth lives in a
Modal Volume, never exposed. Every submission is scored against BOTH; the public
score is returned live, the private score is stored and only revealed (with the
admin token) at the end. Best-per-team state lives in a ``modal.Dict``.

    POST /submit  {team, masks:{id:b64png}}  -> {public_score, public_f1}
    GET  /board                              -> live public ranking (JSON)
    GET  /board?reveal=<admin>               -> + private ranking
    GET  /                                    -> the live board page

Deploy::

    modal deploy -e workshop backend/leaderboard.py

Deploy with Python >=3.10 (FastAPI introspects real annotations; a
`from __future__ import annotations` would break closure-local body models).
"""

import os

import modal

app = modal.App("cell-arena-leaderboard")

gt_vol = modal.Volume.from_name("arena-gt", create_if_missing=True)
bundle_vol = modal.Volume.from_name("arena-bundle", create_if_missing=True)
board_state = modal.Dict.from_name("arena-board", create_if_missing=True)
tokens_secret = modal.Secret.from_name("arena-tokens")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "numpy", "pillow")
    .add_local_python_source("arena")
    .add_local_dir("board", remote_path="/static")
)


@app.function(
    image=image,
    volumes={"/gt": gt_vol, "/bundle": bundle_vol},
    secrets=[tokens_secret],
    timeout=300,
    min_containers=int(os.environ.get("ARENA_WARM_BOARD", "0")),
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import time

    import numpy as np
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    from arena.scoring import score
    from arena.wire import decode_submission

    def _load_gt(name: str) -> dict[str, np.ndarray]:
        gt_vol.reload()
        with np.load(f"/gt/{name}_gt.npz") as data:
            return {k: data[k] for k in data.files}

    public_gt = _load_gt("public")
    private_gt = _load_gt("private")

    def _allowed_tokens() -> set[str]:
        raw = os.environ.get("ARENA_TOKENS", "")
        return {t.strip() for t in raw.replace(",", "\n").split() if t.strip()}

    def _team(authorization: str | None) -> str:
        token = authorization.removeprefix("Bearer ").strip() if authorization else None
        if not token or token not in _allowed_tokens():
            raise HTTPException(status_code=401, detail="invalid or missing team token")
        return token

    def _score_vs(masks: dict, gt: dict) -> dict:
        ids = sorted(gt)
        preds = [masks.get(i, np.zeros_like(gt[i])) for i in ids]
        return score(preds, [gt[i] for i in ids])

    api = FastAPI(title="cell-arena leaderboard")

    class SubmitRequest(BaseModel):
        team: str
        masks: dict[str, str]

    @api.post("/submit")
    def submit(req: SubmitRequest, authorization: str | None = Header(default=None)):
        token = _team(authorization)
        masks = decode_submission(req.masks)
        pub = _score_vs(masks, public_gt)
        priv = _score_vs(masks, private_gt)

        prev = board_state.get(token)
        n_subs = (prev["n_subs"] + 1) if prev else 1
        # Keep each team's BEST-PUBLIC submission; its private score rides along
        # (the classic Kaggle reveal: overfit the public board and you may drop).
        if prev is None or pub["map"] > prev["public"]:
            board_state[token] = {
                "team": req.team,
                "public": pub["map"],
                "public_f1": pub["f1"],
                "private": priv["map"],
                "private_f1": priv["f1"],
                "n_subs": n_subs,
                "last": int(time.time()),
            }
        else:
            rec = dict(prev)
            rec.update(team=req.team, n_subs=n_subs, last=int(time.time()))
            board_state[token] = rec

        return {
            "public_score": pub["map"],
            "public_f1": pub["f1"],
            "your_best_public": board_state[token]["public"],
            "n_public_frames": len(public_gt),
        }

    @api.get("/board")
    def board(reveal: str | None = None):
        revealed = bool(reveal and reveal == os.environ.get("ARENA_ADMIN_TOKEN"))
        rows = []
        for rec in board_state.values():
            row = {
                "team": rec["team"],
                "public": round(rec["public"], 4),
                "public_f1": round(rec["public_f1"], 4),
                "n_subs": rec["n_subs"],
                "last": rec["last"],
            }
            if revealed:
                row["private"] = round(rec["private"], 4)
                row["private_f1"] = round(rec["private_f1"], 4)
            rows.append(row)
        rows.sort(key=lambda r: r.get("private" if revealed else "public", 0), reverse=True)
        return {
            "revealed": revealed,
            "rows": rows,
            "n_public_frames": len(public_gt),
            "n_private_frames": len(private_gt),
            "now": int(time.time()),
        }

    @api.get("/", response_class=HTMLResponse)
    def index():
        # Served from the volume so board tweaks are instant uploads, no redeploy.
        bundle_vol.reload()
        path = "/bundle/board.html" if os.path.isfile("/bundle/board.html") else "/static/index.html"
        with open(path) as fh:
            return fh.read()

    @api.get("/data/{path:path}")
    def data(path: str):
        from fastapi.responses import FileResponse

        if ".." in path or path.startswith("/"):
            raise HTTPException(status_code=400, detail="bad path")
        full = f"/bundle/{path}"
        if not os.path.isfile(full):
            bundle_vol.reload()
            if not os.path.isfile(full):
                raise HTTPException(status_code=404, detail=f"not found: {path}")
        return FileResponse(full)

    return api
