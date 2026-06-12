"""Smoke-test the deployed leaderboard end to end (no GPU backend needed).

Submits the hidden GT as a 'perfect' prediction (expect public ~1.0), an empty
prediction (expect 0), and a public-only prediction (expect public ~1.0,
private ~0), then checks the board + reveal + data route.

    ARENA_LEADERBOARD_URL=https://...modal.run python backend/leaderboard_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import requests

from arena.wire import encode_submission

LB = os.environ.get(
    "ARENA_LEADERBOARD_URL", "https://carnation-workshop--cell-arena-leaderboard-web.modal.run"
).rstrip("/")
ADMIN = os.environ.get("ARENA_ADMIN_TOKEN", "admin-dev-xyz")

server = Path("data/build/server")
public = dict(np.load(server / "public_gt.npz"))
private = dict(np.load(server / "private_gt.npz"))
all_gt = {**public, **private}
print(f"leaderboard = {LB}\npublic={len(public)} private={len(private)} frames\n")


def submit(masks, team, token):
    r = requests.post(
        f"{LB}/submit",
        json={"team": team, "masks": encode_submission(masks)},
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


# 1) perfect (submit GT) on token aaa
perfect = submit(all_gt, "perfect-bot", "devtoken-aaa")
print(f"perfect      -> public={perfect['public_score']:.4f} f1={perfect['public_f1']:.4f}  (expect ~1.0)")

# 2) empty on token bbb
empty = {k: np.zeros_like(v) for k, v in all_gt.items()}
res_empty = submit(empty, "empty-bot", "devtoken-bbb")
print(f"empty        -> public={res_empty['public_score']:.4f}  (expect 0.0)")

# 3) public-only (private frames omitted) re-using aaa — should NOT beat perfect publicly
pub_only = submit(public, "perfect-bot", "devtoken-aaa")
print(f"public-only  -> public={pub_only['public_score']:.4f}  best_public={pub_only['your_best_public']:.4f}")

# 4) auth rejection
bad = requests.post(f"{LB}/submit", json={"team": "x", "masks": {}}, headers={"Authorization": "Bearer nope"})
print(f"bad token    -> HTTP {bad.status_code}  (expect 401)")

# 5) public board (no reveal) — private must be hidden
board = requests.get(f"{LB}/board").json()
print(f"\nboard (public): revealed={board['revealed']}, {len(board['rows'])} teams")
for r in board["rows"]:
    assert "private" not in r, "PRIVATE LEAKED on public board!"
    print(f"   {r['team']:14s} public={r['public']:.4f} subs={r['n_subs']}")

# 6) reveal with admin token
rev = requests.get(f"{LB}/board", params={"reveal": ADMIN}).json()
print(f"\nboard (revealed={rev['revealed']}):")
for r in rev["rows"]:
    print(f"   {r['team']:14s} private={r.get('private')} public={r['public']}")

# 7) data route serves the bundle manifest
man = requests.get(f"{LB}/data/manifest.json")
print(f"\n/data/manifest.json -> HTTP {man.status_code}, sets={list(man.json()['sets']) if man.ok else '-'}")

# 8) board page renders
page = requests.get(f"{LB}/")
print(f"/ (board page) -> HTTP {page.status_code}, {len(page.text)} bytes, has table={'<table' in page.text}")
