# Operator runbook

Everything you do as the host. The infra is two Modal apps (in the `workshop`
environment) plus three Modal volumes; participants only ever touch the notebook.

## One-time setup

```bash
# 0. deps for the build/deploy tooling
uv venv --python 3.11 .venv && uv pip install -e ".[dev]" modal remotezip

# 1. build the dataset (pulls dense SHSY5Y from LIVECell, makes the seed-1234 split).
#    data_prep/ is operator-only and gitignored (not in the public repo): it
#    regenerates the hidden GT, so it must stay off the clone path.
python -m data_prep.build_dataset --out data/build

# 2. push the hidden GT + the participant bundle to Modal volumes
modal volume create arena-gt -e workshop ; modal volume create arena-bundle -e workshop
modal volume put -e workshop arena-gt data/build/server/public_gt.npz  public_gt.npz  --force
modal volume put -e workshop arena-gt data/build/server/private_gt.npz private_gt.npz --force
cd data/build/bundle && modal volume put -e workshop arena-bundle . / --force ; cd -

# 3. team tokens -> Modal secret (writes tokens/tokens.txt, your private handout)
python tokens/generate_tokens.py --push

# 4. SAM-3 weights are gated. The owner of the HF token must request access at
#    https://huggingface.co/facebook/sam3 and be approved BEFORE the event.
#    The token lives in the `carnation-hf` secret (main env); the deploy reads it
#    cross-env. Until approved, model="sam3" 503s and you run with cpsam_v2.
```

## Deploy

```bash
# secret name/env kept out of source so it reads the gated HF token cross-env
export ARENA_HF_SECRET=carnation-hf ARENA_HF_SECRET_ENV=main
modal deploy -e workshop backend/app.py          # segment + finetune
modal deploy -e workshop backend/leaderboard.py  # board + submit + data

# REQUIRED after any notebook or toolkit change — otherwise participants download a
# stale notebook. Rebuilds workshop.ipynb from source + pushes it (and the wheel):
just notebook
just host-toolkit
```

Always deploy with a Python >=3.10 `modal` (use `.venv/bin/modal`) — see the note
in `backend/app.py` about FastAPI annotations.

## During the session — keep a warm pool

Cold start = container spin-up + model load (SAM-3 ~30-90s). Warm a few
containers for the 2-hour window, then scale back to 0 after:

```bash
# warm: a couple of each GPU model + the web/board front doors
ARENA_HF_SECRET=carnation-hf ARENA_HF_SECRET_ENV=main \
  ARENA_WARM_CELLPOSE=2 ARENA_WARM_SAM3=2 ARENA_WARM_WEB=1 \
  modal deploy -e workshop backend/app.py
ARENA_WARM_BOARD=1 modal deploy -e workshop backend/leaderboard.py

# after the event: redeploy with the vars unset (everything scales to 0 idle)
```

Request a Modal concurrent-GPU limit raise before the event if you expect ~10
teams hammering SAM-3 at once.

## The live board

Open the leaderboard URL on the projector:
`https://<...>-cell-arena-leaderboard-web.modal.run/` — it auto-refreshes every 6s.

## Reveal the private board (at the end)

The admin token is the last line of `tokens/tokens.txt`. Either:

- Open `…/?reveal=<ADMIN_TOKEN>` in the browser (re-ranks by private score), or
- `curl ".../board?reveal=<ADMIN_TOKEN>"` for the JSON.

## Reset the board between runs

```bash
modal run backend/admin.py::reset   # clears the modal.Dict of all submissions
```

## Health checks

```bash
curl .../health                       # models backend
ARENA_BACKEND_URL=... ARENA_TOKEN=<a team token> python backend/smoke_test.py
python backend/leaderboard_smoke.py   # submits GT (1.0) + empty (0.0), checks reveal
```
