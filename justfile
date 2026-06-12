# Dense Cell Segmentation Arena — operator commands.
# `just` with no args lists recipes. Deploys use the venv modal (Python >=3.10).

MODAL := ".venv/bin/modal"
ENV := "workshop"
HF := "ARENA_HF_SECRET=carnation-hf ARENA_HF_SECRET_ENV=main"

default:
    @just --list

# one-time: venv + deps
setup:
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -e ".[dev]" modal remotezip tqdm

# build the LIVECell dense-SHSY5Y split into data/build/
build-data:
    .venv/bin/python -m data_prep.build_dataset --out data/build

# push hidden GT + participant bundle to Modal volumes
host-data:
    {{MODAL}} volume create arena-gt -e {{ENV}} || true
    {{MODAL}} volume create arena-bundle -e {{ENV}} || true
    {{MODAL}} volume put -e {{ENV}} arena-gt data/build/server/public_gt.npz public_gt.npz --force
    {{MODAL}} volume put -e {{ENV}} arena-gt data/build/server/private_gt.npz private_gt.npz --force
    cd data/build/bundle && ../../../{{MODAL}} volume put -e {{ENV}} arena-bundle . / --force

# host the built wheel + notebook for download (run after `just notebook` / a version bump)
host-toolkit:
    {{MODAL}} volume put -e {{ENV}} arena-bundle dist/cell_arena-0.2.0-py3-none-any.whl cell_arena-0.2.0-py3-none-any.whl --force
    {{MODAL}} volume put -e {{ENV}} arena-bundle notebook/workshop.ipynb workshop.ipynb --force

# generate team tokens and push the secret
tokens:
    .venv/bin/python tokens/generate_tokens.py --push

# deploy both apps (idle-scaled)
deploy:
    {{HF}} {{MODAL}} deploy -e {{ENV}} backend/app.py
    {{MODAL}} deploy -e {{ENV}} backend/leaderboard.py

# deploy with a warm pool for the live session
deploy-warm:
    {{HF}} ARENA_WARM_CELLPOSE=2 ARENA_WARM_SAM3=2 ARENA_WARM_WEB=1 {{MODAL}} deploy -e {{ENV}} backend/app.py
    ARENA_WARM_BOARD=1 {{MODAL}} deploy -e {{ENV}} backend/leaderboard.py

# clear the leaderboard between runs
reset-board:
    {{MODAL}} run -e {{ENV}} backend/admin.py::reset

# print current board state
board:
    {{MODAL}} run -e {{ENV}} backend/admin.py::dump

# unit tests + smoke tests
test:
    .venv/bin/python -m pytest tests/ -q

smoke:
    .venv/bin/python backend/leaderboard_smoke.py

# rebuild the notebook from its source cells
notebook:
    .venv/bin/python notebook/build_notebook.py

# run the notebook locally: install jupyter (+ widgets for the bbox annotator),
# register the kernel, then launch. Server + kernel share one env so widgets render.
notebook-local:
    uv pip install --python .venv/bin/python jupyterlab ipykernel ipywidgets jupyter-bbox-widget
    .venv/bin/python -m ipykernel install --user --name cell-arena --display-name "cell-arena (.venv 3.11)"
    .venv/bin/jupyter lab notebook/workshop.ipynb
