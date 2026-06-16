# Dense Cell Segmentation Arena

A 2-hour, hands-on **competition workshop**. Teams race on a Kaggle-style
leaderboard to best segment **dense SHSY5Y cells** (label-free phase contrast,
from the public [LIVECell](https://github.com/sartorius-research/LIVECell)
dataset) — a task where even the newest off-the-shelf models fail.

The one idea being taught: **data-efficient adaptation** — adapt a foundation
model with a *handful of labels* instead of training from scratch.

> Tools here are all open source / generic: a Jupyter notebook, open models
> (SAM 3, Cellpose, Omnipose), scikit-image, and an AI coding agent as the
> interface. A GPU backend serves the big models; everything else runs locally.

## Two ways to play

- **Clear (everyone, zero code):** open the notebook, **Run All**, land on the
  leaderboard, beat the out-of-the-box baseline.
- **Reach (power users):** annotate a few frames, chain scikit-image processing,
  fine-tune an adapter, bring your own algorithm — and top the *private* board.

## Run it

**Recommended setup (works everywhere) — clone the repo and make an environment:**

```bash
git clone https://github.com/Carnation-Bio/cell-segmentation-arena && cd cell-segmentation-arena
uv venv --python 3.11 && uv pip install -e .     # or any Python 3.9+: python -m venv .venv && .venv/bin/pip install -e .
export ARENA_TOKEN=wksp_teamNN_xxxx              # your team token
```

Then pick one:

- **Drive it with Claude Code / Codex (recommended):** point your agent at the repo
  and say *"Read AGENTS.md, get me set up, and get my first leaderboard score, then
  let's improve it."* It reads [`AGENTS.md`](AGENTS.md) and takes it from there.
- **Run the notebook from the repo:** `uv pip install jupyterlab ipywidgets`, then
  `uv run jupyter lab notebook/workshop.ipynb`, paste your token, **Run all**.

**Just want to open the notebook?**
[Download it](https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data/workshop.ipynb)
and **Run all** in a Jupyter that has `pip` (Cell 1 installs the toolkit, stock
Python 3.9+ is fine). If your Jupyter has no pip — e.g. installed via `uv tool` —
use the repo setup above instead.

## Repo layout

| Path | What |
|------|------|
| `notebook/workshop.ipynb` | The participant notebook (golden path + the power-user ladder). |
| `arena/` | Participant toolkit: `segment`, `finetune`, `score_local`, `submit`, viz, processing. |
| `backend/` | Modal GPU apps (SAM 3 + Cellpose + Omnipose inference, LoRA fine-tune, leaderboard). |
| `board/` | Static live leaderboard page. |
| `tokens/` | Team-token generation. |

## Docs

- [`docs/participant-guide.md`](docs/participant-guide.md) — the walkthrough handed to teams.
- [`docs/operator-runbook.md`](docs/operator-runbook.md) — deploy, warm pool, reveal the private board.
- [`AGENTS.md`](AGENTS.md) — how an AI coding agent should drive this repo.

MIT licensed. Not affiliated with any commercial product.
