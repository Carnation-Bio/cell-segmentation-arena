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
- **Reach (cookers):** annotate a few frames, chain scikit-image processing,
  fine-tune an adapter, bring your own algorithm — and top the *private* board.

## Run it

The models run on a remote GPU backend, so you need **no GPU and no install** —
the notebook's first cell pulls in the tiny toolkit itself. You only need your
team token.

**Colab — works on any computer (recommended):**
[Download the notebook](https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data/workshop.ipynb),
then in Colab do *File → Upload notebook*, paste your token, **Run all**. Nothing
to install locally.

**Local — if you already have Jupyter:**
[Download the notebook](https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data/workshop.ipynb),
open it in your Jupyter (`jupyter lab workshop.ipynb`), **Run all**. Cell 1
installs the toolkit into your kernel (stock Python 3.9+ is fine). No clone, no
`git`, no `uv`.

**Cookers / agents — work from the repo:**
Clone it and point your Claude Code / Codex agent at [`AGENTS.md`](AGENTS.md):

```bash
git clone https://github.com/Carnation-Bio/cell-segmentation-arena && cd cell-segmentation-arena
uv venv --python 3.11 && uv pip install -e .     # or any Python >=3.9 + pip install -e .
export ARENA_TOKEN=wksp_teamNN_xxxx
```

(`uv`/`just` are conveniences for this path only — participants don't need them.)

## Repo layout

| Path | What |
|------|------|
| `notebook/workshop.ipynb` | The participant notebook (golden path + the cooker ladder). |
| `arena/` | Participant toolkit: `segment`, `finetune`, `score_local`, `submit`, viz, processing. |
| `backend/` | Modal GPU apps (SAM 3 + Cellpose + Omnipose inference, LoRA fine-tune, leaderboard). |
| `board/` | Static live leaderboard page. |
| `tokens/` | Team-token generation. |

## Docs

- [`docs/participant-guide.md`](docs/participant-guide.md) — the walkthrough handed to teams.
- [`docs/operator-runbook.md`](docs/operator-runbook.md) — deploy, warm pool, reveal the private board.
- [`AGENTS.md`](AGENTS.md) — how an AI coding agent should drive this repo.

MIT licensed. Not affiliated with any commercial product.
