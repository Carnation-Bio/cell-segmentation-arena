# Participant guide — start here

You'll get a **team token** at the event. You need nothing installed — the
notebook pulls in the toolkit itself.

## Run it on any computer (Colab)

1. Open the notebook: click the **Open in Colab** badge in the [README](../README.md),
   or [download it](https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data/workshop.ipynb)
   and in Colab do *File → Upload notebook*.
2. Paste your token (and a team name) into the first cell.
3. **Runtime → Run all.** You're on the leaderboard.

## Prefer your own Jupyter?

[Download the notebook](https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data/workshop.ipynb),
open it (`jupyter lab workshop.ipynb`), **Run all**. The first cell installs the
toolkit into your kernel (stock Python 3.9+). No clone, no `git`, no `uv`.

## The loop

The top of the notebook is your **workstation**: edit `my_pipeline`, run
**Evaluate** to see your masks on every frame + your score, then **Submit** when
you're happy. The bottom is a menu of **ideas** to improve your score. Score on
validation first (it's free); the private board decides the winner.

Driving with an agent? Point Claude Code / Codex at [`AGENTS.md`](../AGENTS.md).
