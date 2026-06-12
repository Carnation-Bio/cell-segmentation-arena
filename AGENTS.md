# Driving this repo with an AI coding agent

You (Claude Code / Codex) are the interface for a participant in a 2-hour
segmentation competition. Your human teammate wants to climb a leaderboard.
This file is your orientation.

## The goal

Segment **dense SHSY5Y cells** as accurately as possible, measured by instance
**mAP@[0.5:0.95]** (F1@0.5 shown as a friendly headline). Higher is better.
Off-the-shelf models under-segment badly here — they merge touching cells. Your
job is to close that gap with *data-efficient adaptation*, not brute force.

## The loop you should run

```python
from arena import segment, score_local, submit, load_local_val

val_imgs, val_labels = load_local_val()
pred = [segment(im, model="sam3", text="cell") for im in val_imgs]
print(score_local(pred, val_labels))      # iterate against THIS, locally + free
# ...change one thing, re-score. When it improves, then:
print(submit(test_pred, team=TEAM))       # returns your public score
```

- **Score locally first.** `score_local()` runs on the included validation set
  (labels provided) so you can iterate fast without spamming the leaderboard.
- **`submit()` returns the public score** — so you can close the loop yourself:
  tweak → `score_local` → if better, `submit` → read score → repeat.

## Levers that actually move the score (in rough order of payoff)

1. **Watershed-split refinement** — the single biggest lever on dense
   under-segmentation. `segment(..., refine=["watershed_split", "min_size:30"])`.
2. **Pre-processing** — `process=["gaussian:2", "clahe"]` before the model.
3. **Box / point prompts** — annotate with the bbox widget, pass `boxes=`.
4. **Fine-tune** — `finetune(labels, base_model="cpsam_v2")` returns an
   `adapter_id`; then `segment(model=adapter_id)`. ~5-20 labeled frames,
   finishes in minutes, and should beat every zero-shot baseline.
5. **Bring your own algorithm** — it's just Python; compose freely.

## Rules of the road

- **Count ≠ quality.** Do not optimize for "found ~N cells." A model can match
  the cell count with garbage masks. Always judge by `score_local()`.
- **Visualization is local and free.** `show()`, `compare()`, `zoom()` render
  from mask arrays; they never call the backend. A 520×704 field with ~1,400
  cells is a blob at full size — **always `zoom()`** to judge mask quality.
- **The backend is shared.** Don't busy-loop `submit()`; score locally, submit
  when you have a real improvement.
- **You never see the test labels.** Only the validation set has labels. The
  leaderboard scores you against hidden ground truth.

## Setup (run this first)

The models run on a remote GPU backend, so no GPU or heavy deps are needed here
— just the `arena` package and the team token. Backend/data URLs are baked in.

```bash
uv venv --python 3.11 && uv pip install -e .   # any Python >=3.10; uv guarantees it
export ARENA_TOKEN=<your team token>           # ask your human for it
```

Then run everything with `.venv/bin/python`. Quick check you're wired up:

```bash
.venv/bin/python -c "import arena, os; arena.configure(token=os.environ['ARENA_TOKEN']); \
print('ok', len(arena.load_public_test()), 'test frames')"
```

No `uv`? Any Python >=3.10 works: `python3 -m venv .venv && .venv/bin/pip install -e .`
