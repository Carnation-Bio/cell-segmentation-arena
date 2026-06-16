# Driving this repo with an AI coding agent

You (Claude Code / Codex) are the interface for a participant in a 2-hour
segmentation competition. Your human teammate wants to climb a leaderboard.
This file is your orientation.

## Start here

When your teammate first talks to you (often just "what do I do?"), get them on
the board fast and show your work:

1. Make sure the environment is set up — they cloned the repo and ran
   `uv pip install -e .`. If `import arena` fails, walk them through it.
2. Get their **team token** (looks like `wksp_team07_xxxx`) and run
   `arena.configure(token=...)`.
3. Run the baseline once and **show them the result**: load the data,
   `segment(model="cpsam_v2")` over the validation set, `score_local`, and a
   `gallery()` / `compare()` so they see the masks, not just a number.
4. `submit` it so they're on the leaderboard, then explain the loop — change one
   thing, re-score on validation, submit when it's better — and start iterating.

## Keep your teammate in the loop

You're the interface for a scientist who wants to *understand* this, not just
watch a number go up. The single most important habit: **stop frequently and
show them what you're doing. Do not just keep going.**

- **Stop and show, every cycle.** Try *one* thing, then **pause, render the
  result, and show the user before you do anything else.** Do not run several
  changes back-to-back and report only at the end — that is the most common way
  to get this wrong. Rule of thumb: never go more than a step or two without
  putting a picture in front of them and saying what you see. If you realize
  you've been heads-down for a while with nothing shown, stop right now and show
  them where things stand.
- **Make it visual, and make sure it actually appears.** A picture of the masks
  teaches far more than a number. `show(image, masks)`, `compare(image, yours, truth)`,
  `zoom()` into a dense patch, and `gallery()` across frames all render the masks.
  When you run them from a terminal (how you're driving), they don't just call a
  no-op `plt.show()` — they **save a PNG and open it in the user's image viewer**,
  and print the path. Use them constantly, and after each one tell the user to
  look at the image that just opened.
- **Narrate in plain language.** Before a change, say in a sentence what you're
  trying and why; after it, say what the result means.

## The goal

Segment **dense SHSY5Y cells** as accurately as possible, measured by instance
**mAP@[0.5:0.95]** (F1@0.5 shown as a friendly headline). Higher is better.
Off-the-shelf models leave a big gap here. Your job is to close it — and to
**measure every change on the validation set**, because intuition about what
helps is often wrong on this data.

## The loop you should run

```python
from arena import segment, score_local, submit, load_local_val

val_imgs, val_labels = load_local_val()
pred = [segment(im, model="cpsam_v2") for im in val_imgs]
print(score_local(pred, val_labels))      # iterate against THIS, locally + free
# ...change one thing, re-score. When it improves, then:
print(submit(test_pred, team=TEAM))       # returns your public score
```

- **Score locally first.** `score_local()` runs on the included validation set
  (labels provided) so you can iterate fast without spamming the leaderboard.
- **`submit()` returns the public score** — so you can close the loop yourself:
  tweak → `score_local` → if better, `submit` → read score → repeat.

## Levers available — no ranking; measure each

These all exist. Which ones help on *these* cells is for you to discover with
`score_local`, not to assume — so don't trust any ordering (including this list).
Change one thing, re-score, keep what wins.

- **Model params** — `segment(..., params={...})` passes to the model:
  `flow_threshold`, `diameter`, `cellprob_threshold`, `augment`, `niter`,
  `resample`. Sweep them on validation.
- **Model choice** — `cpsam_v2` (default), `cpsam`, `cyto3`, `sam3` (text/box-promptable).
- **Refine steps** — `refine=["watershed_split", "min_size:30", "remove_edge", "fill_holes"]`.
- **Pre-processing** — `process=[...]` (skimage step names or your own function) before the model.
- **Fine-tune** — `finetune(labels, base_model="cpsam_v2", n_epochs=, learning_rate=)`
  returns an `adapter_id` you `segment(model=adapter_id)`. Sweep its knobs too. Use
  *all* the labeled frames you have for `labels` — the 3 references **plus** the 12
  validation frames (15 total), not just the references; 3 is too few to teach much.
- **Bring your own algorithm** — it's just Python; compose freely.

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
uv venv --python 3.11 && uv pip install -e .   # any Python >=3.9; uv guarantees it
export ARENA_TOKEN=<your team token>           # ask your human for it
```

Then run everything with `.venv/bin/python`. Quick check you're wired up:

```bash
.venv/bin/python -c "import arena, os; arena.configure(token=os.environ['ARENA_TOKEN']); \
print('ok', len(arena.load_public_test()), 'test frames')"
```

No `uv`? Any Python >=3.9 works: `python3 -m venv .venv && .venv/bin/pip install -e .`
