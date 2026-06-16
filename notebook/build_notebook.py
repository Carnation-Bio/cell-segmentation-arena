"""Build notebook/workshop.ipynb from source cells (maintainable > hand-edited JSON).

    python notebook/build_notebook.py

Two parts: a runnable **workstation** (segment -> see masks across all frames ->
score -> submit) and an **ideas** menu (markdown snippets, not auto-run, that you
fold into your pipeline). 'Run all' only runs the workstation, so it never errors.
"""

import json
from pathlib import Path

VERSION = "0.3.1"
WHEEL_URL = f"https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data/cell_arena-{VERSION}-py3-none-any.whl"
_VTUPLE = tuple(int(x) for x in VERSION.split("."))


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


cells = [
    md("""
# 🔬 Dense Cell Segmentation Arena

Segment **dense SHSY5Y cells** as accurately as you can and climb a **live
leaderboard**. Even the newest foundation models miss badly here — the lesson is
*data-efficient adaptation*.

**How this works:** the top is your **workstation** — run it top to bottom to
segment, *see your masks on every frame*, and get your score. Then improve your
approach with the **ideas** at the bottom (a menu, not a script), and submit when
you're happy.

> The models run on a remote GPU, so calls take a few seconds — not instant.
> You'll see a progress counter; just wait for it.
"""),
    md("# 🛠️ Workstation  ·  run these top to bottom"),
    code(f"""
# 1) Setup — installs/updates the toolkit if needed (fresh machine, Colab, anywhere).
#    First run ~20s. If it asks you to restart the kernel, do it once, then re-run.
import sys

def _toolkit_too_old():
    try:
        from importlib.metadata import version
        return tuple(int(x) for x in version("cell-arena").split(".")[:3]) < {_VTUPLE}
    except Exception:
        return True   # not installed

if _toolkit_too_old():
    %pip install -q --upgrade "{WHEEL_URL}"
    if "arena" in sys.modules:   # an old copy is already loaded -> a restart is required
        raise SystemExit("✅ Toolkit updated. Now RESTART THE KERNEL (Kernel menu -> Restart), then run this cell again.")

import arena

TEAM_TOKEN = ""  #@param {{type:"string"}}
TEAM_NAME  = ""  #@param {{type:"string"}}
arena.configure(token=TEAM_TOKEN)

# Check your token NOW, so a typo shows up here and not as a red error later.
def _check_token():
    import requests
    from arena.config import get_config
    if not TEAM_TOKEN.strip():
        return "❌ No team token yet — paste the one your host gave you into TEAM_TOKEN above, then re-run this cell."
    try:
        r = requests.get(get_config().backend_url + "/whoami",
                         headers={{"Authorization": "Bearer " + TEAM_TOKEN}}, timeout=20)
    except Exception:
        return "⚠️  Couldn't reach the backend to check your token (it may be warming up) — continuing."
    if r.status_code == 200:
        return "✅ Token works — team " + str(r.json().get("team", "?"))
    if r.status_code == 401:
        return "❌ Token not accepted — re-paste the exact token your host gave you, then re-run this cell."
    return "⚠️  Couldn't verify your token yet — continuing; Evaluate will catch it if it's wrong."

print(_check_token(), "| display name:", TEAM_NAME or "(set TEAM_NAME above!)")
"""),
    code("""
# 2) Load the data (downloads a few MB, ~10s): the test images you're scored on,
#    3 fully-labeled reference frames, and a labeled validation set (free to score against).
test_images = arena.load_public_test()             # {id: image}
references  = arena.load_reference_frames()         # [(id, image, label), ...]
val_images, val_labels = arena.load_local_val()
print(f"{len(test_images)} test · {len(references)} reference · {len(val_images)} validation frames")
"""),
    md("**What 'good' looks like** — a fully-labeled reference frame, then zoomed in (at full size, a ~1,000-cell field is just a blob):"),
    code("""
ref_id, ref_img, ref_lab = references[0]
arena.show(ref_img, ref_lab, title=f"reference ground truth — {ref_lab.max()} cells")
arena.zoom(ref_img, ref_lab, box=(150, 250, 320, 420))
"""),
    md("""
### 3) Your approach

`my_pipeline` takes an image and returns instance masks. It's the **one thing you
edit**. Start with the plain baseline; improve it with the ideas at the bottom.
"""),
    code("""
def my_pipeline(img):
    return arena.segment(img, model="cpsam_v2")
"""),
    md("""
### 4) Evaluate  ·  run this every time you change `my_pipeline`

Runs your approach on all validation frames, shows **your masks on every frame**,
and prints your score — **higher is better, 1.0 is perfect**. Free and instant
feedback; this is your loop.
"""),
    code("""
val_preds = arena.run_pipeline(my_pipeline, dict(enumerate(val_images)))
print(arena.score_local(val_preds, dict(enumerate(val_labels))))     # mAP@[.5:.95] + F1@0.5
arena.gallery(val_images, val_preds, val_labels)                     # every frame: yours vs should-be
"""),
    md("**Click through each frame, full size** — drag the slider to inspect your masks vs the truth, one frame at a time:"),
    code("""
arena.browse(val_images, val_preds, val_labels)
"""),
    md("""
### 5) Submit ▶  ·  your call

When the gallery + score look good, run this to go on the board. It runs
`my_pipeline` on all 60 test frames (~1 min). Half are scored live on the public
board; the other half are held back for the **private** board that decides the
winner — so trust your validation score, don't just chase the public one. Re-run
any time; your best counts.
"""),
    code("""
test_preds = arena.run_pipeline(my_pipeline, test_images)
arena.submit(test_preds, team=TEAM_NAME or "anon")
"""),
    md("""
---
# 💡 Ideas to improve your score

A **menu, not a script** — nothing below runs on its own, and nothing here is
ranked. Each idea is a lever and *what it does*; which ones help on these cells is
exactly what you're here to find out. To try one: **change `my_pipeline`** (or
paste a snippet into a new cell), then re-run **Evaluate**. Score on validation
first — it's free — and let that number, not your intuition, decide.
"""),
    md("""
### Tune the model's knobs

`segment(..., params={...})` passes straight to the model. Knobs worth a sweep:

- `flow_threshold` (default 0.4) — how strict the model is about what counts as a cell.
- `diameter` (default: auto) — the model guesses cell size and resizes to match; if the guess is off for these cells, set it yourself.
- `cellprob_threshold` (default 0.0) — the confidence cutoff; lower keeps more, higher keeps fewer.
- `augment=True` — average over flipped/rotated views (slower).
- *Advanced:* `niter` (e.g. 250–1000) and `resample=True` change how pixels are grouped into cells.

```python
def my_pipeline(img):
    return arena.segment(img, model="cpsam_v2",
                         params={"flow_threshold": 0.6, "diameter": 8, "augment": True})
```

Sweep any single knob on validation to see what it does:

```python
for d in [6, 8, 12, 17, 30]:
    preds = arena.run_pipeline(lambda im: arena.segment(im, model="cpsam_v2", params={"diameter": d}),
                               dict(enumerate(val_images)))
    print(d, arena.score_local(preds, dict(enumerate(val_labels))))
```
"""),
    md("""
### Fine-tune on your labels

Keep training the model on the labeled frames you have, then segment with the
adapter it returns. Runs on a GPU (a few minutes). `n_epochs` and
`learning_rate` are yours to sweep:

```python
labels = [(img, lab) for (_id, img, lab) in references] + list(zip(val_images, val_labels))
adapter = arena.finetune(labels, base_model="cpsam_v2", n_epochs=200, learning_rate=1e-5)

def my_pipeline(img):
    return arena.segment(img, model=adapter)
```

Sweep the training knobs like any other — score each adapter on validation:

```python
for ep in [50, 200, 500]:
    adapter = arena.finetune(labels, base_model="cpsam_v2", n_epochs=ep)
    preds = arena.run_pipeline(lambda im: arena.segment(im, model=adapter), dict(enumerate(val_images)))
    print(ep, arena.score_local(preds, dict(enumerate(val_labels))))
```
"""),
    md("""
### Try a different model

```python
def my_pipeline(img):
    return arena.segment(img, model="cpsam")     # or "cyto3"
```

`sam3` is a text-promptable foundation model. Fair warning: on these cells it's
weak out of the box (it's a general model) — lower its confidence to find more:

```python
arena.segment(img, model="sam3", text="cell", params={"confidence": 0.1})
```
"""),
    md("""
### Refine the masks

`refine` steps post-process the model's output: `watershed_split[:N]` (cut a blob
into pieces), `min_size:N` (drop fragments under N pixels), `remove_edge` (drop
cells touching the border), `fill_holes`. Compose any; measure on validation.

```python
def my_pipeline(img):
    return arena.segment(img, model="cpsam_v2", refine=["watershed_split:10", "min_size:30"])
```
"""),
    md("""
### Bring your own algorithm

`segment` is plain array-in / array-out, and a `process`/`refine` step can be your
own function. Compose anything:

```python
from skimage import filters
def my_step(img):
    return filters.unsharp_mask(img, radius=2, amount=1.0)

def my_pipeline(img):
    return arena.segment(img, model="cpsam_v2", process=["clahe", my_step],
                         params={"flow_threshold": 0.6})
```
"""),
    md("""
### Advanced: draw boxes, or have SAM-3 fill the gaps

SAM-3's strength is *boxes*, not text. Draw boxes on a frame and feed them in
(needs a widget-enabled kernel):

```python
train_images = arena.load_train_frames()
tid, timg = next(iter(train_images.items()))
from jupyter_bbox_widget import BBoxWidget
import base64, io
from PIL import Image
buf = io.BytesIO(); Image.fromarray(timg).save(buf, format="PNG")
widget = BBoxWidget(image="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
widget   # draw boxes; then:
boxes = [[b["x"], b["y"], b["x"]+b["width"], b["y"]+b["height"]] for b in widget.bboxes]
arena.show(timg, arena.segment(timg, model="sam3", boxes=boxes))
```

A hint to explore: run `cpsam_v2` first, find where it left gaps, auto-drop boxes
there, ask SAM-3 to fill them in, and merge the two results. You write the merge —
that's the fun part.
"""),
    md("""
### Let your agent cook 🍳

Point your Claude Code / Codex agent at this repo (it reads `AGENTS.md`) and let it
run the `tweak → evaluate → submit` loop for you.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "workshop.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells, {sum(c['cell_type']=='code' for c in cells)} runnable)")
