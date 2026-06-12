"""Build notebook/workshop.ipynb from source cells (maintainable > hand-edited JSON).

    python notebook/build_notebook.py

Two parts: a runnable **workstation** (segment -> see masks across all frames ->
score -> submit) and an **ideas** menu (markdown snippets, not auto-run, that you
fold into your pipeline). 'Run all' only runs the workstation, so it never errors.
"""

import json
from pathlib import Path

REPO = "owenauch/cell-segmentation-arena"
VERSION = "0.3.1"
WHEEL_URL = f"https://carnation-workshop--cell-arena-leaderboard-web.modal.run/data/cell_arena-{VERSION}-py3-none-any.whl"
_VTUPLE = tuple(int(x) for x in VERSION.split("."))


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


cells = [
    md(f"""
# 🔬 Dense Cell Segmentation Arena

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebook/workshop.ipynb)

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
print("ready:", arena.__version__, "| team:", TEAM_NAME or "(set your team name above!)")
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
and prints your score. Free and instant feedback — this is your loop.
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
`my_pipeline` on all 60 hidden test frames (~1 min). The public board updates
live, but the **private** board decides the winner — trust your validation score,
don't just chase the public one. Re-run any time; your best counts.
"""),
    code("""
test_preds = arena.run_pipeline(my_pipeline, test_images)
arena.submit(test_preds, team=TEAM_NAME or "anon")
"""),
    md("""
---
# 💡 Ideas to improve your score

A **menu, not a script** — nothing below runs on its own. Each idea is a direction
and the shape of the code. To use one: **change `my_pipeline` in cell 3** (or paste
a snippet into a new cell), then re-run **Evaluate**. Submit when it's better.
Score on validation first; it's free.
"""),
    md("""
### Tune the model's knobs  ·  *a real win here*

`flow_threshold` (default 0.4) controls how many cells the model keeps — raise it
to recover faint, crowded cells it was dropping. Edit `my_pipeline`:

```python
def my_pipeline(img):
    return arena.segment(img, model="cpsam_v2", params={"flow_threshold": 0.6})
```

Sweep it on validation to find the best value:

```python
for ft in [0.4, 0.6, 0.8]:
    preds = arena.run_pipeline(lambda im: arena.segment(im, model="cpsam_v2", params={"flow_threshold": ft}),
                               dict(enumerate(val_images)))
    print(ft, arena.score_local(preds, dict(enumerate(val_labels))))
```
"""),
    md("""
### Fine-tune on a few labels  ·  *the core lesson*

Adapt the model with the labels you already have. Runs on a GPU (give it a few
minutes), returns an adapter you segment with:

```python
labels = [(img, lab) for (_id, img, lab) in references] + list(zip(val_images, val_labels))
adapter = arena.finetune(labels, base_model="cpsam_v2", n_epochs=100)

def my_pipeline(img):
    return arena.segment(img, model=adapter)
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

`refine` steps post-process: `watershed_split` (cut merged blobs apart),
`min_size:N`, `remove_edge`, `fill_holes`. `watershed_split` helps when a model
*merges* cells — measure it, it can over-split a sharp model:

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
### Annotate your own frames (box prompts)

Draw boxes on a frame and feed them to SAM-3 (its real strength). Best in Colab,
where the widget renders out of the box:

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
