"""Local visualization — render masks from arrays, never touches the backend.

A 520x704 field with ~1,400 cells is a featureless blob at full size, so
``zoom`` is the tool you actually judge quality with: crop a small box and look
at whether touching cells are split.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

Box = tuple[int, int, int, int]  # (row0, col0, row1, col1)


def _colorize(masks: np.ndarray) -> np.ndarray:
    """Map a label image to a stable random RGBA overlay (background transparent)."""
    masks = np.asarray(masks)
    n = int(masks.max())
    rng = np.random.default_rng(12345)
    colors = rng.random((n + 1, 4))
    colors[:, 3] = 1.0
    colors[0] = (0, 0, 0, 0)  # background transparent
    return colors[masks]


def _draw(ax, image, masks, alpha: float = 0.45, boundaries: bool = True) -> None:
    ax.imshow(image, cmap="gray")
    masks = None if masks is None else np.asarray(masks)
    if masks is not None and masks.max() > 0:
        ax.imshow(_colorize(masks), alpha=alpha)
        if boundaries:
            edges = find_boundaries(masks, mode="outer")
            overlay = np.zeros((*masks.shape, 4))
            overlay[edges] = (1, 1, 1, 0.9)
            ax.imshow(overlay)
    ax.set_xticks([])
    ax.set_yticks([])


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return [x[k] for k in sorted(x)]
    return list(x)


def _count(x):
    return int(np.asarray(x).max())


def gallery(images, masks=None, labels=None, ncols: int = 4, max_frames: int = 12) -> None:
    """Grid of many frames with their masks overlaid — see the model's output
    across *all* your data at a glance.

    Each tile is titled with your cell count, and (if you pass ``labels``) the
    count it *should* have. ``images``/``masks``/``labels`` can be lists or
    ``{id: array}`` dicts (e.g. straight from ``run_pipeline``).
    """
    imgs = _as_list(images)
    ms = _as_list(masks)
    ls = _as_list(labels)
    n = min(len(imgs), max_frames)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 3.1))
    axes = np.atleast_1d(axes).reshape(-1)
    for k, ax in enumerate(axes):
        if k < n:
            _draw(ax, imgs[k], None if ms is None else ms[k], alpha=0.5, boundaries=False)
            if ms is not None:
                title = f"{_count(ms[k])} cells" if ls is None else f"yours {_count(ms[k])} / should be {_count(ls[k])}"
                ax.set_title(title, fontsize=9)
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def browse(images, masks=None, labels=None, panel_size: float = 8.0) -> None:
    """Click through every frame, big, with a zoom — your masks vs the truth.

    A ``frame`` slider steps through frames; a ``zoom`` slider blows up the centre
    so you can actually see individual cells in a dense field. Each panel is large
    (``panel_size`` inches). Falls back to ``gallery`` if widgets aren't available.
    """
    imgs = _as_list(images)
    ms = _as_list(masks)
    ls = _as_list(labels)
    n = len(imgs)

    def render(frame: int = 0, zoom: int = 1) -> None:
        img = np.asarray(imgs[frame])
        h, w = img.shape[:2]
        if zoom > 1:
            rh, rw = h // (2 * zoom), w // (2 * zoom)
            r, c = h // 2, w // 2
            crop = (slice(max(0, r - rh), r + rh), slice(max(0, c - rw), c + rw))
        else:
            crop = (slice(None), slice(None))

        panels = []
        if ms is not None:
            panels.append((f"yours — {_count(ms[frame])} cells", np.asarray(ms[frame])))
        if ls is not None:
            panels.append((f"should have — {_count(ls[frame])} cells", np.asarray(ls[frame])))
        if not panels:
            panels = [(f"frame {frame + 1} / {n}", None)]

        fig, axes = plt.subplots(1, len(panels), figsize=(panel_size * len(panels), panel_size))
        axes = np.atleast_1d(axes)
        for ax, (title, m) in zip(axes, panels):
            _draw(ax, img[crop], None if m is None else m[crop])
            ax.set_title(f"{title}    (frame {frame + 1}/{n}, zoom {zoom}×)", fontsize=11)
        plt.tight_layout()
        plt.show()

    try:
        import ipywidgets as widgets
    except ImportError:
        gallery(images, masks, labels)  # static fallback
        return
    widgets.interact(
        render,
        frame=widgets.IntSlider(min=0, max=max(n - 1, 0), step=1, value=0, description="frame"),
        zoom=widgets.SelectionSlider(options=[1, 2, 4, 8], value=1, description="zoom"),
    )


def show(image, masks=None, title: str | None = None, figsize=(9, 7)) -> None:
    """Show an image with its instance masks overlaid."""
    _, ax = plt.subplots(figsize=figsize)
    _draw(ax, image, masks)
    n = 0 if masks is None else int(np.asarray(masks).max())
    ax.set_title(title or (f"{n} instances" if masks is not None else "image"))
    plt.tight_layout()
    plt.show()


def compare(image, mine, gt, figsize=(16, 6)) -> None:
    """Side by side: image | your masks | ground truth."""
    _, axes = plt.subplots(1, 3, figsize=figsize)
    _draw(axes[0], image, None)
    axes[0].set_title("image")
    _draw(axes[1], image, mine)
    axes[1].set_title(f"yours — {int(np.asarray(mine).max())} cells")
    _draw(axes[2], image, gt)
    axes[2].set_title(f"ground truth — {int(np.asarray(gt).max())} cells")
    plt.tight_layout()
    plt.show()


def zoom(image, masks=None, box: Box | None = None, figsize=(9, 9)) -> None:
    """Zoom into ``box = (row0, col0, row1, col1)`` to inspect dense regions.

    Defaults to a 150 px crop at the image center if no box is given.
    """
    image = np.asarray(image)
    if box is None:
        h, w = image.shape[:2]
        r, c = h // 2, w // 2
        box = (max(0, r - 75), max(0, c - 75), min(h, r + 75), min(w, c + 75))
    r0, c0, r1, c1 = box
    crop_img = image[r0:r1, c0:c1]
    crop_mask = None if masks is None else np.asarray(masks)[r0:r1, c0:c1]
    _, ax = plt.subplots(figsize=figsize)
    _draw(ax, crop_img, crop_mask, alpha=0.4)
    ax.set_title(f"zoom {box}")
    plt.tight_layout()
    plt.show()
