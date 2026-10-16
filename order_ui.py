"""
order_ui.py -- putting a mod's uploaded pictures and videos in order
(fork addition, used by both extract tabs).

The order is a list of ["img" | "vid", path] pairs, sent with the extraction
job as ``order``; see apply_order(). Files added later go to the end, files
removed drop out, and the rest keep the order you gave them.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import gradio as gr


def _paths(files) -> List[str]:
    return [str(getattr(f, "name", f)) for f in (files or []) if f]


def merged_order(order, images, videos) -> List[list]:
    """``order`` brought up to date with the current uploads."""
    present = [["img", p] for p in _paths(images)] + [["vid", p] for p in _paths(videos)]
    keys = {tuple(item) for item in present}
    kept = [list(item) for item in (order or []) if tuple(item) in keys]
    seen = {tuple(item) for item in kept}
    return kept + [item for item in present if tuple(item) not in seen]


def apply_order(order, image_paths, video_paths):
    """[(kind, path)] for every upload, in the chosen order (uploads the
    order doesn't mention follow, images first, as before)."""
    return [tuple(item) for item in merged_order(order, image_paths, video_paths)]


_THUMBS = {}


def _video_thumb(path: str) -> Optional[str]:
    if path in _THUMBS:
        return _THUMBS[path]
    thumb = None
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            thumb = os.path.join(tempfile.gettempdir(), f"refmods_thumb_{abs(hash(path))}.jpg")
            cv2.imwrite(thumb, frame)
    except Exception:
        thumb = None
    _THUMBS[path] = thumb
    return thumb


def _blank_thumb() -> str:
    path = os.path.join(tempfile.gettempdir(), "refmods_thumb_video.png")
    if not os.path.isfile(path):
        from PIL import Image
        Image.new("RGB", (160, 90), (60, 60, 60)).save(path)
    return path


def gallery_items(order):
    """One gallery item per upload, always (so gallery and order line up)."""
    items = []
    for n, (kind, path) in enumerate(order, start=1):
        name = os.path.basename(path)
        caption = f"{n} · {'🎬 ' if kind == 'vid' else ''}{name}"
        picture = path if kind == "img" else (_video_thumb(path) or _blank_thumb())
        items.append((picture, caption))
    return items


def build(images, videos, note: str = ""):
    """The Order panel under an extractor's uploads. Returns the gr.State
    holding the order (pass it to the extract handler)."""
    order_state = gr.State([])
    selected = gr.State(None)
    with gr.Accordion("Order", open=True, visible=False) as panel:
        gr.Markdown("Click one, then move it. This is the order the mod keeps." + (f" {note}" if note else ""))
        gallery = gr.Gallery(show_label=False, columns=6, height="auto", allow_preview=False,
                             object_fit="contain")
        with gr.Row():
            first_btn = gr.Button("⏮ First", size="sm", scale=0, min_width=90)
            up_btn = gr.Button("◀ Earlier", size="sm", scale=0, min_width=90)
            down_btn = gr.Button("Later ▶", size="sm", scale=0, min_width=90)
            last_btn = gr.Button("Last ⏭", size="sm", scale=0, min_width=90)
            reset_btn = gr.Button("Reset", size="sm", scale=0, min_width=90)

    def refresh(order, imgs, vids):
        order = merged_order(order, imgs, vids)
        return (order, None, gr.update(value=gallery_items(order), selected_index=None),
                gr.update(visible=len(order) > 1))

    for files in (images, videos):
        files.change(fn=refresh, inputs=[order_state, images, videos],
                     outputs=[order_state, selected, gallery, panel], queue=False)

    def on_select(evt: gr.SelectData):
        return evt.index if isinstance(evt.index, int) else None

    gallery.select(fn=on_select, inputs=None, outputs=[selected], queue=False)

    def move(order, index, how):
        if index is None or not order or not 0 <= index < len(order):
            return order, index, gr.update()
        order = [list(item) for item in order]
        item = order.pop(index)
        target = {"first": 0, "up": max(0, index - 1), "down": min(len(order), index + 1),
                  "last": len(order)}[how]
        order.insert(target, item)
        return order, target, gr.update(value=gallery_items(order), selected_index=target)

    for btn, how in ((first_btn, "first"), (up_btn, "up"), (down_btn, "down"), (last_btn, "last")):
        btn.click(fn=lambda order, index, _how=how: move(order, index, _how), inputs=[order_state, selected],
                  outputs=[order_state, selected, gallery], queue=False)

    reset_btn.click(fn=lambda imgs, vids: refresh([], imgs, vids)[:3], inputs=[images, videos],
                    outputs=[order_state, selected, gallery], queue=False)
    return order_state
