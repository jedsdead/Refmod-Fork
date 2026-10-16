"""
flux_ui.py -- the Flux RefMods panel (Media Generator) and the plugin's Flux
tab (fork addition). The mechanism is in flux.py.
"""

from __future__ import annotations

import json

import gradio as gr

from . import flux

NONE_CHOICE = "(none)"
RESTORE_KEY = "_fluxrefmod_restore"   # = outputinfo.RESTORE_FLUX
BUDGET_UI = [("Full (up to ~4096 tokens)", 4096), ("2048 tokens", 2048),
             ("1024 tokens", 1024), ("512 tokens", 512)]


def _choices():
    return [NONE_CHOICE] + flux.list_mods()


def _rows_payload(vals):
    rows = []
    for i in range(0, len(vals), 2):
        name, strength = vals[i], vals[i + 1]
        if name and name != NONE_CHOICE:
            rows.append({"mod": name, "strength": float(strength if strength is not None else 1.0)})
    return rows


def _readout(rows):
    if not rows:
        return "*No Flux RefMods selected.*"
    lines, n = [], 0
    for row in rows:
        meta = flux.read_meta(row["mod"]) or {}
        pictures = int(meta.get("pictures") or 1)
        first, last = n + 1, n + pictures
        lines.append(f"**{row['mod']}**: +{first}" + (f" to +{last}" if last > first else "")
                     + f" ({int(meta.get('tokens') or 0):,} tokens)")
        n = last
    return ("  \n".join(lines) + "  \n*+1 is the reference after your own. "
            "The console logs each mod's exact prompt label.*")


def _model_choices(api_session):
    try:
        records = api_session.list_model_defs(base_model_type=list(flux.FLUX2_TYPES + flux.FLUX3_TYPES))
    except Exception as e:
        print(f"[Refmods] could not list Flux models: {e!r}")
        records = []
    return [(r.get("name") or r.get("model_type"), r.get("model_type")) for r in records]


def submit_extract(plugin, api_session, model_type, spec):
    """Run a Flux extraction as a short task on ``model_type``.
    Returns (ok, message)."""
    log = []

    class Callbacks:
        def on_status(self, message):
            if message:
                log.append(str(message))

        def on_progress(self, update):
            pass

    try:
        plugin._submit(api_session, model_type,
                       {"custom_settings": {flux.FLUX_SETTING: json.dumps({flux.FLUX_EXTRACT_KEY: spec})}},
                       Callbacks())
    except Exception as e:
        return False, f"The task failed to run: {e!r}"
    done = [l for l in log if "finished" in l.lower()]
    if done:
        return True, done[-1]
    return False, (log[-1] if log else "No result -- see the console.")


def build_inline(plugin):
    """The Flux panel under LoRAs on the Media Generator page."""
    target = getattr(plugin, "model_choice_target", None) or getattr(plugin, "wangp_model_choice_target", None)
    get_base_mt = getattr(plugin, "get_base_model_type", None)
    with gr.Accordion("Refmods: Flux", open=False) as accordion:
        gr.Markdown("Works with **Flux 2** (Dev, Klein 4B / 9B) and **Flux 3 Klein**. "
                    "Make mods in the **Refmods** tab (Extract (Flux)).")
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh mod list", size="sm", scale=0)
            clear_btn = gr.Button("Clear", size="sm", scale=0)
        readout = gr.Markdown(_readout([]))
        dropdowns, rows_ui, widgets = [], [], []
        for i in range(flux.MAX_ROWS):
            with gr.Row(visible=i == 0) as row_ui:
                mdd = gr.Dropdown(choices=_choices(), value=NONE_CHOICE, label=f"Flux mod {i + 1}", scale=3)
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Strength", scale=2)
            dropdowns.append(mdd)
            rows_ui.append(row_ui)
            widgets += [mdd, strength]
        with gr.Row():
            add_btn = gr.Button("➕ Add", size="sm", scale=0)
            remove_btn = gr.Button("➖ Remove", size="sm", scale=0)
        shown = gr.State(1)

    def apply(state, *vals):
        rows = _rows_payload(vals)
        if isinstance(state, dict):
            if rows:
                state[flux.FLUX_STASH_KEY] = json.dumps({"rows": rows})
            else:
                state.pop(flux.FLUX_STASH_KEY, None)
        return state, _readout(rows)

    gr.on(triggers=[w.change for w in widgets], fn=apply, inputs=[plugin.state] + widgets,
          outputs=[plugin.state, readout], queue=False)

    def add(n):
        n = min(flux.MAX_ROWS, int(n) + 1)
        return [n] + [gr.update(visible=i < n) for i in range(flux.MAX_ROWS)]

    def remove(n):
        n = max(1, int(n) - 1)
        updates = [n] + [gr.update(visible=i < n) for i in range(flux.MAX_ROWS)]
        cleared = []
        for i in range(flux.MAX_ROWS):
            cleared += ([gr.update(value=NONE_CHOICE), gr.update(value=1.0)] if i >= n
                        else [gr.update(), gr.update()])
        return updates + cleared

    add_btn.click(fn=add, inputs=[shown], outputs=[shown] + rows_ui, queue=False)
    remove_btn.click(fn=remove, inputs=[shown], outputs=[shown] + rows_ui + widgets, queue=False)

    def clear(state):
        if isinstance(state, dict):
            state.pop(flux.FLUX_STASH_KEY, None)
        resets = []
        for _ in range(flux.MAX_ROWS):
            resets += [gr.update(value=NONE_CHOICE), gr.update(value=1.0)]
        return [state, _readout([]), 1] + [gr.update(visible=i == 0) for i in range(flux.MAX_ROWS)] + resets

    clear_btn.click(fn=clear, inputs=[plugin.state],
                    outputs=[plugin.state, readout, shown] + rows_ui + widgets, queue=False)

    def refresh(*current):
        choices = _choices()
        return [gr.update(choices=choices, value=v if v in choices else NONE_CHOICE) for v in current]

    refresh_btn.click(fn=refresh, inputs=dropdowns, outputs=dropdowns, queue=False)

    # fork.34: loading a file's settings refills the panel (see outputinfo).
    refresh_trigger = getattr(plugin, "refresh_form_trigger", None)
    triggers = [t.change for t in (refresh_trigger, target) if t is not None]
    if triggers:
        restore_outputs = [readout, shown] + rows_ui + widgets

        def restore_panel(state):
            if not isinstance(state, dict) or RESTORE_KEY not in state:
                return [gr.update()] * len(restore_outputs)
            raw = state.pop(RESTORE_KEY)
            try:
                rows = (json.loads(raw).get("rows") or []) if raw else []
            except Exception:
                rows = []
            available = set(flux.list_mods())
            missing = [r.get("mod") for r in rows if r.get("mod") not in available]
            rows = [r for r in rows if r.get("mod") in available][:flux.MAX_ROWS]
            n = max(1, len(rows))
            values = []
            for i in range(flux.MAX_ROWS):
                if i < len(rows):
                    values += [gr.update(choices=_choices(), value=rows[i]["mod"]),
                               gr.update(value=float(rows[i].get("strength", 1.0) or 0.0))]
                else:
                    values += [gr.update(value=NONE_CHOICE), gr.update(value=1.0)]
            text = _readout(rows)
            if missing:
                text += "  \nNot found here: " + ", ".join(str(m) for m in missing) + "."
            return [text, n] + [gr.update(visible=i < n) for i in range(flux.MAX_ROWS)] + values

        gr.on(triggers=triggers, fn=restore_panel, inputs=[plugin.state], outputs=restore_outputs, queue=False)

    if target is not None:
        def visibility(target_value):
            model_type = str(target_value or "").split("|", 1)[0].strip()
            return gr.update(visible=flux.is_flux_refmod_capable(model_type, get_base_mt))
        target.change(fn=visibility, inputs=[target], outputs=[accordion], queue=False)
    return accordion


LAYOUT_UI = [("Separate references", flux.LAYOUT_SEPARATE), ("Contact sheet (one reference)", flux.LAYOUT_SHEET),
             ("Merge into one picture", flux.LAYOUT_MERGE)]
LAYOUT_INFO = "Several pictures: one reference each, side by side in one, or averaged into one (smaller, softer)."


def build_tab(plugin, api_session):
    """The Extract (Flux) tab."""
    choices = _model_choices(api_session)
    gr.Markdown("### Extract a Flux RefMod\nFor **Flux 2** (Dev, Klein 4B / 9B) and **Flux 3 Klein**: one mod "
                "works in all of them. Saved mods are listed in the **Library** tab.")
    model_dd = gr.Dropdown(label="Flux model (its VAE does the encoding)", choices=choices,
                           value=choices[0][1] if choices else None)
    with gr.Row():
        name = gr.Textbox(label="Mod name", value="my_flux_mod", info="A path (people/anna) makes a subfolder.")
        budget = gr.Dropdown(label="Size per picture", choices=BUDGET_UI, value=flux.DEFAULT_BUDGET,
                             info="Fewer tokens = faster, less detail.")
        concept = gr.Dropdown(label="Concept type", choices=list(flux.CONCEPT_TYPES), value="identity")
    with gr.Row():
        images = gr.Files(label="Pictures", file_types=["image"], file_count="multiple")
        with gr.Column():
            videos = gr.Files(label="Videos (become pictures)", file_types=["video"], file_count="multiple")
            per_second = gr.Slider(0.25, 4, value=1.0, step=0.25, label="Pictures per second of video",
                                   info=f"Sharpest frame of each slot; up to {flux.MAX_VIDEO_PICTURES} per video.")
    from . import order_ui
    order = order_ui.build(images, videos, note="Separate references are numbered in this order.")
    layout = gr.Radio(label="Layout", choices=LAYOUT_UI, value=flux.LAYOUT_SEPARATE, info=LAYOUT_INFO)
    description = gr.Textbox(label="Description", lines=1, info="A note for you; the model never reads it.")
    extract_btn = gr.Button("Extract Flux RefMod", variant="primary")
    extract_status = gr.Markdown("")

    def do_extract(model_type, mod_name, size, concept_type, files, video_files, rate, how, note, sequence=None):
        paths = [getattr(f, "name", f) for f in (files or [])]
        video_paths = [getattr(f, "name", f) for f in (video_files or [])]
        if not model_type:
            return "⚠️ Pick a Flux model first (none installed?)."
        if not paths and not video_paths:
            return "⚠️ Add at least one picture or video."
        ok, message = submit_extract(plugin, api_session, model_type,
                                     {"name": mod_name, "images": paths, "videos": video_paths,
                                      "pictures_per_second": float(rate or 1.0), "layout": how,
                                      "order": order_ui.merged_order(sequence, files, video_files),
                                      "budget": int(size or flux.DEFAULT_BUDGET),
                                      "concept_type": concept_type, "description": note or ""})
        return ("✅ " if ok else "❌ ") + message

    extract_btn.click(fn=do_extract,
                      inputs=[model_dd, name, budget, concept, images, videos, per_second, layout, description, order],
                      outputs=[extract_status])
