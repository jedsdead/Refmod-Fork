"""
Refmods For Wan GP -- Wan2GP plugin (formerly MiniMax H3 RefMods).

No-training "reference mods" for MiniMax H3 Ref2VA: compress an image/video
reference into a small .safetensors file once (Extract tab), then reuse it at
any strength in later generations without re-encoding it every time (Generate
tab), optionally blending several mods together. Ported from the ComfyUI
custom-node pack ComfyUI-MiniMaxH3Mod (MIT, (c) 2026 Luisa/luisacaotica) --
see README.md for how the mechanism works and its current limitations.

Design note on why this plugin has its own "Generate" section instead of
hooking the main Media Generator form: the per-model "Custom Settings" fields
Wan2GP auto-builds from a model's definition (the channel this plugin uses to
carry a RefMod selection all the way to the pipeline) are not given a stable
elem_id, so a plugin cannot bind its own rich widgets to them. Submitting a
self-contained task through the API session (the same mechanism the bundled
Sample Plugin demonstrates) sidesteps that limitation entirely and is the
supported, documented way for a plugin to drive a full generation.

The Generate panel below covers the fields most people tune day to day
(prompt, resolution, frame count, steps, flow shift, sampler, reference-image
budget, LoRAs, step-skipping accelerators, sliding window, and the sol-attn
sparsity dial) plus the RefMods themselves. Anything not exposed here still
gets a valid value: "Sync from the main form" copies every setting from
Wan2GP's own Media Generator tab (for the same model) as the starting point,
and `api_session.merge_settings_with_defaults(...)` fills in the rest from
that model's own factory defaults before submission -- so no field is ever
missing, even ones this panel doesn't have a dedicated widget for.
"""

from __future__ import annotations

import functools
import json
import math
import os

import gradio as gr
from PIL import Image

from shared.utils.plugins import WAN2GPPlugin

from . import core, encframes, storage
from . import flux, flux_ui   # fork.31
from . import outputinfo   # fork.34
from . import order_ui   # fork.37
import time
from .patches import (EXTRACT_WRAPPER_KEY, SETTING_GENERATE, STASH_KEY, install_patches,
                      install_get_model_settings_patch, install_prepare_inputs_dict_patch,
                      is_minimax_h3_refmod_capable,
                      DEFAULT_AUDIO_SECONDS, MIN_AUDIO_SECONDS)

try:
    from gradio_rangeslider import RangeSlider   # shipped with Wan2GP (fork: video trim)
except Exception:                                 # pragma: no cover
    RangeSlider = None

PlugIn_Name = "Refmods For Wan GP"
PlugIn_Tab = "Refmods"   # fork.36: the tab's label
PlugIn_Id = "H3RefMods"

# Row counts are just how many picker slots the UI draws -- they are NOT the
# model's limits. Wan2GP's own "at most 12 references: 9 images, 2 videos, 2
# audio" check is bypassed for RefMods (see patches.py's _place_refmod_ref):
# those numbers are a UI/product cap, not an architectural one. MiniMax H3
# uses RoPE positions computed at runtime, an unbounded reference loop, and
# free-running <Picture N> labels, and the ComfyUI community has verified 15
# image references working correctly. Raise these if you want more slots --
# the live counter below will keep telling you what you're actually sending.
IMAGE_ROWS = 20
VIDEO_ROWS = 6
AUDIO_ROWS = 4
NONE_CHOICE = "(none)"

# How video-kind mods behave on sliding windows after the first one. Image and
# audio mods always go to every window (that is what WanGP does with its own
# live references: src_ref_images / audio_guide are set once, before the window
# loop, and passed to every generate() call), so only the video case is a
# choice. The values are what travels in the serialized state; see
# patches.WINDOW_MODE_* for what each one does at injection time.
WINDOW_MODE_CHOICES = [
    ("First frame as a still (recommended)", "still"),
    ("Whole clip, like a live reference video", "full"),
    ("Not at all (first window only)", "first"),
]
WINDOW_MODE_DEFAULT = "still"

# Mirrors models/minimax_h3/minimax_h3_handler.py -- kept as a local constant
# so this UI doesn't need a live import of Wan2GP internals just to draw a
# dropdown. If a future Wan2GP version changes these, only this dropdown's
# labels/values would need updating, nothing else in the plugin depends on it.
FIRST_BLOCK_CACHE_STRENGTHS = [
    ("Low (0.06)", 0.06),
    ("Balanced (0.08, upstream default)", 0.08),
    ("High (0.10)", 0.10),
    ("Very High (0.12)", 0.12),
    ("Maximum (0.14)", 0.14),
]
STEPS_SKIPPING_CHOICES = [
    ("None", ""),
    ("Spectrum Feature Forecasting", "spectrum"),
    ("First Block Cache", "first_block"),
]
SAMPLE_SOLVER_CHOICES = [
    ("Euler", "euler"),
    ("RES Multistep", "res_multistep"),
    ("Ralston 2S (~2x slower)", "ralston_2s"),
]


def _diagnose(api_session, model_type, patch_error):
    """Confirm the plugin's monkeypatches are actually wired up for the
    selected model, *before* running an extraction/generation -- surfaces the
    exact failure mode that made the original bug so hard to notice (nothing
    ever raised an error; generate() just quietly ran a normal render)."""
    if not model_type:
        return "⚠️ Pick a model above first."
    lines = []
    if patch_error:
        lines.append(f"❌ Pipeline patch failed at plugin startup: {patch_error}")
    else:
        lines.append("✅ Pipeline patched (extraction / injection hooks are active).")
    try:
        model_def = api_session.get_model_def(model_type) or {}
        declared_ids = {s.get("id") for s in (model_def.get("custom_settings") or []) if isinstance(s, dict)}
        missing = [sid for sid in (SETTING_GENERATE,) if sid not in declared_ids]
        if missing:
            lines.append(f"❌ This model does NOT declare {missing} under custom_settings -- RefMods will "
                         f"silently no-op for it (a real generation will run instead of an extraction, or "
                         f"without the mods applied). Check the terminal log for '[Refmods]' lines at "
                         f"Wan2GP startup -- the model-definition patch did not take effect for "
                         f"'{model_type}'.")
        else:
            lines.append(f"✅ '{model_type}' declares both custom_settings ids -- RefMod payloads will "
                         f"survive task submission (both this plugin's own Generate tab and, "
                         f"combined with the check below, the inline panel).")
    except Exception as e:
        lines.append(f"⚠️ Could not read the model definition for '{model_type}': {e!r}")
    try:
        from . import patches
        if getattr(patches.install_prepare_inputs_dict_patch, patches._PREPARE_INPUTS_PATCH_MARKER, False):
            lines.append("✅ Inline panel hook active (prepare_inputs_dict patched) -- RefMods selected "
                         "in the 'Refmods: MiniMax H3' accordion on the Media Generator page will "
                         "apply to that page's own Generate button.")
        else:
            lines.append("❌ Inline panel hook NOT active -- the inline accordion on the Media Generator "
                         "page will be visible but selections made there will have no effect. Use this "
                         "plugin's own 'Generate' tab instead, or check the terminal log for a "
                         "'[Refmods] prepare_inputs_dict' line explaining why.")
    except Exception as e:
        lines.append(f"⚠️ Could not check the inline panel hook status: {e!r}")
    return "\n".join(lines)


ALL_FOLDERS_CHOICE = "(all folders)"
ROOT_FOLDER_CHOICE = "(main folder only)"


def _folder_choices():
    """Choices for the folder picker above the mod rows: browse everything,
    just the root, or any one subfolder (recursively discovered, shown as
    "characters/voices"-style relative paths). Picking "(all folders)" is
    how you get back to seeing every mod again after narrowing down."""
    return [ALL_FOLDERS_CHOICE, ROOT_FOLDER_CHOICE] + storage.list_mod_folders()


def _mod_choices(kind=None, folder=ALL_FOLDERS_CHOICE):
    """[(none), name, name, ...], optionally restricted to mods of a given
    "image"/"video"/"audio" kind so a slot can only ever offer mods that
    fit it, and to a chosen folder. Names are folder-relative paths
    ("characters/tanya") so they stay unambiguous across subfolders and
    remain directly usable with storage.load_refmod()."""
    if folder == ROOT_FOLDER_CHOICE:
        list_folder, recursive = None, False
    elif not folder or folder == ALL_FOLDERS_CHOICE:
        list_folder, recursive = None, True
    else:
        list_folder, recursive = folder, True
    if kind in ("image", "video", "audio"):
        names = storage.list_refmods_by_kind(kind, folder=list_folder, recursive=recursive)
    else:
        names = storage.list_refmods(folder=list_folder, recursive=recursive)
    return [NONE_CHOICE] + names


def _refresh_mod_dropdown_updates(folder=ALL_FOLDERS_CHOICE, *current_values):
    """gr.update(...) for the folder picker followed by every mod-picker
    dropdown built by _build_mod_picker_rows, in the same
    image-then-video-then-audio order.

    ``current_values`` are those dropdowns' *current* selections, in the
    same order. Each one is kept selected and force-included in its own
    choices even when it lives outside the folder now being browsed --
    otherwise narrowing to a folder would silently drop selections made
    from other folders (and Gradio would then reject the stale value with
    "Value: x is not in the list of choices"). That's what makes it
    possible to browse folder by folder and still combine mods from
    several different folders in one generation."""
    def row_update(kind, index):
        choices = _mod_choices(kind, folder)
        current = current_values[index] if index < len(current_values) else None
        if current and current != NONE_CHOICE and current not in choices:
            choices = choices + [current]
        return gr.update(choices=choices, value=current if current else NONE_CHOICE)

    updates = [gr.update(choices=_folder_choices(), value=folder)]
    i = 0
    for _ in range(IMAGE_ROWS):
        updates.append(row_update("image", i)); i += 1
    for _ in range(VIDEO_ROWS):
        updates.append(row_update("video", i)); i += 1
    for _ in range(AUDIO_ROWS):
        updates.append(row_update("audio", i)); i += 1
    return updates


FPS_ASSUMED_FOR_DURATION_ESTIMATE = 24  # MiniMax H3's own default fps -- only used to turn a
                                        # video-kind mod's latent frame count into an estimated
                                        # seconds figure for the counter below; the real cap
                                        # generate() enforces is duration-based (<=15s), not a
                                        # frame count, and uses whatever fps the render actually
                                        # runs at.

MAX_LATENT_FRAMES = 90  # the highest "latent frames" value that still stays under MiniMax H3's
                        # native 15s reference cap once the video VAE's causal 4:1 temporal
                        # compression is accounted for -- see latent_frames_to_seconds() below;
                        # 91 already estimates to just over 15s.


def latent_frames_to_seconds(latent_frames, fps: int = FPS_ASSUMED_FOR_DURATION_ESTIMATE) -> float:
    """Same causal-VAE math the live reference-budget counter uses (see
    _format_ref_counter): approximately how many real seconds of source
    video a given "latent frames" extraction setting corresponds to."""
    latent_frames = max(1, int(latent_frames))
    t_px = (latent_frames - 1) * 4 + 1 if latent_frames > 1 else 1
    return round(t_px / fps, 1)


def seconds_to_latent_frames(seconds, fps: int = FPS_ASSUMED_FOR_DURATION_ESTIMATE) -> int:
    """Inverse of latent_frames_to_seconds() -- what "latent frames" value
    to actually extract with so the result is close to the requested number
    of seconds. Round-trips exactly for every value latent_frames_to_seconds
    itself can produce."""
    seconds = max(0.0, float(seconds))
    t_px = seconds * fps
    if t_px <= 1:
        return 1
    return max(1, round((t_px - 1) / 4) + 1)


AUDIO_LATENTS_PER_SECOND = 40  # MiniMax H3's own audio VAE: encoder downsamples by 800x at
                               # 32kHz = 40 latents/s exactly (models/minimax_h3/components/
                               # audio_autoencoder.py's own docstring) -- unlike the video
                               # estimate below, this is an exact, fps-independent rate, not
                               # an approximation.


def _source_file_count(meta, which):
    """How many source files of a given kind ("img" / "vid") went into a
    mod, read from the "{n} img, {m} vid" tag every extraction writes.
    Falls back to 1 when the tag is missing or unparseable (mods from
    older versions, or hand-made ones), since a mod always came from at
    least one file."""
    for tag in (meta.get("tags") or []):
        counts = core.parse_source_counts(tag)
        if counts is not None:
            return counts[0] if which == "img" else counts[1]
    return 1


def _safe_choice_update(choices, keep=None):
    """gr.update for a single-select dropdown whose options just changed.

    Gradio raises "Value: X is not in the list of choices" the next time a
    dropdown is read if its value no longer appears in its choices -- which
    happens whenever a mod is moved, renamed or deleted while it is selected.
    Keeping the value only when it survived avoids that.
    """
    choices = list(choices or [])
    return gr.update(choices=choices, value=keep if keep in choices else None)


def _trim_summary(trims, names, durations=None):
    """What each video will give the mod, with a warning for spans under
    H3's 2-second audio minimum and over its 15-second reference limit."""
    durations = durations or {}
    trims = trims or {}
    parts, warnings = [], []
    for n in names:
        if n in trims:
            parts.append(f"`{n}` {trims[n][0]:g}-{trims[n][1]:g}s")
            span = float(trims[n][1]) - float(trims[n][0])
        else:
            span = float(durations.get(n) or 0.0)
        if 0 < span < 2.0 - 0.05:
            warnings.append(f"⚠️ `{n}`: only {span:.1f}s will be used. Under 2 seconds is too short "
                            f"for its own audio to work as a voice reference.")
        elif span > 15.05:
            warnings.append(f"`{n}`: {span:.1f}s -- only the first 15s of it is used (H3's limit "
                            f"for a reference video). Trim it to choose which 15s.")
    text = ("Trimmed: " + ", ".join(parts)) if parts else "*No trims set - every video is used whole.*"
    return "\n\n".join([text] + warnings)


# Fork panel. Values per picker row: mod, strength, "Send as one video",
# "Include audio", prompt label, soundtrack's prompt label.
ROW_WIDTH = 6
# Panel-wide controls after the mod rows: picture mode, N, video mode, N,
# fit to output, mods in phase 2, shrink method, video mods on sliding windows,
# mods in phase 1 (fork.30, last so older saved orders still line up).
STACK_CONTROLS = 9

# Label dropdowns (fork.27): how far each one counts.
LABEL_MAX = {"image": 20, "video": 9, "audio": 9}
LABEL_WORDS = {"image": "Picture", "video": "Video", "audio": "Audio"}
AUTO_LABEL = "auto"
PHASE2_MODS_UI = [("Full", "full"), ("Fit to tile", "tile"), ("Off", "off")]
PHASE1_MODS_UI = [("Full size", "full"), ("Fit to phase 1", "fit")]   # fork.30; Full is the default since fork.40
SHRINK_METHOD_UI = [("Re-encode (best quality)", "reencode"), ("Fast (resize the stored mod)", "fast")]


def _label_choices(kind):
    """Choices for a prompt-label dropdown: Auto, then <Picture 1>, <Picture 2>..."""
    return [("Auto", AUTO_LABEL)] + [(f"<{LABEL_WORDS[kind]} {n}>", n)
                                      for n in range(1, LABEL_MAX[kind] + 1)]


# fork.34: values a panel restore is putting into a row, so the row's own
# change handlers (which start a newly picked mod fresh) keep them.
_RESTORE_PENDING = {}
_RESTORE_SECONDS = 15.0


def _pending_row(key, name):
    item = _RESTORE_PENDING.get(key)
    if not item or time.time() - item[0] > _RESTORE_SECONDS or item[1].get("mod") != name:
        return None
    return item[1]


def _row_restore_updates(row, kind):
    """The six row widgets set to a saved row (see _rows_payload)."""
    name = row["mod"]
    meta = _row_meta(name, 1.0)
    multi = (kind == "image" and bool(meta) and _is_picture_mod(meta)
             and int(meta.get("latent_t", 1) or 1) > 1)
    has_audio = bool(meta) and meta.get("kind") != "audio" and bool(meta.get("has_audio"))
    send_as = (("sheet" if row.get("send_as") == "sheet" else "video" if row.get("as_video") else "pictures")
               if multi else "pictures")
    as_video = send_as == "video"
    use_audio = bool(row.get("use_audio", True))
    choices = _mod_choices(kind)
    return [gr.update(choices=choices if name in choices else choices + [name], value=name),
            gr.update(value=float(row.get("strength", 1.0) or 0.0)),
            gr.update(visible=multi, value=send_as),
            gr.update(visible=has_audio, value=use_audio),
            gr.update(visible=True, choices=_label_choices("video" if as_video else kind),
                      value=_label_number(row.get("label")) or AUTO_LABEL),
            gr.update(visible=has_audio and use_audio, value=_label_number(row.get("audio_label")) or AUTO_LABEL)]


def _row_boxes_update(name, kind="image", key=None):
    pending = _pending_row(key, name) if key is not None else None
    if pending is not None:
        return tuple(_row_restore_updates(pending, kind)[2:])
    """The per-row options after a row's mod changes, starting fresh for the
    newly picked mod (nothing chosen for the previous mod carries over).

    "Send as one video" (image rows only) shows only while the row holds a
    mod made from several pictures, unticked. "Include audio" (image and
    video rows) shows only while the mod carries a soundtrack, ticked, and
    with it the soundtrack's own label dropdown. The label dropdown shows
    once a mod is picked."""
    meta = _row_meta(name, 1.0)
    multi = (kind == "image" and bool(meta) and _is_picture_mod(meta)
             and int(meta.get("latent_t", 1) or 1) > 1)
    has_audio = bool(meta) and meta.get("kind") != "audio" and bool(meta.get("has_audio"))
    return (gr.update(visible=multi, value="pictures"), gr.update(visible=has_audio, value=True),
            gr.update(visible=bool(meta), choices=_label_choices(kind), value=AUTO_LABEL),
            gr.update(visible=has_audio, value=AUTO_LABEL))


def _panel_restore_updates(raw):
    """fork.34: the inline panel refilled from a loaded file's selection --
    status, folder picker, every row widget, the panel controls, row
    visibility and the per-kind shown counts (see restore_panel)."""
    try:
        payload = json.loads(raw) if raw else {}
    except Exception:
        payload = {}
    counts = {"image": IMAGE_ROWS, "video": VIDEO_ROWS, "audio": AUDIO_ROWS}
    slots = {k: [] for k in counts}
    missing = []
    for row in payload.get("rows") or []:
        name = row.get("mod")
        meta = storage.read_refmod_meta(storage.mod_path(name)) if name else None
        if meta is None:
            missing.append(str(name))
            continue
        kind = meta.get("kind", "image")
        if kind == "video" and core.is_still_stack(kind, int(meta.get("latent_t", 1) or 1),
                                                   meta.get("tags"), meta.get("source", "")):
            kind = "image"
        kind = kind if kind in counts else "image"
        if len(slots[kind]) < counts[kind]:
            slots[kind].append(row)
        else:
            missing.append(str(name))
    _RESTORE_PENDING.clear()
    now = time.time()
    widgets, ui, shown = [], [], []
    for kind, count in counts.items():
        for i in range(count):
            if i < len(slots[kind]):
                _RESTORE_PENDING[(kind, i)] = (now, slots[kind][i])
                widgets += _row_restore_updates(slots[kind][i], kind)
            else:
                widgets += _cleared_row_updates()
        n = max(1, len(slots[kind]))
        ui += [gr.update(visible=i < n) for i in range(count)]
        shown.append(n)
    controls = [payload.get("stack_pictures", encframes.STACK_ALL),
                payload.get("stack_pictures_n", encframes.STACK_DEFAULT_N),
                payload.get("video_frames", encframes.STACK_UP_TO_N),
                payload.get("video_frames_n", encframes.MAX_CLIP_FRAMES_SHOWN),
                bool(payload.get("fit_to_output", False)), payload.get("phase2_mods", "full"),
                payload.get("shrink_method", "reencode"), payload.get("window_mode", WINDOW_MODE_DEFAULT),
                payload.get("phase1_mods", "full")]
    restored = sum(len(v) for v in slots.values())
    status = (f"↺ {restored} RefMod(s) restored from the loaded settings." if restored
              else "*No RefMods selected.*")
    if missing:
        status += " Not found here: " + ", ".join(missing) + "."
    return ([status, gr.update(value=ALL_FOLDERS_CHOICE)] + widgets + [gr.update(value=v) for v in controls]
            + ui + shown)


def _as_video_label_update(on, name, key=None):
    """The label dropdown after "Send as one video" flips: <Video N> or
    <Picture N> choices, back to Auto (unless a restore is setting it)."""
    pending = _pending_row(key, name) if key is not None else None
    value = (_label_number(pending.get("label")) or AUTO_LABEL) if pending else AUTO_LABEL
    return gr.update(choices=_label_choices("video" if _send_as(on) == "video" else "image"), value=value)


def _soundtrack_label_update(include_audio, name):
    """The soundtrack's label dropdown: shown while the row's mod has a
    soundtrack and "Include audio" is ticked."""
    meta = _row_meta(name, 1.0)
    has_audio = bool(meta) and meta.get("kind") != "audio" and bool(meta.get("has_audio"))
    return gr.update(visible=bool(include_audio and has_audio))


def _cleared_row_updates():
    """gr.update()s returning one picker row to empty."""
    return [gr.update(value=NONE_CHOICE), gr.update(value=1.0),
            gr.update(value="pictures", visible=False), gr.update(value=True, visible=False),
            gr.update(value=AUTO_LABEL, visible=False), gr.update(value=AUTO_LABEL, visible=False)]


def _label_number(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


# fork.38: how a multi-picture image row is sent (was the "Send as one video" box).
SEND_AS_UI = [("Separate pictures", "pictures"), ("One video", "video"), ("Contact sheet", "sheet")]


def _send_as(value) -> str:
    """"pictures" / "video" / "sheet" from the row's Send-as value (older
    callers pass the old checkbox's True / False)."""
    if value in ("pictures", "video", "sheet"):
        return value
    return "video" if value is True else "pictures"


def _rows_payload(values):
    """The selection payload's "rows" from the pickers' flat values
    (ROW_WIDTH per row): one entry per row with a mod and strength > 0."""
    rows = []
    for i in range(0, len(values) - ROW_WIDTH + 1, ROW_WIDTH):
        name, strength, as_video, use_audio, label, audio_label = values[i:i + ROW_WIDTH]
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            continue
        if name and name != NONE_CHOICE and strength > 0:
            send_as = _send_as(as_video)
            row = {"mod": name, "strength": strength, "as_video": send_as == "video",
                   "use_audio": bool(use_audio)}
            if send_as == "sheet":
                row["send_as"] = "sheet"
            if _label_number(label):
                row["label"] = _label_number(label)
            if use_audio and _label_number(audio_label):
                row["audio_label"] = _label_number(audio_label)
            rows.append(row)
    return rows


def _normalize_rows(rows):
    """Picker rows as (name, strength, send-as-one-video, include-audio,
    label, soundtrack label); shorter rows (other plugins, older builds) mean
    "not sent as one video", "audio included" and automatic labels."""
    out = []
    for row in rows or []:
        row = tuple(row)
        out.append((row[0], row[1],
                    _send_as(row[2]) if len(row) > 2 else "pictures",
                    bool(row[3]) if len(row) > 3 else True,
                    _label_number(row[4]) if len(row) > 4 else None,
                    _label_number(row[5]) if len(row) > 5 else None))
    return out


def _audio_file_seconds(path):
    """Length of an audio file in seconds, or None if it can't be read."""
    if not path:
        return None
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        return None


def _row_meta(name, strength):
    """Metadata for a picker row that will actually be sent, else None."""
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        return None
    if not name or name == NONE_CHOICE or strength <= 0:
        return None
    try:
        return storage.read_refmod_meta(storage.mod_path(name))
    except Exception:
        return None


def _is_picture_mod(meta):
    kind = meta.get("kind", "image")
    latent_t = max(1, int(meta.get("latent_t", 1)))
    return kind == "image" or core.is_still_stack(kind, latent_t, meta.get("tags"),
                                                    meta.get("source", ""))


def _stack_payload(stack_mode, stack_n,
                   video_mode=encframes.STACK_UP_TO_N, video_n=encframes.MAX_CLIP_FRAMES_SHOWN,
                   fit_to_output=False, phase2_mods="full", shrink_method="reencode",
                   window_mode=WINDOW_MODE_DEFAULT, phase1_mods="full"):
    """The selection payload's text-encoder fields (see patches.py's
    _build_refmod_sentinels): how many pictures of a mod sent as one video,
    and how many frames of a video mod, the text encoder is shown, plus the
    panel's sizing and sliding-window options. Whether a mod is sent as one
    video is per row ("as_video" in each row)."""
    def count(value, default):
        try:
            return max(1, int(value or default))
        except (TypeError, ValueError):
            return default
    mode = stack_mode if stack_mode in encframes.STACK_CHOICES else encframes.STACK_ALL
    vmode = video_mode if video_mode in encframes.STACK_CHOICES else encframes.STACK_UP_TO_N
    return {"stack_pictures": mode,
            "stack_pictures_n": count(stack_n, encframes.STACK_DEFAULT_N),
            "video_frames": vmode,
            "video_frames_n": count(video_n, encframes.MAX_CLIP_FRAMES_SHOWN),
            # "Fit mods to the output size" (off by default).
            "fit_to_output": bool(fit_to_output),
            # "Mods in phase 2": full / tile / off.
            "phase2_mods": phase2_mods if phase2_mods in ("full", "tile", "off") else "full",
            # "Mods in phase 1": fit / full (fork.30).
            "phase1_mods": phase1_mods if phase1_mods in ("fit", "full") else "full",
            # "Shrink method": reencode / fast.
            "shrink_method": shrink_method if shrink_method in ("reencode", "fast") else "reencode",
            # "Video mods on sliding windows 2, 3, ...": still / full / first.
            "window_mode": window_mode if window_mode in [v for _, v in WINDOW_MODE_CHOICES]
            else WINDOW_MODE_DEFAULT}


def _ordered_labels(items, word):
    """Number one kind's mods the way patches._order_by_label does: chosen
    numbers where asked, Auto ones filling the gaps in row order."""
    explicit = sorted((i for i in items if i[2]), key=lambda i: i[2])
    auto = [i for i in items if not i[2]]
    out, position = [], 1
    while explicit or auto:
        name, count, want = (explicit.pop(0) if explicit and (explicit[0][2] <= position or not auto)
                             else auto.pop(0))
        last = position + count - 1
        span = f"<{word} {position}>" if count == 1 else f"<{word} {position}>–<{word} {last}>"
        out.append(f"`{name}` → {span}" + (f" (asked for <{word} {want}>)" if want and want != position else ""))
        position = last + 1
    return out


def _format_label_line(rows):
    """Which <Picture N> / <Video N> / <Audio N> each picked mod will get in
    window 1 -- its chosen number, or the next free one on Auto. Assumes no
    start image, Continue Video or references of the generation's own: those
    take the first numbers, so with a start image pick <Picture 2> onwards (the
    console reports the labels each generation really used)."""
    pictures, videos, audios = [], [], []    # (mod name, count, wanted number)
    for name, strength, as_video, use_audio, label, audio_label in _normalize_rows(rows):
        meta = _row_meta(name, strength)
        if meta is None:
            continue
        kind = meta.get("kind", "image")
        latent_t = max(1, int(meta.get("latent_t", 1)))
        if _is_picture_mod(meta):
            if as_video == "video" and latent_t > 1:
                videos.append((name, 1, label))
            elif as_video == "sheet" and latent_t > 1:
                pictures.append((name, 1, label))
            else:
                pictures.append((name, latent_t, label))
        elif kind == "video":
            videos.append((name, 1, label))
        else:
            audios.append((name, 1, label))
            continue
        if meta.get("has_audio") and use_audio:
            audios.append((name + " (soundtrack)", 1, audio_label))

    parts = (_ordered_labels(pictures, "Picture") + _ordered_labels(videos, "Video")
             + _ordered_labels(audios, "Audio"))
    if not parts:
        return ""
    extra = ""
    if len(videos) > 3:
        extra = " (videos past the third have no label -- only 3 reference-video slots)"
    return ("**Prompt labels** (with no start image, Continue Video or references of your own -- "
            "those come first): " + " · ".join(parts) + extra)


def _format_ref_counter(rows):
    """A live 'how close to MiniMax H3 Ref2VA's own native reference caps am
    I' readout, from the current (mod_name, strength, send-as-one-video) values of every
    picker row (image rows, then video rows, then audio rows, any order
    internally). Mirrors exactly what _inject_refmods will actually send at
    generation time: rows with no mod picked or strength<=0 are skipped,
    and -- critically -- an image-kind mod counts once *per frame it
    contains* (see patches.py's _inject_refmods: a multi-image stack gets
    split into one reference per frame), not once per mod. Video- and
    audio-kind mods, by contrast, each go into their own native reference
    slot no matter how many are picked, so what actually matters for them
    is total duration (each against its own, separate 15-second budget --
    video and audio don't share one)."""
    n_images = 0
    n_stacks = 0
    video_latent_frames = 0
    audio_seconds_total = 0.0
    n_video_files = 0
    n_audio_files = 0
    n_soundtracks = 0
    for name, strength, as_video, use_audio, _label, _audio_label in _normalize_rows(rows):
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            continue
        if not name or name == NONE_CHOICE or strength <= 0:
            continue
        try:
            meta = storage.read_refmod_meta(storage.mod_path(name))
        except Exception:
            meta = None
        if meta is None:
            continue
        kind = meta.get("kind", "image")
        latent_t = max(1, int(meta.get("latent_t", 1)))
        if _is_picture_mod(meta) and as_video == "sheet" and latent_t > 1:
            n_images += 1          # fork.38: one contact sheet
        elif _is_picture_mod(meta) and as_video == "video" and latent_t > 1:
            n_stacks += 1          # one video reference; pictures have no duration
        elif _is_picture_mod(meta):
            n_images += latent_t
        elif kind == "video":
            t_px = (latent_t - 1) * 4 + 1 if latent_t > 1 else 1  # undo the causal 4:1 compression
            video_latent_frames += t_px
            n_video_files += _source_file_count(meta, "vid")
        else:  # "audio"
            audio_seconds_total += latent_t / AUDIO_LATENTS_PER_SECOND
            # Audio extraction takes exactly one file per mod, so each
            # selected audio mod is one source file.
            n_audio_files += 1
        # A soundtrack attached to an image or video mod is injected as its
        # own audio reference, so it counts toward the audio budget too --
        # unless the row's "Include audio" is unticked.
        if kind != "audio" and meta.get("has_audio") and use_audio:
            audio_seconds_total += int(meta.get("audio_t", 0) or 0) / AUDIO_LATENTS_PER_SECOND
            n_soundtracks += 1
    video_seconds = video_latent_frames / FPS_ASSUMED_FOR_DURATION_ESTIMATE
    # The 9 / 15s / 15s figures are MiniMax's *documented* reference budget,
    # which this plugin no longer enforces on RefMods (see patches.py's
    # _place_refmod_ref). They stay here purely as a reference point: the
    # model demonstrably works past them (15 image references verified by
    # the ComfyUI community), but that is past what MiniMax documents, so
    # the counter flags it as "beyond documented" rather than as an error.
    img_mark = "🔶" if n_images > 9 else "▫️" if n_images == 0 else "✅"
    vid_mark = "🔶" if video_seconds > 15 else "▫️" if video_seconds == 0 else "✅"
    aud_mark = "🔶" if audio_seconds_total > 15 else "▫️" if audio_seconds_total == 0 else "✅"
    notes = []
    if n_images > 9:
        notes.append(f"{n_images} images is past MiniMax's documented 9 (a multi-image mod counts "
                     f"once per image it contains). Verified to work up to ~15; expect more VRAM "
                     f"use and, at some point, attention dilution.")
    if video_seconds > 15:
        notes.append("video total is past the documented 15s budget.")
    if audio_seconds_total > 15:
        notes.append("audio total is past the documented 15s budget.")
    note = (" -- " + " ".join(notes)) if notes else ""
    label_line = _format_label_line(rows)
    vid_files = f" from {n_video_files} file{'s' if n_video_files != 1 else ''}" if n_video_files else ""
    if n_stacks:
        vid_files += f" + {n_stacks} picture stack{'s' if n_stacks != 1 else ''} as video"
    aud_parts = []
    if n_audio_files:
        aud_parts.append(f"{n_audio_files} file{'s' if n_audio_files != 1 else ''}")
    if n_soundtracks:
        aud_parts.append(f"{n_soundtracks} mod soundtrack{'s' if n_soundtracks != 1 else ''}")
    aud_files = (" from " + " + ".join(aud_parts)) if aud_parts else ""
    return (f"{img_mark} **Images: {n_images}** (documented budget: 9)&nbsp;&nbsp;&nbsp;"
           f"{vid_mark} **Video: ~{video_seconds:.1f}s**{vid_files} "
           f"(documented: 15s, at {FPS_ASSUMED_FOR_DURATION_ESTIMATE}fps)&nbsp;&nbsp;&nbsp;"
           f"{aud_mark} **Audio: {audio_seconds_total:.1f}s**{aud_files} (documented: 15s)"
           f"{note}"
           + (f"\n\n{label_line}" if label_line else ""))


def _hidden_from_api(build):
    """Run a UI builder and keep every event handler it registers out of
    Gradio's API listing (show_api=False).

    On the first page load Gradio builds that listing for every handler, and
    for each input it searches the whole component list -- so the work grows
    with handlers x inputs x components. The panel's handlers take every
    picker as input (~160 of them), and with Wan2GP's own large UI around it
    the first page load could pass the 3 seconds Gradio allows when it checks
    that localhost is reachable at launch. It then refuses to start with "When
    localhost is not accessible, a shareable link must be created" (seen with
    fork.20, fixed in fork.21). Nothing here is meant to be called through the API -- the
    plugin runs generations through Wan2GP's own session -- so hiding them
    loses nothing."""
    @functools.wraps(build)
    def wrapper(*args, **kwargs):
        try:
            from gradio.context import Context
            root = Context.root_block
            before = set(root.fns.keys()) if root is not None else None
        except Exception:
            root, before = None, None
        try:
            return build(*args, **kwargs)
        finally:
            if root is not None and before is not None:
                hidden = 0
                for key, fn in list(root.fns.items()):
                    if key not in before and getattr(fn, "show_api", False):
                        fn.show_api = False
                        hidden += 1
                if hidden:
                    print(f"[Refmods] {hidden} panel event handler(s) kept out of Gradio's API "
                          f"listing (faster first page load)")
    return wrapper


def _model_choices(api_session):
    try:
        records = api_session.list_model_defs(
            base_model_type=["minimax_h3_ref2va", "minimax_h3_ref2va_pruned"])
    except Exception as e:
        print(f"[Refmods] could not list MiniMax H3 Ref2VA models: {e!r}")
        records = []
    return [(r.get("name") or r.get("model_type"), r.get("model_type")) for r in records]


DEFAULT_MODEL_TYPE = "minimax_h3_ref2va_pruned"  # "MiniMax H3 Ref2VA Pruned 20B"


def _default_model_choice(model_choices):
    """Preselect MiniMax H3 Ref2VA Pruned 20B at startup -- the lighter,
    faster variant most people running this plugin day to day will have
    installed. Falls back to whatever's first if that exact model, or any
    other model whose name mentions "pruned", isn't available."""
    for _, model_type in model_choices:
        if model_type == DEFAULT_MODEL_TYPE:
            return model_type
    for display_name, model_type in model_choices:
        if "pruned" in str(display_name).lower():
            return model_type
    return model_choices[0][1] if model_choices else None


def _decompile_details(mod):
    """A plain summary of what a mod holds, for the Decompile tab."""
    counts = next((core.parse_source_counts(t) for t in mod.tags
                   if core.parse_source_counts(t) is not None), None)
    made = []
    if counts:
        if counts[0]:
            made.append(f"{counts[0]} picture{'s' if counts[0] != 1 else ''}")
        if counts[1]:
            made.append(f"{counts[1]} video{'s' if counts[1] != 1 else ''}")
    extras = [t for t in mod.tags if core.parse_source_counts(t) is None]
    lines = [f"**{mod.name}** — {mod.kind} mod, **{mod.token_count:,} tokens**"]
    if made:
        lines.append("Made from: " + " and ".join(made)
                     + (f" ({', '.join(extras)})" if extras else ""))
    if mod.kind == "audio":
        lines.append(f"Audio: ~{mod.latent_t / AUDIO_LATENTS_PER_SECOND:.1f}s")
    else:
        px = f"~{mod.latent_w * 16}x{mod.latent_h * 16}px"
        if mod.mode == "training":
            lines.append(f"Mode: **training** — pooled to a {mod.latent_h}x{mod.latent_w} grid "
                         f"(decodes to a small, blurry {px} picture; the fine detail was pooled "
                         f"away at extraction)")
        else:
            lines.append(f"Mode: **encode** — full detail at {px}")
        if mod.still_stack() or mod.kind == "image" or mod.latent_t <= 1:
            lines.append(f"Pictures: {mod.latent_t}"
                         + (" (one merged picture)" if mod.source == "merge" else ""))
        else:
            seconds = ((mod.latent_t - 1) * 4 + 1) / FPS_ASSUMED_FOR_DURATION_ESTIMATE
            lines.append(f"Video: {mod.latent_t} latent frames, ~{seconds:.1f}s")
        if mod.audio_latent is not None:
            lines.append(f"Soundtrack: ~{mod.audio_latent.shape[-1] / AUDIO_LATENTS_PER_SECOND:.1f}s")
        lines.append("Encoder frames: " + (f"{len(mod.enc_times)} stored" if mod.has_encoder_frames()
                                           else "not stored"))
    if mod.concept_type and mod.concept_type != "generic":
        lines.append(f"Concept type: {mod.concept_type}")
    if mod.description:
        lines.append(f"Description: {mod.description}")
    return "  \n".join(lines)


# fork.32: the Library lists H3 and Flux mods together. Its folder picker
# reads "H3", "H3/<subfolder>", "Flux", "Flux/<subfolder>" (or every folder);
# H3 mods live in loras/refmods_plugin/minimax_h3/, Flux mods in .../flux2/.
LIB_H3, LIB_FLUX = "H3", "Flux"


def _library_folder_choices():
    return ([ALL_FOLDERS_CHOICE, LIB_H3] + [f"{LIB_H3}/{f}" for f in storage.list_mod_folders()]
            + [LIB_FLUX] + [f"{LIB_FLUX}/{f}" for f in flux.list_folders()])


def _library_destinations():
    return ([(f"{LIB_H3} (top)", LIB_H3)] + [f"{LIB_H3}/{f}" for f in storage.list_mod_folders()]
            + [(f"{LIB_FLUX} (top)", LIB_FLUX)] + [f"{LIB_FLUX}/{f}" for f in flux.list_folders()])


def _library_subfolders():
    return ([f"{LIB_H3}/{f}" for f in storage.list_mod_folders()]
            + [f"{LIB_FLUX}/{f}" for f in flux.list_folders()])


def _split_library_folder(folder):
    """(model, subfolder) for a Library folder value. model None = both;
    subfolder "" = all of that model's mods, None = its top level only."""
    if not folder or folder == ALL_FOLDERS_CHOICE:
        return None, ""
    if folder == ROOT_FOLDER_CHOICE:
        return "h3", None
    head, _, rest = str(folder).partition("/")
    if head == LIB_FLUX:
        return "flux", rest
    if head == LIB_H3:
        return "h3", rest
    return "h3", str(folder)   # a plain H3 subfolder, as older code passes


def _flux_library_names(folder=None):
    model, sub = _split_library_folder(folder)
    if model == "h3":
        return []
    names = flux.list_mods()
    return [n for n in names if n.startswith(sub + "/")] if sub else names


def _qualified_choices(folder=None):
    """Dropdown choices naming H3 and Flux mods unambiguously."""
    return ([(f"{LIB_H3} · {n}", f"h3:{n}") for n in _library_mod_names(folder)]
            + [(f"{LIB_FLUX} · {n}", f"flux:{n}") for n in _flux_library_names(folder)])


def _split_qualified(value):
    model, _, name = str(value or "").partition(":")
    return (model, name) if model in ("h3", "flux") else ("h3", str(value or ""))


def _library_rows(folder=None):
    """Table rows for the Library tab: H3 mods, then Flux mods, filtered by
    the Library's folder picker (see _split_library_folder)."""
    model, _ = _split_library_folder(folder)
    rows = []
    if model != "flux":
        args = _h3_list_args(folder)
        rows += [[LIB_H3, i["name"], i["kind"], i["mode"], i["tokens"],
                  (f'{i["prompt_view"]}px' if i.get("prompt_view") else "--"),
                  i["size_mb"], i["concept_type"], i["description"]]
                 for i in storage.list_refmods_info(folder=args[0], recursive=args[1])]
    for name in _flux_library_names(folder):
        meta = flux.read_meta(name) or {}
        rows.append([LIB_FLUX, name, "image", meta.get("mode", "encode"), meta.get("tokens", "?"), "--",
                     meta.get("size_mb", "?"), meta.get("concept_type", ""),
                     meta.get("description", "") if meta.get("usable", True)
                     else "not a Flux 2 latent -- won't load"])
    return rows


def _h3_list_args(folder):
    model, sub = _split_library_folder(folder)
    if sub is None:
        return None, False
    return (sub or None), True


def _convertible(folder=None):
    """H3 picture mods (single pictures and picture stacks) Convert to Flux
    can take, honouring the Library's folder picker."""
    model, _ = _split_library_folder(folder)
    if model == "flux":
        return []
    list_folder, recursive = _h3_list_args(folder)
    return storage.list_refmods_by_kind("image", folder=list_folder, recursive=recursive)


CLEAN_AUDIO_INFO = "Keeps voices only (Wan2GP's own separator); music and effects go."


def _audio_mod_names(folder=None):
    """fork.41: H3 audio mods and mods with a soundtrack, for Clean audio."""
    model, _ = _split_library_folder(folder)
    if model == "flux":
        return []
    list_folder, recursive = _h3_list_args(folder)
    return storage.list_refmods_with_audio(folder=list_folder, recursive=recursive)


def _library_mod_names(folder=None):
    """H3 mod names (folder-relative) for the H3-only tools, honouring the
    Library's folder picker; none when a Flux folder is picked."""
    model, _ = _split_library_folder(folder)
    if model == "flux":
        return []
    list_folder, recursive = _h3_list_args(folder)
    return storage.list_refmods(folder=list_folder, recursive=recursive)


def _lora_choices(api_session, model_type):
    if not model_type:
        return []
    try:
        info = api_session.list_loras(model_type)
    except Exception as e:
        print(f"[Refmods] could not list LoRAs for {model_type}: {e!r}")
        return []
    return list(info.get("loras") or [])


def _patches_module():
    from . import patches
    return patches


def _tag_job(overrides):
    """(overrides, job id) -- the id put into a plugin job's spec, or None
    for an ordinary generation."""
    import uuid
    custom = dict((overrides or {}).get("custom_settings") or {})
    for key, wrapper in ((SETTING_GENERATE, EXTRACT_WRAPPER_KEY), (flux.FLUX_SETTING, flux.FLUX_EXTRACT_KEY)):
        try:
            data = json.loads(custom.get(key) or "")
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get(wrapper), dict):
            job_id = uuid.uuid4().hex
            data[wrapper] = dict(data[wrapper], job_id=job_id)
            custom[key] = json.dumps(data)
            return dict(overrides, custom_settings=custom), job_id
    return overrides, None


_NO_OUTPUT_ERROR = "no output with client_id"


class _JobCallbacks:
    """Holds a job's status lines and errors until it ends, then hands the
    caller the job's own record (or, if the job never ran, what Wan2GP said)."""

    def __init__(self, inner):
        self.inner, self.status, self.errors = inner, [], []

    def on_status(self, message):
        if message:
            self.status.append(str(message))

    def on_error(self, error):
        self.errors.append(getattr(error, "message", None) or str(error))

    def on_progress(self, update):
        method = getattr(self.inner, "on_progress", None)
        if callable(method):
            method(update)

    def finish(self, recorded):
        relay_status = getattr(self.inner, "on_status", None)
        relay_error = getattr(self.inner, "on_error", None)
        lines = list(recorded) if recorded else self.status
        for line in lines:
            if callable(relay_status):
                relay_status(line)
        for message in self.errors:
            if recorded and _NO_OUTPUT_ERROR in message:
                continue   # expected: a job makes no video
            if callable(relay_error):
                relay_error(message)
            elif callable(relay_status):
                relay_status(f"error: {message}")


class MiniMaxH3RefModsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.38.1-fork.44"
        self.description = ("No-training reference mods for MiniMax H3: compress a reference "
                            "into a small file once, reuse it at any strength without "
                            "re-encoding it every generation.")
        self._patch_error = install_patches()
        self._flux_error = flux.install()   # fork.31: Flux 2 / Flux 3
        try:   # fork.32: every mod file in its model's folder (H3 / Flux)
            flux.sort_mod_files()
        except Exception as e:
            print(f"[Refmods] could not sort mod files into their model folders: {e!r}")

    def setup_ui(self):
        self.request_component("state")
        self.request_component("model_choice_target")
        self.request_component("wangp_model_choice_target")  # legacy/alternate name, harmless if absent
        self.request_global("get_current_model_settings")
        self.request_global("refresh_model_defs")
        self.request_global("prepare_inputs_dict")
        self.request_global("get_state_model_type")
        self.request_global("get_model_settings")
        self.request_global("get_base_model_type")
        self.request_global("set_model_settings")          # fork.34: restore on load
        self.request_global("collect_custom_settings_from_inputs")   # fork.42
        self.request_component("refresh_form_trigger")     # fork.34
        try:   # fork.34: a readable RefMods line in each output's info
            self.register_data_hook("before_metadata_save", outputinfo.before_metadata_save)
        except Exception as e:
            print(f"[Refmods] could not register the metadata hook ({e!r}); outputs won't list their RefMods")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Tab, component_constructor=self.create_ui)
        self.insert_after(target_component_id="loras_multipliers",
                          new_component_constructor=self._build_inline_refmods_section)
        self.insert_after(target_component_id="loras_multipliers",   # fork.31
                          new_component_constructor=self._build_inline_flux_section)

    def post_ui_setup(self, components):
        """Wan2GP builds its whole model catalog (``models_def``, what
        ``get_model_def()`` reads from) once, at import time, well before any
        plugin is loaded -- so the ``family_handler.query_model_def`` patch
        installed in ``__init__`` (see patches.py) has no effect on entries
        already cached by then, even though the patch itself is correctly in
        place. ``self.refresh_model_defs`` (a global, only available once
        Wan2GP finishes injecting plugin globals -- i.e. here, not in
        ``__init__``) is Wan2GP's own supported way to rebuild that catalog
        on demand; calling it once now forces every MiniMax H3 model
        definition to be recomputed through our now-patched function, so the
        two RefMod custom_settings actually end up in the cache the rest of
        Wan2GP reads from."""
        if getattr(self, "_post_ui_setup_done", False):
            return {}  # Wan2GP appears to call post_ui_setup more than once at startup;
                      # everything below is safe to repeat but noisy/wasteful, so skip it.
        self._post_ui_setup_done = True

        refresh = getattr(self, "refresh_model_defs", None)
        if callable(refresh):
            try:
                refresh()
                print("[Refmods] refreshed Wan2GP's model catalog so the RefMod custom_settings "
                     "declaration takes effect on cached MiniMax H3 model definitions")
            except Exception as e:
                print(f"[Refmods] refresh_model_defs() failed ({e!r}); MiniMax H3 model "
                     "definitions may still be missing the RefMod custom_settings -- use "
                     "'Check setup for this model' in the plugin tab to confirm.")
        else:
            print("[Refmods] refresh_model_defs was not exposed by Wan2GP's plugin globals; "
                 "RefMods will likely not take effect until Wan2GP is restarted after this "
                 "plugin was enabled. Use 'Check setup for this model' in the plugin tab to confirm.")

        get_base_mt = getattr(self, "get_base_model_type", None)
        if not callable(get_base_mt):
            print("[Refmods] get_base_model_type was not exposed by Wan2GP's plugin globals; "
                 "a MiniMax H3 Ref2VA finetune whose own model_type name doesn't start with "
                 "'minimax_h3_ref2va' (e.g. a custom-named checkpoint) may not be recognized -- "
                 "the inline panel would stay hidden for it, and RefMods wouldn't apply even if "
                 "selected through the plugin's own 'Generate' tab.")
            get_base_mt = None

        orig_prepare = getattr(self, "prepare_inputs_dict", None)
        get_state_mt = getattr(self, "get_state_model_type", None)
        if callable(orig_prepare) and callable(get_state_mt):
            err = install_prepare_inputs_dict_patch(orig_prepare, get_state_mt, self.set_global, get_base_mt)
            if err:
                print(f"[Refmods] {err}")
        else:
            print("[Refmods] prepare_inputs_dict / get_state_model_type were not exposed by "
                 "Wan2GP's plugin globals; the inline RefMods panel on the Media Generator page "
                 "will be visible but will not affect generations from that page's own Generate "
                 "button. Use the plugin's own 'Generate' tab instead.")

        orig_set_model_settings = getattr(self, "set_model_settings", None)   # fork.34
        if callable(orig_set_model_settings):
            err3 = outputinfo.install_set_model_settings_patch(orig_set_model_settings, self.set_global, get_base_mt)
            if err3:
                print(f"[Refmods] {err3}")

        orig_collect = getattr(self, "collect_custom_settings_from_inputs", None)   # fork.42
        if callable(orig_collect):
            from . import patches
            err4 = patches.install_collect_custom_settings_patch(orig_collect, self.set_global)
            if err4:
                print(f"[Refmods] {err4}")

        orig_get_model_settings = getattr(self, "get_model_settings", None)
        if callable(orig_get_model_settings):
            err2 = install_get_model_settings_patch(orig_get_model_settings, self.set_global, get_base_mt)
            if err2:
                print(f"[Refmods] {err2}")
        else:
            print("[Refmods] get_model_settings was not exposed by Wan2GP's plugin globals; "
                 "a RefMod change right before clicking Generate (with no other field touched "
                 "in between) may not always be picked up from the inline panel -- change any "
                 "other field (e.g. click into the prompt box) once after picking mods as a "
                 "workaround, or use the plugin's own 'Generate' tab instead.")
        return {}

    # ── Inline panel injected onto the Media Generator page ───────────────

    @_hidden_from_api
    def _build_inline_refmods_section(self):
        # Wan2GP hands plugins their requested components out of generate_media_tab's
        # own locals(), so the key is the Python *variable* name ("model_choice_target"),
        # not the elem_id ("wangp_model_choice_target"). Try both so this works across
        # Wan2GP versions regardless of which name a given build exposes.
        target = getattr(self, "model_choice_target", None) or getattr(self, "wangp_model_choice_target", None)
        # Default to visible (matches the previous, always-shown behavior) rather than
        # hidden: there's no confirmed signal that fires on the very first page load
        # (only on an actual model switch), so defaulting to hidden could strand the
        # panel out of sight for anyone who already has a MiniMax H3 Ref2VA model
        # selected by default and hasn't touched the model dropdown yet. The .change()
        # handler below still correctly hides it the moment any model switch happens.
        with gr.Accordion("Refmods: MiniMax H3", open=False) as accordion:
            gr.Markdown(
                "Works with **MiniMax H3 Ref2VA, FL2VA and FL2VA ControlNet**. Make mods in the **Refmods** tab.")
            # Fork: the full panel (per-row options, sizing, text encoder).
            mod_rows, panel_controls = self._build_mod_picker_rows_full()
            with gr.Row():
                status = gr.Markdown("*No RefMods selected.*")
                clear_btn = gr.Button("Clear RefMods", size="sm", scale=0)

        widgets = [c for row in mod_rows for c in row] + list(panel_controls)

        def apply_selection(state, *vals):
            n_rows = IMAGE_ROWS + VIDEO_ROWS + AUDIO_ROWS
            rows = _rows_payload(vals[:n_rows * ROW_WIDTH])
            payload = {"rows": rows, "retention": 1.0, "scramble_seed": -1, "curve": None,
                       **_stack_payload(*vals[n_rows * ROW_WIDTH:n_rows * ROW_WIDTH + STACK_CONTROLS])}
            if not isinstance(state, dict):
                return state, "⚠️ Could not access session state -- try reloading the page."
            if rows:
                state[STASH_KEY] = json.dumps(payload)
                msg = f"✅ {len(rows)} RefMod(s) armed for the next MiniMax H3 generation from this page."
            else:
                state.pop(STASH_KEY, None)
                msg = "*No RefMods selected.*"
            return state, msg

        def clear_selection(state):
            # Fork: empties the slots as well, so the panel never shows mods
            # that aren't armed.
            if isinstance(state, dict):
                state.pop(STASH_KEY, None)
            resets = []
            for _ in mod_rows:
                resets += _cleared_row_updates()
            return [state, "*No RefMods selected.*"] + resets

        clear_btn.click(fn=clear_selection, inputs=[self.state],
                        outputs=[self.state, status] + [c for row in mod_rows for c in row], queue=False)

        # fork.34: loading a file's settings refills the panel (outputinfo).
        parts = getattr(self, "_panel_parts", None)
        refresh_trigger = getattr(self, "refresh_form_trigger", None)
        triggers = [t.change for t in (refresh_trigger, target) if t is not None]
        if parts is not None and triggers:
            kinds = ("image", "video", "audio")
            restore_outputs = ([status, parts["folder_dd"]] + widgets
                               + [ui for k in kinds for ui in parts["kind_ui"][k]]
                               + [parts["shown"][k] for k in kinds])

            def restore_panel(state):
                if not isinstance(state, dict) or outputinfo.RESTORE_H3 not in state:
                    return [gr.update()] * len(restore_outputs)
                return _panel_restore_updates(state.pop(outputinfo.RESTORE_H3))

            gr.on(triggers=triggers, fn=restore_panel, inputs=[self.state], outputs=restore_outputs,
                  queue=False)

        # One handler for every widget (gr.on), not one per widget: each takes
        # every widget as input, and Gradio's first page load does work for
        # every handler x input (see _hidden_from_api).
        gr.on(triggers=[w.change for w in widgets], fn=apply_selection,
              inputs=[self.state] + widgets, outputs=[self.state, status], queue=False)

        if target is not None:
            get_base_mt = getattr(self, "get_base_model_type", None)

            def update_visibility(target_value):
                model_type = str(target_value or "").split("|", 1)[0].strip()
                return gr.update(visible=is_minimax_h3_refmod_capable(model_type, get_base_mt))

            target.change(fn=update_visibility, inputs=[target], outputs=[accordion], queue=False)
        else:
            print("[Refmods] neither 'model_choice_target' nor 'wangp_model_choice_target' was "
                 "exposed by Wan2GP -- the inline panel will stay visible for every model instead "
                 "of only MiniMax H3 Ref2VA ones (it still has no effect on other models, this "
                 "only affects whether it's shown).")

        return accordion

    @_hidden_from_api
    def _build_inline_flux_section(self):   # fork.31
        return flux_ui.build_inline(self)

    def _submit(self, api_session, model_type, overrides, callbacks):
        """fork.39: a plugin job (extract, decompile, Library job, Flux
        extraction) is tagged with an id, and its own record of what it
        reported is handed to ``callbacks`` once it ends -- Wan2GP's API can
        miss a job's last status line, and reports "no output ... in the
        gallery" for every job, since jobs make no video on purpose."""
        overrides, job_id = _tag_job(overrides)
        if not job_id:
            return self._submit_task(api_session, model_type, overrides, callbacks)
        relay = _JobCallbacks(callbacks)
        try:
            return self._submit_task(api_session, model_type, overrides, relay)
        finally:
            relay.finish(_patches_module().JOB_RESULTS.pop(job_id, None))

    def _submit_task(self, api_session, model_type, overrides, callbacks):
        """Build a full, validated settings dict for ``model_type`` -- starting
        from that model's own factory defaults, then applying ``overrides`` --
        and submit it. Any field this plugin doesn't have a widget for still
        gets a sane, model-correct value this way."""
        payload = dict(overrides)
        payload["model_type"] = model_type  # always wins, even if overrides (e.g. a synced
                                            # main-form snapshot) carried a different one
        merged = api_session.merge_settings_with_defaults(payload)
        merged["model_type"] = model_type
        job = api_session.submit_task(merged, callbacks=callbacks)
        return job.result()

    # ── Extract ─────────────────────────────────────────────────────────

    @_hidden_from_api
    def _build_mod_picker_rows(self):
        """A folder picker + "Refresh mod list" button + a live
        reference-budget counter, followed by IMAGE_ROWS image-kind +
        VIDEO_ROWS video-kind + AUDIO_ROWS audio-kind mod picker rows
        (dropdown restricted to that kind, plus strength), matching MiniMax
        H3 Ref2VA's own native reference caps. The folder picker, refresh
        button and counter live here (above the rows, for visibility) and
        are fully wired before returning. Returns the flat list of
        (dropdown, strength) pairs, image rows first, then video rows, then
        audio rows -- callers must keep that same order when reading values
        back (_refresh_mod_dropdown_updates() above does too)."""
        with gr.Row():
            folder_dd = gr.Dropdown(choices=_folder_choices(), value=ALL_FOLDERS_CHOICE,
                                    label="Folder", scale=3,
                                    info="Show one subfolder only.")
            refresh_btn = gr.Button("🔄 Refresh mod list", size="sm", scale=1)
        counter = gr.Markdown(_format_ref_counter([]))
        mod_rows = []
        gr.Markdown(f"**Image RefMods** (up to {IMAGE_ROWS}; leave a row on `{NONE_CHOICE}` to skip it)")
        for i in range(IMAGE_ROWS):
            with gr.Row():
                mdd = gr.Dropdown(choices=_mod_choices("image"), value=NONE_CHOICE,
                                  label=f"Image Mod {i + 1}", scale=3)
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength", scale=2)
                mod_rows.append((mdd, strength))
        gr.Markdown(f"**Video RefMods** (up to {VIDEO_ROWS}; leave a row on `{NONE_CHOICE}` to skip it)")
        for i in range(VIDEO_ROWS):
            with gr.Row():
                mdd = gr.Dropdown(choices=_mod_choices("video"), value=NONE_CHOICE,
                                  label=f"Video Mod {i + 1}", scale=3)
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength", scale=2)
                mod_rows.append((mdd, strength))
        gr.Markdown(f"**Audio RefMods** (up to {AUDIO_ROWS}; leave a row on `{NONE_CHOICE}` to skip it)")
        for i in range(AUDIO_ROWS):
            with gr.Row():
                mdd = gr.Dropdown(choices=_mod_choices("audio"), value=NONE_CHOICE,
                                  label=f"Audio Mod {i + 1}", scale=3)
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength", scale=2)
                mod_rows.append((mdd, strength))

        picker_outputs = [folder_dd] + [r[0] for r in mod_rows]
        # The current selections are passed in as well so switching folders
        # can preserve them (see _refresh_mod_dropdown_updates) -- without
        # that, narrowing to a folder would drop any mod picked from a
        # different one and Gradio would reject the now-stale value.
        picker_inputs = [folder_dd] + [r[0] for r in mod_rows]
        refresh_btn.click(fn=_refresh_mod_dropdown_updates, inputs=picker_inputs,
                          outputs=picker_outputs, queue=False)
        folder_dd.change(fn=_refresh_mod_dropdown_updates, inputs=picker_inputs,
                         outputs=picker_outputs, queue=False)

        def update_counter(*vals):
            n = len(mod_rows)
            pairs = [(vals[i * 2], vals[i * 2 + 1]) for i in range(n)]
            return _format_ref_counter(pairs)

        row_widgets = [c for row in mod_rows for c in row]
        for w in row_widgets:
            w.change(fn=update_counter, inputs=row_widgets, outputs=[counter], queue=False)

        window_mode = gr.Dropdown(
            label="Video mods on sliding windows 2, 3, ...",
            choices=WINDOW_MODE_CHOICES, value=WINDOW_MODE_DEFAULT,
            info="Runs longer than one window only. Image and audio mods reach every window.")

        # Appended as a one-element pseudo-row so both callers (inline panel
        # and the plugin's own Generate tab) pick it up through the same flat
        # widget list; each reads it back at index IMAGE_ROWS+VIDEO_ROWS+AUDIO_ROWS)*2.
        # (Fork: this plugin's own panels use _build_mod_picker_rows_full below;
        # this one is kept as it is for other plugins that build their own
        # copy of the panel, e.g. 2 Phases Plus' H3 Upscaler tab.)
        return mod_rows + [(window_mode,)]

    def _build_mod_picker_rows_full(self):
        """Fork. The mod pickers: a folder picker + "Refresh mod list" button, the
        live budget/label readout, the "Text encoder options" accordion, then
        one slot per kind (image, video, audio) with Add / Remove buttons
        revealing up to IMAGE_ROWS / VIDEO_ROWS / AUDIO_ROWS -- all wired
        before returning. Returns the flat list of (dropdown, strength,
        send-as-one-video, include-audio, max-size) rows (ROW_WIDTH values each), image
        rows first, then video rows, then audio rows. "Send as one video" only
        shows on image rows holding a multi-picture mod, "Include audio" only
        on image/video rows holding a mod with a soundtrack. Callers must keep
        that order when reading values
        back (_refresh_mod_dropdown_updates() does too) -- plus the encoder
        controls (picture mode, N, video mode, N) and the "Fit mods to the
        output size" box, the "Mods in phase 2" and "Shrink method" choices -- STACK_CONTROLS
        values -- which callers put into
        the selection payload with _stack_payload()."""
        with gr.Row():
            folder_dd = gr.Dropdown(choices=_folder_choices(), value=ALL_FOLDERS_CHOICE,
                                    label="Folder", scale=3,
                                    info="Show one subfolder only.")
            refresh_btn = gr.Button("🔄 Refresh mod list", size="sm", scale=1)
        # Above the pickers, where they can't be missed: the readout of what
        # is selected (with each mod's prompt label), then the text encoder
        # options.
        counter = gr.Markdown(_format_ref_counter([]))
        with gr.Accordion("Text encoder options", open=True):
            gr.Markdown("Pictures the text encoder sees per mod, to bind its label. The model always gets the whole mod.")
            with gr.Row():
                stack_mode = gr.Dropdown(
                    choices=list(encframes.STACK_CHOICES), value=encframes.STACK_ALL, scale=3,
                    label="Mods sent as one video: pictures shown",
                    info="'up to N' saves memory on big stacks.")
                stack_n = gr.Number(value=encframes.STACK_DEFAULT_N, precision=0, minimum=1,
                                    maximum=256, label="N", visible=False, scale=1)
            with gr.Row():
                video_mode = gr.Dropdown(
                    choices=[encframes.STACK_UP_TO_N, encframes.STACK_ALL],
                    value=encframes.STACK_UP_TO_N, scale=3,
                    label="Video mods: frames shown",
                    info="Two frames a second are stored; 'up to N' spreads N across the clip.")
                video_n = gr.Number(value=encframes.MAX_CLIP_FRAMES_SHOWN, precision=0, minimum=1,
                                    maximum=256, label="N", scale=1)
        with gr.Row():
            fit_to_output = gr.Checkbox(
                False, label="Fit mods to the output size (faster)", scale=2,
                info="Shrink mods larger than the output. Less detail.")
            shrink_method = gr.Dropdown(
                choices=SHRINK_METHOD_UI, value="reencode", label="Shrink method", scale=1,
                info="Re-encode: best. Fast: instant, rougher.")
        with gr.Row():
            # fork.30: two-phase runs only.
            phase1_mods = gr.Dropdown(
                choices=PHASE1_MODS_UI, value="full", label="Mods in phase 1 (two-phase)", scale=1,
                info="Fit: shrink to phase 1's half canvas, if phase 1 gets slow.")
            phase2_mods = gr.Dropdown(
                choices=PHASE2_MODS_UI, value="full", label="Mods in phase 2 (two-phase)", scale=1,
                info="Off: no visual mods (fastest).")
        window_mode = gr.Dropdown(
            label="Video mods on sliding windows 2, 3, ...",
            choices=WINDOW_MODE_CHOICES, value=WINDOW_MODE_DEFAULT,
            info="Runs longer than one window only. Image and audio mods reach every window.")

        # One slot per kind to start with; "Add" reveals the next one and
        # "Remove" clears and hides the last. Every slot is still built up
        # front (Gradio can't create components on the fly) -- the hidden ones
        # are simply empty, and the order returned is unchanged: image rows,
        # then video rows, then audio rows, each a (dropdown, strength,
        # send-as-one-video) triple.
        mod_rows = []
        kind_ui_all, shown_all = {}, {}   # fork.34
        counts = (IMAGE_ROWS, VIDEO_ROWS, AUDIO_ROWS)
        for kind, label, count in (("image", "Image RefMods", counts[0]),
                                   ("video", "Video RefMods", counts[1]),
                                   ("audio", "Audio RefMods", counts[2])):
            gr.Markdown(f"**{label}**" + (" -- a start image, Continue Video or your own references "
                                           "take the first label numbers." if kind == "image" else ""))
            kind_rows, kind_ui = [], []
            for i in range(count):
                with gr.Row(visible=i == 0) as row_ui:
                    mdd = gr.Dropdown(choices=_mod_choices(kind), value=NONE_CHOICE,
                                      label=f"{kind.capitalize()} mod {i + 1}", scale=4)
                    strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength",
                                         scale=3, min_width=200)
                    # fork.27: the prompt label this mod gets (Auto = next free).
                    label_dd = gr.Dropdown(choices=_label_choices(kind), value=AUTO_LABEL,
                                           label="Label", visible=False, scale=2, min_width=130)
                    with gr.Column(scale=2, min_width=150):
                        as_video = gr.Dropdown(choices=SEND_AS_UI, value="pictures", label="Send as",
                                               visible=False)   # fork.38 (was a checkbox)
                        use_audio = gr.Checkbox(True, label="Include audio", visible=False)
                        audio_label_dd = gr.Dropdown(choices=_label_choices("audio"), value=AUTO_LABEL,
                                                     label="Soundtrack label", visible=False)
                kind_rows.append((mdd, strength, as_video, use_audio, label_dd, audio_label_dd))
                kind_ui.append(row_ui)
                mdd.change(fn=lambda name, _kind=kind, _key=(kind, i): _row_boxes_update(name, _kind, _key),
                           inputs=[mdd], outputs=[as_video, use_audio, label_dd, audio_label_dd],
                           queue=False)
                if kind == "image":
                    # One video reference: the label becomes a <Video N>.
                    as_video.change(fn=lambda on, name, _key=(kind, i): _as_video_label_update(on, name, _key),
                                    inputs=[as_video, mdd], outputs=[label_dd], queue=False)
                if kind != "audio":
                    use_audio.change(fn=_soundtrack_label_update, inputs=[use_audio, mdd],
                                     outputs=[audio_label_dd], queue=False)
            with gr.Row():
                add_btn = gr.Button(f"➕ Add {kind} mod", size="sm", scale=0, min_width=150)
                remove_btn = gr.Button("➖ Remove last", size="sm", scale=0, min_width=150)
            shown = gr.State(1)
            kind_ui_all[kind], shown_all[kind] = kind_ui, shown   # fork.34
            kind_widgets = [c for row in kind_rows for c in row]

            def add_slot(n, _count=count):
                n = min(_count, int(n or 1) + 1)
                return [n] + [gr.update(visible=k < n) for k in range(_count)]

            def remove_slot(n, _count=count, _width=ROW_WIDTH):
                n = max(1, int(n or 1))
                last = n - 1                       # the slot being cleared
                n = max(1, n - 1)
                cleared = []
                for k in range(_count):
                    if k == last:
                        cleared += _cleared_row_updates()[:_width]
                    else:
                        cleared += [gr.update()] * _width
                return [n] + [gr.update(visible=k < n) for k in range(_count)] + cleared

            add_btn.click(fn=add_slot, inputs=[shown], outputs=[shown] + kind_ui, queue=False)
            remove_btn.click(fn=remove_slot, inputs=[shown],
                             outputs=[shown] + kind_ui + kind_widgets, queue=False)
            mod_rows.extend(kind_rows)

        picker_outputs = [folder_dd] + [r[0] for r in mod_rows]
        # The current selections are passed in as well so switching folders
        # can preserve them (see _refresh_mod_dropdown_updates) -- without
        # that, narrowing to a folder would drop any mod picked from a
        # different one and Gradio would reject the now-stale value.
        picker_inputs = [folder_dd] + [r[0] for r in mod_rows]
        refresh_btn.click(fn=_refresh_mod_dropdown_updates, inputs=picker_inputs,
                          outputs=picker_outputs, queue=False)
        folder_dd.change(fn=_refresh_mod_dropdown_updates, inputs=picker_inputs,
                         outputs=picker_outputs, queue=False)

        def update_counter(*vals):
            rows = [tuple(vals[i * ROW_WIDTH:(i + 1) * ROW_WIDTH]) for i in range(len(mod_rows))]
            return _format_ref_counter(rows)

        row_widgets = [c for row in mod_rows for c in row]
        gr.on(triggers=[w.change for w in row_widgets], fn=update_counter, inputs=row_widgets,
              outputs=[counter], queue=False)
        stack_mode.change(fn=lambda mode: gr.update(visible=mode == encframes.STACK_UP_TO_N),
                          inputs=[stack_mode], outputs=[stack_n], queue=False)
        video_mode.change(fn=lambda mode: gr.update(visible=mode == encframes.STACK_UP_TO_N),
                          inputs=[video_mode], outputs=[video_n], queue=False)

        # fork.34: what a restore (outputinfo) needs to refill this panel.
        self._panel_parts = {"folder_dd": folder_dd, "kind_ui": kind_ui_all, "shown": shown_all}
        return mod_rows, (stack_mode, stack_n, video_mode, video_n, fit_to_output, phase2_mods,
                          shrink_method, window_mode, phase1_mods)


    def _build_extract_section(self, api_session, model_dd):
        gr.Markdown("### Extract a RefMod\nRuns a short task on the model above to use its VAE. No video is made; the mod file is the result.")
        with gr.Row():
            name = gr.Textbox(label="Mod name", value="my_concept",
                              info="Use a path (characters/tanya) for a subfolder.")
            mode = gr.Radio(label="Mode", choices=["training", "encode"], value="training",
                            info="encode: exact, best for faces. training: pooled and smaller, for general concepts.")
        with gr.Row():
            ref_images = gr.Files(label="Reference image(s)", file_types=["image"], file_count="multiple")
            with gr.Column():
                ref_videos = gr.Files(label="Reference video(s)", file_types=["video"], file_count="multiple")
                # Fork: videos as still pictures instead of clips.
                video_as_pictures = gr.Checkbox(
                    False, label="Turn reference videos into a picture stack",
                    info="Sharpest frame per time slot: identity at fewer tokens, no motion.")
                pictures_per_second = gr.Slider(
                    0.25, 12, value=1.0, step=0.25, label="Pictures per second of video",
                    visible=False,
                    info="Each picture costs about 3-4 clip frames.")
        stills_estimate = gr.Markdown("")
        gr.Markdown("*Every image and video added goes into this one mod.*")
        source_order = order_ui.build(ref_images, ref_videos)   # fork.37
        with gr.Accordion("Preview / trim reference videos", open=True):   # fork
            gr.Markdown("Untrimmed videos are used whole, up to 15 s.")
            with gr.Row():
                video_pick = gr.Dropdown(label="Video", choices=[], value=None, scale=2)
                video_duration = gr.Markdown("")
            video_preview = gr.Video(label="Preview", height=260)
            if RangeSlider is not None:
                video_trim = RangeSlider(minimum=0.0, maximum=15.0, value=(0.0, 15.0), step=0.1,
                                         label="Trim (seconds)", info="Start - End")
            else:
                video_trim = gr.Slider(0.0, 15.0, value=0.0, step=0.1, label="Trim start (seconds)",
                                       info="gradio_rangeslider isn't installed, so only a start "
                                            "offset is available here.")
            trim_state = gr.State({})
            trim_status = gr.Markdown("")
        with gr.Row():
            ref_audio = gr.Audio(label="Reference audio (optional)", type="filepath", scale=2)
            with gr.Column(scale=1):
                use_clip_audio = gr.Checkbox(False, label="Use the clip's own audio",
                                             info="Ignored if an audio file is set.")
                clean_audio = gr.Checkbox(False, label="Remove background music / noise",   # fork.41
                                          info=CLEAN_AUDIO_INFO)
        soundtrack_seconds = gr.Slider(
            MIN_AUDIO_SECONDS, 15.0, value=DEFAULT_AUDIO_SECONDS, step=0.5,
            label="Audio length (seconds)",
            info=f"At least {MIN_AUDIO_SECONDS:g}s. ~80 tokens a second.")
        gr.Markdown("*Audio alone makes an audio-only mod; with images or video it becomes the mod's soundtrack (its own `<Audio N>`).*")
        audio_duration_warning = gr.Markdown("")
        remove_background_images_ref = gr.Dropdown(
            choices=[("Keep Backgrounds behind all Reference Images", 0),
                    ("Remove Background behind People / Objects", 1)],
            value=0,
            label="Automatic Removal of Background behind People or Objects in Reference Images",
            info="Images only.")
        with gr.Accordion("Advanced", open=True):
            with gr.Row():
                ref_resolution = gr.Slider(256, 2048, value=1024, step=64, label="Ref resolution (short edge, px)",
                                           info="A cap; never upscales.")
                enc_resolution = gr.Slider(256, 2048, value=core.ENC_MAX_EDGE, step=64,
                                           label="Prompt-view resolution (long edge, px)",
                                           info="What the text encoder sees, to bind the label. Lower = fewer tokens.")
                pool_h = gr.Slider(2, 64, value=16, step=2, label="Pool grid height (training mode)",
                                   info="Training mode. Bigger grid = more detail, more tokens.")
                pool_w = gr.Slider(2, 64, value=16, step=2, label="Pool grid width (training mode)",
                                   info="Training mode. Same as height, other axis.")
            pool_warning = gr.Markdown("")
            encode_mode_note = gr.Markdown(
                "ℹ️ Pool grid and Identity refinement only apply in `training` mode.", visible=False)
            with gr.Row():
                # fork.27: no duration slider -- a video is used whole, or as
                # trimmed in "Preview / trim reference videos", up to H3's 15 s.
                latent_frames = gr.State(latent_frames_to_seconds(MAX_LATENT_FRAMES))
                identity = gr.Slider(0, 2000, value=500, step=50, label="Identity refinement steps (training mode)",
                                     info="Flat after ~200 steps. 0 skips it.")
                multiplier = gr.Slider(1, 10, value=1, step=1, label="Repeat multiplier",
                                       info="Repeats the mod. Higher = stronger effect, bigger file.")
                max_tokens = gr.Slider(0, 65536, value=65536, step=512, label="Max tokens (0 = no cap)",
                                       info="Max tokens allowed. 0 = unlimited.")
            merge = gr.Checkbox(   # fork
                False, label="Merge into one picture (training mode, several sources)",
                info="Combine the pictures into one that keeps what they share. Best on similarly framed shots.")
            description = gr.Textbox(
                label="keyword - description", lines=2,
                info="Text note only. Model never reads it.")
        with gr.Accordion("Misc.", open=False):
            concept_type = gr.Dropdown(
                label="Concept type", choices=list(core.CONCEPT_TYPES), value="generic",
                info="Label only. No effect on generation.")
            save = gr.Checkbox(label="Save to disk", value=True,
                               info="Off = test run, nothing saved.")
        extract_btn = gr.Button("Extract & Save RefMod", variant="primary")
        extract_status = gr.Textbox(label="Status", interactive=False, lines=3)

        def update_pool_warning(concept_type, mode, pool_h, pool_w):
            w = core.identity_training_pool_warning(concept_type, mode, int(pool_h), int(pool_w))
            return f"⚠️ {w}" if w else ""

        for w in (concept_type, mode, pool_h, pool_w):
            w.change(fn=update_pool_warning, inputs=[concept_type, mode, pool_h, pool_w],
                    outputs=[pool_warning], queue=False)

        def update_mode_controls(mode_value):
            # These three are read only on the "training" branch of the
            # extractor (see patches._run_extraction: encode keeps the raw VAE
            # latent and records optimize_steps=0). Showing them as live
            # controls in encode mode invites setting them and getting an
            # identical file back, so hide them and say why.
            training = mode_value == "training"
            return (gr.update(visible=training), gr.update(visible=training),
                    gr.update(visible=training), gr.update(visible=not training))

        mode.change(fn=update_mode_controls, inputs=[mode],
                    outputs=[pool_h, pool_w, identity, encode_mode_note], queue=False)
        mode.change(fn=lambda m: gr.update(visible=m == "training"), inputs=[mode],   # fork
                    outputs=[merge], queue=False)

        def update_audio_duration_warning(audio_path, current_seconds):
            real_seconds = _audio_file_seconds(audio_path)
            if real_seconds is None:
                return ""
            if real_seconds < MIN_AUDIO_SECONDS - 0.05:
                return (f"⛔ This audio file is only ~{real_seconds:.1f}s long. H3 needs at least "
                        f"{MIN_AUDIO_SECONDS:g}s of audio to use it as a reference, so extraction "
                        f"will be refused -- use a longer clip.")
            if real_seconds > float(current_seconds) + 0.15:
                return (f"⚠️ This audio file is ~{real_seconds:.1f}s long, but 'Audio length' "
                        f"is set to {float(current_seconds):.1f}s -- only the first "
                        f"{float(current_seconds):.1f}s will be kept. Raise it (up to 15s) to use "
                        f"more of this recording.")
            return ""

        for w in (ref_audio, soundtrack_seconds):
            w.change(fn=update_audio_duration_warning, inputs=[ref_audio, soundtrack_seconds],
                    outputs=[audio_duration_warning], queue=False)

        def update_stills_estimate(enabled, per_second, videos, images, trims, duration_s,
                                   mode_value, resolution, grid_h, grid_w, cap, merged=False):
            if not enabled:
                return ""
            paths = [str(f.name if hasattr(f, "name") else f) for f in (videos or []) if f]
            if not paths:
                return "*Add a reference video to see how many pictures it will make.*"
            per_second = max(0.05, float(per_second or 1.0))
            parts, total_pictures, total_tokens = [], len(images or []), 0

            def picture_tokens(width, height):
                if mode_value != "encode":
                    return max(1, int(grid_h) // 2) * max(1, int(grid_w) // 2)
                scale = min(1.0, float(resolution) / min(width, height))
                return max(1, round(height * scale / 32)) * max(1, round(width * scale / 32))

            for image in images or []:
                try:
                    with Image.open(str(image.name if hasattr(image, "name") else image)) as im:
                        total_tokens += picture_tokens(*im.size)
                except Exception:
                    pass
            for path in paths:
                name = os.path.basename(path)
                seconds = storage.video_duration_seconds(path) or 0.0
                start, end = (trims or {}).get(name, (0.0, seconds or float(duration_s)))
                span = max(0.0, min(float(end) - float(start), float(duration_s)))
                count = max(1, math.ceil(span * per_second - 1e-9))
                per_picture = picture_tokens(*(storage.video_size(path) or (1280, 720)))
                total_pictures += count
                total_tokens += count * per_picture
                parts.append(f"`{name}` → {count} picture{'s' if count != 1 else ''} "
                             f"({span:.1f}s)")
            if merged and mode_value == "training" and total_pictures > 1:
                one = max(1, int(grid_h) // 2) * max(1, int(grid_w) // 2)
                return ("**Picture stack:** " + " · ".join(parts)
                        + f" — **{total_pictures} pictures merged into one, ~{one:,} tokens**")
            line = ("**Picture stack:** " + " · ".join(parts)
                    + f" — **{total_pictures} pictures in this mod, ~{total_tokens:,} tokens**")
            if cap and int(cap) > 0 and total_tokens > int(cap):
                line += (f"\n\n⚠️ Over 'Max tokens' ({int(cap):,}): some pictures will be "
                         f"dropped to fit.")
            if total_pictures > 9:
                line += ("\n\n⚠️ More than 9 pictures. H3's documented limit is 9 picture "
                         "references, so used as separate pictures some go past it. Ticking "
                         "*send as one video reference* in the RefMod panel avoids that, but is "
                         "itself outside H3's documented limits. Fewer pictures per second, or a "
                         "shorter trim, keeps it within 9.")
            return line

        def on_videos_uploaded(paths, trims):
            """Populate the picker when files are uploaded and select the first."""
            paths = [p for p in (paths or []) if p]
            names = [os.path.basename(str(p)) for p in paths]
            trims = {k: v for k, v in (trims or {}).items() if k in names}   # forget removed files
            if not names:
                return (gr.update(choices=[], value=None), None, "", trims, "")
            first_path, first_name = str(paths[0]), names[0]
            seconds = storage.video_duration_seconds(first_path) or 15.0
            start, end = trims.get(first_name, (0.0, seconds))
            durations = {os.path.basename(str(p)): storage.video_duration_seconds(str(p)) for p in paths}
            return (gr.update(choices=names, value=first_name), first_path,
                    f"**{seconds:.2f}s**", trims,
                    _trim_summary(trims, names, durations))

        def on_video_picked(selected, paths, trims):
            paths = [p for p in (paths or []) if p]
            lookup = {os.path.basename(str(p)): str(p) for p in paths}
            path = lookup.get(selected)
            if path is None:
                return None, "", gr.update()
            seconds = storage.video_duration_seconds(path) or 15.0
            start, end = (trims or {}).get(selected, (0.0, seconds))
            value = (float(start), float(min(end, seconds))) if RangeSlider is not None else float(start)
            return path, f"**{seconds:.2f}s**", gr.update(maximum=round(seconds, 2), value=value)

        def on_trim_changed(value, selected, paths, trims):
            if not selected:
                return trims or {}, ""
            trims = dict(trims or {})
            lookup = {os.path.basename(str(p)): str(p) for p in (paths or []) if p}
            seconds = storage.video_duration_seconds(lookup.get(selected, "")) or 0.0
            if RangeSlider is not None and isinstance(value, (tuple, list)) and len(value) == 2:
                start, end = float(value[0]), float(value[1])
            else:                                  # start-only fallback
                start, end = float(value or 0.0), seconds
            if start <= 0.0 and (not seconds or end >= seconds - 0.05):
                trims.pop(selected, None)          # whole clip -> no trim recorded
            else:
                trims[selected] = (round(start, 2), round(end, 2))
            durations = {n: storage.video_duration_seconds(p) for n, p in lookup.items()}
            return trims, _trim_summary(trims, list(lookup), durations)

        def do_extract(model_type, name, mode, concept_type, ref_images, ref_videos, ref_audio,
                       remove_background_images_ref, ref_resolution, enc_resolution, pool_h, pool_w,
                       latent_frames, identity, multiplier, max_tokens, description, save,
                       use_clip_audio=False, trims=None, as_pictures=False, per_second=1.0,
                       merge_refs=False, audio_seconds=DEFAULT_AUDIO_SECONDS, order=None,
                       clean=False):
            if not model_type:
                return "Pick a MiniMax H3 Ref2VA model above first."
            image_paths = [f.name if hasattr(f, "name") else f for f in (ref_images or [])]
            video_paths = [f.name if hasattr(f, "name") else f for f in (ref_videos or [])]
            if not image_paths and not video_paths and not ref_audio:
                return "Add at least one reference image, video, or audio file."
            # Fork: audio with images/video becomes the mod's soundtrack. An
            # audio file too short to use is refused before anything runs.
            audio_len = _audio_file_seconds(ref_audio)
            if audio_len is not None and audio_len < MIN_AUDIO_SECONDS - 0.05:
                return (f"Extraction refused: the reference audio is only ~{audio_len:.1f}s long. "
                        f"H3 needs at least {MIN_AUDIO_SECONDS:g}s of audio to use it as a "
                        f"reference -- use a longer clip.")
            spec = {
                "name": name, "mode": mode, "concept_type": concept_type,
                "image_paths": image_paths, "video_paths": video_paths,
                "audio_path": ref_audio,
                "remove_background_images_ref": int(remove_background_images_ref or 0),
                "ref_resolution": int(ref_resolution), "enc_resolution": int(enc_resolution),
                "pool_h": int(pool_h), "pool_w": int(pool_w),
                "latent_frames": seconds_to_latent_frames(latent_frames), "identity": int(identity),
                "multiplier": int(multiplier), "max_tokens": int(max_tokens),
                "description": description or "", "save": bool(save),
                # fork
                "use_clip_audio": bool(use_clip_audio),
                "video_trims": {k: list(v) for k, v in (trims or {}).items()},
                "video_as_pictures": bool(as_pictures), "pictures_per_second": float(per_second or 1.0),
                "merge": bool(merge_refs),
                "audio_seconds": max(MIN_AUDIO_SECONDS, float(audio_seconds or DEFAULT_AUDIO_SECONDS)),
                "order": order_ui.merged_order(order, ref_images, ref_videos),   # fork.37
                "clean_audio": bool(clean),   # fork.41
            }
            log = {"lines": []}

            class ExtractCallbacks:
                def on_status(self, status):
                    if status:
                        log["lines"].append(str(status))

                def on_progress(self, update):
                    pass

            try:
                self._submit(api_session, model_type,
                            {"video_length": 107,
                             # Rides inside the single declared slot: Wan2GP keeps only the
                             # first 5 custom settings per model and MiniMax H3 now declares
                             # 4 of its own, so a second slot of ours would be dropped.
                             "custom_settings": {SETTING_GENERATE: json.dumps({EXTRACT_WRAPPER_KEY: spec})}},
                            ExtractCallbacks())
            except Exception as e:
                return f"Extraction task failed to run: {e!r}"
            lines = log["lines"]
            refused = [l for l in lines if "extraction refused" in l.lower()]   # fork
            if refused:
                return refused[-1]
            warnings = [l for l in lines if "⚠️" in l]   # fork: kept visible however many follow
            tail = "\n".join(warnings + [l for l in lines[-6:] if l not in warnings])
            ok = ("Done -- check the Library tab (Refresh) to see the saved mod." if any(
                    "saved" in l.lower() for l in log["lines"]) else
                  "Task finished, but no confirmation line was captured -- check the terminal log.")
            return (tail + ("\n" if tail else "") + ok) if tail else ok

        extract_btn.click(
            fn=do_extract,
            inputs=[model_dd, name, mode, concept_type, ref_images, ref_videos, ref_audio,
                   remove_background_images_ref, ref_resolution, enc_resolution, pool_h, pool_w,
                   latent_frames, identity, multiplier, max_tokens, description, save,
                   use_clip_audio, trim_state, video_as_pictures, pictures_per_second, merge,
                   soundtrack_seconds, source_order, clean_audio],
            outputs=[extract_status],
            queue=False,
        )

        # Fork: picture-stack estimate, and the video preview / trim.
        estimate_inputs = [video_as_pictures, pictures_per_second, ref_videos, ref_images,
                           trim_state, latent_frames, mode, ref_resolution, pool_h, pool_w,
                           max_tokens, merge]
        gr.on(triggers=[w.change for w in estimate_inputs], fn=update_stills_estimate,
              inputs=estimate_inputs, outputs=[stills_estimate], queue=False)
        video_as_pictures.change(fn=lambda on: gr.update(visible=bool(on)),
                                 inputs=[video_as_pictures], outputs=[pictures_per_second], queue=False)
        ref_videos.change(fn=on_videos_uploaded, inputs=[ref_videos, trim_state],
                          outputs=[video_pick, video_preview, video_duration, trim_state, trim_status],
                          queue=False)
        video_pick.change(fn=on_video_picked, inputs=[video_pick, ref_videos, trim_state],
                          outputs=[video_preview, video_duration, video_trim], queue=False)
        video_trim.change(fn=on_trim_changed, inputs=[video_trim, video_pick, ref_videos, trim_state],
                          outputs=[trim_state, trim_status], queue=False)

    # ── Library ─────────────────────────────────────────────────────────

    def _build_library_section(self, api_session=None, model_dd=None):
        gr.Markdown("### Saved RefMods\nH3 mods are in `loras/refmods_plugin/minimax_h3/`, Flux mods in "
                    "`loras/refmods_plugin/flux2/`; both can have subfolders. Files are sorted into the "
                    "right one when Wan2GP starts.")
        with gr.Row():
            library_folder_dd = gr.Dropdown(choices=_library_folder_choices(), value=ALL_FOLDERS_CHOICE,
                                            label="Folder", scale=3,
                                            info="Show one subfolder only.")
            refresh_btn = gr.Button("Refresh", scale=1)
        table = gr.Dataframe(
            headers=["model", "name", "kind", "mode", "tokens", "prompt view", "size (MB)",
                     "concept type", "description"],
            value=_library_rows(), interactive=False, wrap=True)
        with gr.Row():
            delete_name = gr.Dropdown(label="Mod to delete", choices=_qualified_choices(), value=None,
                                      scale=2)
            delete_btn = gr.Button("Delete", variant="stop")
        delete_status = gr.Textbox(label="", interactive=False, show_label=False)
        gr.Markdown(
            "Re-check every mod's image/video kind (fixes old multi-image mods).")
        with gr.Row():
            fix_btn = gr.Button("Fix classification (image vs video)")
            fix_status = gr.Markdown("")

        gr.Markdown(
            "**Rename, move or edit a description.** A path (`characters/tanya`) moves the mod; a leading `/` moves it to the main folder.")
        with gr.Row():
            edit_name_dd = gr.Dropdown(label="Mod to edit", choices=_qualified_choices(), scale=2)
            edit_load_btn = gr.Button("Load")
        with gr.Row():
            edit_name_field = gr.Textbox(label="Name (optionally with a folder path)")
            edit_description_field = gr.Textbox(label="Description", lines=2)
        edit_save_btn = gr.Button("Save changes", variant="primary")
        edit_status = gr.Textbox(label="", interactive=False, show_label=False)

        # ── wiring (all components above already exist) ──
        def _names_keeping(folder, current):
            """Mod names for the chosen folder, with the currently-selected
            one force-included even if it lives elsewhere -- otherwise
            switching folders would leave a stale value Gradio rejects."""
            names = _qualified_choices(folder)
            if current and current not in [v for _, v in names]:
                model, name = _split_qualified(current)
                names = names + [(f"{LIB_FLUX if model == 'flux' else LIB_H3} · {name}", current)]
            return names

        def do_refresh(folder, current):
            return (_library_rows(folder),
                    _safe_choice_update(_library_folder_choices(), folder or ALL_FOLDERS_CHOICE),
                    gr.update(choices=_names_keeping(folder, current), value=current or None))

        refresh_btn.click(fn=do_refresh, inputs=[library_folder_dd, edit_name_dd],
                          outputs=[table, library_folder_dd, edit_name_dd], queue=False)
        library_folder_dd.change(fn=do_refresh, inputs=[library_folder_dd, edit_name_dd],
                                 outputs=[table, library_folder_dd, edit_name_dd], queue=False)

        def do_delete(value, folder):
            if not value:
                return (_library_rows(folder), gr.update(), gr.update(), "Pick a mod first.")
            model, name = _split_qualified(value)
            ok = flux.delete_mod(name) if model == "flux" else storage.delete_refmod(name)
            label = f"{LIB_FLUX if model == 'flux' else LIB_H3} mod '{name}'"
            choices = _qualified_choices(folder)
            return (_library_rows(folder), gr.update(choices=choices, value=None),
                    gr.update(choices=choices, value=None),
                    (f"Deleted the {label}." if ok else f"No {label} found."))

        delete_btn.click(fn=do_delete, inputs=[delete_name, library_folder_dd],
                         outputs=[table, edit_name_dd, delete_name, delete_status], queue=False)

        def do_fix(folder):
            fixed, checked = storage.reclassify_all_mods()
            msg = (f"Checked {checked} mod(s), fixed {fixed}." if fixed else
                  f"Checked {checked} mod(s), all already correctly classified.")
            return _library_rows(folder), gr.update(choices=_qualified_choices(folder)), msg

        fix_btn.click(fn=do_fix, inputs=[library_folder_dd],
                      outputs=[table, edit_name_dd, fix_status], queue=False)

        def do_load_for_edit(value):
            if not value:
                return "", "", "Pick a mod first."
            model, name = _split_qualified(value)
            meta = (flux.read_meta(name) if model == "flux"
                    else storage.read_refmod_meta(storage.mod_path(name)))
            if meta is None:
                return "", "", f"No mod named '{name}' found."
            # Show the full folder-relative path, not the metadata's bare
            # "name" field, so editing it can move the mod (and so a plain
            # save round-trips without silently relocating anything).
            return name, meta.get("description", "") or "", ""

        edit_load_btn.click(fn=do_load_for_edit, inputs=[edit_name_dd],
                            outputs=[edit_name_field, edit_description_field, edit_status], queue=False)

        def do_save_edit(old_value, new_name, new_description, folder):
            if not old_value:
                return _library_rows(folder), gr.update(), gr.update(), "Pick a mod first (use Load)."
            model, old_name = _split_qualified(old_value)
            typed = str(new_name or "").strip().replace("\\", "/")
            head, _, rest = typed.lstrip("/").partition("/")
            if rest and head in (LIB_H3, LIB_FLUX):   # "Flux/people/anna", as the folder picker reads
                if (head == LIB_FLUX) != (model == "flux"):
                    return (_library_rows(folder), gr.update(), gr.update(),
                            "H3 and Flux mods stay in their own folders (use Convert to Flux).")
                new_name = "/" + rest
            try:
                if model == "flux":
                    final_name = flux.rename_and_update(old_name, new_name=new_name,
                                                        description=new_description)
                else:
                    final_name = storage.rename_and_update_mod(old_name, new_name=new_name,
                                                                new_description=new_description)
            except Exception as e:
                return _library_rows(folder), gr.update(), gr.update(), f"Could not save: {e}"
            msg = f"Saved as '{final_name}'." if final_name != old_name else "Saved."
            names = _qualified_choices(folder)
            final_value = f"{model}:{final_name}"
            return (_library_rows(folder), _safe_choice_update(_library_folder_choices(), folder),
                    gr.update(choices=names, value=final_value if final_value in [v for _, v in names] else None),
                    msg)

        edit_save_btn.click(fn=do_save_edit,
                            inputs=[edit_name_dd, edit_name_field, edit_description_field, library_folder_dd],
                            outputs=[table, library_folder_dd, edit_name_dd, edit_status], queue=False)

        gr.Markdown(
            "**Prompt hint.** Builds a prompt string from mods' descriptions, which the model never reads on its own.")
        with gr.Row():
            hint_names = gr.Textbox(label="Mod name(s), comma-separated, in the order you want them combined",
                                    scale=2)
            hint_btn = gr.Button("Build Prompt Hint")
        hint_output = gr.Textbox(label="Prompt hint -- copy this into your prompt", interactive=False, lines=2)

        def do_build_hint(names_str):
            names = [n.strip() for n in (names_str or "").split(",") if n.strip()]
            if not names:
                return "Enter at least one mod name."
            metas = []
            for n in names:
                meta = storage.read_refmod_meta(storage.mod_path(n))
                if meta is None:
                    return f"No mod named '{n}' found."
                metas.append(meta)
            hint = core.build_prompt_hint(metas)
            return hint or "(none of the selected mod(s) have a description set -- nothing to hint)"

        hint_btn.click(fn=do_build_hint, inputs=[hint_names], outputs=[hint_output], queue=False)

        # ── Fork: folders, soundtracks and encoder frames ──
        with gr.Accordion("Folders and moving", open=False):
            gr.Markdown("Moves keep everything in the mod and never overwrite.")
            with gr.Row():
                new_folder_name = gr.Textbox(label="New folder", scale=2,
                                             placeholder="H3/characters or Flux/people (/ to nest)")
                create_folder_btn = gr.Button("Create folder")
            with gr.Row():
                move_dd = gr.Dropdown(label="Mods to move", choices=_qualified_choices(),
                                      value=[], multiselect=True, scale=2)
                move_target = gr.Dropdown(label="Destination", choices=_library_destinations(),
                                          value=LIB_H3, scale=1)
                move_btn = gr.Button("Move", variant="primary")
            with gr.Row():
                delete_folder_dd = gr.Dropdown(label="Delete an empty folder",
                                               choices=_library_subfolders(), value=None, scale=2)
                delete_folder_btn = gr.Button("Delete folder", variant="stop")
            folder_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Soundtrack: add, replace or remove", open=False):
            gr.Markdown("Adds a soundtrack as its own `<Audio N>`. Runs a short task on the model above.")
            with gr.Row():
                audio_target_dd = gr.Dropdown(label="Mods to update", choices=_library_mod_names(),
                                              value=[], multiselect=True, scale=2)
                attach_btn = gr.Button("Apply soundtrack", variant="primary")
            with gr.Row():
                audio_file = gr.File(label="Audio or video file (a video's soundtrack is used)",
                                     file_types=["audio", "video"], type="filepath", scale=2)
                audio_seconds = gr.Slider(MIN_AUDIO_SECONDS, 15.0, value=DEFAULT_AUDIO_SECONDS,
                                          step=0.5, label="Seconds to use", scale=1,
                                          info=f"At least {MIN_AUDIO_SECONDS:g}s; shorter is refused.")
                with gr.Column(scale=1):
                    audio_clean = gr.Checkbox(False, label="Remove background music / noise",   # fork.41
                                              info=CLEAN_AUDIO_INFO)
                    audio_remove = gr.Checkbox(False, label="Remove existing soundtrack instead")
            audio_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Clean audio", open=False):   # fork.41
            gr.Markdown("Removes background music / noise from audio mods and soundtracks you already have. "
                        "Voices are kept. Runs a short task on the model above.")
            with gr.Row():
                clean_dd = gr.Dropdown(label="Mods with audio", choices=_audio_mod_names(), value=[],
                                       multiselect=True, scale=2)
                clean_copy = gr.Checkbox(True, label="Save as copy",
                                         info="Saves name_clean and keeps the original. Off: overwrites.")
                clean_btn = gr.Button("Clean audio", variant="primary")
            clean_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Encoder frames", open=False):
            gr.Markdown("Stores prompt-view frames in mods that lack them, so they aren't decoded every session.")
            with gr.Row():
                frames_target_dd = gr.Dropdown(label="Mods to update", choices=_library_mod_names(),
                                               value=[], multiselect=True, scale=2)
                frames_btn = gr.Button("Store for selected")
                frames_all_btn = gr.Button("Store for every mod missing them", variant="primary")
            frames_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Convert to Flux", open=False):   # fork.32
            gr.Markdown("Decodes H3 picture mods with the H3 model above, then encodes the pictures with "
                        "a Flux model. Each picture becomes one Flux reference.")
            with gr.Row():
                convert_dd = gr.Dropdown(label="H3 picture mods", choices=_convertible(), value=[],
                                         multiselect=True, scale=3)
                flux_choices = flux_ui._model_choices(api_session)
                convert_model = gr.Dropdown(label="Flux model", choices=flux_choices, scale=2,
                                            value=flux_choices[0][1] if flux_choices else None)
                convert_budget = gr.Dropdown(label="Size per picture", choices=flux_ui.BUDGET_UI,
                                             value=flux.DEFAULT_BUDGET, scale=2)
            convert_layout = gr.Radio(label="Layout", choices=flux_ui.LAYOUT_UI, value=flux.LAYOUT_SEPARATE,
                                      info=flux_ui.LAYOUT_INFO)
            convert_btn = gr.Button("Convert", variant="primary")
            convert_status = gr.Textbox(label="", interactive=False, show_label=False, lines=2)

        def run_job(spec, model_type):
            """Run a Library job as a short task on the chosen model (it needs
            the model's VAEs). Returns its last status line."""
            if api_session is None or not model_type:
                return ("Pick a model at the top of this tab first -- this needs the model's "
                        "VAEs, so it runs as a task.")
            log = {"lines": []}

            class JobCallbacks:
                def on_status(self, message):
                    if message:
                        log["lines"].append(str(message))

                def on_progress(self, update):
                    pass

            try:
                self._submit(api_session, model_type,
                             {"video_length": 107,
                              "custom_settings": {SETTING_GENERATE: json.dumps({EXTRACT_WRAPPER_KEY: spec})}},
                             JobCallbacks())
            except Exception as e:
                return f"The task failed to run: {e!r}"
            done = [l for l in log["lines"] if "finished" in l.lower() or "nothing to do" in l.lower()]
            return done[-1] if done else (log["lines"][-1] if log["lines"] else "Finished -- see the console.")

        def picked(names):
            return [n for n in (names if isinstance(names, list) else [names]) if n]

        fork_lists = [move_dd, move_target, delete_folder_dd, audio_target_dd, frames_target_dd,
                      delete_name, convert_dd, clean_dd]

        def fork_list_updates(folder):
            names, qualified = _library_mod_names(folder), _qualified_choices(folder)
            return [gr.update(choices=qualified, value=[]),
                    gr.update(choices=_library_destinations(), value=LIB_H3),
                    gr.update(choices=_library_subfolders(), value=None),
                    gr.update(choices=names, value=[]), gr.update(choices=names, value=[]),
                    gr.update(choices=qualified, value=None),
                    gr.update(choices=_convertible(folder), value=[]),
                    gr.update(choices=_audio_mod_names(folder), value=[])]

        def after_change(folder, message):
            folder = folder if folder in _library_folder_choices() else ALL_FOLDERS_CHOICE
            return [_library_rows(folder), gr.update(choices=_library_folder_choices(), value=folder),
                    *fork_list_updates(folder), message]

        change_outputs = [table, library_folder_dd] + fork_lists

        def do_create_folder(new_name, folder):
            raw = str(new_name or "").strip().strip("/")
            model, sub = _split_library_folder(raw) if raw.partition("/")[0] in (LIB_H3, LIB_FLUX) \
                else ("h3", raw)
            if not sub:
                return after_change(folder, "Enter a folder name after H3/ or Flux/.")
            try:
                made = flux.create_folder(sub) if model == "flux" else storage.create_mod_folder(sub)
                return after_change(folder, f"Created '{LIB_FLUX if model == 'flux' else LIB_H3}/{made}'.")
            except Exception as e:
                return after_change(folder, f"Could not create the folder: {e}")

        def do_move_mods(values, destination, folder):
            values = picked(values)
            if not values:
                return after_change(folder, "Pick at least one mod to move.")
            dest_model, dest_sub = _split_library_folder(destination or LIB_H3)
            dest_sub = dest_sub or ""
            same = [_split_qualified(v)[1] for v in values if _split_qualified(v)[0] == dest_model]
            other = [_split_qualified(v)[1] for v in values if _split_qualified(v)[0] != dest_model]
            moved, skipped = (flux.move_mods(same, dest_sub) if dest_model == "flux"
                              else storage.move_mods(same, dest_sub))
            where = f"{LIB_FLUX if dest_model == 'flux' else LIB_H3}" + (f"/{dest_sub}" if dest_sub else "")
            parts = [f"Moved {len(moved)} mod(s) to {where}."] if moved else []
            if other:
                parts.append(f"Not moved: {', '.join(other)} -- H3 and Flux mods stay in their own "
                             f"folders (use Convert to Flux).")
            if skipped:
                parts.append("Skipped: " + "; ".join(f"'{n}' ({why})" for n, why in skipped))
            return after_change(folder, " ".join(parts) or "Nothing to do.")

        def do_delete_folder(value, folder):
            if not value:
                return after_change(folder, "Pick a folder to delete.")
            model, sub = _split_library_folder(value)
            try:
                gone = flux.delete_folder(sub) if model == "flux" else storage.delete_mod_folder(sub)
                return after_change(folder, f"Deleted '{value}'." if gone else "Nothing to do.")
            except Exception as e:
                return after_change(folder, f"Could not delete the folder: {e}")

        create_folder_btn.click(fn=do_create_folder, inputs=[new_folder_name, library_folder_dd],
                                outputs=change_outputs + [folder_status], queue=False)
        move_btn.click(fn=do_move_mods, inputs=[move_dd, move_target, library_folder_dd],
                       outputs=change_outputs + [folder_status], queue=False)
        delete_folder_btn.click(fn=do_delete_folder, inputs=[delete_folder_dd, library_folder_dd],
                                outputs=change_outputs + [folder_status], queue=False)

        def do_attach_audio(names, audio_path, seconds, remove, folder, clean=False, model_type=None):
            names = picked(names)
            if not names:
                return gr.update(), "Pick at least one mod first."
            if not audio_path and not remove:
                return gr.update(), "Choose an audio file, or tick 'Remove existing soundtrack'."
            message = run_job({"op": "attach_audio", "names": names, "audio_path": audio_path,
                               "seconds": float(seconds), "remove": bool(remove),
                               "clean": bool(clean) and not remove}, model_type)
            return gr.update(choices=_library_mod_names(folder), value=[]), message

        def do_store_frames(names, all_missing, folder, model_type=None):
            names = picked(names)
            if not all_missing and not names:
                return _library_rows(folder), gr.update(), "Pick at least one mod first."
            if all_missing and not storage.list_refmods_missing_frames():
                return _library_rows(folder), gr.update(), "Every mod already has encoder frames."
            message = run_job({"op": "store_frames", "names": names, "all_missing": bool(all_missing)},
                              model_type)
            return _library_rows(folder), gr.update(choices=_library_mod_names(folder), value=[]), message

        model_input = [model_dd] if model_dd is not None else []
        attach_btn.click(fn=do_attach_audio,
                         inputs=[audio_target_dd, audio_file, audio_seconds, audio_remove,
                                 library_folder_dd, audio_clean] + model_input,
                         outputs=[audio_target_dd, audio_status], queue=False)
        def do_clean_audio(names, as_copy, folder, model_type=None):   # fork.41
            names = picked(names)
            if not names:
                return after_change(folder, "Pick at least one mod with audio.")
            message = run_job({"op": "clean_audio", "names": names, "as_copy": bool(as_copy)}, model_type)
            return after_change(folder, message)

        clean_btn.click(fn=do_clean_audio, inputs=[clean_dd, clean_copy, library_folder_dd] + model_input,
                        outputs=change_outputs + [clean_status], queue=False)
        frames_btn.click(fn=do_store_frames,
                         inputs=[frames_target_dd, gr.State(False), library_folder_dd] + model_input,
                         outputs=[table, frames_target_dd, frames_status], queue=False)
        frames_all_btn.click(fn=do_store_frames,
                             inputs=[frames_target_dd, gr.State(True), library_folder_dd] + model_input,
                             outputs=[table, frames_target_dd, frames_status], queue=False)
        def do_convert(names, flux_model, budget, layout, folder, model_type=None):   # fork.32
            names = picked(names)
            if not names:
                return after_change(folder, "Pick at least one H3 picture mod.")
            if not model_type:
                return after_change(folder, "Pick the H3 model at the top of this tab (it decodes the mods).")
            if not flux_model:
                return after_change(folder, "Pick a Flux model (it encodes the pictures).")
            lines, decoded = [], []
            for name in names:   # every decode first, so each model loads once
                message = run_job({"op": "decompile", "name": name}, model_type)
                pictures = storage.list_decompiled(name)[0]
                if pictures:
                    decoded.append((name, pictures))
                else:
                    lines.append(f"'{name}': could not decode it ({message})")
            for name, pictures in decoded:
                meta = storage.read_refmod_meta(storage.mod_path(name)) or {}
                target = flux.free_name(name)
                ok, message = flux_ui.submit_extract(
                    self, api_session, flux_model,
                    {"name": target, "images": pictures, "budget": int(budget or flux.DEFAULT_BUDGET),
                     "layout": layout or flux.LAYOUT_SEPARATE,
                     "concept_type": meta.get("concept_type") or "identity",
                     "description": meta.get("description") or "", "converted_from": name})
                soft = (" Made in training mode, so its pictures are soft." if meta.get("mode") == "training"
                        else "")
                lines.append(f"'{name}' -> Flux/{target}: {message}{soft}" if ok
                             else f"'{name}': {message}")
            return after_change(folder, "\n".join(lines))

        # Deleting or renaming changes the names every list offers.
        delete_btn.click(fn=fork_list_updates, inputs=[library_folder_dd], outputs=fork_lists, queue=False)
        edit_save_btn.click(fn=fork_list_updates, inputs=[library_folder_dd], outputs=fork_lists, queue=False)
        convert_btn.click(fn=do_convert,
                          inputs=[convert_dd, convert_model, convert_budget, convert_layout,
                                  library_folder_dd] + model_input,
                          outputs=change_outputs + [convert_status])
        # Keep these lists in step whenever the Library is refreshed or browsed.
        refresh_btn.click(fn=fork_list_updates, inputs=[library_folder_dd], outputs=fork_lists, queue=False)
        library_folder_dd.change(fn=fork_list_updates, inputs=[library_folder_dd], outputs=fork_lists,
                                 queue=False)
        return table

    # ── Generate ────────────────────────────────────────────────────────

    def _build_decompile_section(self, api_session, model_dd):   # fork
        gr.Markdown("### Decompile a RefMod\nRebuilds a mod's pictures, video and audio as files: reconstructions, not the originals.")
        with gr.Row():
            mod_dd = gr.Dropdown(label="Mod", choices=_library_mod_names(), value=None, scale=4)
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0, min_width=110)
        details = gr.Markdown("*Pick a mod to see what's inside it.*")
        with gr.Accordion("Quick look (stored encoder frames, no decoding)", open=True):
            quick_note = gr.Markdown("")
            quick_gallery = gr.Gallery(label="Encoder frames", columns=6, height="auto",
                                       show_label=False, visible=False)
        decompile_btn = gr.Button("Decompile (full decode)", variant="primary")
        decompile_status = gr.Textbox(label="", interactive=False, show_label=False)
        out_gallery = gr.Gallery(label="Pictures", columns=4, height="auto", visible=False)
        out_video = gr.Video(label="Video", visible=False)
        out_audio = gr.Audio(label="Audio", type="filepath", visible=False)

        def show_outputs(name):
            pictures, videos, audios = storage.list_decompiled(name) if name else ([], [], [])
            return (gr.update(value=pictures or None, visible=bool(pictures)),
                    gr.update(value=videos[0] if videos else None, visible=bool(videos)),
                    gr.update(value=audios[0] if audios else None, visible=bool(audios)))

        def on_pick(name):
            if not name:
                return ("*Pick a mod to see what's inside it.*", "", gr.update(visible=False),
                        "", *show_outputs(None))
            try:
                mod = storage.load_refmod(name)
            except Exception as e:
                return (f"Could not read this mod: {e}", "", gr.update(visible=False), "",
                        *show_outputs(None))
            if mod.has_encoder_frames():
                frames = (mod.enc_frames * 255).round().clamp(0, 255).byte().numpy()
                times = list(mod.enc_times) or list(range(len(frames)))
                label = ("picture {}" if mod.enc_layout == core.ENC_LAYOUT_STILLS
                         else "{:.1f}s")
                items = [(frames[i], label.format(i + 1 if "picture" in label else times[i]))
                         for i in range(len(frames))]
                note = (f"{len(items)} stored frame(s): what the text encoder is shown for this "
                        f"mod, at a reduced size.")
                gallery = gr.update(value=items, visible=True)
            else:
                note = ("*No stored encoder frames"
                        + (" -- audio mods have none.*" if mod.kind == "audio"
                           else ". Decompile below, or store them from the Library.*"))
                gallery = gr.update(value=None, visible=False)
            already = any(storage.list_decompiled(name))
            return (_decompile_details(mod), note, gallery,
                    "Showing the files from an earlier decompile." if already else "",
                    *show_outputs(name))

        def do_decompile(name, model_type=None):
            if not name:
                return ("Pick a mod first.", *show_outputs(None))
            if api_session is None or not model_type:
                return ("Pick a model at the top of this tab first -- decoding needs the "
                        "model's VAEs, so this runs as a task.", *show_outputs(name))
            spec = {"op": "decompile", "name": name}
            log = {"lines": []}

            class DecompileCallbacks:
                def on_status(self, message):
                    log["lines"].append(str(message))
                def on_error(self, message):
                    log["lines"].append(f"error: {message}")
                def on_progress(self, update):
                    pass

            try:
                self._submit(api_session, model_type,
                             {"video_length": 107,
                              "custom_settings": {SETTING_GENERATE: json.dumps({EXTRACT_WRAPPER_KEY: spec})}},
                             DecompileCallbacks())
            except Exception as e:
                return (f"Decompile failed to run: {e!r}", *show_outputs(name))
            done = [l for l in log["lines"] if "Decompile" in l]
            return ((done[-1] if done else
                     (log["lines"][-1] if log["lines"] else "Finished -- see the console.")),
                    *show_outputs(name))

        pick_outputs = [details, quick_note, quick_gallery, decompile_status,
                        out_gallery, out_video, out_audio]
        mod_dd.change(fn=on_pick, inputs=[mod_dd], outputs=pick_outputs, queue=False)
        refresh_btn.click(fn=lambda current: _safe_choice_update(_library_mod_names(), current),
                          inputs=[mod_dd], outputs=[mod_dd], queue=False)
        decompile_btn.click(fn=do_decompile,
                            inputs=[mod_dd] + ([model_dd] if model_dd is not None else []),
                            outputs=[decompile_status, out_gallery, out_video, out_audio],
                            queue=False)

    def _build_generate_section(self, api_session, model_dd):
        gr.Markdown(
            "**Tip:** the **Media Generator** tab has every option, and RefMods are there too under *Refmods: MiniMax H3*.")
        gr.Markdown("### Generate with RefMods\nA simpler form. Other settings come from the Media Generator (*Sync from the main form*) or the model's defaults.")
        synced = gr.State({})
        with gr.Row():
            sync_btn = gr.Button("⟲ Sync from the main form (copies every current setting)")
            sync_status = gr.Markdown("*Not synced yet -- using this model's factory defaults.*")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=4, scale=3)
            with gr.Column(scale=1):
                resolution = gr.Textbox(label="Resolution (WxH)", value="1280x720")
                video_length = gr.Slider(107, 737, value=124, step=17, label="Number of frames")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                repeat_generation = gr.Slider(1, 25, value=1, step=1, label="Videos per prompt")

        with gr.Row():
            num_inference_steps = gr.Slider(1, 100, value=20, step=1, label="Number of inference steps")
            flow_shift = gr.Slider(0.0, 25.0, value=12.0, step=0.1, label="Flow shift")
            sample_solver = gr.Dropdown(label="Sampler solver / scheduler", choices=SAMPLE_SOLVER_CHOICES,
                                        value="euler")
            image_refs_relative_size = gr.Slider(50, 400, value=100, step=1,
                                                 label="Reference image budget (% of output pixels)",
                                                 info="Higher = more reference detail kept, slower.")

        mod_rows, panel_controls = self._build_mod_picker_rows_full()   # fork: the full panel

        with gr.Accordion("Advanced Mode", open=False):
            with gr.Tab("LoRAs"):
                lora_choices = gr.State([])
                activated_loras = gr.Dropdown(label="Activated LoRAs", choices=[], multiselect=True, value=[])
                loras_multipliers = gr.Textbox(
                    label="LoRAs multipliers (1.0 by default) separated by spaces or line breaks", value="")
                refresh_loras_btn = gr.Button("Refresh LoRA list")

            with gr.Tab("Steps Skipping"):
                skip_steps_cache_type = gr.Dropdown(label="Skip steps cache type", choices=STEPS_SKIPPING_CHOICES,
                                                    value="")
                skip_steps_multiplier = gr.Dropdown(label="First Block Cache threshold",
                                                    choices=FIRST_BLOCK_CACHE_STRENGTHS, value=0.08,
                                                    visible=False)
                skip_steps_start_step_perc = gr.Slider(0, 100, value=25, step=1,
                                                       label="Skip steps starting moment (% of generation)")
                skip_steps_cache_type.change(
                    fn=lambda t: gr.update(visible=t == "first_block"),
                    inputs=[skip_steps_cache_type], outputs=[skip_steps_multiplier], queue=False)

            with gr.Tab("Sliding Window"):
                gr.Markdown("Used automatically once **Number of frames** needs more than one window.")
                sliding_window_size = gr.Slider(124, 481, value=362, step=17, label="Sliding window size (frames)")
                sliding_window_overlap = gr.Slider(1, 120, value=18, step=17, label="Sliding window overlap (frames)")

            with gr.Tab("Attention"):
                override_attention = gr.Dropdown(
                    label="Override attention mode", value="",
                    choices=[("Auto (recommended)", ""), ("Sol-Attn (sparse)", "sol")],
                    allow_custom_value=True,
                    info="Leave on Auto unless you know a specific backend is installed.")
                attention_sparsity = gr.Slider(
                    0.0, 4.0, value=1.3, step=0.05, label="Sol-Attn Start Tau",
                    info="Sol-Attn only. Higher = sparser, faster.")

        generate_btn = gr.Button("Generate", variant="primary")
        output_video = gr.Video(label="Output")
        gen_status = gr.Textbox(label="Status", interactive=False, lines=3)

        def do_sync(state, model_type):
            settings = dict(self.get_current_model_settings(state) or {})
            note = ("*Synced from the main form.*" if settings else
                   "*Main form has no settings for the current model yet -- using factory defaults.*")
            return (settings, note, settings.get("prompt", ""), settings.get("resolution", "1280x720"),
                   settings.get("video_length", 124), settings.get("seed", -1),
                   settings.get("repeat_generation", 1), settings.get("num_inference_steps", 20),
                   settings.get("flow_shift", 12.0), settings.get("sample_solver", "euler"),
                   settings.get("image_refs_relative_size", 100),
                   settings.get("activated_loras", []), settings.get("loras_multipliers", ""),
                   settings.get("skip_steps_cache_type", ""), settings.get("skip_steps_multiplier", 0.08),
                   settings.get("skip_steps_start_step_perc", 25),
                   settings.get("sliding_window_size", 362), settings.get("sliding_window_overlap", 18),
                   settings.get("override_attention", ""), settings.get("attention_sparsity", 1.3))

        def do_refresh_loras(model_type):
            choices = _lora_choices(api_session, model_type)
            return choices, gr.update(choices=choices)

        def do_generate(model_type, synced_settings, prompt, resolution, video_length, seed, repeat_generation,
                        num_inference_steps, flow_shift, sample_solver, image_refs_relative_size,
                        activated_loras, loras_multipliers,
                        skip_steps_cache_type, skip_steps_multiplier, skip_steps_start_step_perc,
                        sliding_window_size, sliding_window_overlap,
                        override_attention, attention_sparsity, *row_values):
            if not model_type:
                raise gr.Error("Pick a MiniMax H3 Ref2VA model above first.")
            n_rows = IMAGE_ROWS + VIDEO_ROWS + AUDIO_ROWS
            rows = _rows_payload(row_values[:n_rows * ROW_WIDTH])
            state_payload = {"rows": rows, "retention": 1.0, "scramble_seed": -1, "curve": None,
                             **_stack_payload(*row_values[n_rows * ROW_WIDTH:
                                                          n_rows * ROW_WIDTH + STACK_CONTROLS])}

            overrides = dict(synced_settings or {})
            overrides.update({
                "prompt": prompt, "resolution": resolution, "video_length": int(video_length),
                "seed": int(seed), "repeat_generation": int(repeat_generation),
                "num_inference_steps": int(num_inference_steps), "flow_shift": float(flow_shift),
                "sample_solver": sample_solver, "image_refs_relative_size": int(image_refs_relative_size),
                "activated_loras": list(activated_loras or []), "loras_multipliers": loras_multipliers or "",
                "skip_steps_cache_type": skip_steps_cache_type, "skip_steps_multiplier": float(skip_steps_multiplier),
                "skip_steps_start_step_perc": int(skip_steps_start_step_perc),
                "sliding_window_size": int(sliding_window_size), "sliding_window_overlap": int(sliding_window_overlap),
                "override_attention": override_attention, "attention_sparsity": float(attention_sparsity),
                "custom_settings": {SETTING_GENERATE: json.dumps(state_payload)},
            })

            log = {"lines": []}

            class GenCallbacks:
                def on_status(self, status):
                    if status:
                        log["lines"].append(str(status))

                def on_progress(self, update):
                    pass

            try:
                result = self._submit(api_session, model_type, overrides, GenCallbacks())
            except Exception as e:
                return gr.update(), f"Generation task failed to run: {e!r}"
            tail = "\n".join(log["lines"][-6:])
            if result.success and result.generated_files:
                return result.generated_files[0], tail or "Done."
            if result.cancelled:
                return gr.update(), ((tail + "\nCancelled.") if tail else "Cancelled.")
            errors = list(result.errors or [])
            return gr.update(), (tail + "\n" if tail else "") + str(errors[0] if errors else "No output produced.")

        sync_btn.click(
            fn=do_sync, inputs=[self.state, model_dd],
            outputs=[synced, sync_status, prompt, resolution, video_length, seed, repeat_generation,
                    num_inference_steps, flow_shift, sample_solver, image_refs_relative_size,
                    activated_loras, loras_multipliers,
                    skip_steps_cache_type, skip_steps_multiplier, skip_steps_start_step_perc,
                    sliding_window_size, sliding_window_overlap, override_attention, attention_sparsity],
            queue=False,
        )
        refresh_loras_btn.click(fn=do_refresh_loras, inputs=[model_dd], outputs=[lora_choices, activated_loras],
                               queue=False)
        flat_rows = [c for row in mod_rows for c in row] + list(panel_controls)
        generate_btn.click(
            fn=do_generate,
            inputs=[model_dd, synced, prompt, resolution, video_length, seed, repeat_generation,
                   num_inference_steps, flow_shift, sample_solver, image_refs_relative_size,
                   activated_loras, loras_multipliers,
                   skip_steps_cache_type, skip_steps_multiplier, skip_steps_start_step_perc,
                   sliding_window_size, sliding_window_overlap, override_attention, attention_sparsity] + flat_rows,
            outputs=[output_video, gen_status],
            queue=False,
        )

    # ── Tab assembly ────────────────────────────────────────────────────

    @_hidden_from_api
    def create_ui(self, api_session):
        with gr.Column() as root:
            if self._patch_error:
                gr.Markdown(f"⚠️ **RefMods could not hook into MiniMax H3**: {self._patch_error}")
            gr.Markdown(f"## {PlugIn_Name}\nReference mods for MiniMax H3 and Flux. See the README for details.")
            model_choices = _model_choices(api_session)
            model_dd = gr.Dropdown(
                label="MiniMax H3 Ref2VA model (for the H3 tabs and for decoding in Convert to Flux)",
                choices=model_choices, value=_default_model_choice(model_choices))
            with gr.Row():
                refresh_models_btn = gr.Button("Refresh model list", size="sm")
                diagnose_btn = gr.Button("Check setup for this model", size="sm")
            diagnose_status = gr.Markdown("")
            refresh_models_btn.click(fn=lambda: gr.update(choices=_model_choices(api_session)),
                                     outputs=[model_dd], queue=False)
            diagnose_btn.click(fn=lambda mt: _diagnose(api_session, mt, self._patch_error),
                              inputs=[model_dd], outputs=[diagnose_status], queue=False)

            with gr.Tabs():
                with gr.Tab("Extract (H3)"):
                    self._build_extract_section(api_session, model_dd)
                with gr.Tab("Extract (Flux)"):   # fork.31
                    if getattr(self, "_flux_error", None):
                        gr.Markdown(f"⚠️ **Flux RefMods are off**: {self._flux_error}")
                    flux_ui.build_tab(self, api_session)
                with gr.Tab("Library"):
                    self._build_library_section(api_session, model_dd)
                with gr.Tab("Decompile"):   # fork
                    self._build_decompile_section(api_session, model_dd)
                with gr.Tab("Generate"):
                    self._build_generate_section(api_session, model_dd)
        return root
