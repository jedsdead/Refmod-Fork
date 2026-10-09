"""
MiniMax H3 RefMods -- Wan2GP plugin.

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

import json
import math
import os

import gradio as gr
from PIL import Image

from shared.utils.plugins import WAN2GPPlugin

from . import core, encframes, storage
from .patches import (SETTING_COMBINED, SETTING_EXTRACT, SETTING_GENERATE, STASH_KEY,
                      install_patches, pack_refmod_setting, set_pending_extract,
                      install_get_model_settings_patch, install_prepare_inputs_dict_patch,
                      is_minimax_h3_ref2va, is_minimax_h3_refmod_capable,
                      DEFAULT_AUDIO_SECONDS, MIN_AUDIO_SECONDS)

PlugIn_Name = "MiniMax H3 RefMods"
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
try:
    from gradio_rangeslider import RangeSlider          # shipped with Wan2GP
except Exception:                                        # pragma: no cover
    RangeSlider = None

NONE_CHOICE = "(none)"
ROOT_FOLDER_CHOICE = "(top level)"

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
        missing = [] if SETTING_COMBINED in declared_ids else [SETTING_COMBINED]
        if not missing and len(declared_ids) > 5:
            lines.append(f"⚠️ This model declares {len(declared_ids)} custom settings; Wan2GP keeps only "
                         f"the first 5, so '{SETTING_COMBINED}' may still be dropped at task time.")
        if missing:
            lines.append(f"❌ This model does NOT declare {missing} under custom_settings -- RefMods will "
                         f"silently no-op for it (a real generation will run instead of an extraction, or "
                         f"without the mods applied). Check the terminal log for '[H3RefMod]' lines at "
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
                         "in the 'MiniMax H3 RefMods (inline)' accordion on the Media Generator page will "
                         "apply to that page's own Generate button.")
        else:
            lines.append("❌ Inline panel hook NOT active -- the inline accordion on the Media Generator "
                         "page will be visible but selections made there will have no effect. Use this "
                         "plugin's own 'Generate' tab instead, or check the terminal log for a "
                         "'[H3RefMod] prepare_inputs_dict' line explaining why.")
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

    updates = [_safe_choice_update(_folder_choices(), folder)]
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


def _trim_summary(trims, names):
    if not trims:
        return "*No trims set - every video is used from its start.*"
    parts = [f"`{n}` {trims[n][0]:g}-{trims[n][1]:g}s" for n in names if n in trims]
    return "Trimmed: " + ", ".join(parts) if parts else ""


# Values per picker row: mod, strength, "Send as one video", "Include audio",
# "Max size" (fork.19).
ROW_WIDTH = 5
# Panel-wide controls after the mod rows: picture mode, N, video mode, N,
# fit to output, mods in phase 2, shrink method.
STACK_CONTROLS = 7

# "Max size" slider: short edge in pixels, in steps of 32, down to this.
MAX_SIZE_FLOOR = 128
MAX_SIZE_STEP = 32
PHASE2_MODS_UI = [("Full", "full"), ("Fit to tile", "tile"), ("Off", "off")]
SHRINK_METHOD_UI = [("Re-encode (best quality)", "reencode"), ("Fast (resize the stored mod)", "fast")]


def _mod_short_edge(meta):
    """A picture or video mod's short edge in pixels (16 px per latent cell),
    or 0 when it has no size (audio, or unknown)."""
    if not meta or meta.get("kind") == "audio":
        return 0
    try:
        return min(int(meta.get("latent_h", 0)), int(meta.get("latent_w", 0))) * 16
    except (TypeError, ValueError):
        return 0


def _max_size_update(meta):
    """The "Max size" slider for a freshly picked mod: its range runs from the
    mod's own short edge ("full size", where it starts) down to
    MAX_SIZE_FLOOR, in MAX_SIZE_STEP steps. Hidden for audio mods and for mods
    already too small to shrink."""
    short = _mod_short_edge(meta)
    if short <= MAX_SIZE_FLOOR:
        return gr.update(visible=False, value=0)
    lowest = short - MAX_SIZE_STEP * ((short - MAX_SIZE_FLOOR) // MAX_SIZE_STEP)
    return gr.update(visible=True, minimum=lowest, maximum=short, value=short)


def _shrunk_size(meta, max_size):
    """(old w, old h, new w, new h) in pixels when the row's "Max size" shrinks
    the mod -- the same sizing patches.py uses -- else None."""
    short = _mod_short_edge(meta)
    try:
        cap = int(max_size or 0)
    except (TypeError, ValueError):
        return None
    if not short or cap < 32 or cap >= short:
        return None
    lat_h, lat_w = int(meta.get("latent_h", 0)), int(meta.get("latent_w", 0))
    scale = cap / float(short)
    new_w = max(32, int(lat_w * 16 * scale) // 32 * 32)
    new_h = max(32, int(lat_h * 16 * scale) // 32 * 32)
    if (new_w // 16) * (new_h // 16) >= lat_h * lat_w:
        return None
    return lat_w * 16, lat_h * 16, new_w, new_h


def _row_boxes_update(name, kind="image"):
    """The two per-row boxes after a row's mod changes, starting fresh for
    the newly picked mod (a tick made for the previous mod never carries
    over).

    "Send as one video" (image rows only) shows only while the row holds a
    mod made from several pictures -- a single picture is already one
    reference -- unticked. "Include audio" (image and video rows) shows only
    while the mod carries a soundtrack, ticked."""
    meta = _row_meta(name, 1.0)
    multi = (kind == "image" and bool(meta) and _is_picture_mod(meta)
             and int(meta.get("latent_t", 1) or 1) > 1)
    has_audio = bool(meta) and meta.get("kind") != "audio" and bool(meta.get("has_audio"))
    return (gr.update(visible=multi, value=False), gr.update(visible=has_audio, value=True),
            _max_size_update(meta))


def _cleared_row_updates():
    """gr.update()s returning one picker row to empty."""
    return [gr.update(value=NONE_CHOICE), gr.update(value=1.0),
            gr.update(value=False, visible=False), gr.update(value=True, visible=False),
            gr.update(value=0, visible=False)]


def _rows_payload(values):
    """The selection payload's "rows" from the pickers' flat values
    (ROW_WIDTH per row): one entry per row with a mod and strength > 0."""
    rows = []
    for i in range(0, len(values) - ROW_WIDTH + 1, ROW_WIDTH):
        name, strength, as_video, use_audio, max_size = values[i:i + ROW_WIDTH]
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            continue
        if name and name != NONE_CHOICE and strength > 0:
            row = {"mod": name, "strength": strength, "as_video": bool(as_video),
                   "use_audio": bool(use_audio)}
            # Only when it actually shrinks the mod: "full size" is left out.
            if _shrunk_size(_row_meta(name, strength), max_size):
                row["max_size"] = int(max_size)
            rows.append(row)
    return rows


def _normalize_rows(rows):
    """Picker rows as (name, strength, send-as-one-video, include-audio,
    max-size); rows with fewer values mean "not sent as one video", "audio
    included" and "full size"."""
    out = []
    for row in rows or []:
        row = tuple(row)
        out.append((row[0], row[1],
                    bool(row[2]) if len(row) > 2 else False,
                    bool(row[3]) if len(row) > 3 else True,
                    row[4] if len(row) > 4 else None))
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
                   fit_to_output=False, phase2_mods="full", shrink_method="reencode"):
    """The selection payload's text-encoder fields (see patches.py's
    _build_refmod_sentinels): how many pictures of a mod sent as one video,
    and how many frames of a video mod, the text encoder is shown. Whether a
    mod is sent as one video is per row ("as_video" in each row)."""
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
            # "Fit mods to the output size" (fork.18, off by default).
            "fit_to_output": bool(fit_to_output),
            # "Mods in phase 2" (fork.19): full / tile / off.
            "phase2_mods": phase2_mods if phase2_mods in ("full", "tile", "off") else "full",
            # "Shrink method" (fork.20): reencode / fast.
            "shrink_method": shrink_method if shrink_method in ("reencode", "fast") else "reencode"}


def _format_label_line(rows):
    """Which <Picture N> / <Video N> / <Audio N> each picked mod will get in
    window 1, in the order the pipeline assigns them (pictures, then videos,
    then audio, each in row order). Assumes no start image and no references
    of the generation's own, which are numbered ahead of the mods."""
    pictures, videos, audios = [], [], []    # (mod name, count)
    for name, strength, as_video, use_audio, _max_size in _normalize_rows(rows):
        meta = _row_meta(name, strength)
        if meta is None:
            continue
        kind = meta.get("kind", "image")
        latent_t = max(1, int(meta.get("latent_t", 1)))
        if _is_picture_mod(meta):
            if as_video and latent_t > 1:
                videos.append((name, 1))
            else:
                pictures.append((name, latent_t))
        elif kind == "video":
            videos.append((name, 1))
        else:
            audios.append((name, 1))
            continue
        if meta.get("has_audio") and use_audio:
            audios.append((name + " (soundtrack)", 1))

    def labels(items, word):
        out, n = [], 0
        for name, count in items:
            first, n = n + 1, n + count
            span = f"<{word} {first}>" if count == 1 else f"<{word} {first}>–<{word} {n}>"
            out.append(f"`{name}` → {span}")
        return out

    parts = labels(pictures, "Picture") + labels(videos, "Video") + labels(audios, "Audio")
    if not parts:
        return ""
    extra = ""
    if len(videos) > 3:
        extra = " (videos past the third have no label -- only 3 reference-video slots)"
    return ("**Prompt labels** (window 1, with no start image or other references): "
            + " · ".join(parts) + extra)


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
    sizes = []
    for name, strength, as_video, use_audio, max_size in _normalize_rows(rows):
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
        shrunk = _shrunk_size(meta, max_size)
        if shrunk:
            ow, oh, nw, nh = shrunk
            sizes.append(f"`{name}` {ow}×{oh} → {nw}×{nh}px "
                         f"(~{100.0 * (nw // 16) * (nh // 16) / max(1, (ow // 16) * (oh // 16)):.0f}% of its tokens)")
        if _is_picture_mod(meta) and as_video and latent_t > 1:
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
           + (f"\n\n{label_line}" if label_line else "")
           + ("\n\n**Max size:** " + " · ".join(sizes) if sizes else ""))


def _model_choices(api_session):
    try:
        records = api_session.list_model_defs(
            base_model_type=["minimax_h3_ref2va", "minimax_h3_ref2va_pruned"])
    except Exception as e:
        print(f"[H3RefMod] could not list MiniMax H3 Ref2VA models: {e!r}")
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


def _library_rows(folder=None):
    """Table rows for the Library tab. ``folder`` follows the same picker
    convention as _mod_choices: None/ALL_FOLDERS_CHOICE lists every mod in
    every subfolder, ROOT_FOLDER_CHOICE lists only the main folder, and any
    other value lists that subfolder (and below)."""
    if folder == ROOT_FOLDER_CHOICE:
        list_folder, recursive = None, False
    elif not folder or folder == ALL_FOLDERS_CHOICE:
        list_folder, recursive = None, True
    else:
        list_folder, recursive = folder, True
    return [[i["name"], i["kind"], i["mode"], i["tokens"], i["size_mb"], i.get("frames", ""),
             i["concept_type"], i["description"]]
            for i in storage.list_refmods_info(folder=list_folder, recursive=recursive)]


def _library_mod_names(folder=None):
    """Just the folder-relative names for the Library's own dropdowns,
    honouring the same folder filter as _library_rows()."""
    if folder == ROOT_FOLDER_CHOICE:
        return storage.list_refmods(folder=None, recursive=False)
    if not folder or folder == ALL_FOLDERS_CHOICE:
        return storage.list_refmods(folder=None, recursive=True)
    return storage.list_refmods(folder=folder, recursive=True)



def _lora_choices(api_session, model_type):
    if not model_type:
        return []
    try:
        info = api_session.list_loras(model_type)
    except Exception as e:
        print(f"[H3RefMod] could not list LoRAs for {model_type}: {e!r}")
        return []
    return list(info.get("loras") or [])


class MiniMaxH3RefModsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.31.0-fork.20"
        self.description = ("No-training reference mods for MiniMax H3: compress a reference "
                            "into a small file once, reuse it at any strength without "
                            "re-encoding it every generation.")
        self._patch_error = install_patches()

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
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)
        self.insert_after(target_component_id="loras_multipliers",
                          new_component_constructor=self._build_inline_refmods_section)

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
                print("[H3RefMod] refreshed Wan2GP's model catalog so the RefMod custom_settings "
                     "declaration takes effect on cached MiniMax H3 model definitions")
            except Exception as e:
                print(f"[H3RefMod] refresh_model_defs() failed ({e!r}); MiniMax H3 model "
                     "definitions may still be missing the RefMod custom_settings -- use "
                     "'Check setup for this model' in the plugin tab to confirm.")
        else:
            print("[H3RefMod] refresh_model_defs was not exposed by Wan2GP's plugin globals; "
                 "RefMods will likely not take effect until Wan2GP is restarted after this "
                 "plugin was enabled. Use 'Check setup for this model' in the plugin tab to confirm.")

        get_base_mt = getattr(self, "get_base_model_type", None)
        if not callable(get_base_mt):
            print("[H3RefMod] get_base_model_type was not exposed by Wan2GP's plugin globals; "
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
                print(f"[H3RefMod] {err}")
        else:
            print("[H3RefMod] prepare_inputs_dict / get_state_model_type were not exposed by "
                 "Wan2GP's plugin globals; the inline RefMods panel on the Media Generator page "
                 "will be visible but will not affect generations from that page's own Generate "
                 "button. Use the plugin's own 'Generate' tab instead.")

        orig_get_model_settings = getattr(self, "get_model_settings", None)
        if callable(orig_get_model_settings):
            err2 = install_get_model_settings_patch(orig_get_model_settings, self.set_global, get_base_mt)
            if err2:
                print(f"[H3RefMod] {err2}")
        else:
            print("[H3RefMod] get_model_settings was not exposed by Wan2GP's plugin globals; "
                 "a RefMod change right before clicking Generate (with no other field touched "
                 "in between) may not always be picked up from the inline panel -- change any "
                 "other field (e.g. click into the prompt box) once after picking mods as a "
                 "workaround, or use the plugin's own 'Generate' tab instead.")
        return {}

    # ── Inline panel injected onto the Media Generator page ───────────────

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
        with gr.Accordion("MiniMax H3 RefMods (inline)", open=False) as accordion:
            gr.Markdown(
                "Applies when a **MiniMax H3 Ref2VA, FL2VA or FL2VA ControlNet** model is selected "
                "above -- every other "
                "field on this page (resolution, frame count, steps, attention mode, memory "
                "profile, output filename, etc.) is untouched and works exactly as normal. "
                "Extract new RefMods from the **MiniMax H3 RefMods** tab. Selecting a mod here "
                "applies immediately to generations started from *this page's* own Generate "
                "button -- no separate submission needed.")
            mod_rows, stack_controls = self._build_mod_picker_rows()
            with gr.Row():
                status = gr.Markdown("*No RefMods selected.*")
                clear_btn = gr.Button("Clear armed RefMods", size="sm", scale=0)

        widgets = [c for row in mod_rows for c in row] + list(stack_controls)

        def apply_selection(state, *vals):
            n_rows = IMAGE_ROWS + VIDEO_ROWS + AUDIO_ROWS
            rows = _rows_payload(vals[:n_rows * ROW_WIDTH])
            payload = {"rows": rows, "retention": 1.0, "scramble_seed": -1, "curve": None,
                       **_stack_payload(*vals[n_rows * ROW_WIDTH:n_rows * ROW_WIDTH + STACK_CONTROLS])}
            try:
                from . import patches as _h3p
                _h3p.set_armed_selection(json.dumps(payload) if rows else None)
                print(f"[H3RefMod] inline panel: {len(rows)} RefMod(s) armed"
                      if rows else "[H3RefMod] inline panel: selection cleared")
            except Exception:
                pass
            if not isinstance(state, dict):
                return state, "⚠️ Could not access session state -- try reloading the page."
            if rows:
                state[STASH_KEY] = json.dumps(payload)
                msg = f"✅ {len(rows)} RefMod(s) armed for the next MiniMax H3 generation from this page."
            else:
                # Clearing the pickers clears the selection. (This used to hold
                # on to the last selection, from when the task payload was
                # being dropped and this was the only path that worked -- an
                # empty panel then kept injecting mods into every generation.)
                state.pop(STASH_KEY, None)
                msg = "*No RefMods selected.*"
            return state, msg

        def clear_selection(state):
            try:
                from . import patches as _h3p
                _h3p.set_armed_selection(None, force=True)
            except Exception:
                pass
            if isinstance(state, dict):
                state.pop(STASH_KEY, None)
            print("[H3RefMod] inline panel: armed RefMods cleared")
            # Empty the slots too, so the panel never shows mods that aren't
            # armed (the next edit would otherwise silently re-arm them all).
            resets = []
            for _ in mod_rows:
                resets += _cleared_row_updates()
            return [state, "*No RefMods armed.*"] + resets

        clear_btn.click(fn=clear_selection, inputs=[self.state],
                        outputs=[self.state, status] + [c for row in mod_rows for c in row],
                        queue=False)

        for w in widgets:
            w.change(fn=apply_selection, inputs=[self.state] + widgets, outputs=[self.state, status], queue=False)

        if target is not None:
            get_base_mt = getattr(self, "get_base_model_type", None)

            def update_visibility(target_value):
                model_type = str(target_value or "").split("|", 1)[0].strip()
                return gr.update(visible=is_minimax_h3_refmod_capable(model_type, get_base_mt))

            target.change(fn=update_visibility, inputs=[target], outputs=[accordion], queue=False)
        else:
            print("[H3RefMod] neither 'model_choice_target' nor 'wangp_model_choice_target' was "
                 "exposed by Wan2GP -- the inline panel will stay visible for every model instead "
                 "of only MiniMax H3 Ref2VA ones (it still has no effect on other models, this "
                 "only affects whether it's shown).")

        return accordion

    def _submit(self, api_session, model_type, overrides, callbacks):
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

    def _build_mod_picker_rows(self):
        """The mod pickers: a folder picker + "Refresh mod list" button, the
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
                                    info="Narrow the mod lists below to one subfolder of "
                                         "loras/refmods_plugin/minimax_h3/. Pick "
                                         f"'{ALL_FOLDERS_CHOICE}' to go back to seeing everything.")
            refresh_btn = gr.Button("🔄 Refresh mod list", size="sm", scale=1)
        # Above the pickers, where they can't be missed: the readout of what
        # is selected (with each mod's prompt label), then the text encoder
        # options.
        counter = gr.Markdown(_format_ref_counter([]))
        with gr.Accordion("Text encoder options", open=True):
            gr.Markdown("How many pictures of each mod the text encoder is shown, to bind its "
                        "prompt label. The model always receives every mod in full.")
            with gr.Row():
                stack_mode = gr.Dropdown(
                    choices=list(encframes.STACK_CHOICES), value=encframes.STACK_ALL, scale=3,
                    label="Mods sent as one video: pictures shown",
                    info="For multi-picture mods with 'Send as one video' ticked. 'all' is best "
                         "for identity; 'up to N' caps text-encoder memory and time on big "
                         "stacks.")
                stack_n = gr.Number(value=encframes.STACK_DEFAULT_N, precision=0, minimum=1,
                                    maximum=256, label="N", visible=False, scale=1)
            with gr.Row():
                video_mode = gr.Dropdown(
                    choices=[encframes.STACK_UP_TO_N, encframes.STACK_ALL],
                    value=encframes.STACK_UP_TO_N, scale=3,
                    label="Video mods: frames shown",
                    info="A video mod keeps two frames per second of its clip. 'up to N' shows N "
                         "of them, spread from first to last; 'all' shows every one. Fewer = a "
                         "lighter text encode and slightly faster steps.")
                video_n = gr.Number(value=encframes.MAX_CLIP_FRAMES_SHOWN, precision=0, minimum=1,
                                    maximum=256, label="N", scale=1)
        with gr.Row():
            fit_to_output = gr.Checkbox(
                False, label="Fit mods to the output size (faster)", scale=2,
                info="Off: every picture and video mod goes in at the size it was extracted "
                     "at. On: a mod bigger than the video you're making is shrunk to its size "
                     "first, keeping its shape. Fewer tokens, so generation is faster -- but a "
                     "shrunk mod carries less fine detail, faces first. Mods already that size or "
                     "smaller are left alone. Works in Ref2VA, FL2VA and FL2VA ControlNet.")
            phase2_mods = gr.Dropdown(
                choices=PHASE2_MODS_UI, value="full", label="Mods in phase 2", scale=1,
                info="Two-phase runs only. Full: mods as set above. Fit to tile: with phase 2 "
                     "tiling on, each mod is shrunk to one tile, so the 4 tiles pay a quarter "
                     "each. Off (experimental): phase 2 runs without the visual mods -- fastest, "
                     "but faces may drift. Ref2VA keeps them if you also use references of your "
                     "own.")
            shrink_method = gr.Dropdown(
                choices=SHRINK_METHOD_UI, value="reencode", label="Shrink method", scale=1,
                info="How a mod is made smaller (Max size, Fit mods, phase 1, tiles). Re-encode: "
                     "decoded, resized and re-encoded with the loaded VAE -- best quality, takes a "
                     "few seconds per mod once. Fast: the stored mod is resized directly -- "
                     "instant, no VAE, but rougher.")

        # One slot per kind to start with; "Add" reveals the next one and
        # "Remove" clears and hides the last. Every slot is still built up
        # front (Gradio can't create components on the fly) -- the hidden ones
        # are simply empty, and the order returned is unchanged: image rows,
        # then video rows, then audio rows, each a (dropdown, strength,
        # send-as-one-video) triple.
        mod_rows = []
        for kind, label, count in (("image", "Image RefMods", IMAGE_ROWS),
                                   ("video", "Video RefMods", VIDEO_ROWS),
                                   ("audio", "Audio RefMods", AUDIO_ROWS)):
            note = (" -- a mod made from several pictures also gets a **Send as one video** "
                    "box: ticked, it is one `<Video N>` in the prompt instead of one "
                    "`<Picture N>` per picture." if kind == "image" else "")
            gr.Markdown(f"**{label}**{note}")
            kind_rows, kind_ui = [], []
            for i in range(count):
                with gr.Row(visible=i == 0) as row_ui:
                    mdd = gr.Dropdown(choices=_mod_choices(kind), value=NONE_CHOICE,
                                      label=f"{kind.capitalize()} mod {i + 1}", scale=4)
                    with gr.Column(scale=3, min_width=220):
                        strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength")
                        # "Max size" (fork.19): shown once a picture or video
                        # mod is picked; starts at the mod's own size (full).
                        max_size = gr.Slider(MAX_SIZE_FLOOR, 1024, value=0, step=MAX_SIZE_STEP,
                                             label="Max size (short edge, px)", visible=False,
                                             info="Starts at the mod's own size. Lower = fewer "
                                                  "tokens and faster, less fine detail.")
                    with gr.Column(scale=1, min_width=130):
                        as_video = gr.Checkbox(False, label="Send as one video", visible=False)
                        use_audio = gr.Checkbox(True, label="Include audio", visible=False)
                kind_rows.append((mdd, strength, as_video, use_audio, max_size))
                kind_ui.append(row_ui)
                if kind != "audio":
                    mdd.change(fn=lambda name, _kind=kind: _row_boxes_update(name, _kind),
                               inputs=[mdd], outputs=[as_video, use_audio, max_size], queue=False)
            with gr.Row():
                add_btn = gr.Button(f"➕ Add {kind} mod", size="sm", scale=0, min_width=150)
                remove_btn = gr.Button("➖ Remove last", size="sm", scale=0, min_width=150)
            shown = gr.State(1)
            kind_widgets = [c for row in kind_rows for c in row]

            def add_slot(n, _count=count):
                n = min(_count, int(n or 1) + 1)
                return [n] + [gr.update(visible=k < n) for k in range(_count)]

            def remove_slot(n, _count=count):
                n = max(1, int(n or 1))
                last = n - 1                       # the slot being cleared
                n = max(1, n - 1)
                cleared = []
                for k in range(_count):
                    if k == last:
                        cleared += _cleared_row_updates()
                    else:
                        cleared += [gr.update()] * ROW_WIDTH
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
        for w in row_widgets:
            w.change(fn=update_counter, inputs=row_widgets, outputs=[counter], queue=False)
        stack_mode.change(fn=lambda mode: gr.update(visible=mode == encframes.STACK_UP_TO_N),
                          inputs=[stack_mode], outputs=[stack_n], queue=False)
        video_mode.change(fn=lambda mode: gr.update(visible=mode == encframes.STACK_UP_TO_N),
                          inputs=[video_mode], outputs=[video_n], queue=False)

        return mod_rows, (stack_mode, stack_n, video_mode, video_n, fit_to_output, phase2_mods,
                          shrink_method)

    def _build_extract_section(self, api_session, model_dd):
        gr.Markdown("### Extract a RefMod\n"
                    "Turn one or more reference images (and/or one reference video) into a small "
                    "saved file. This briefly runs a real generation task on the model selected "
                    "above so it can reuse its already-loaded VAE -- you'll see the usual progress "
                    "bar for a few seconds, then **no video is produced on purpose**: the mod file "
                    "is what was created. Check the status line below for confirmation.")
        with gr.Row():
            name = gr.Textbox(label="Mod name", value="my_concept", scale=2,
                              info="A plain name saves to the main folder. Include a path "
                                   "(characters/tanya) to save into a subfolder instead -- it's "
                                   "created automatically if it doesn't exist.")
            mode = gr.Radio(label="Mode", choices=["training", "encode"], value="training", scale=2,
                            info="encode = full fidelity (identical to a live reference), best for a "
                                 "precise face/identity. training = approximate (pooling discards "
                                 "fine detail), best for a general concept/style/pose where "
                                 "approximation is fine.")
        with gr.Accordion("Description and concept type (optional labels)", open=False):
            with gr.Row():
                description = gr.Textbox(
                    label="keyword - description", lines=2, scale=3,
                    info="Text note only. The model never reads it; the Library's prompt hint "
                         "uses it.")
                concept_type = gr.Dropdown(
                    label="Concept type", choices=list(core.CONCEPT_TYPES), value="generic", scale=1,
                    info="Label only. No effect on generation.")

        gr.Markdown("#### Sources\n"
                    "*Add as many reference images and videos as you like -- every one is encoded "
                    "and stacked into this single mod.*")
        with gr.Row():
            with gr.Column():
                ref_images = gr.Files(label="Reference image(s)", file_types=["image"],
                                      file_count="multiple")
                remove_background_images_ref = gr.Dropdown(
                    choices=[("Keep backgrounds", 0),
                             ("Remove background behind people / objects", 1)],
                    value=0, label="Image backgrounds",
                    info="Same background removal as Wan2GP's own reference images. Images "
                         "only -- not videos or pictures taken from them.")
            with gr.Column():
                ref_videos = gr.Files(label="Reference video(s)", file_types=["video"],
                                      file_count="multiple")
                video_as_pictures = gr.Checkbox(
                    False, label="Turn reference videos into a picture stack",
                    info="Instead of a clip, take still pictures from each video: the sharpest "
                         "frame from every slot of time, so motion blur and blinks are skipped. "
                         "Same span as a clip (the trim, up to 'Reference duration to use'). "
                         "Keeps identity at a fraction of a clip's tokens, but not motion.")
                pictures_per_second = gr.Slider(
                    0.25, 12, value=1.0, step=0.25, label="Pictures per second of video",
                    visible=False,
                    info="1 = one picture per second (5 from a 5s span). Each picture costs about "
                         "as much as 3-4 frames of a clip, so high values approach a clip's cost.")
        stills_estimate = gr.Markdown("")
        with gr.Accordion("Preview / trim reference videos", open=False):
            gr.Markdown("Pick an uploaded video to watch it, and trim the span this mod is built "
                        "from. Trims are per video and remembered while the page stays open; "
                        "untrimmed videos are used from their start as before. A trim also "
                        "limits the soundtrack taken by *Use the clip's own audio*.")
            with gr.Row():
                video_pick = gr.Dropdown(label="Video", choices=[], value=None, scale=2)
                video_duration = gr.Markdown("")
            video_preview = gr.Video(label="Preview", height=260)
            if RangeSlider is not None:
                video_trim = RangeSlider(minimum=0.0, maximum=15.0, value=(0.0, 15.0), step=0.1,
                                         label="Trim (seconds)", info="Start - End")
            else:
                video_trim = gr.Slider(0.0, 15.0, value=0.0, step=0.1,
                                       label="Trim start (seconds)",
                                       info="gradio_rangeslider isn't installed, so only a start "
                                            "offset is available here.")
            trim_state = gr.State({})
            trim_status = gr.Markdown("")

        gr.Markdown("#### Soundtrack (optional)")
        with gr.Row():
            ref_audio = gr.Audio(label="Reference audio", type="filepath", scale=2)
            use_clip_audio = gr.Checkbox(False, label="Use the clip's own audio", scale=1,
                                         info="Take the soundtrack from the first reference video "
                                              "that has one, so a single clip of someone talking "
                                              "gives this mod both the look and the voice. Ignored "
                                              "if you choose an audio file, or if no video source "
                                              "has an audio track. A trimmed clip's audio starts "
                                              "at the trim start.")
        soundtrack_seconds = gr.Slider(
            MIN_AUDIO_SECONDS, 15.0, value=DEFAULT_AUDIO_SECONDS, step=0.5,
            label="Soundtrack length (seconds)",
            info=f"How much audio this mod keeps -- an audio file, or a clip's own audio -- "
                 f"independent of the video length. At least {MIN_AUDIO_SECONDS:g}s, H3's shortest "
                 f"audio reference: an audio file shorter than that is refused; a clip's own audio "
                 f"is kept with a warning. Longer gives the model more of a voice to copy, at ~80 "
                 f"tokens per second.")
        audio_duration_warning = gr.Markdown("")
        with gr.Accordion("How soundtracks work", open=False):
            gr.Markdown("*Audio on its own makes an audio-only mod. Audio **together with** images "
                       "or video gives that mod a soundtrack as well as a look: the two latents are "
                       "stored side by side (they can't be stacked -- their shapes differ) and the "
                       "audio is injected as its own `<Audio N>` reference, so refer to it that way "
                       "in prompts while `<Picture N>`/`<Video N>` stays the picture. Every "
                       "soundtrack, and every audio-only mod, is as long as 'Soundtrack length' "
                       "(4s by default), whatever the video length. An audio file shorter than 2s, "
                       "H3's shortest audio reference, is refused; a clip's own audio that short is "
                       "kept with a warning. It adds 2 tokens per audio latent "
                       "(~80/second). Audio is always extracted at full fidelity; 'Mode' above "
                       "doesn't apply to it. You can also add or replace a soundtrack later from the "
                       "Library tab.*")

        with gr.Accordion("Settings", open=True):
            with gr.Row():
                ref_resolution = gr.Slider(256, 2048, value=1024, step=64,
                                           label="Ref resolution (short edge, px)",
                                           info="Size before encoding. Smaller = faster, fewer "
                                                "tokens, less detail.")
                latent_frames = gr.Slider(0.1, latent_frames_to_seconds(MAX_LATENT_FRAMES),
                                          value=latent_frames_to_seconds(16), step=0.1,
                                          label="Reference duration to use (seconds) -- video",
                                          info="Approximate real seconds kept from a video source "
                                               "(and the span pictures are taken from, for a picture "
                                               "stack). Audio has its own 'Soundtrack length' above. "
                                               "MiniMax H3's native 15s reference cap is a **shared** "
                                               "budget -- **if a generation combines two video mods, "
                                               "their durations add together, so one mod near 14.9s "
                                               "leaves no room for a second one alongside it.**")
                max_tokens = gr.Slider(0, 65536, value=65536, step=512, label="Max tokens (0 = no cap)",
                                       info="Max tokens allowed. 0 = unlimited.")
            with gr.Row(visible=True) as training_row:
                pool_h = gr.Slider(2, 64, value=16, step=2, label="Pool grid height",
                                   info="Training mode. Bigger grid = more detail, more tokens.")
                pool_w = gr.Slider(2, 64, value=16, step=2, label="Pool grid width",
                                   info="Training mode. Same as height, other axis.")
                identity = gr.Slider(0, 2000, value=500, step=50, label="Identity refinement steps",
                                     info="Training mode. Higher = truer to original.")
            merge = gr.Checkbox(
                False, label="Merge into one picture (training mode, several sources)",
                visible=True,
                info="Instead of keeping every picture, combine them into ONE small picture of "
                     "what they have in common -- one reference's tokens however many go in. "
                     "What agrees across the pictures at the same place in the frame survives; "
                     "backgrounds and unique details are averaged away. Works best on similarly "
                     "framed shots, for a general look rather than a precise face. Identity "
                     "refinement steps refine the merge.")
            pool_warning = gr.Markdown("")
            multiplier = gr.Slider(1, 10, value=1, step=1, label="Repeat multiplier",
                                   info="Repeats the mod. Higher = stronger effect, bigger file.")

        with gr.Row():
            save = gr.Checkbox(label="Save to disk", value=True,
                               info="Off = test run, nothing saved.")
            store_frames = gr.Checkbox(
                label="Store encoder frames", value=True,
                info="Saves the pictures the text encoder is shown inside the mod, so "
                     "generations never have to decode it. Makes the file a little bigger. "
                     "Off = decoded once per session instead; the Library can add them later.")
        extract_btn = gr.Button("Extract & Save RefMod", variant="primary")
        extract_status = gr.Textbox(label="Status", interactive=False, lines=3)
        mode.change(fn=lambda m: (gr.update(visible=m == "training"),
                                  gr.update(visible=m == "training")),
                    inputs=[mode], outputs=[training_row, merge], queue=False)

        def update_pool_warning(concept_type, mode, pool_h, pool_w):
            w = core.identity_training_pool_warning(concept_type, mode, int(pool_h), int(pool_w))
            return f"⚠️ {w}" if w else ""

        for w in (concept_type, mode, pool_h, pool_w):
            w.change(fn=update_pool_warning, inputs=[concept_type, mode, pool_h, pool_w],
                    outputs=[pool_warning], queue=False)

        def update_audio_duration_warning(audio_path, current_seconds):
            real_seconds = _audio_file_seconds(audio_path)
            if real_seconds is None:
                return ""
            if real_seconds < MIN_AUDIO_SECONDS - 0.05:
                return (f"⛔ This audio file is only ~{real_seconds:.1f}s long. H3 needs at least "
                        f"{MIN_AUDIO_SECONDS:g}s of audio to use it as a reference, so extraction "
                        f"will be refused -- use a longer clip.")
            if real_seconds > float(current_seconds) + 0.15:
                return (f"⚠️ This audio file is ~{real_seconds:.1f}s long, but 'Soundtrack length' "
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
            return (gr.update(choices=names, value=first_name), first_path,
                    f"**{seconds:.2f}s**", trims,
                    _trim_summary(trims, names))

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
            return trims, _trim_summary(trims, list(lookup))

        def do_extract(model_type, name, mode, concept_type, ref_images, ref_videos, ref_audio,
                       remove_background_images_ref, ref_resolution, pool_h, pool_w, latent_frames,
                       identity, multiplier, max_tokens, use_clip_audio, description, save,
                       store_frames=True, trims=None, as_pictures=False, per_second=1.0,
                       merge_refs=False, audio_seconds=DEFAULT_AUDIO_SECONDS):
            if not model_type:
                return "Pick a MiniMax H3 Ref2VA model above first."
            image_paths = [f.name if hasattr(f, "name") else f for f in (ref_images or [])]
            video_paths = [f.name if hasattr(f, "name") else f for f in (ref_videos or [])]
            if not image_paths and not video_paths and not ref_audio:
                return "Add at least one reference image, video, or audio file."
            # Refuse an audio file too short to use before running anything
            # (a clip's own audio is checked by the task, also before any
            # slow encoding).
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
                "ref_resolution": int(ref_resolution), "pool_h": int(pool_h), "pool_w": int(pool_w),
                "latent_frames": seconds_to_latent_frames(latent_frames), "identity": int(identity),
                "multiplier": int(multiplier), "max_tokens": int(max_tokens),
                "use_clip_audio": bool(use_clip_audio),
                "video_trims": {k: list(v) for k, v in (trims or {}).items()},
                "description": description or "", "save": bool(save),
                "store_encoder_frames": bool(store_frames),
                "video_as_pictures": bool(as_pictures),
                "pictures_per_second": float(per_second or 1.0),
                "merge": bool(merge_refs),
                "audio_seconds": max(MIN_AUDIO_SECONDS, float(audio_seconds or DEFAULT_AUDIO_SECONDS)),
            }
            log = {"lines": []}

            class ExtractCallbacks:
                def on_status(self, status):
                    if status:
                        log["lines"].append(str(status))

                def on_progress(self, update):
                    pass

            spec_json = json.dumps(spec)
            # Wan2GP strips unknown custom settings on the way to the queue in
            # some setups, and an extract task that loses its payload renders a
            # video instead. patched_generate picks this up only if the payload
            # is missing, consumes it once, and expires it.
            set_pending_extract(spec_json)
            try:
                self._submit(api_session, model_type,
                            {"video_length": 107,
                             "custom_settings": {SETTING_COMBINED:
                                                 pack_refmod_setting(extract_json=spec_json)}},
                            ExtractCallbacks())
            except Exception as e:
                set_pending_extract(None)
                return f"Extraction task failed to run: {e!r}"
            finally:
                set_pending_extract(None)
            lines = log["lines"]
            refused = [l for l in lines if "extraction refused" in l.lower()]
            if refused:
                return refused[-1]
            # Warnings stay visible however many lines follow them.
            warnings = [l for l in lines if "⚠️" in l]
            tail = [l for l in lines[-6:] if l not in warnings]
            ok = ("Done -- check the Library tab (Refresh) to see the saved mod." if any(
                    "saved" in l.lower() for l in lines) else
                  "Task finished, but no confirmation line was captured -- check the terminal log.")
            return "\n".join(warnings + tail + [ok])

        estimate_inputs = [video_as_pictures, pictures_per_second, ref_videos, ref_images,
                           trim_state, latent_frames, mode, ref_resolution, pool_h, pool_w,
                           max_tokens, merge]
        for w in (video_as_pictures, pictures_per_second, ref_videos, ref_images, trim_state,
                  latent_frames, mode, ref_resolution, pool_h, pool_w, max_tokens, merge):
            w.change(fn=update_stills_estimate, inputs=estimate_inputs, outputs=[stills_estimate],
                     queue=False)
        video_as_pictures.change(fn=lambda on: gr.update(visible=bool(on)),
                                 inputs=[video_as_pictures], outputs=[pictures_per_second],
                                 queue=False)

        ref_videos.change(fn=on_videos_uploaded, inputs=[ref_videos, trim_state],
                          outputs=[video_pick, video_preview, video_duration, trim_state,
                                   trim_status], queue=False)
        video_pick.change(fn=on_video_picked, inputs=[video_pick, ref_videos, trim_state],
                          outputs=[video_preview, video_duration, video_trim], queue=False)
        video_trim.change(fn=on_trim_changed, inputs=[video_trim, video_pick, ref_videos, trim_state],
                          outputs=[trim_state, trim_status], queue=False)

        extract_btn.click(
            fn=do_extract,
            inputs=[model_dd, name, mode, concept_type, ref_images, ref_videos, ref_audio,
                   remove_background_images_ref, ref_resolution, pool_h, pool_w, latent_frames,
                   identity, multiplier, max_tokens, use_clip_audio, description, save,
                   store_frames, trim_state, video_as_pictures, pictures_per_second, merge,
                   soundtrack_seconds],
            outputs=[extract_status],
            queue=False,
        )

    # ── Library ─────────────────────────────────────────────────────────

    def _build_library_section(self, api_session=None, model_dd=None):
        gr.Markdown("### Saved RefMods\n"
                    "Mods live in `loras/refmods_plugin/minimax_h3/` (subfolders are supported, "
                    "and mods in them are named by their path, e.g. `characters/tanya`). Mods "
                    "from the ComfyUI RefMod packs can be dropped in directly.")
        with gr.Row():
            library_folder_dd = gr.Dropdown(choices=_folder_choices(), value=ALL_FOLDERS_CHOICE,
                                            label="Folder", scale=3,
                                            info=f"Pick '{ALL_FOLDERS_CHOICE}' to go back to "
                                                 f"seeing every mod again.")
            refresh_btn = gr.Button("Refresh", scale=1)
        table = gr.Dataframe(
            headers=["name", "kind", "mode", "tokens", "size (MB)", "encoder frames", "concept type",
                     "description"],
            value=_library_rows(), interactive=False, wrap=True)
        with gr.Accordion("Prompt hint", open=False):
            gr.Markdown("Builds a prompt line from the *keyword - description* of one or more "
                        "mods (`concept_type: description; ...`), since the model never reads "
                        "those descriptions by itself.")
            with gr.Row():
                hint_names = gr.Textbox(label="Mod name(s), comma-separated, in the order you want them combined",
                                        scale=2)
                hint_btn = gr.Button("Build Prompt Hint")
            hint_output = gr.Textbox(label="Prompt hint -- copy this into your prompt", interactive=False, lines=2)
        with gr.Accordion("Rename or edit a description", open=False):
            gr.Markdown("Pick a mod, click Load, edit, then Save. A plain name keeps the mod in "
                        "its folder; a path (`characters/tanya`) moves it there, and a leading "
                        "`/` moves it back to the main folder. Only the name and description "
                        "change.")
            with gr.Row():
                edit_name_dd = gr.Dropdown(label="Mod to edit", choices=_library_mod_names(), scale=2)
                edit_load_btn = gr.Button("Load")
            with gr.Row():
                edit_name_field = gr.Textbox(label="Name (optionally with a folder path)")
                edit_description_field = gr.Textbox(label="Description", lines=2)
            edit_save_btn = gr.Button("Save changes", variant="primary")
            edit_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Folders and moving", open=False):
            gr.Markdown("Folders are subfolders of the mods directory. A move rewrites the mod "
                        "in place, keeping everything in it, and never overwrites a mod of the "
                        "same name.")
            with gr.Row():
                new_folder_name = gr.Textbox(label="New folder", scale=2,
                                             placeholder="characters, or characters/female to nest")
                create_folder_btn = gr.Button("Create folder")
            with gr.Row():
                move_dd = gr.Dropdown(label="Mods to move", choices=_library_mod_names(),
                                      value=[], multiselect=True, scale=2)
                move_target = gr.Dropdown(label="Destination",
                                          choices=[ROOT_FOLDER_CHOICE] + storage.list_mod_folders(),
                                          value=ROOT_FOLDER_CHOICE, scale=1)
                move_btn = gr.Button("Move", variant="primary")
            with gr.Row():
                delete_folder_dd = gr.Dropdown(label="Delete an empty folder",
                                               choices=storage.list_mod_folders(), value=None, scale=2)
                delete_folder_btn = gr.Button("Delete folder", variant="stop")
            folder_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Soundtrack: add, replace or remove", open=False):
            gr.Markdown("Give an image or video mod its own voice without rebuilding it: the "
                        "audio is injected as its own `<Audio N>` reference. Audio-only mods are "
                        "skipped. Runs a short task on the model selected at the top of this tab.")
            with gr.Row():
                audio_target_dd = gr.Dropdown(label="Mods to update", choices=_library_mod_names(),
                                              value=[], multiselect=True, scale=2,
                                              info="Pick one or more image/video mods.")
                attach_btn = gr.Button("Apply soundtrack", variant="primary")
            with gr.Row():
                audio_file = gr.File(label="Audio or video file (a video's soundtrack is used)",
                                     file_types=["audio", "video"], type="filepath", scale=2)
                audio_seconds = gr.Slider(MIN_AUDIO_SECONDS, 15.0, value=DEFAULT_AUDIO_SECONDS,
                                          step=0.5, label="Seconds to use", scale=1,
                                          info=f"At least {MIN_AUDIO_SECONDS:g}s; a clip shorter "
                                               f"than that is refused.")
                audio_remove = gr.Checkbox(False, label="Remove existing soundtrack instead")
            audio_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Encoder frames", open=False):
            gr.Markdown("Stores the pictures the text encoder is shown inside mods that don't "
                        "have them yet (older mods, or ones from other tools), so generations "
                        "never decode them. Nothing else in the mod changes. The table's "
                        "*encoder frames* column shows which have them. Runs a short task on the "
                        "model selected at the top of this tab.")
            with gr.Row():
                frames_target_dd = gr.Dropdown(label="Mods to update", choices=_library_mod_names(),
                                               value=[], multiselect=True, scale=2)
                frames_btn = gr.Button("Store for selected")
                frames_all_btn = gr.Button("Store for every mod missing them", variant="primary")
            frames_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Accordion("Maintenance", open=False):
            gr.Markdown("Older versions saved mods made only from several still images as "
                        "`video`. This rescans every mod (all folders) and corrects the label in "
                        "place; the latent is untouched. Safe to run anytime.")
            with gr.Row():
                fix_btn = gr.Button("Fix classification (image vs video)")
                fix_status = gr.Markdown("")
        with gr.Accordion("Delete mods", open=False):
            with gr.Row():
                delete_dd = gr.Dropdown(label="Mods to delete", choices=_library_mod_names(),
                                        value=[], multiselect=True, scale=2,
                                        info="Pick one or more from the list above. Follows the "
                                             "Folder picker; deletion is permanent.")
                delete_btn = gr.Button("Delete selected", variant="stop")
            delete_confirm = gr.Checkbox(False, label="Yes, delete permanently",
                                         info="Required, since deletion can't be undone. Resets "
                                              "after each delete.")
            delete_status = gr.Textbox(label="", interactive=False, show_label=False)

        # ── wiring (all components above already exist) ──
        def _names_keeping(folder, current):
            """Mod names for the chosen folder, with the currently-selected
            one force-included even if it lives elsewhere -- otherwise
            switching folders would leave a stale value Gradio rejects."""
            names = _library_mod_names(folder)
            if current and current not in names:
                names = names + [current]
            return names

        def do_attach_audio(names, audio_path, seconds, remove, folder, model_type=None):
            names = [n for n in (names if isinstance(names, list) else [names]) if n]
            if not names:
                return gr.update(), "Pick at least one mod first."
            if not audio_path and not remove:
                return gr.update(), "Choose an audio file, or tick 'Remove existing soundtrack'."
            if api_session is None or not model_type:
                return gr.update(), ("Pick a model at the top of this tab first -- encoding audio "
                                     "needs the model's audio VAE, so this runs as a task.")
            spec = {"op": "attach_audio", "names": names, "audio_path": audio_path,
                    "seconds": float(seconds), "remove": bool(remove)}
            log = {"lines": []}

            class AttachCallbacks:
                def on_status(self, message):
                    log["lines"].append(str(message))
                def on_error(self, message):
                    log["lines"].append(f"error: {message}")

            spec_json = json.dumps(spec)
            set_pending_extract(spec_json)   # same payload-loss guard extraction uses
            try:
                self._submit(api_session, model_type,
                             {"video_length": 107,
                              "custom_settings": {SETTING_COMBINED:
                                                  pack_refmod_setting(extract_json=spec_json)}},
                             AttachCallbacks())
            except Exception as e:
                return gr.update(), f"Soundtrack update failed to run: {e!r}"
            finally:
                set_pending_extract(None)
            tail = [line for line in log["lines"] if "Attach audio finished" in line]
            return (gr.update(choices=_library_mod_names(folder), value=[]),
                    tail[-1] if tail else (log["lines"][-1] if log["lines"]
                                           else "Finished -- see the console for details."))

        def do_store_frames(names, all_missing, folder, model_type=None):
            names = [n for n in (names if isinstance(names, list) else [names]) if n]
            if not all_missing and not names:
                return _library_rows(folder), gr.update(), "Pick at least one mod first."
            if api_session is None or not model_type:
                return (_library_rows(folder), gr.update(),
                        "Pick a model at the top of this tab first -- decoding needs the "
                        "model's video VAE, so this runs as a task.")
            if all_missing and not storage.list_refmods_missing_frames():
                return _library_rows(folder), gr.update(), "Every mod already has encoder frames."
            spec = {"op": "store_frames", "names": names, "all_missing": bool(all_missing)}
            log = {"lines": []}

            class FramesCallbacks:
                def on_status(self, message):
                    log["lines"].append(str(message))
                def on_error(self, message):
                    log["lines"].append(f"error: {message}")
                def on_progress(self, update):
                    pass

            spec_json = json.dumps(spec)
            set_pending_extract(spec_json)
            try:
                self._submit(api_session, model_type,
                             {"video_length": 107,
                              "custom_settings": {SETTING_COMBINED:
                                                  pack_refmod_setting(extract_json=spec_json)}},
                             FramesCallbacks())
            except Exception as e:
                return _library_rows(folder), gr.update(), f"Storing encoder frames failed to run: {e!r}"
            finally:
                set_pending_extract(None)
            tail = [line for line in log["lines"] if "Encoder frames finished" in line
                    or "nothing to do" in line]
            return (_library_rows(folder), gr.update(choices=_library_mod_names(folder), value=[]),
                    tail[-1] if tail else (log["lines"][-1] if log["lines"]
                                           else "Finished -- see the console for details."))

        frames_inputs = [frames_target_dd, gr.State(False), library_folder_dd] + \
            ([model_dd] if model_dd is not None else [])
        frames_all_inputs = [frames_target_dd, gr.State(True), library_folder_dd] + \
            ([model_dd] if model_dd is not None else [])
        frames_btn.click(fn=do_store_frames, inputs=frames_inputs,
                         outputs=[table, frames_target_dd, frames_status], queue=False)
        frames_all_btn.click(fn=do_store_frames, inputs=frames_all_inputs,
                             outputs=[table, frames_target_dd, frames_status], queue=False)

        attach_btn.click(fn=do_attach_audio,
                         inputs=([audio_target_dd, audio_file, audio_seconds, audio_remove,
                                  library_folder_dd, model_dd] if model_dd is not None else
                                 [audio_target_dd, audio_file, audio_seconds, audio_remove,
                                  library_folder_dd]),
                         outputs=[audio_target_dd, audio_status], queue=False)

        def _folder_refresh(message, folder):
            """Everything that depends on the folder list or the mod list."""
            folders = storage.list_mod_folders()
            names = _library_mod_names(folder)
            return (_library_rows(folder),
                    _safe_choice_update(_folder_choices(), folder),
                    _safe_choice_update([ROOT_FOLDER_CHOICE] + folders, ROOT_FOLDER_CHOICE),
                    gr.update(choices=folders, value=None),
                    gr.update(choices=names, value=[]),
                    gr.update(choices=names, value=[]),
                    gr.update(choices=names, value=[]),
                    _safe_choice_update(_names_keeping(folder, None)),
                    gr.update(choices=names, value=[]),
                    message)

        def do_create_folder(name, folder):
            try:
                created = storage.create_mod_folder(name)
            except Exception as e:
                return _folder_refresh(f"Could not create the folder: {e}", folder)
            return _folder_refresh(f"Created '{created}'.", folder)

        def do_move_mods(names, destination, folder):
            names = [n for n in (names if isinstance(names, list) else [names]) if n]
            if not names:
                return _folder_refresh("Pick at least one mod to move.", folder)
            target = "" if destination in (None, ROOT_FOLDER_CHOICE) else destination
            moved, skipped = storage.move_mods(names, target)
            where = "the top level" if not target else f"'{target}'"
            parts = []
            if moved:
                parts.append(f"Moved {len(moved)} mod(s) to {where}.")
            if skipped:
                parts.append("Skipped: " + "; ".join(f"'{n}' ({why})" for n, why in skipped))
            return _folder_refresh(" ".join(parts) or "Nothing to do.", folder)

        def do_delete_folder(name, folder):
            if not name:
                return _folder_refresh("Pick a folder to delete.", folder)
            try:
                removed = storage.delete_mod_folder(name)
            except Exception as e:
                return _folder_refresh(f"Could not delete the folder: {e}", folder)
            return _folder_refresh(f"Deleted '{removed}'.", folder)

        def do_refresh(folder, current):
            # Every picker that lists mods or folders. (The move picker and the
            # two folder pickers used to be missing here, so a newly extracted
            # mod or new folder didn't show up in them until a restart.)
            names = _library_mod_names(folder)
            folders = storage.list_mod_folders()
            return (_library_rows(folder),
                    _safe_choice_update(_folder_choices(), folder),
                    _safe_choice_update(_names_keeping(folder, current), current),
                    gr.update(choices=names, value=[]),
                    gr.update(choices=names, value=[]),
                    gr.update(choices=names, value=[]),
                    gr.update(choices=names, value=[]),
                    _safe_choice_update([ROOT_FOLDER_CHOICE] + folders, ROOT_FOLDER_CHOICE),
                    gr.update(choices=folders, value=None))

        create_folder_btn.click(fn=do_create_folder, inputs=[new_folder_name, library_folder_dd],
                                outputs=[table, library_folder_dd, move_target, delete_folder_dd, delete_dd, move_dd, audio_target_dd, edit_name_dd, frames_target_dd, folder_status], queue=False)
        move_btn.click(fn=do_move_mods, inputs=[move_dd, move_target, library_folder_dd],
                       outputs=[table, library_folder_dd, move_target, delete_folder_dd, delete_dd, move_dd, audio_target_dd, edit_name_dd, frames_target_dd, folder_status], queue=False)
        delete_folder_btn.click(fn=do_delete_folder, inputs=[delete_folder_dd, library_folder_dd],
                                outputs=[table, library_folder_dd, move_target, delete_folder_dd, delete_dd, move_dd, audio_target_dd, edit_name_dd, frames_target_dd, folder_status], queue=False)

        refresh_outputs = [table, library_folder_dd, edit_name_dd, delete_dd, audio_target_dd,
                           frames_target_dd, move_dd, move_target, delete_folder_dd]
        refresh_btn.click(fn=do_refresh, inputs=[library_folder_dd, edit_name_dd],
                          outputs=refresh_outputs, queue=False)
        library_folder_dd.change(fn=do_refresh, inputs=[library_folder_dd, edit_name_dd],
                                 outputs=refresh_outputs, queue=False)

        def do_delete(names, folder, confirmed):
            names = [n for n in (names if isinstance(names, list) else [names]) if n]
            if not names:
                return (_library_rows(folder), gr.update(), gr.update(), gr.update(),
                        "Pick at least one mod to delete.")
            if not confirmed:
                return (_library_rows(folder), gr.update(), gr.update(), gr.update(),
                        f"Tick 'Yes, delete permanently' to delete {len(names)} mod(s).")
            deleted = [n for n in names if storage.delete_refmod(n)]
            missing = [n for n in names if n not in deleted]
            remaining = _library_mod_names(folder)
            parts = []
            if deleted:
                parts.append(f"Deleted {len(deleted)}: " + ", ".join(f"'{n}'" for n in deleted) + ".")
            if missing:
                parts.append("Not found: " + ", ".join(f"'{n}'" for n in missing) + ".")
            return (_library_rows(folder),
                    gr.update(choices=remaining, value=[]),          # delete picker
                    _safe_choice_update(remaining),                  # edit picker
                    gr.update(value=False),                          # re-arm the confirmation
                    " ".join(parts))

        delete_btn.click(fn=do_delete, inputs=[delete_dd, library_folder_dd, delete_confirm],
                         outputs=[table, delete_dd, edit_name_dd, delete_confirm, delete_status],
                         queue=False)

        def do_fix(folder):
            fixed, checked = storage.reclassify_all_mods()
            msg = (f"Checked {checked} mod(s), fixed {fixed}." if fixed else
                  f"Checked {checked} mod(s), all already correctly classified.")
            names = _library_mod_names(folder)
            return (_library_rows(folder), _safe_choice_update(names),
                    gr.update(choices=names, value=[]), msg)

        fix_btn.click(fn=do_fix, inputs=[library_folder_dd],
                      outputs=[table, edit_name_dd, delete_dd, fix_status], queue=False)

        def do_load_for_edit(name):
            if not name:
                return "", "", "Pick a mod first."
            meta = storage.read_refmod_meta(storage.mod_path(name))
            if meta is None:
                return "", "", f"No mod named '{name}' found."
            # Show the full folder-relative path, not the metadata's bare
            # "name" field, so editing it can move the mod (and so a plain
            # save round-trips without silently relocating anything).
            return name, meta.get("description", "") or "", ""

        edit_load_btn.click(fn=do_load_for_edit, inputs=[edit_name_dd],
                            outputs=[edit_name_field, edit_description_field, edit_status], queue=False)

        def do_save_edit(old_name, new_name, new_description, folder):
            if not old_name:
                return (_library_rows(folder), gr.update(), gr.update(), gr.update(),
                        "Pick a mod first (use Load).")
            try:
                final_name = storage.rename_and_update_mod(old_name, new_name=new_name,
                                                            new_description=new_description)
            except Exception as e:
                return (_library_rows(folder), gr.update(), gr.update(), gr.update(),
                        f"Could not save: {e!r}")
            msg = f"Saved as '{final_name}'." if final_name != old_name else "Saved."
            names = _library_mod_names(folder)
            return (_library_rows(folder), _safe_choice_update(_folder_choices(), folder),
                    gr.update(choices=names, value=final_name if final_name in names else None),
                    gr.update(choices=names, value=[]), msg)

        edit_save_btn.click(fn=do_save_edit,
                            inputs=[edit_name_dd, edit_name_field, edit_description_field, library_folder_dd],
                            outputs=[table, library_folder_dd, edit_name_dd, delete_dd, edit_status],
                            queue=False)


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
        return table

    # ── Generate ────────────────────────────────────────────────────────

    def _build_decompile_section(self, api_session, model_dd):
        gr.Markdown("### Decompile a RefMod\n"
                    "See what a mod holds: its pictures, video and audio, rebuilt as ordinary "
                    "files you can view, play and save. These are **reconstructions of what the "
                    "mod stores, not the original sources** -- anything extraction discarded "
                    "(the original resolution, the parts outside a trim, the fine detail pooled "
                    "away in training mode, the individual pictures behind a merge) can't come "
                    "back. Encode-mode mods come closest to their sources.")
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
                frames = mod.encoder_frames_uint8().numpy()
                times = list(mod.enc_times)
                label = ("picture {}" if mod.encoder_layout() == encframes.LAYOUT_STILLS
                         else "{:.1f}s")
                items = [(frames[i], label.format(i + 1 if "picture" in label else times[i]))
                         for i in range(len(frames))]
                note = (f"{len(items)} stored frame(s): what the text encoder is shown for this "
                        f"mod, at a reduced size.")
                gallery = gr.update(value=items, visible=True)
            else:
                note = ("*No stored encoder frames"
                        + (" -- audio mods have none." if mod.kind == "audio"
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

            spec_json = json.dumps(spec)
            set_pending_extract(spec_json)
            try:
                self._submit(api_session, model_type,
                             {"video_length": 107,
                              "custom_settings": {SETTING_COMBINED:
                                                  pack_refmod_setting(extract_json=spec_json)}},
                             DecompileCallbacks())
            except Exception as e:
                return (f"Decompile failed to run: {e!r}", *show_outputs(name))
            finally:
                set_pending_extract(None)
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
            "# ⚠️ Prefer the **Media Generator** tab for actual generation\n"
            "**It has far more options (resolution categories, attention mode, memory profile, "
            "output filename, and everything else Wan2GP's main form offers) -- and your RefMods "
            "are available there too, under LoRAs → MiniMax H3 RefMods (inline).** This panel below "
            "is kept as a simpler fallback with only a subset of the fields.")
        gr.Markdown("### Generate with RefMods\n"
                    "Covers the fields you're most likely to tune here: prompt, resolution, frame "
                    "count, steps, sampler, reference-image budget, LoRAs, step-skipping "
                    "accelerators, sliding window, and sol-attn sparsity, plus up to "
                    f"{IMAGE_ROWS} image + {VIDEO_ROWS} video RefMods (MiniMax H3 Ref2VA's own "
                    "native caps). Anything else keeps whatever you last set on the main "
                    "**Media Generator** tab for this model (use *Sync from the main form* below), "
                    "or that model's own factory defaults otherwise -- so no setting is ever left "
                    "unset, even ones without a widget here.")
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

        mod_rows, stack_controls = self._build_mod_picker_rows()

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
                    info="Only used when Sol-Attn is selected above. Higher = sparser/faster, "
                         "lower = denser/more faithful. End Tau is fixed at 0.8.")

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
                "custom_settings": {SETTING_COMBINED:
                                    pack_refmod_setting(state_json=json.dumps(state_payload))},
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
        flat_rows = [c for row in mod_rows for c in row] + list(stack_controls)
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

    def create_ui(self, api_session):
        with gr.Column() as root:
            if self._patch_error:
                gr.Markdown(f"⚠️ **RefMods could not hook into MiniMax H3**: {self._patch_error}")
            gr.Markdown(f"## {PlugIn_Name}\n"
                       "No-training reference mods for MiniMax H3 Ref2VA. See the README bundled "
                       "with this plugin for how the mechanism works and its current limitations.")
            model_choices = _model_choices(api_session)
            model_dd = gr.Dropdown(
                label="MiniMax H3 Ref2VA model (used for Extract and Generate below)",
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
                with gr.Tab("Extract"):
                    self._build_extract_section(api_session, model_dd)
                with gr.Tab("Library"):
                    self._build_library_section(api_session, model_dd)
                with gr.Tab("Decompile"):
                    self._build_decompile_section(api_session, model_dd)
                with gr.Tab("Generate"):
                    self._build_generate_section(api_session, model_dd)
        return root
