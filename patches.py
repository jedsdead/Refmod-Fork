"""
patches.py -- wires the RefMod mechanism into Wan2GP's MiniMax H3 pipeline.

Why monkeypatching, and why *these* three methods
---------------------------------------------------
Wan2GP's own reference-conditioning code (models/minimax_h3/pipeline.py) builds
two parallel lists inside ``MiniMaxH3Pipeline.generate()``: ``refs`` (kind /
shape metadata) and ``visual_latents`` (the actual VAE latents). Both are
local variables assembled by ``_add_image_reference`` / ``_add_video_reference``
and are only "closed off" into the packed conditioning right before sampling
starts. A plugin living outside wgp.py's source tree cannot reach into those
locals -- so instead of re-implementing (and fatally risking drifting out of
sync with) the ~300-line ``generate()`` method, this module:

1. Wraps ``_add_image_reference`` / ``_add_video_reference`` so that, when
   handed one of our own lightweight sentinel objects instead of a real
   image/video tensor, they append the *precomputed* RefMod latent straight
   into ``refs`` / ``visual_latents`` -- skipping the pixel resize + VAE
   encode, but otherwise going through the exact same, unmodified code path
   a live reference would. This is the entire point of a "RefMod": skip the
   repeated encode, not the model's own attention mechanism.

2. Wraps ``generate()`` itself so that, right before calling the original,
   it reads a small JSON blob out of the ``custom_settings`` dict (Wan2GP's
   existing generic per-model settings channel, already plumbed end to end
   from a submitted task to ``pipeline.generate(**kwargs)``) and turns it
   into sentinel objects appended to ``input_ref_images`` / ``input_frames``
   / ``input_frames2`` -- the same public parameters a live reference would
   use. This means RefMods work even when the user supplies *no* live
   reference at all, which is the main point of the feature.

3. The same ``generate()`` wrapper also recognizes a second, distinct
   ``custom_settings`` key that means "this call is a RefMod *extraction*
   job, not a real render": it runs the VAE encode + optional compression
   directly (a few seconds of work), saves the .safetensors file, and
   returns ``None`` immediately -- the same graceful no-output outcome
   Wan2GP already produces when a user aborts a generation, so nothing
   downstream needs to change to handle it.

4. Declares our two ``custom_settings`` ids on MiniMax H3's model
   definition (see ``_install_model_def_patch`` below). This one is not
   optional: Wan2GP validates *every* submitted task's ``custom_settings``
   against the ids the target model declares, and silently drops anything
   else -- so without this, points 2 and 3 above would never actually see
   our payload, with no error anywhere to explain why.

None of this touches Wan2GP's own files on disk; it is applied at import
time to the already-loaded ``MiniMaxH3Pipeline`` class, and is idempotent
(safe to call ``install_patches()`` more than once).
"""

from __future__ import annotations

import collections
import functools
import json
import math
import os
import random
import traceback
from typing import Optional

import torch
import torch.nn.functional as F

from . import core, encframes, storage

SETTING_GENERATE = "h3_refmod_state"     # custom_settings key: mods to inject into a real render
SETTING_EXTRACT = "h3_refmod_extract"    # legacy custom_settings key, still read for queued tasks
EXTRACT_WRAPPER_KEY = "__h3refmod_extract__"  # extraction spec carried inside SETTING_GENERATE
_CUSTOM_SETTINGS_MAX_GUESS = 5           # Wan2GP's own CUSTOM_SETTINGS_MAX, for the warning below
STASH_KEY = "_h3refmod_selection"        # key inside the session `state` dict for the inline panel

# Fork: a letter Wan2GP never puts in a video prompt type. A custom setting
# keyed to it is never shown on the Media Generator form, but still travels
# with every task, queue entry and saved settings file.
HIDDEN_SETTING_FLAG = "§"
# Fork: earlier fork builds (up to fork.23) wrote {"state": ..., "extract": ...}
# into SETTING_GENERATE, and forks 12-21 used the key "h3_refmod". Both are
# still read (see read_refmod_payload) so their saved settings and queued
# tasks keep working; nothing writes them any more.
LEGACY_FORK_SETTING = "h3_refmod"

# Fork: audio in a mod (an audio mod, or a visual mod's soundtrack) -- how much
# is kept by default, and the shortest allowed. H3 documents 2s as the minimum
# length of an audio reference, so a shorter clip is refused at extraction.
DEFAULT_AUDIO_SECONDS = 4.0
MIN_AUDIO_SECONDS = 2.0
REFERENCE_SECONDS_MAX = 15.0   # fork: H3's limit for a reference video


class ExtractionBlocked(ValueError):
    """An extraction refused for a reason the user can fix; its message is
    shown in the status box instead of a traceback."""


def read_refmod_payload(custom_settings):
    """(selection json, extraction spec) from a task's custom_settings.

    The current form is the one generate() writes and reads below: a plain
    selection, or {EXTRACT_WRAPPER_KEY: spec}, in SETTING_GENERATE (plus the
    legacy SETTING_EXTRACT key). Also accepted, for tasks and settings saved
    by earlier fork builds: {"state": ..., "extract": ...} in either
    SETTING_GENERATE or LEGACY_FORK_SETTING."""
    if not isinstance(custom_settings, dict):
        return None, None
    state, extract = None, custom_settings.get(SETTING_EXTRACT) or None
    for key in (LEGACY_FORK_SETTING, SETTING_GENERATE):   # the declared key wins
        blob = custom_settings.get(key)
        if not blob:
            continue
        try:
            parsed = json.loads(blob) if isinstance(blob, str) else blob
        except Exception:
            _log(f"could not parse {key}; ignoring it")
            continue
        if not isinstance(parsed, dict):
            continue
        if EXTRACT_WRAPPER_KEY in parsed:
            extract = parsed[EXTRACT_WRAPPER_KEY] or extract
        elif "state" in parsed or "extract" in parsed:
            state = parsed.get("state") or state
            extract = parsed.get("extract") or extract
        elif "rows" in parsed:
            state = blob if isinstance(blob, str) else json.dumps(parsed)
    if isinstance(extract, str):
        try:
            extract = json.loads(extract)
        except Exception:
            pass
    return state, extract
FPS_ASSUMED_FOR_DURATION_ESTIMATE = 24   # matches plugin.py's own constant of the same name --
                                         # MiniMax H3's own default fps, used only to turn a
                                         # latent frame count into an estimated-seconds figure
                                         # for status messages.
AUDIO_LATENTS_PER_SECOND = 40            # matches plugin.py's own constant of the same name --
                                         # MiniMax H3's own audio VAE encoder downsamples by
                                         # 800x at 32kHz = 40 latents/s exactly (see
                                         # components/audio_autoencoder.py's own docstring).

_PATCH_MARKER = "_h3refmod_plugin_patched"


def is_minimax_h3_ref2va(model_type, get_base_model_type_fn=None) -> bool:
    """True if ``model_type`` -- which may be an arbitrary finetune name,
    not necessarily prefixed with "minimax_h3_ref2va" -- is actually a
    MiniMax H3 Ref2VA-family model once resolved to its true underlying
    architecture.

    A finetune's model_type identifier is an arbitrary string chosen at
    finetune-creation time (often derived from a checkpoint filename or a
    display name the user typed) and is *not* guaranteed to start with the
    base architecture's own name -- so a plain prefix check on model_type
    alone is unreliable and will silently misclassify some finetunes.
    ``get_base_model_type_fn`` (Wan2GP's own ``get_base_model_type``, when
    available) resolves to ``model_def["architecture"]`` instead, which
    always correctly reflects a finetune's true base regardless of how it
    was named. Falls back to a plain prefix check on ``model_type`` itself
    if the resolver isn't available (e.g. an older Wan2GP build without it)
    -- still correct for the common case where a finetune's own name does
    happen to start with the architecture name, just not for others.
    """
    model_type = str(model_type or "")
    if not model_type:
        return False
    if callable(get_base_model_type_fn):
        try:
            base = get_base_model_type_fn(model_type)
            if base:
                return str(base).startswith("minimax_h3_ref2va")
        except Exception:
            pass
    return model_type.startswith("minimax_h3_ref2va")


# Fork: RefMods on FL2VA (and finetunes built on it, e.g. VDN), and on the
# FL2VA ControlNet-Union model, which is FL2VA plus a control branch. See the
# "FL2VA support" section below for how mods reach a model with no reference
# inputs of its own.
FL2VA_REFMODS = True
CONTROLNET_REFMODS = True


def _fl2va_family_prefixes():
    return ("minimax_h3_fl2va",) + (("minimax_h3_control",) if CONTROLNET_REFMODS else ())


def is_minimax_h3_refmod_capable(model_type, get_base_model_type_fn=None) -> bool:
    """Models RefMods can be used with: Ref2VA, plus (fork) FL2VA, its
    finetunes and the FL2VA ControlNet model. Resolves finetunes to their
    architecture exactly as is_minimax_h3_ref2va does."""
    if is_minimax_h3_ref2va(model_type, get_base_model_type_fn):
        return True
    model_type = str(model_type or "")
    if not FL2VA_REFMODS or not model_type:
        return False
    if callable(get_base_model_type_fn):
        try:
            base = get_base_model_type_fn(model_type)
            if base:
                return str(base).startswith(_fl2va_family_prefixes())
        except Exception:
            pass
    return model_type.startswith(_fl2va_family_prefixes())


class _RefModImageSentinel:
    """Stands in for a single-frame ("image kind") reference. Carries an
    already-encoded, already-weighted VAE latent [1, 24, 1, H, W]."""
    __slots__ = ("latent", "hide_ref", "enc", "max_short", "label_first")

    def __init__(self, latent: torch.Tensor):
        self.latent = latent
        self.hide_ref = False
        self.enc = None   # presentation payload: dict or None
        self.max_short = None     # fork: the row's "Max size" (short edge, px)
        self.label_first = False  # fork: see STABLE_REFMOD_LABELS

    @property
    def dtype(self):
        # Wan2GP's shared guide_to_float()/guide_to_dtype() helpers probe
        # `.dtype` to decide whether a reference is a uint8 control guide
        # that needs converting; reporting the latent's own (floating) dtype
        # makes them treat a sentinel as an already-float tensor and hand it
        # straight back untouched, instead of raising AttributeError.
        return self.latent.dtype


class _RefModVideoSentinel:
    """Stands in for a multi-frame ("video kind") reference. Carries an
    already-encoded, already-weighted VAE latent [1, 24, T, H, W].

    ``generate()`` touches ``input_frames``/``input_frames2`` more than once
    before ever reaching ``_add_video_reference`` (which is the only place
    that's patched to actually recognize this sentinel and use its latent):
    it runs every entry through ``_as_video()`` first (also patched, to pass
    a sentinel through untouched), then computes a total-duration budget
    check via ``sum(video.shape[1] for video in video_sources) / fps``. The
    ``.shape`` property below exists purely to survive *that* second access
    without crashing -- it's a plausible reconstructed pixel-space
    [C, T, H, W] shape derived from the latent's own dims (undoing the video
    VAE's causal 4:1 temporal compression and 16x spatial downsampling), not
    real pixel data (there is none for a RefMod)."""
    __slots__ = ("latent", "hide_ref", "enc", "is_stack", "max_short", "picture_encs")

    def __init__(self, latent: torch.Tensor):
        self.latent = latent
        self.hide_ref = False
        self.enc = None   # presentation payload: dict or None
        # Fork: a multi-picture mod sent as ONE video reference ("Send as one
        # video"). Its latent frames are separate pictures, not footage: no
        # duration, never trimmed by the reference-video budget, and split
        # back into pictures where a video can't go.
        self.is_stack = False
        self.picture_encs = None  # fork: a stack's per-picture prompt entries
        self.max_short = None     # fork: the row's "Max size" (short edge, px)

    @property
    def dtype(self):
        # See _RefModImageSentinel.dtype. This one matters in practice: on
        # image/single-frame output WanGP itself now folds the first
        # reference video into the image references with
        # guide_to_float(input_frames), which reads `.dtype` first.
        return self.latent.dtype

    @property
    def shape(self):
        if self.is_stack:
            # Fork: pictures have no duration. Report the shortest valid
            # reference (5 frames) so Wan2GP's 15-second reference-video
            # budget neither counts a stack as footage nor trims it.
            return (3, 5, self.latent.shape[3] * 16, self.latent.shape[4] * 16)
        t = self.latent.shape[2]
        t_px = (t - 1) * 4 + 1 if t > 1 else 1
        h_px = self.latent.shape[3] * 16
        w_px = self.latent.shape[4] * 16
        return (3, t_px, h_px, w_px)

    def __getitem__(self, key):
        """Support the one slicing shape Wan2GP applies to reference videos:
        ``video[:, :max_frames]``, used to share the 15s reference budget
        between several reference videos (only when "-" is in
        video_prompt_type). A RefMod carries a latent rather than pixels, so
        the pixel-frame count is converted back to latent frames -- undoing
        the video VAE's causal 4:1 temporal compression -- and a trimmed
        sentinel is returned. Any other indexing is refused loudly rather
        than silently returning something wrong."""
        if (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None)
                and isinstance(key[1], slice) and key[1].start in (None, 0) and key[1].step is None):
            stop_px = key[1].stop
            if stop_px is None or self.is_stack:
                return self
            keep_t = max(1, (int(stop_px) - 1) // 4 + 1)
            if keep_t >= self.latent.shape[2]:
                return self
            trimmed = _RefModVideoSentinel(self.latent[:, :, :keep_t])
            trimmed.hide_ref = self.hide_ref
            # Fork: keep the prompt entry (only the frames still present) and
            # the row's Max size across the trim.
            trimmed.enc = _enc_trimmed(self.enc, ((keep_t - 1) * 4 + 1) / FPS_ASSUMED_FOR_DURATION_ESTIMATE)
            trimmed.max_short = self.max_short
            return trimmed
        raise TypeError(f"_RefModVideoSentinel supports only [:, :n] slicing, got {key!r}")


class _RefModAudioSentinel:
    """Stands in for an audio-kind reference. Carries an already-encoded,
    already-weighted audio VAE latent [1, 32, 2, T] in ``.latent``.

    Newer Wan2GP builds route every audio reference through
    ``_prepare_audio_references(sources)`` (patched below) before
    ``_add_audio_reference`` ever sees it -- that function computes a
    duration budget up front (``torch.is_tensor(source)`` vs.
    ``sf.info(source).duration``) and can truncate whichever "waveform" it
    produces per source if the combined total exceeds 15s. A plain object
    fails the ``torch.is_tensor()`` check and would be treated as a file
    path (crashing), and even a real ``torch.Tensor`` subclass isn't safe
    here: the function's own truncation slicing on a real waveform doesn't
    reliably preserve a Tensor subclass or custom attributes across the
    op, which would silently detach ``.latent`` from the result. Patching
    ``_prepare_audio_references`` itself to recognize this sentinel by
    ``isinstance`` *before* any of its tensor-vs-path logic runs sidesteps
    both problems -- see install_patches() below."""
    __slots__ = ("latent", "hide_ref", "enc")

    def __init__(self, latent: torch.Tensor):
        self.latent = latent
        self.hide_ref = False
        self.enc = None   # presentation payload: dict or None

    @property
    def dtype(self):
        # See _RefModImageSentinel.dtype.
        return self.latent.dtype


def _log(msg: str) -> None:
    print(f"[Refmods] {msg}")


# Which reference kwargs this Wan2GP build's generate() actually accepts.
# Older builds expose two native reference-video slots (input_frames,
# input_frames2) and two audio ones (audio_guide, audio_guide2); newer ones
# add a third of each (input_frames3 / "*", audio_guide3 / "D"). Filled in by
# install_patches() so injection uses whatever exists and falls back to
# direct injection for the rest.
_generate_params = set()


_STATE_ATTR = "_h3refmod_gen_state"

# The pipeline currently inside generate(). _resize_video is a module-level
# helper with no `self`, so this is how the video path reaches that call's
# bookkeeping. Generation is single-threaded per pipeline, so one slot is
# enough; it is only ever read, never trusted to exist.
_ACTIVE = {"pipeline": None}

# Wan2GP's own native reference caps, enforced inline in generate() between
# the reference-building loop and the point where `refs` is read. They are a
# UI/product limit, not an architectural one: MiniMax H3 uses RoPE positions
# computed at runtime, an unbounded reference loop, and free-running
# <Picture N> labels, and the ComfyUI community verified 15 image refs
# working. RefMods are kept out of this check -- live references are not.
_NATIVE_CAP_TOTAL, _NATIVE_CAP_IMAGE, _NATIVE_CAP_VIDEO, _NATIVE_CAP_AUDIO = 12, 9, 3, 3


def _reset_gen_state(pipeline_self) -> dict:
    """Fresh per-generate() bookkeeping. Called at the top of every
    patched generate() call (each sliding window is its own call)."""
    state = {
        "phase": 1,               # 1 until the first _prepare_condition_rows, then 2
        "refs": None,             # phase-1 `refs` list object
        "refs2": None,            # phase-2 `phase_2_refs` list object
        "hidden": [],             # [(position_among_visible_refs, entry)] -- phase 1 only
        "overflow_visual": [],    # [(latent, entry)] video mods past the 2 native kwargs
        "overflow_audio": [],     # [(latent, entry)] audio mods past the 2 native kwargs
        "phase2_overflow_done": False,
    }
    setattr(pipeline_self, _STATE_ATTR, state)
    return state


def _gen_state(pipeline_self) -> dict:
    state = getattr(pipeline_self, _STATE_ATTR, None)
    return state if isinstance(state, dict) else _reset_gen_state(pipeline_self)


def _place_refmod_ref(pipeline_self, sentinel, refs, entry) -> None:
    """Place one RefMod's `refs` entry. The matching latent has already been
    appended by the caller, in natural order.

    Phase 1: if this sentinel was marked `hide_ref` by _inject_refmods (it
    would push Wan2GP's inline cap check over 12/9/2/2), record where it
    belongs instead of appending -- _restore_hidden_refs puts it back at that
    exact position right after the check. Everything else is appended now.

    Phase 2 (the latent-upscaler refinement pass): always append. There is
    no cap check there, and phase 2 aligns its refs with phase 1's *by
    position* (`zip(phase_2_refs, visual_refs)` copies each `kind` across),
    so both phases must end up in the same natural order -- which is why
    hidden entries are restored in place rather than appended at the end."""
    state = _gen_state(pipeline_self)
    if state["phase"] == 1:
        state["refs"] = refs
        if getattr(sentinel, "hide_ref", False):
            state["hidden"].append((len(refs), entry))
            return
    else:
        state["refs2"] = refs
    refs.append(entry)


def _note_refs(pipeline_self, refs) -> None:
    """Remember the refs list object seen by a *live* (non-RefMod) reference
    too -- needed when every RefMod is overflow and none of them passes
    through _add_*_reference itself."""
    state = _gen_state(pipeline_self)
    state["refs" if state["phase"] == 1 else "refs2"] = refs


def _restore_hidden_refs(pipeline_self, visual_latents, audio_latents) -> None:
    """Called from _prepare_condition_rows. On the first call of a generate()
    (end of phase 1, just past the cap check), reinsert hidden entries at
    their original positions and append overflow mods. On a later call
    (phase 2), append overflow *video* mods once more, since phase 2 only
    re-adds input_ref_images and the two native video kwargs."""
    state = _gen_state(pipeline_self)
    if state["phase"] == 1:
        state["phase"] = 2
        refs = state["refs"]
        pending = len(state["hidden"]) + len(state["overflow_visual"]) + len(state["overflow_audio"])
        if not pending:
            return
        if refs is None:
            _log(f"could not place {pending} RefMod reference(s): generate() never exposed its "
                 f"reference list. Generation continues without them.")
            return
        # Positions were recorded among *visible* refs; each earlier insertion
        # shifts later ones by one, hence `+ i`.
        for i, (pos, entry) in enumerate(state["hidden"]):
            refs.insert(pos + i, entry)
        for latent, entry in state["overflow_visual"]:
            visual_latents.append(latent)
            refs.append(entry)
        for latent, entry in state["overflow_audio"]:
            audio_latents.append(latent)
            refs.append(entry)
        if state["hidden"] or state["overflow_visual"] or state["overflow_audio"]:
            _log(f"placed {len(state['hidden'])} RefMod ref(s) past Wan2GP's 12/9/2/2 reference "
                 f"check, plus {len(state['overflow_visual'])} video / "
                 f"{len(state['overflow_audio'])} audio beyond its two native slots each")
    elif not state["phase2_overflow_done"] and state["overflow_visual"]:
        state["phase2_overflow_done"] = True
        if _phase2_mods_off(pipeline_self):   # fork: "Mods in phase 2: Off"
            return
        refs2 = state["refs2"]
        if refs2 is None:
            return
        for latent, entry in state["overflow_visual"]:
            visual_latents.append(latent)
            refs2.append(dict(entry))


# ═══════════════════════════════════════════════════════════════════════════
# Installation
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_DEF_PATCH_MARKER = "_h3refmod_model_def_patched"


def _count_refmod_visual_refs(state_json: str) -> int:
    """How many "visual reference" slots the RefMods named in a
    SETTING_GENERATE payload will occupy once injected -- mirrors
    _inject_refmods()'s own counting exactly: an image-kind mod contributes
    one slot per stacked frame times its copies (each frame becomes its own
    image reference, see core.H3RefMod's docstring), a video-kind mod
    contributes exactly one slot regardless of copies (copies repeats its
    frames within that one slot, doesn't spawn extra ones). Metadata-only
    (no tensor loading) since this only needs kind/latent_t."""
    try:
        state = json.loads(state_json)
    except Exception:
        return 0
    total = 0
    for row in state.get("rows") or []:
        name = row.get("mod")
        strength = float(row.get("strength", 1.0) or 0)
        if not name or strength <= 0:
            continue
        copies = max(1, min(10, int(row.get("copies", 1))))
        meta = storage.read_refmod_meta(storage.mod_path(name))
        if meta is None:
            continue
        kind = meta.get("kind", "image")
        latent_t = max(1, int(meta.get("latent_t", 1)))
        # Fork: a picture stack (including one saved as "video" by ComfyUI)
        # is pictures -- unless the row sends it as one video reference.
        pictures = kind == "image" or core.is_still_stack(kind, latent_t, meta.get("tags"),
                                                          meta.get("source", ""))
        if pictures and _row_send_as(row) == "sheet" and latent_t > 1:
            total += copies   # fork.38: one contact sheet per copy
        elif pictures and row.get("as_video") and latent_t > 1:
            total += 1
        elif pictures:
            total += latent_t * copies
        elif kind == "video":
            total += 1
        # "audio" kind mods don't count as visual references at all -- they
        # go into audio_guide/audio_guide2, not input_ref_images/input_frames.
    return total



_VALIDATE_PATCH_MARKER = "_h3refmod_validate_patched"
_INSUFFICIENT_VISUAL_MARKER = "at least as many reference images and videos as audio references"


def _install_model_def_patch() -> None:
    """Declare our two custom_settings IDs on MiniMax H3's model definition.

    Without this, Wan2GP's own task-submission validation
    (``collect_custom_settings_from_inputs``, called from ``validate_settings``
    for *every* task, including ones submitted through the API/plugin path)
    only keeps ``custom_settings`` entries whose id is declared in the
    model's own ``model_def["custom_settings"]`` list -- anything else is
    silently replaced with ``None`` before the task ever reaches
    ``pipeline.generate()``. MiniMax H3 doesn't declare any custom settings
    of its own, so without this patch our RefMod payload never survives the
    trip from a submitted task to ``generate()``'s ``kwargs`` (it *looks*
    like it worked -- no error anywhere -- generate() just silently runs a
    completely normal render instead of seeing our payload).

    This adds two plain "text" custom settings (ids ``h3_refmod_state`` /
    ``h3_refmod_extract``, matching the SETTING_GENERATE / SETTING_EXTRACT
    keys above) to MiniMax H3 Ref2VA's model definition. They will show up
    as two extra (normally empty) text boxes under "Custom Settings" on
    Wan2GP's own Media Generator form for this model -- harmless, and not
    meant to be hand-edited there; this plugin's own Generate/Extract
    buttons fill them in on submission.
    """
    try:
        from models.minimax_h3 import minimax_h3_handler
    except Exception as e:
        _log(f"could not import minimax_h3_handler to declare custom settings ({e!r}); "
             f"RefMod extraction/injection will likely silently no-op instead of taking effect.")
        return

    FamilyHandler = minimax_h3_handler.family_handler
    if getattr(FamilyHandler, _MODEL_DEF_PATCH_MARKER, False):
        return

    _orig_query_model_def = FamilyHandler.query_model_def

    # ONE declared slot, not two. Wan2GP keeps only the first
    # CUSTOM_SETTINGS_MAX (5) custom settings of a model
    # (get_model_custom_settings truncates with `custom_settings[:MAX]`), and
    # MiniMax H3 Ref2VA now declares four of its own (Mask Denoising Mode,
    # Audio Refinement, plus the two excerpt-position settings). Declaring two
    # here made six, so the last one -- the extraction job -- was silently
    # dropped and an extraction ran as an ordinary generation instead.
    # Extraction jobs now travel inside this same slot (see EXTRACT_WRAPPER_KEY).
    extra_settings = [
        {"id": SETTING_GENERATE, "name": "H3RefModState",
         "label": "RefMods selection (managed by Refmods For Wan GP -- leave blank)",
         "type": "text", "default": "",
         # Fork: hidden on the Media Generator form (see HIDDEN_SETTING_FLAG);
         # it still travels with every task.
         "video_prompt_type": HIDDEN_SETTING_FLAG},
    ]

    @staticmethod
    def patched_query_model_def(base_model_type, model_def):
        result = _orig_query_model_def(base_model_type, model_def)
        if isinstance(result, dict):
            existing = result.get("custom_settings")
            existing = list(existing) if isinstance(existing, list) else []
            existing_ids = {e.get("id") for e in existing if isinstance(e, dict)}
            merged = existing + [s for s in extra_settings if s["id"] not in existing_ids]
            result["custom_settings"] = merged
            # Loud warning rather than another silent drop if Wan2GP ever
            # declares enough of its own settings to push ours past the cap.
            if len(merged) > _CUSTOM_SETTINGS_MAX_GUESS and not getattr(patched_query_model_def, "_warned", False):
                patched_query_model_def._warned = True
                _log(f"warning: this model declares {len(merged)} custom settings and Wan2GP keeps "
                     f"only the first {_CUSTOM_SETTINGS_MAX_GUESS}; RefMods may be ignored. "
                     f"Please report this -- the plugin needs to claim its slot differently.")
        return result

    FamilyHandler.query_model_def = patched_query_model_def
    setattr(FamilyHandler, _MODEL_DEF_PATCH_MARKER, True)
    _log("declared the h3_refmod_state custom setting on MiniMax H3's model definition")

    if getattr(FamilyHandler, _VALIDATE_PATCH_MARKER, False):
        return
    _orig_validate = getattr(FamilyHandler, "validate_generative_settings", None)
    if _orig_validate is None:
        return

    @staticmethod
    def patched_validate_generative_settings(base_model_type, model_def, inputs):
        error = _orig_validate(base_model_type, model_def, inputs)
        # Wan2GP's own pre-flight check (called before pipeline.generate() ever
        # runs) only counts *native* image_refs/video_guide fields -- it has no
        # visibility into RefMods, which only turn into visual references much
        # later, inside our own generate() patch. Rather than re-implementing
        # every rule this function enforces (durations, per-type caps, control-
        # video-specific checks...), only step in for this one specific failure:
        # if RefMods would supply enough visual references to satisfy it, clear
        # it; every other check the original function performs is untouched.
        if error and _INSUFFICIENT_VISUAL_MARKER in error:
            state_json = read_refmod_payload(inputs.get("custom_settings"))[0]
            if state_json:
                refmod_visual_count = _count_refmod_visual_refs(state_json)
                if refmod_visual_count > 0:
                    video_prompt_type = str(inputs.get("video_prompt_type") or "")
                    audio_prompt_type = str(inputs.get("audio_prompt_type") or "")
                    image_count = len(inputs.get("image_refs") or [])
                    video_count = (1 if "V" in video_prompt_type else 0) + (1 if "+" in video_prompt_type else 0)
                    audio_count = (1 if "A" in audio_prompt_type else 0) + (1 if "B" in audio_prompt_type else 0)
                    visual_count = image_count + video_count + refmod_visual_count
                    if audio_count <= visual_count:
                        _log(f"RefMods supply {refmod_visual_count} visual reference(s) -- "
                             f"clearing the native '{audio_count} audio vs "
                             f"{image_count + video_count} visual' pre-flight check "
                             f"({visual_count} visual once RefMods are counted).")
                        return None
        return error

    FamilyHandler.validate_generative_settings = patched_validate_generative_settings
    setattr(FamilyHandler, _VALIDATE_PATCH_MARKER, True)
    _log("patched validate_generative_settings so RefMods count as visual references "
         "against MiniMax H3's audio-reference pre-flight check")


def install_patches() -> Optional[str]:
    """Apply the monkeypatches. Returns None on success, or an error string
    (also printed) if Wan2GP's MiniMax H3 module could not be found -- e.g.
    a very different Wan2GP version. Safe to call more than once."""
    try:
        from models.minimax_h3 import pipeline as h3_pipeline
    except Exception as e:
        msg = (f"could not import models.minimax_h3.pipeline ({e!r}); this Wan2GP "
               f"install may not include MiniMax H3, or its layout has changed. "
               f"RefMod extraction/injection will not be available.")
        _log(msg)
        return msg

    _install_model_def_patch()

    Pipeline = h3_pipeline.MiniMaxH3Pipeline
    if getattr(Pipeline, _PATCH_MARKER, False):
        return None  # already patched (e.g. plugin reloaded)

    _orig_add_image_reference = Pipeline._add_image_reference
    _orig_add_video_reference = Pipeline._add_video_reference
    _orig_add_audio_reference = getattr(Pipeline, "_add_audio_reference", None)
    _orig_load_audio_reference = getattr(Pipeline, "_load_audio_reference", None)
    _orig_generate = Pipeline.generate
    try:
        import inspect as _inspect
        _generate_params.clear()
        _generate_params.update(_inspect.signature(_orig_generate).parameters)
    except Exception:
        pass
    _orig_as_video = getattr(h3_pipeline, "_as_video", None)

    @functools.wraps(_orig_add_image_reference)
    def patched_add_image_reference(self, image, target_width, target_height,
                                     image_refs_relative_size, presentation, visual_latents, refs):
        if isinstance(image, _RefModImageSentinel):
            latent = (_fit_phase_1(self, image, "image", image.latent, target_width, target_height,
                                   image_refs_relative_size)
                      if _should_fit(self, target_width, target_height) else image.latent)
            if _phase2_mods_off(self):
                # Fork, "Mods in phase 2: Off": the label stays in the prompt
                # so the numbering is unchanged, but no reference rows go in.
                _present_refmod(presentation, image)
                return
            latent = _apply_size_limits(self, image, "image", latent, target_width, target_height)
            _present_refmod(presentation, image)
            visual_latents.append(latent)
            _place_refmod_ref(self, image, refs,
                              {"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})
            return
        if isinstance(image, _RefModVideoSentinel):
            # A video-kind RefMod arriving on the *image* reference path. This
            # plugin never puts one there -- something else moved it, e.g. the
            # MiniMax H3 Image Mode plugin, which folds reference videos into
            # reference images for single-frame (text-to-image) output, where a
            # moving reference has no meaning. Passing it through to the
            # original would crash in _to_pil ("int() argument must be ... not
            # '_RefModVideoSentinel'"), so use the clip's first frame as a
            # still image reference instead -- a valid single-frame latent, and
            # the closest thing to what was asked for.
            latent = (_fit_phase_1(self, image, "image", image.latent[:, :, :1], target_width, target_height,
                                   image_refs_relative_size)
                      if _should_fit(self, target_width, target_height) else image.latent[:, :, :1])
            image.enc = _enc_as_still(image.enc)  # one still latent -> "<Picture N>", not "<Video N>"
            latent = _apply_size_limits(self, image, "image", latent, target_width, target_height)  # fork
            _present_refmod(presentation, image)
            visual_latents.append(latent)
            _place_refmod_ref(self, image, refs,
                              {"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})
            if not getattr(patched_add_image_reference, "_warned_video_as_image", False):
                patched_add_image_reference._warned_video_as_image = True
                _log("a video-kind RefMod was handed to the image-reference path (image/"
                     "single-frame output mode); using its first frame as a still reference.")
            return
        _note_refs(self, refs)
        return _orig_add_image_reference(self, image, target_width, target_height,
                                          image_refs_relative_size, presentation, visual_latents, refs)

    @functools.wraps(_orig_add_video_reference)
    def patched_add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs):
        if isinstance(video, _RefModImageSentinel):
            # Symmetric guard: a still RefMod on the video path is simply a
            # one-frame clip, which is a valid video latent.
            latent = video.latent
            _present_refmod(presentation, video)
            visual_latents.append(latent)
            _place_refmod_ref(self, video, refs,
                              {"kind": "video", "latent_t": latent.shape[2],
                               "latent_h": latent.shape[-2], "latent_w": latent.shape[-1], "ref_audio_t": 0})
            return
        if isinstance(video, _RefModVideoSentinel):
            if _phase2_mods_off(self):   # fork, "Mods in phase 2: Off" (see the image hook)
                _present_refmod(presentation, video)
                return
            latent = _apply_size_limits(self, video, "video", video.latent,
                                        *_pass_canvas(self, _gen_state(self)["phase"]))  # fork
            _present_refmod(presentation, video)
            visual_latents.append(latent)
            _place_refmod_ref(self, video, refs,
                              {"kind": "video", "latent_t": latent.shape[2],
                               "latent_h": latent.shape[-2], "latent_w": latent.shape[-1], "ref_audio_t": 0})
            return
        _note_refs(self, refs)
        return _orig_add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs)

    if _orig_add_audio_reference is not None and _orig_load_audio_reference is not None:
        @functools.wraps(_orig_load_audio_reference)
        def patched_load_audio_reference(self, path):
            # generate() calls self._load_audio_reference(audio_guide) inline,
            # *before* _add_audio_reference (which is the only place that
            # recognizes a RefMod sentinel) ever sees it -- the original does a
            # real soundfile.read(path, ...), which would crash on a sentinel,
            # exactly the same problem _as_video patch solves for video.
            if isinstance(path, _RefModAudioSentinel):
                return path
            return _orig_load_audio_reference(self, path)
        Pipeline._load_audio_reference = patched_load_audio_reference

        @functools.wraps(_orig_add_audio_reference)
        def patched_add_audio_reference(self, waveform, presentation, audio_latents, refs):
            if isinstance(waveform, _RefModAudioSentinel):
                latent = waveform.latent
                _present_refmod(presentation, waveform)
                audio_latents.append(latent)
                _place_refmod_ref(self, waveform, refs,
                                  {"kind": "audio", "ref_audio_t": latent.shape[-1]})
                return
            _note_refs(self, refs)
            return _orig_add_audio_reference(self, waveform, presentation, audio_latents, refs)
        Pipeline._add_audio_reference = patched_add_audio_reference
    else:
        _log("could not find _add_audio_reference / _load_audio_reference on MiniMaxH3Pipeline "
             "to patch; audio-kind RefMods will not be available (image and video RefMods are "
             "unaffected). This Wan2GP build may not support direct audio references yet.")

    _orig_prepare_condition_rows = getattr(Pipeline, "_prepare_condition_rows", None)
    if _orig_prepare_condition_rows is not None:
        @functools.wraps(_orig_prepare_condition_rows)
        def patched_prepare_condition_rows(self, visual_latents, audio_latents, generator, *args, **kwargs):
            _append_fl2va_rows(self, visual_latents, audio_latents)   # fork: FL2VA RefMods
            _restore_hidden_refs(self, visual_latents, audio_latents)
            return _orig_prepare_condition_rows(self, visual_latents, audio_latents, generator, *args, **kwargs)
        Pipeline._prepare_condition_rows = patched_prepare_condition_rows
    else:
        _log("no _prepare_condition_rows found on MiniMaxH3Pipeline -- RefMod references will be "
             "added immediately instead of after Wan2GP's own reference-count check, so the "
             "native 9-image / 2-video / 2-audio caps will still apply to RefMods on this build.")

    _orig_prepare_audio_references = getattr(Pipeline, "_prepare_audio_references", None)
    if _orig_prepare_audio_references is not None:
        @functools.wraps(_orig_prepare_audio_references)
        def patched_prepare_audio_references(self, sources):
            # Newer Wan2GP builds route every audio reference through this
            # function *before* _add_audio_reference ever sees it. It computes
            # a shared 15s duration budget across all sources up front
            # (`torch.is_tensor(source)` vs. `sf.info(source).duration`, which
            # crashes on a plain sentinel object) and can truncate each
            # resulting "waveform" if the combined total goes over. A RefMod
            # sentinel is neither a real waveform tensor nor a file path, and
            # even if it were made a torch.Tensor subclass, the function's own
            # slicing on truncation isn't guaranteed to preserve a subclass or
            # a custom .latent attribute across the op -- so this replicates
            # the original function's exact duration/truncation logic, with a
            # dedicated isinstance-checked branch for our own sentinels that
            # operates on the real .latent tensor's own time axis instead,
            # and re-wraps the (possibly truncated) result back into a
            # sentinel so _add_audio_reference downstream still recognizes it.
            # Every non-sentinel source is delegated to the untouched original
            # per-source logic, unchanged.
            import soundfile as sf

            sources = [source for source in sources if source is not None]
            durations = []
            for source in sources:
                if isinstance(source, _RefModAudioSentinel):
                    durations.append(source.latent.shape[-1] / AUDIO_LATENTS_PER_SECOND)
                elif torch.is_tensor(source):
                    durations.append(source.shape[-1] / storage.AUDIO_SAMPLE_RATE)
                else:
                    durations.append(sf.info(source).duration)
            max_duration = 15 / len(sources) if sources and sum(durations) > 15 else None

            waveforms = []
            for source, own_duration in zip(sources, durations):
                if isinstance(source, _RefModAudioSentinel):
                    latent = source.latent
                    if max_duration is not None and own_duration > max_duration:
                        keep_t = max(1, round(max_duration * AUDIO_LATENTS_PER_SECOND))
                        latent = latent[..., :keep_t]
                    rewrapped = _RefModAudioSentinel(latent)
                    rewrapped.hide_ref = getattr(source, "hide_ref", False)  # keep the cap-check decision
                    rewrapped.enc = getattr(source, "enc", None)             # keep the <Audio N> prompt entry
                    waveforms.append(rewrapped)
                    continue
                if torch.is_tensor(source):
                    waveform = source
                else:
                    info = sf.info(source)
                    frames = -1 if max_duration is None else round(max_duration * info.samplerate)
                    audio, sample_rate = sf.read(source, frames=frames, dtype="float32", always_2d=True)
                    waveform = self._waveform(audio, sample_rate)
                if max_duration is not None:
                    waveform = waveform[..., :round(max_duration * storage.AUDIO_SAMPLE_RATE)]
                waveforms.append(waveform)
            return waveforms
        Pipeline._prepare_audio_references = patched_prepare_audio_references
    # else: older Wan2GP builds don't have this function at all -- generate()
    # calls _load_audio_reference/_add_audio_reference directly instead
    # (both already patched above), so nothing extra is needed there.

    if _orig_as_video is not None:
        @functools.wraps(_orig_as_video)
        def patched_as_video(source):
            # generate() runs every entry of `video_sources` (= [input_frames,
            # input_frames2]) through _as_video() itself, *before* looping over
            # them to call _add_video_reference -- so our sentinel has to survive
            # this call too, not just the one inside _add_video_reference, or it
            # crashes here first with "'_RefModVideoSentinel' object has no
            # attribute 'ndim'" before our other patch ever gets a chance to run.
            if isinstance(source, _RefModVideoSentinel):
                return source
            return _orig_as_video(source)
        h3_pipeline._as_video = patched_as_video
    else:
        _log("could not find _as_video in models.minimax_h3.pipeline to patch; "
             "video-kind RefMods will likely fail with an AttributeError at generation time. "
             "Image-kind RefMods and extraction are unaffected.")

    _orig_resize_video = getattr(h3_pipeline, "_resize_video", None)
    if _orig_resize_video is not None:
        @functools.wraps(_orig_resize_video)
        def patched_resize_video(video, height, width):
            # Newer Wan2GP builds resize each reference video to the output
            # resolution right before handing it to _add_video_reference. A
            # RefMod sentinel carries an already-VAE-encoded latent, not pixels
            # -- there is nothing meaningful to bicubic-resize (and no pixel
            # tensor to .permute()), so it passes through untouched and the
            # latent goes into the packed sequence at its own saved
            # resolution, exactly as it did before this step existed.
            if isinstance(video, _RefModVideoSentinel):
                active = _ACTIVE.get("pipeline")
                if active is None or not _should_fit(active, width, height):
                    return video
                fitted = _fit_phase_1(active, video, "video", video.latent, width, height, 100.0)
                if fitted is video.latent:
                    return video
                shrunk = _RefModVideoSentinel(fitted)
                shrunk.hide_ref = video.hide_ref
                shrunk.enc = video.enc
                # fork: the stack flag and the row's options travel with it
                shrunk.is_stack, shrunk.max_short = video.is_stack, video.max_short
                shrunk.picture_encs = video.picture_encs
                return shrunk
            return _orig_resize_video(video, height, width)
        h3_pipeline._resize_video = patched_resize_video
    else:
        _log("no _resize_video found in models.minimax_h3.pipeline -- fine on older Wan2GP "
             "builds that don't have it; on newer ones video-kind RefMods would fail with "
             "\"'_RefModVideoSentinel' object has no attribute 'permute'\".")

    _install_fl2va_hooks(Pipeline)   # fork: _encode_prompt, _prepare_control_rows, forward

    @functools.wraps(_orig_generate)
    def patched_generate(self, *args, **kwargs):
        custom_settings = kwargs.get("custom_settings")
        custom_settings = custom_settings if isinstance(custom_settings, dict) else {}

        # The extraction spec rides inside the generate slot (or the legacy
        # SETTING_EXTRACT key); read_refmod_payload also accepts the shapes
        # earlier fork builds wrote.
        state_json, extract_job = read_refmod_payload(custom_settings)
        if extract_job:
            set_progress_status = kwargs.get("set_progress_status")
            try:
                _run_extract_job(self, extract_job, set_progress_status=set_progress_status)
            except ExtractionBlocked as blocked:   # fork: a refusal the user can fix
                _log(f"extraction refused: {blocked}")
                if set_progress_status is not None:
                    try:
                        set_progress_status(f"RefMod extraction refused: {blocked}")
                    except Exception:
                        pass
            except Exception:
                _log("extraction job failed:\n" + traceback.format_exc())
                if set_progress_status is not None:
                    try:
                        set_progress_status("RefMod extraction failed -- see the console/log for details")
                    except Exception:
                        pass
            return None  # graceful no-output outcome, same as a user-initiated abort

        # Fresh bookkeeping for this call -- every generate() (and every
        # sliding window, each of which is its own call) starts at phase 1.
        state = _reset_gen_state(self)
        _ACTIVE["pipeline"] = self
        # The full output canvas for this call. Phase 1 of a two-phase run is
        # handed a *smaller* canvas (H3_TWO_PHASE_SCALE); comparing against
        # this is how _should_fit tells the phases apart, so single-phase and
        # phase-2 runs are left exactly as they were.
        try:
            state["full_canvas"] = (int(kwargs.get("width")), int(kwargs.get("height")))
        except (TypeError, ValueError):
            state["full_canvas"] = None

        # Fork: this run's sizing options and which route the mods take.
        _clear_fl2va()
        fl2va = FL2VA_REFMODS and _is_fl2va_pipeline(self)
        _set_run_options(self, kwargs, state_json, fl2va)

        # state_json was already read (and cleared if it turned out to carry an
        # extraction job) at the top of this function -- don't re-read it here.
        if state_json and fl2va:
            # Fork: FL2VA has no reference inputs; the mods are staged and
            # added by the FL2VA hooks instead (see "FL2VA support").
            if not kwargs.get("refinement_mode"):
                try:
                    _stage_fl2va_refmods(self, kwargs, state_json)
                except Exception:
                    _clear_fl2va()
                    _log("could not stage RefMods for FL2VA, continuing without them:\n"
                         + traceback.format_exc())
        elif state_json:
            try:
                kwargs = _inject_refmods(self, kwargs, state_json)
            except Exception:
                _log("could not inject RefMods, continuing without them:\n" + traceback.format_exc())

        try:
            return _orig_generate(self, *args, **kwargs)
        finally:
            _clear_fl2va()   # fork: never let staged FL2VA mods leak into the next generation
            _veda_restore(self)
            _REFMODS_IN_USE["on"] = False

    Pipeline._add_image_reference = patched_add_image_reference
    Pipeline._add_video_reference = patched_add_video_reference
    Pipeline.generate = patched_generate
    setattr(Pipeline, _PATCH_MARKER, True)
    _log("patches installed on MiniMaxH3Pipeline (RefMod extraction + injection enabled)")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Inline Media Generator panel support
# ═══════════════════════════════════════════════════════════════════════════

_PREPARE_INPUTS_PATCH_MARKER = "_h3refmod_prepare_inputs_patched"


def install_prepare_inputs_dict_patch(orig_prepare_inputs_dict, get_state_model_type_fn, set_global_fn,
                                      get_base_model_type_fn=None) -> Optional[str]:
    """Lets the inline RefMods panel (injected onto Wan2GP's own Media
    Generator page, see plugin.py's ``_build_inline_refmods_section``) affect
    generations started from *that page's own* Generate button, without
    duplicating a single one of its fields.

    The inline panel writes the user's RefMod selection into
    ``state[STASH_KEY]`` -- a namespaced key on the session state dict this
    plugin owns, untouched by anything else in Wan2GP. It's stored there
    (rather than directly into the model's settings dict) because *that*
    dict gets rebuilt from the live form fields on every edit (Wan2GP
    autosaves the form continuously via ``save_inputs``/``prepare_inputs_dict
    (target="state")``), which would silently wipe a one-off injection the
    next time the user touched an unrelated field like the prompt.

    This wraps ``prepare_inputs_dict`` (a plain function in wgp.py, patched
    here via ``set_global`` -- Wan2GP's own supported way for a plugin to
    replace one of its globals) so that, on every call, it re-reads
    ``state[STASH_KEY]`` and folds it into that call's ``custom_settings``
    -- surviving exactly the autosave cycle that would otherwise erase it.
    The original function's behavior is fully preserved for every other
    model, and for MiniMax H3 whenever the panel's selection is empty.
    ``get_base_model_type_fn`` (see ``is_minimax_h3_ref2va``) makes this
    correctly recognize a MiniMax H3 Ref2VA finetune even when its own
    model_type name doesn't start with "minimax_h3_ref2va".
    """
    if getattr(install_prepare_inputs_dict_patch, _PREPARE_INPUTS_PATCH_MARKER, False):
        return None

    def patched_prepare_inputs_dict(target, inputs, model_type=None, model_filename=None):
        state = inputs.get("state") if isinstance(inputs, dict) else None
        result = orig_prepare_inputs_dict(target, inputs, model_type, model_filename)
        try:
            # fork.34: an output's metadata keeps the task's own selection --
            # the panel may have changed while the task waited in the queue.
            if isinstance(state, dict) and isinstance(result, dict) and target != "metadata":
                resolved_type = model_type or get_state_model_type_fn(state)
                if is_minimax_h3_refmod_capable(resolved_type, get_base_model_type_fn):
                    stash = state.get(STASH_KEY)
                    existing = dict(result.get("custom_settings") or {})
                    if stash:
                        existing[SETTING_GENERATE] = stash
                        result["custom_settings"] = existing
                    elif SETTING_GENERATE in existing:
                        existing.pop(SETTING_GENERATE, None)
                        result["custom_settings"] = existing or None
                else:   # fork.31: Flux 2 / Flux 3 panel
                    from . import flux
                    result = flux.fold_stash(state, result, resolved_type, get_base_model_type_fn)
        except Exception:
            _log("prepare_inputs_dict patch: RefMod injection failed, leaving settings "
                 "untouched for this call:\n" + traceback.format_exc())
        return result

    try:
        set_global_fn("prepare_inputs_dict", patched_prepare_inputs_dict)
    except Exception as e:
        msg = f"could not patch prepare_inputs_dict ({e!r}); the inline Media Generator panel will not work."
        _log(msg)
        return msg
    setattr(install_prepare_inputs_dict_patch, _PREPARE_INPUTS_PATCH_MARKER, True)
    _log("hooked prepare_inputs_dict so the inline RefMods panel on the Media Generator page "
         "persists across form edits")
    return None


_GET_MODEL_SETTINGS_PATCH_MARKER = "_h3refmod_get_model_settings_patched"


def install_get_model_settings_patch(orig_get_model_settings, set_global_fn,
                                     get_base_model_type_fn=None) -> Optional[str]:
    """Closes a gap the ``prepare_inputs_dict`` patch above doesn't cover:
    `Generate` (``process_prompt_and_add_tasks`` in wgp.py) does *not* call
    ``prepare_inputs_dict`` again -- it reads the task straight out of
    ``get_model_settings(state, model_type)``, a plain cache lookup
    (``state["all_settings"][model_type]``) last refreshed whenever
    ``save_inputs``/``prepare_inputs_dict`` most recently ran. Since that
    only happens on a *native* form field changing (see the docstring
    above), if the very last thing the user touched before clicking
    Generate was a RefMod picker in the inline panel -- not any native
    field -- that cache is stale and the task would be built from
    whatever RefMod selection existed the last time a native field changed,
    not the current one.

    This wraps ``get_model_settings`` itself (also via ``set_global``) so
    the exact same freshest-``state[STASH_KEY]`` injection happens one more
    time, right at the point the task is actually assembled -- the last
    possible moment before it's queued. Every other model, and MiniMax H3
    with an empty selection, are returned completely unchanged.
    ``get_base_model_type_fn`` (see ``is_minimax_h3_ref2va``) makes this
    correctly recognize a MiniMax H3 Ref2VA finetune even when its own
    model_type name doesn't start with "minimax_h3_ref2va".
    """
    if getattr(install_get_model_settings_patch, _GET_MODEL_SETTINGS_PATCH_MARKER, False):
        return None

    def patched_get_model_settings(state, model_type):
        settings = orig_get_model_settings(state, model_type)
        try:
            if (isinstance(settings, dict) and isinstance(state, dict)
                    and is_minimax_h3_refmod_capable(model_type, get_base_model_type_fn)):
                stash = state.get(STASH_KEY)
                existing = dict(settings.get("custom_settings") or {})
                if stash:
                    existing[SETTING_GENERATE] = stash
                    settings = dict(settings)
                    settings["custom_settings"] = existing
                elif SETTING_GENERATE in existing:
                    existing.pop(SETTING_GENERATE, None)
                    settings = dict(settings)
                    settings["custom_settings"] = existing or None
            elif isinstance(settings, dict) and isinstance(state, dict):   # fork.31: Flux
                from . import flux
                settings = flux.fold_stash(state, settings, model_type, get_base_model_type_fn)
        except Exception:
            _log("get_model_settings patch: RefMod freshness check failed, leaving settings "
                 "untouched for this call:\n" + traceback.format_exc())
        return settings

    try:
        set_global_fn("get_model_settings", patched_get_model_settings)
    except Exception as e:
        msg = f"could not patch get_model_settings ({e!r}); a RefMod change right before " \
             f"clicking Generate (with no other field touched in between) may not always " \
             f"be picked up."
        _log(msg)
        return msg
    setattr(install_get_model_settings_patch, _GET_MODEL_SETTINGS_PATCH_MARKER, True)
    _log("hooked get_model_settings so the inline panel's RefMod selection is always fresh "
         "at the moment Generate actually queues the task")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Generation-time injection
# ═══════════════════════════════════════════════════════════════════════════

def _build_refmod_sentinels(pipeline_self, state_json: str, window_no=None, image_output=False,
                            visual_labels=True, offsets=None):
    """Load a RefMod selection and turn it into sentinels, one list per kind.

    Shared by the Ref2VA route (_inject_refmods, which hands the sentinels to
    generate()'s own reference inputs) and, in this fork, the FL2VA route
    (_stage_fl2va_refmods, which can't use those inputs at all). Returns
    (image_sentinels, video_sentinels, audio_sentinels, total_tokens,
    retention, window_mode), or None when there is nothing to inject.
    ``visual_labels`` False skips decoding prompt-view frames for mods saved
    without them. ``offsets`` ({"image", "video", "audio"}) is how many labels
    of each kind the generation's own pictures/references take ahead of the
    mods; each kind's mods are ordered so the prompt labels picked in the
    panel (a row's "label" / "audio_label") come out where asked."""
    later_window = isinstance(window_no, int) and window_no > 1
    try:
        state = json.loads(state_json)
    except Exception as e:
        _log(f"could not parse RefMod state ({e!r}); ignoring")
        return None
    if EXTRACT_WRAPPER_KEY in state:
        return None  # an extraction spec, not a mod selection (already handled upstream)
    rows = state.get("rows") or []
    if not rows:
        return None
    retention = float(state.get("retention", 1.0))
    curve = state.get("curve")  # [direction, shape, value] or None
    if isinstance(curve, list):
        curve = tuple(curve)
    seed = int(state.get("scramble_seed", -1))
    window_mode = _window_mode_from_state(state)
    shown = _shown_frames_from_state(state)   # fork: "Text encoder options"

    loaded_rows = []  # (mod, strength, copies, row), one entry per selected row, in order
    for row in rows:
        name = row.get("mod")
        strength = float(row.get("strength", 1.0))
        copies = max(1, min(10, int(row.get("copies", 1))))
        if not name or strength <= 0.0:
            continue
        try:
            mod = _load_refmod_cached(name)
        except Exception as e:
            _log(f"could not load mod {name!r}, skipping ({e!r})")
            continue
        loaded_rows.append((mod, strength, copies, row))

    if not loaded_rows:
        return None

    if seed >= 0 and len(loaded_rows) > 1:
        rng = random.Random(seed)
        rng.shuffle(loaded_rows)
        keep = rng.randint(max(1, len(loaded_rows) // 2), len(loaded_rows))
        loaded_rows = loaded_rows[:keep]
        _log(f"scramble seed={seed}: kept {len(loaded_rows)}/{len(rows)} row(s), order shuffled")

    # Fork: sentinels are collected per mod ("groups") so each kind can be put
    # in the order of the labels picked in the panel. A group is
    # [wanted label number or None, display name, [sentinels]].
    image_groups, video_groups, audio_groups = [], [], []
    total_tokens = 0
    for mod, strength, copies, row in loaded_rows:
        eff = max(0.0, min(2.0, strength * retention))
        latent = _weighted_latent_cached(mod, eff, curve)
        if latent is None:
            continue
        total_tokens += mod.token_count
        if visual_labels:
            _ensure_enc_frames(pipeline_self, mod)   # fork: mods saved without encoder frames
        want = _row_label(row, "label")
        image_sentinels, video_sentinels = [], []
        # A picture stack ComfyUI saved as kind "video" is still pictures (fork).
        pictures = mod.kind == "image" or mod.still_stack()
        send_as = _row_send_as(row)   # fork.38
        if pictures and send_as == "sheet" and latent.shape[2] > 1:
            # Fork.38, "Send as: Contact sheet": every picture side by side
            # in ONE picture reference (one <Picture N>), laid out in latent
            # space -- no VAE pass. The text encoder is shown the same grid.
            sheet = _tile_stack(latent)
            enc = _enc_sheet(mod, eff, latent.shape[2])
            for _ in range(copies):
                sentinel = _RefModImageSentinel(sheet)
                sentinel.enc = dict(enc) if enc else None
                image_sentinels.append(sentinel)
        elif pictures and send_as == "video" and latent.shape[2] > 1:
            # Fork, "Send as one video": every picture of the mod in ONE video
            # reference -- one <Video N> label instead of one <Picture N> per
            # picture. "Copies" repeats the pictures inside that one slot.
            if copies > 1:
                latent = latent.repeat(1, 1, copies, 1, 1)
            sentinel = _RefModVideoSentinel(latent)
            sentinel.is_stack = True
            sentinel.enc = _enc_stack(mod, eff, shown)
            sentinel.picture_encs = [_enc_item(mod, "image", j, eff) for j in range(latent.shape[2])]
            video_sentinels.append(sentinel)
        elif pictures:
            # An "image" mod can still have more than one latent frame if
            # several still images were stacked together at extraction time
            # (see core.H3RefMod's docstring) -- each frame is an
            # independent image reference, so split it back into one
            # sentinel per frame here rather than sending Wan2GP's
            # single-frame-only image-reference path a multi-frame latent.
            # "Copies" duplicates the whole row's worth of image sentinels.
            t = latent.shape[2]
            for _ in range(copies):
                for j in range(t):
                    sentinel = _RefModImageSentinel(latent[:, :, j:j + 1])
                    # One prompt entry per image, so a two-image mod really
                    # does become <Picture N> and <Picture N+1>.
                    sentinel.enc = _enc_item(mod, "image", j, eff)
                    image_sentinels.append(sentinel)
        elif mod.kind == "video":
            # Video-kind mods go through Wan2GP's own native reference-video
            # slots directly -- the exact same mechanism "Use Two Reference
            # Videos" uses, one mod per slot -- rather than being merged
            # into a single combined tensor. "Copies" here repeats this
            # mod's own frames within its one slot (extending its own
            # duration) instead of spawning extra slot-consuming instances,
            # since MiniMax H3 only exposes 2 such slots in total.
            if later_window and window_mode == WINDOW_MODE_FIRST:
                pass  # this mod sits out every window after the first
            elif later_window and window_mode == WINDOW_MODE_STILL:
                # Keep the identity, drop the motion: one still reference,
                # the mod's own first latent frame. A whole clip handed to a
                # later window can surface a few of its frames at the window
                # boundary; a single still cannot.
                still = _RefModImageSentinel(latent[:, :, :1])
                still.enc = _enc_as_still(_enc_item(mod, "video", 0, eff))
                image_sentinels.append(still)
                want = None   # a picture now: its video label number no longer applies
            else:
                if copies > 1:
                    latent = latent.repeat(1, 1, copies, 1, 1)
                sentinel = _RefModVideoSentinel(latent)
                sentinel.enc = _enc_clip_shown(_enc_item(mod, "video", 0, eff), shown)
                video_sentinels.append(sentinel)
        else:  # "audio"
            # Same pattern as video-kind mods, one slot per mod (2 native
            # audio-reference slots total); "copies" repeats this mod's own
            # time-axis frames within its one slot rather than spawning a
            # second one.
            if copies > 1:
                latent = latent.repeat(1, 1, 1, copies)
            sentinel = _RefModAudioSentinel(latent)
            # Audio prompt entries carry no pixels at all -- the label is the
            # whole thing -- so an audio mod always gets one.
            sentinel.enc = {"type": "audio"} if PROMPT_LABELS_FOR_REFMODS else None
            audio_groups.append([want, mod.name, [_named(sentinel, mod.name, want)]])

        if image_sentinels:
            image_groups.append([want, mod.name, [_named(s, mod.name, want) for s in image_sentinels]])
        if video_sentinels:
            video_groups.append([want, mod.name, [_named(s, mod.name, want) for s in video_sentinels]])
        # Fork: a visual mod's own soundtrack is injected exactly like an
        # audio mod -- its own native audio slot and <Audio N> label -- unless
        # the row's "Include audio" is unticked.
        if mod.kind != "audio" and mod.audio_latent is not None:
            if row.get("use_audio", True):
                soundtrack = _RefModAudioSentinel(mod.audio_latent)
                soundtrack.enc = {"type": "audio"} if PROMPT_LABELS_FOR_REFMODS else None
                audio_want = _row_label(row, "audio_label")
                audio_groups.append([audio_want, mod.name + " (soundtrack)",
                                     [_named(soundtrack, mod.name + " (soundtrack)", audio_want)]])
                _log(f"'{mod.name}' carries a soundtrack (~{mod.audio_latent.shape[-1] / AUDIO_LATENTS_PER_SECOND:.1f}s) "
                     f"-- injected as its own <Audio N> reference")
            else:
                _log(f"'{mod.name}': soundtrack left out (Include audio unticked)")

    if not image_groups and not video_groups and not audio_groups:
        return None

    # Image / single-frame output: a reference *video* has no place to go.
    # WanGP handles its own live case by folding only the first reference
    # video into the image references and switching reference videos off
    # entirely ("if image_outputs and video_references: input_ref_images =
    # [..., guide_to_float(input_frames)]; video_references = False" in
    # pipeline.generate). If we put video RefMods in the native video slots
    # here, that same code would keep the first one and silently drop any
    # others. Convert them ourselves instead -- one still reference per mod,
    # from its first latent frame, exactly what the image path can use.
    if image_output and video_groups:
        for _want, name, sentinels in video_groups:
            converted = []
            for sentinel in sentinels:
                if sentinel.is_stack:
                    # Fork: a picture stack sent as one video goes back to
                    # being its pictures -- all of them, not just the first.
                    converted.extend(_stack_as_pictures(sentinel))
                    continue
                still = _RefModImageSentinel(sentinel.latent[:, :, :1])
                still.enc = _enc_as_still(sentinel.enc)
                still.max_short = sentinel.max_short
                converted.append(still)
            image_groups.append([None, name, [_named(s, name, None) for s in converted]])
        _log(f"image output mode: {len(video_groups)} video-kind RefMod(s) converted to "
             f"still image references (first frame each) -- a reference video has no meaning "
             f"for a single-frame output, and WanGP would otherwise keep only the first one.")
        video_groups = []

    offsets = offsets or {}
    ordered = {}
    for kind, groups in (("image", image_groups), ("video", video_groups), ("audio", audio_groups)):
        ordered[kind] = _order_by_label(groups, int(offsets.get(kind, 0) or 0), kind)
    flat = lambda groups: [s for _w, _n, sentinels in groups for s in sentinels]
    return (flat(ordered["image"]), flat(ordered["video"]), flat(ordered["audio"]),
            total_tokens, retention, window_mode)


_LABEL_WORDS = {"image": "Picture", "video": "Video", "audio": "Audio"}


def _row_label(row, key):
    """A row's chosen label number (fork.27), or None for automatic."""
    try:
        value = int(row.get(key) or 0)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _named(sentinel, name, want):
    """Tag a sentinel's prompt entry with its mod's name and wanted label
    number, so the labels it really got can be reported. The text encoder
    reads only "type", "frames" and "timestamps"; extra keys are ignored."""
    if isinstance(getattr(sentinel, "enc", None), dict):
        sentinel.enc["refmod"] = name
        sentinel.enc["want"] = want
    for enc in getattr(sentinel, "picture_encs", None) or []:
        if isinstance(enc, dict):
            enc["refmod"], enc["want"] = name, None
    return sentinel


def _order_by_label(groups, offset, kind):
    """Order one kind's mods so each lands on the label number picked for it.

    Labels are numbered in order, after the ``offset`` the generation's own
    pictures / references take. Mods with a chosen number go where they
    asked; mods left on Auto fill the numbers in between, in row order. Two
    mods asking for the same number, or a number past a gap, close up --
    the labels actually given are logged when the prompt is encoded."""
    explicit = sorted((g for g in groups if g[0]), key=lambda g: g[0])
    auto = [g for g in groups if not g[0]]
    ordered, position = [], offset + 1
    while explicit or auto:
        if explicit and (explicit[0][0] <= position or not auto):
            group = explicit.pop(0)
        else:
            group = auto.pop(0)
        ordered.append(group)
        position += len(group[2])
    if any(g[0] for g in groups):
        plan, position = [], offset + 1
        for want, name, sentinels in ordered:
            count = len(sentinels)
            label = (f"<{_LABEL_WORDS[kind]} {position}>" if count == 1
                     else f"<{_LABEL_WORDS[kind]} {position}>-<{_LABEL_WORDS[kind]} {position + count - 1}>")
            plan.append(f"'{name}' -> {label}" + (f" (asked for {want})" if want and want != position else ""))
            position += count
        _log(f"{_LABEL_WORDS[kind].lower()} label order: " + ", ".join(plan)
             + (f" -- after {offset} of the generation's own" if offset else ""))
    return ordered


def _inject_refmods(pipeline_self, kwargs: dict, state_json: str) -> dict:
    window_no = kwargs.get("window_no")
    later_window = isinstance(window_no, int) and window_no > 1
    stable = STABLE_REFMOD_LABELS and not kwargs.get("input_ref_images")
    built = _build_refmod_sentinels(
        pipeline_self, state_json, window_no,
        image_output=int(kwargs.get("image_mode") or 0) > 0 and not getattr(pipeline_self, "audio_only", False),
        offsets=_label_offsets(kwargs, later_window, stable, ref2va=True))
    if built is None:
        return kwargs
    image_sentinels, video_sentinels, audio_sentinels, total_tokens, retention, window_mode = built

    state = _gen_state(pipeline_self)

    # Live (non-RefMod) references already in this generation -- they still
    # count against Wan2GP's inline 12/9/2/2 check, so RefMods get whatever
    # room is left before we start hiding them from it.
    video_prompt_type = str(kwargs.get("video_prompt_type") or "")
    audio_prompt_type = str(kwargs.get("audio_prompt_type") or "")
    live_img = len(kwargs.get("input_ref_images") or [])
    live_vid = ((kwargs.get("input_frames") is not None and "V" in video_prompt_type)
                + (kwargs.get("input_frames2") is not None and "+" in video_prompt_type)
                + (kwargs.get("input_frames3") is not None and "*" in video_prompt_type))
    live_aud = ((kwargs.get("audio_guide") is not None and "A" in audio_prompt_type)
                + (kwargs.get("audio_guide2") is not None and "B" in audio_prompt_type)
                + (kwargs.get("audio_guide3") is not None and "D" in audio_prompt_type))

    if image_sentinels:
        existing = kwargs.get("input_ref_images") or []
        kwargs["input_ref_images"] = list(existing) + image_sentinels
        if stable:
            for sentinel in image_sentinels:   # fork: see STABLE_REFMOD_LABELS
                sentinel.label_first = "keep" if later_window else "record"

    native_videos = []
    if video_sentinels:
        remaining = list(video_sentinels)
        if remaining and kwargs.get("input_frames") is None:
            kwargs["input_frames"] = remaining.pop(0)
            native_videos.append(kwargs["input_frames"])
            if "V" not in video_prompt_type:
                video_prompt_type += "V"
        if remaining and kwargs.get("input_frames2") is None:
            kwargs["input_frames2"] = remaining.pop(0)
            native_videos.append(kwargs["input_frames2"])
            if "V" not in video_prompt_type:
                video_prompt_type += "V"
            if "+" not in video_prompt_type:
                video_prompt_type += "+"
        # Newer Wan2GP builds expose a third native reference-video slot
        # (input_frames3, enabled by "*"). Use it when it exists, so a third
        # video RefMod travels the native path -- including through the
        # phase-2 refinement pass, which re-reads video_sources -- instead of
        # being appended directly.
        if remaining and "input_frames3" in _generate_params and kwargs.get("input_frames3") is None:
            kwargs["input_frames3"] = remaining.pop(0)
            native_videos.append(kwargs["input_frames3"])
            if "V" not in video_prompt_type:
                video_prompt_type += "V"
            if "*" not in video_prompt_type:
                video_prompt_type += "*"
        # Wan2GP's generate() only reads two video kwargs; the rest are added
        # straight into refs/visual_latents after the cap check instead.
        for sentinel in remaining:
            # Fork: these never pass through the hooks, so size them here, for
            # the first pass (fitted to phase 1's canvas, then the size limits).
            canvas = _pass_canvas(pipeline_self, 1)
            latent = sentinel.latent
            if getattr(pipeline_self, "_h3_two_phase", False) and canvas[0]:
                latent = _fit_phase_1(pipeline_self, sentinel, "video", latent, canvas[0], canvas[1], 100.0)
            latent = _apply_size_limits(pipeline_self, sentinel, "video", latent, *canvas)
            state["overflow_visual"].append((latent, {
                "kind": "video", "latent_t": latent.shape[2],
                "latent_h": latent.shape[-2], "latent_w": latent.shape[-1], "ref_audio_t": 0}))
        if remaining:
            _log(f"{len(remaining)} video-kind RefMod(s) beyond Wan2GP's two native "
                 f"reference-video slots will be added directly")
        kwargs["video_prompt_type"] = video_prompt_type

    native_audios = []
    if audio_sentinels:
        if "K" in audio_prompt_type:
            # "Use reference-video soundtrack(s)" (K) reads audio_guide/audio_guide2
            # as the soundtrack for the reference video(s) -- the exact slots
            # audio-kind RefMods need. Skip rather than silently overwrite it.
            _log(f"skipped {len(audio_sentinels)} audio-kind RefMod(s) -- 'Use reference-video "
                 f"soundtrack(s)' is already using the audio reference slots for this generation. "
                 f"Turn that off to use audio-kind RefMods instead.")
        else:
            remaining_audio = list(audio_sentinels)
            if remaining_audio and kwargs.get("audio_guide") is None:
                kwargs["audio_guide"] = remaining_audio.pop(0)
                native_audios.append(kwargs["audio_guide"])
                if "A" not in audio_prompt_type:
                    audio_prompt_type += "A"
            if remaining_audio and kwargs.get("audio_guide2") is None:
                kwargs["audio_guide2"] = remaining_audio.pop(0)
                native_audios.append(kwargs["audio_guide2"])
                if "B" not in audio_prompt_type:
                    audio_prompt_type += "B"
            # Third native audio slot on newer builds (audio_guide3, flag "D").
            if remaining_audio and "audio_guide3" in _generate_params and kwargs.get("audio_guide3") is None:
                kwargs["audio_guide3"] = remaining_audio.pop(0)
                native_audios.append(kwargs["audio_guide3"])
                if "D" not in audio_prompt_type:
                    audio_prompt_type += "D"
            for sentinel in remaining_audio:
                latent = sentinel.latent
                state["overflow_audio"].append((latent, {"kind": "audio", "ref_audio_t": latent.shape[-1]}))
            if remaining_audio:
                _log(f"{len(remaining_audio)} audio-kind RefMod(s) beyond Wan2GP's two native "
                     f"audio-reference slots will be added directly")
            kwargs["audio_prompt_type"] = audio_prompt_type

    # Decide which RefMods stay visible to Wan2GP's inline cap check and
    # which get hidden from it (and restored in place right after). Visible
    # as many as fit -- that keeps the separate "at least as many visual as
    # audio references" check satisfied whenever a live audio clip relies on
    # RefMod visuals -- in the same order generate() will process them.
    room_total = _NATIVE_CAP_TOTAL - (live_img + live_vid + live_aud)
    room = {"image": _NATIVE_CAP_IMAGE - live_img, "video": _NATIVE_CAP_VIDEO - live_vid,
            "audio": _NATIVE_CAP_AUDIO - live_aud}
    visible_visual = live_img + live_vid
    visible_audio = live_aud
    for kind, sentinels in (("image", image_sentinels), ("video", native_videos), ("audio", native_audios)):
        for sentinel in sentinels:
            fits = room_total > 0 and room[kind] > 0
            if kind == "audio":
                fits = fits and visible_audio + 1 <= visible_visual
            sentinel.hide_ref = not fits
            if fits:
                room_total -= 1
                room[kind] -= 1
                if kind == "audio":
                    visible_audio += 1
                else:
                    visible_visual += 1

    _REFMODS_IN_USE["on"] = True   # fork: see _veda_full_references
    where = f" [sliding window {window_no}, video mods: {window_mode}]" if later_window else ""
    _log(f"injecting {len(image_sentinels)} image-kind + {len(video_sentinels)} video-kind + "
         f"{len(audio_sentinels)} audio-kind RefMod reference(s), "
         f"retention={retention:.2f} (~{total_tokens} tokens){where}")
    return kwargs


# ═══════════════════════════════════════════════════════════════════════════
# Fork additions: panel options, mod sizing, FL2VA
# ═══════════════════════════════════════════════════════════════════════════
#
# Everything in this section is new in the jedsdead fork and only reached from
# a few marked lines in the code above. It reads the extra fields the fork's
# panel puts in a selection ("as_video", "use_audio", "max_size" per row;
# "fit_to_output", "phase2_mods", "shrink_method", "stack_pictures",
# "video_frames" for the whole panel) and does nothing when they are absent,
# so a plain {"rows": [{"mod", "strength"}]} selection behaves exactly as it
# does without the fork.

# In windows 2+ the frame carried over from the previous window is added to the
# prompt as a picture BEFORE any reference, so it takes <Picture 1> and pushes
# every reference's label up by one -- a prompt written for window 1 then points
# at the wrong picture. With this on, picture RefMods are presented ahead of
# that frame instead, so their labels are the same in every window. Only when
# the generation has no reference images of its own.
STABLE_REFMOD_LABELS = True

# Prompt labels in FL2VA: every mod is labelled as in Ref2VA -- picture and
# video mods get <Picture N> / <Video N> (their stored frames shown to the
# text encoder), audio mods <Audio N>. FL2VA presents its own start/end frames
# as <Picture N> first, so a picture mod's number follows them. Set this to
# "none" to label nothing at all, like the ComfyUI original.
FL2VA_PROMPT_LABELS = "on"

# Earlier fork builds called the window modes whole / picture / first.
_LEGACY_WINDOW_MODES = {"whole": "full", "picture": "still", "first": "first"}


def _window_mode_from_state(state) -> str:
    mode = state.get("window_mode")
    if not mode and state.get("window_video"):
        mode = _LEGACY_WINDOW_MODES.get(str(state.get("window_video")).lower())
    mode = str(mode or WINDOW_MODE_DEFAULT)
    return mode if mode in (WINDOW_MODE_STILL, WINDOW_MODE_FULL, WINDOW_MODE_FIRST) else WINDOW_MODE_DEFAULT


def _shown_frames_from_state(state) -> dict:
    """The panel's "Text encoder options": how many pictures of a stack sent
    as one video, and how many frames of a video mod, the encoder is shown."""
    def count(key, default):
        try:
            return max(1, int(state.get(key) or default))
        except (TypeError, ValueError):
            return default
    stack_mode = state.get("stack_pictures") or encframes.STACK_ALL
    clip_mode = state.get("video_frames") or encframes.STACK_UP_TO_N
    return {"stack_mode": stack_mode if stack_mode in encframes.STACK_CHOICES else encframes.STACK_ALL,
            "stack_n": count("stack_pictures_n", encframes.STACK_DEFAULT_N),
            "clip_mode": clip_mode if clip_mode in encframes.STACK_CHOICES else encframes.STACK_UP_TO_N,
            "clip_n": count("video_frames_n", encframes.MAX_CLIP_FRAMES_SHOWN)}


# Label bookkeeping for one generation (all its windows): how many pictures
# of the generation's own came ahead of the mods' pictures in window 1, so
# later windows can put the mods' pictures back at the same numbers.
_LABEL_BASE = {"image": None}
_LABEL_LOG = {"done": False, "text": ""}   # fork.34: "text" goes into the output's info

# Fork.28: whether this generation carries RefMods (either route), for the
# Veda Sparse adjustment below.
_REFMODS_IN_USE = {"on": False}


def _veda_full_references(transformer) -> None:
    """Veda Sparse attention (the "Veda Sparse" model variants) attends to
    reference rows sparsely, by a learned predictor's choice, at the
    budget its predictor file sets. The FL2VA variants load the text-to-video
    predictor, which was never trained with references and keeps only ~10%
    of them -- so a mod reached the model but was mostly never looked at.
    While a generation carries RefMods, references are attended in full
    instead (Veda's own "keeps all" mode for references: they become global
    rows, like the text); the generated video stays sparse, so most of the
    speed-up remains. Restored when the generation ends."""
    policy = getattr(transformer, "veda_attention", None)
    if policy is None or not _REFMODS_IN_USE["on"] or getattr(policy, "_h3refmod_saved_reference", None):
        return
    reference = getattr(policy, "reference", None)
    if reference is None or getattr(reference, "keeps_all", False):
        return
    try:
        from models.minimax_h3.veda.selection import Budget
        policy._h3refmod_saved_reference = reference
        policy.reference = Budget(ratio=1.0)
        policy.reset()
        _log(f"Veda Sparse: references attended in full while RefMods are in use "
             f"(this model's predictor had them at {reference.describe()})")
    except Exception:
        policy._h3refmod_saved_reference = None
        _log("could not switch Veda's references to full attention; mods may be weak on this "
             "model:\n" + traceback.format_exc())


def _veda_restore(pipeline_self) -> None:
    policy = getattr(getattr(pipeline_self, "transformer", None), "veda_attention", None)
    saved = getattr(policy, "_h3refmod_saved_reference", None) if policy is not None else None
    if saved is not None:
        policy.reference = saved
        policy._h3refmod_saved_reference = None
        policy.reset()


def _label_offsets(kwargs, later_window, stable, ref2va):
    """How many labels of each kind the generation's own inputs take ahead of
    the mods: start/end frames (a sliding window's carried-over frame takes
    the start frame's place) and, in Ref2VA, its own reference images, videos
    and audio. Used to put each mod on the label number picked for it."""
    video_prompt_type = str(kwargs.get("video_prompt_type") or "")
    audio_prompt_type = str(kwargs.get("audio_prompt_type") or "")
    if later_window and stable and _LABEL_BASE["image"] is not None:
        images = _LABEL_BASE["image"]
    else:
        images = (1 if (later_window or kwargs.get("image_start") is not None) else 0) \
            + (1 if kwargs.get("image_end") is not None else 0)
    if not ref2va:
        return {"image": images, "video": 0, "audio": 0}
    return {"image": images + len(kwargs.get("input_ref_images") or []),
            "video": ((kwargs.get("input_frames") is not None and "V" in video_prompt_type)
                      + (kwargs.get("input_frames2") is not None and "+" in video_prompt_type)
                      + (kwargs.get("input_frames3") is not None and "*" in video_prompt_type)),
            "audio": ((kwargs.get("audio_guide") is not None and "A" in audio_prompt_type)
                      + (kwargs.get("audio_guide2") is not None and "B" in audio_prompt_type)
                      + (kwargs.get("audio_guide3") is not None and "D" in audio_prompt_type))}


def _is_ours(item) -> bool:
    return isinstance(item, dict) and "refmod" in item


def _place_picture_label(presentation, entry, mode) -> None:
    """Add a mod picture's prompt entry so its <Picture N> is the same on every
    sliding window. Window 1 ("record") appends it and notes how many of the
    generation's own pictures came first. Later windows ("keep") put it back
    after that many: the frame carried over from the previous window would
    otherwise sit ahead of it and push its number up by one."""
    own = [i for i, item in enumerate(presentation) if isinstance(item, dict)
           and item.get("type") == "image" and not _is_ours(item)]
    if mode == "record":
        if _LABEL_BASE["image"] is None:
            _LABEL_BASE["image"] = len(own)
        presentation.append(entry)
        return
    base = _LABEL_BASE["image"] or 0
    if len(own) > base:
        presentation.insert(own[base], entry)
    else:
        presentation.append(entry)


def _log_prompt_labels(presentation) -> None:
    """Report, once per generation, the label each mod actually got."""
    if _LABEL_LOG["done"] or not isinstance(presentation, list):
        return
    counts = {"image": 0, "video": 0, "audio": 0}
    named = []
    for item in presentation:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind not in counts:
            continue
        counts[kind] += 1
        if _is_ours(item):
            want = item.get("want")
            named.append(f"'{item['refmod']}' -> <{_LABEL_WORDS[kind]} {counts[kind]}>"
                         + (f" (asked for <{_LABEL_WORDS[kind]} {want}>)" if want and want != counts[kind] else ""))
    if named:
        _LABEL_LOG["done"] = True
        _LABEL_LOG["text"] = ", ".join(named)
        _log("RefMods in the prompt: " + _LABEL_LOG["text"])


# ── Caches ────────────────────────────────────────────────────────────────
#
# Every sliding window is its own generate() call, so without these each
# window re-read every mod off disk and re-ran the blur in weighted_latent().
# Keyed by file mtime, so re-extracting a mod under the same name is picked up.
_MOD_CACHE = collections.OrderedDict()
_MOD_CACHE_MAX = 8
_WEIGHTED_CACHE = collections.OrderedDict()
_WEIGHTED_CACHE_MAX = 16
_SHRINK_CACHE = collections.OrderedDict()
_SHRINK_CACHE_MAX = 16


def _cache_put(cache, key, value, limit):
    cache[key] = value
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _load_refmod_cached(name):
    try:
        stamp = os.path.getmtime(storage.mod_path(name) + ".safetensors")
    except Exception:
        stamp = None
    key = (name, stamp)
    mod = _MOD_CACHE.get(key) if stamp is not None else None
    if mod is None:
        mod = storage.load_refmod(name)
        if stamp is not None:
            _cache_put(_MOD_CACHE, key, mod, _MOD_CACHE_MAX)
    else:
        _MOD_CACHE.move_to_end(key)
    mod._h3_cache_key = key if stamp is not None else None
    return mod


def _weighted_latent_cached(mod, strength, curve):
    base = getattr(mod, "_h3_cache_key", None)
    key = (base, round(float(strength), 6), tuple(curve) if isinstance(curve, (list, tuple)) else curve)
    if base is not None and key in _WEIGHTED_CACHE:
        _WEIGHTED_CACHE.move_to_end(key)
        latent = _WEIGHTED_CACHE[key]
    else:
        latent = mod.weighted_latent(strength, curve=curve)
        if base is not None and latent is not None:
            _cache_put(_WEIGHTED_CACHE, key, latent, _WEIGHTED_CACHE_MAX)
    # A clone on the way out: at strength 1.0 this is the cached mod's own
    # tensor, and one in-place write downstream would corrupt it for every
    # later generation.
    return None if latent is None else latent.clone()


# ── Prompt entries ────────────────────────────────────────────────────────

def _decode_cthw(pipeline_self, latent) -> torch.Tensor:
    """VAE-decode a visual latent to [C, T, H, W] float in [-1, 1] on the CPU."""
    decoded = pipeline_self.vae.decode(latent.to(device=pipeline_self.device, dtype=torch.float32))
    return decoded.float().clamp(-1.0, 1.0)[0].cpu()


def decode_encoder_frames(pipeline_self, mod):
    """A visual mod's encoder frames rebuilt from its latent:
    ([N, H, W, 3] float 0..1, timestamps, layout, fps). Pictures (single or
    stacked) are decoded one latent frame at a time -- they were encoded
    separately, so decoding them as one clip would smear them together. A
    video mod is decoded as a clip and sampled at two frames a second."""
    layout = core.expected_enc_layout(mod.kind, mod.latent_t, mod.still_stack())
    if layout == core.ENC_LAYOUT_STILLS:
        pictures = [encframes.cap_long_edge(_qwen_frames_from_cthw(
                        _decode_cthw(pipeline_self, mod.latent[:, :, j:j + 1])[:, :1]), core.ENC_MAX_EDGE)
                    for j in range(mod.latent.shape[2])]
        size = pictures[0].shape[1:3]
        pictures = [p if p.shape[1:3] == size else
                    F.interpolate(p.permute(0, 3, 1, 2), size=tuple(size), mode="bilinear",
                                  align_corners=False).permute(0, 2, 3, 1) for p in pictures]
        frames = torch.cat(pictures)
        return frames, [float(j) for j in range(frames.shape[0])], layout, 0.0
    video = _decode_cthw(pipeline_self, mod.latent)
    indices, times = encframes.clip_picks(video.shape[1], encframes.ENC_FPS)
    keep = encframes.thin_evenly(len(indices), core.ENC_MAX_CLIP_FRAMES)
    indices, times = [indices[i] for i in keep], [times[i] for i in keep]
    frames = encframes.cap_long_edge(_qwen_frames_from_cthw(video[:, indices]), core.ENC_MAX_EDGE_CLIP)
    return frames, times, layout, encframes.ENC_FPS


def _ensure_enc_frames(pipeline_self, mod) -> None:
    """Give a mod saved without encoder frames (older mods, or ones from other
    tools) a set decoded from its latent, once per session -- the mod is
    cached, so later windows and runs reuse them. Without frames a mod still
    conditions the DiT but gets no <Picture N> / <Video N> label."""
    if not PROMPT_LABELS_FOR_REFMODS or mod.kind == "audio":
        return
    wanted = core.expected_enc_layout(mod.kind, mod.latent_t, mod.still_stack())
    if mod.has_encoder_frames() and mod.enc_layout == wanted:
        return
    if getattr(pipeline_self, "vae", None) is None:
        return
    try:
        frames, times, layout, fps = decode_encoder_frames(pipeline_self, mod)
        mod.enc_frames, mod.enc_times, mod.enc_layout, mod.enc_fps = frames, times, layout, fps
        _log(f"'{mod.name}': no stored encoder frames -- decoded {frames.shape[0]} for its prompt "
             f"label (once per session; Library > Encoder frames can store them in the mod)")
    except Exception:
        _log(f"could not decode encoder frames for '{mod.name}'; it is injected without a prompt "
             f"label:\n" + traceback.format_exc())


def _dimmed(frames, strength):
    """The same dimming _enc_item applies: a weaker mod shows the encoder a
    weaker picture."""
    return frames * strength + 0.5 * (1.0 - strength) if strength < 1.0 else frames


def _row_send_as(row) -> str:
    """fork.38: how a multi-picture row is sent -- "pictures" (one reference
    each), "video" (one video reference) or "sheet" (one contact sheet)."""
    if str(row.get("send_as") or "") == "sheet":
        return "sheet"
    return "video" if row.get("as_video") else "pictures"


def _sheet_grid(n: int):
    """(rows, cols) of a contact sheet of n pictures: as square as possible."""
    cols = max(1, math.ceil(n ** 0.5))
    return math.ceil(n / cols), cols


def _tile_stack(latent):
    """A picture stack [1, C, T, H, W] laid out as one picture latent
    [1, C, 1, rows*H, cols*W]; empty cells hold the stack's average."""
    _, c, t, h, w = latent.shape
    rows, cols = _sheet_grid(t)
    fill = latent.float().mean(dim=(2, 3, 4), keepdim=True).to(latent.dtype)
    sheet = fill.expand(1, c, 1, rows * h, cols * w).clone()
    for j in range(t):
        r, k = divmod(j, cols)
        sheet[:, :, 0, r * h:(r + 1) * h, k * w:(k + 1) * w] = latent[:, :, j]
    return sheet


def _enc_sheet(mod, strength, count):
    """The prompt entry for a contact sheet: the stored prompt-view pictures
    in the same grid, as one picture (capped like any prompt-view frame)."""
    frames = getattr(mod, "enc_frames", None)
    if not PROMPT_LABELS_FOR_REFMODS or frames is None or not frames.shape[0]:
        return None
    try:
        frames = frames[:count].float()
        n, h, w, _ = frames.shape
        rows, cols = _sheet_grid(n)
        grid = torch.full((1, rows * h, cols * w, 3), 0.5)
        for j in range(n):
            r, k = divmod(j, cols)
            grid[0, r * h:(r + 1) * h, k * w:(k + 1) * w] = frames[j]
        grid = encframes.cap_long_edge(grid, core.ENC_MAX_EDGE)
        _log(f"'{mod.name}': {n} pictures sent as one contact sheet ({rows}x{cols}) -- one <Picture N>")
        return {"type": "image", "frames": _dimmed(grid, strength).clone()}
    except Exception:
        _log("could not build a contact sheet's prompt entry:\n" + traceback.format_exc())
        return None


def _enc_stack(mod, strength, shown):
    """The prompt entry for a picture stack sent as one video: its pictures
    (all, or up to N) as one video item."""
    frames = getattr(mod, "enc_frames", None)
    if not PROMPT_LABELS_FOR_REFMODS or frames is None or not frames.shape[0]:
        return None
    picks = encframes.stack_picks(frames.shape[0], shown["stack_mode"], shown["stack_n"])
    chosen, stamps = encframes.stack_presentation(frames, picks)
    _log(f"'{mod.name}': {mod.latent_t} pictures sent as one video reference "
         f"(text encoder shown {len(picks)} of {frames.shape[0]})")
    return {"type": "video", "frames": _dimmed(chosen, strength).clone(), "timestamps": stamps}


def _enc_clip_shown(enc, shown):
    """A video mod's prompt entry thinned to the panel's "frames shown"."""
    if not isinstance(enc, dict) or enc.get("type") != "video" or shown["clip_mode"] == encframes.STACK_ALL:
        return enc
    frames = enc.get("frames")
    keep = encframes.thin_evenly(frames.shape[0], shown["clip_n"])
    if len(keep) == frames.shape[0]:
        return enc
    out = dict(enc, frames=frames[keep].clone())
    if enc.get("timestamps"):
        out["timestamps"] = [enc["timestamps"][i] for i in keep]
    return out


def _enc_trimmed(enc, seconds):
    """A clip's prompt entry cut to the part a budget trim left."""
    if not isinstance(enc, dict) or enc.get("type") != "video" or not enc.get("timestamps"):
        return enc
    keep = [i for i, t in enumerate(enc["timestamps"]) if t < seconds + 1e-6] or [0]
    if len(keep) == len(enc["timestamps"]):
        return enc
    return dict(enc, frames=enc["frames"][keep].clone(), timestamps=[enc["timestamps"][i] for i in keep])


def _stack_as_pictures(stack):
    """A picture stack sent as one video, split back into its pictures."""
    pictures = []
    encs = stack.picture_encs or []
    for j in range(stack.latent.shape[2]):
        still = _RefModImageSentinel(stack.latent[:, :, j:j + 1])
        still.enc = encs[j] if j < len(encs) else None
        still.hide_ref = stack.hide_ref
        still.max_short = stack.max_short
        pictures.append(still)
    return pictures


# ── Mod sizing: "Max size", "Fit mods to the output size", phase 2 ────────

PHASE2_MODS_FULL, PHASE2_MODS_TILE, PHASE2_MODS_OFF = "full", "tile", "off"
PHASE1_MODS_FIT, PHASE1_MODS_FULL = "fit", "full"   # fork.30
SHRINK_REENCODE, SHRINK_FAST = "reencode", "fast"


def _row_max_short(row):
    """A row's "Max size" as a short edge in pixels, or None for full size."""
    try:
        value = int(row.get("max_size") or 0)
    except (TypeError, ValueError):
        return None
    return value if value >= 32 else None


def _state_option(state_json, key, default, allowed=None):
    try:
        value = json.loads(state_json).get(key, default) if state_json else default
    except Exception:
        return default
    if allowed is not None:
        value = str(value or default).lower()
        return value if value in allowed else default
    return value


def _dims_for_short_edge(latent, short_px):
    """(width, height) in pixels that bring a mod's short edge down to
    ``short_px``, keeping its shape -- or (None, None) if it already is."""
    try:
        lat_h, lat_w, short_px = int(latent.shape[-2]), int(latent.shape[-1]), int(short_px)
    except Exception:
        return None, None
    stored_short = min(lat_h, lat_w) * _VAE_SPATIAL_FACTOR
    if short_px <= 0 or stored_short <= short_px:
        return None, None
    scale = short_px / float(stored_short)
    new_w = max(32, int(lat_w * _VAE_SPATIAL_FACTOR * scale) // 32 * 32)
    new_h = max(32, int(lat_h * _VAE_SPATIAL_FACTOR * scale) // 32 * 32)
    if (new_w // _VAE_SPATIAL_FACTOR) * (new_h // _VAE_SPATIAL_FACTOR) >= lat_h * lat_w:
        return None, None
    return new_w, new_h


def _fit_output_dims(latent, target_width, target_height):
    """(width, height) in pixels that bring a mod down to a canvas's pixel
    area, keeping its own aspect ratio -- or (None, None) if it fits already."""
    try:
        lat_h, lat_w = int(latent.shape[-2]), int(latent.shape[-1])
        out_w, out_h = int(target_width), int(target_height)
    except Exception:
        return None, None
    stored = lat_h * lat_w
    canvas = (out_w / _VAE_SPATIAL_FACTOR) * (out_h / _VAE_SPATIAL_FACTOR)
    if out_w <= 0 or out_h <= 0 or stored <= canvas:
        return None, None
    scale = math.sqrt(canvas / stored)
    new_w = max(32, int(lat_w * _VAE_SPATIAL_FACTOR * scale) // 32 * 32)
    new_h = max(32, int(lat_h * _VAE_SPATIAL_FACTOR * scale) // 32 * 32)
    if (new_w // _VAE_SPATIAL_FACTOR) * (new_h // _VAE_SPATIAL_FACTOR) >= stored:
        return None, None
    return new_w, new_h


def _phase_1_canvas(width, height):
    """The half-size canvas phase 1 of a two-phase run renders at
    (pipeline.py: round(target / H3_TWO_PHASE_SCALE / 32) * 32)."""
    try:
        from models.minimax_h3 import pipeline as h3_pipeline
        scale = float(getattr(h3_pipeline, "H3_TWO_PHASE_SCALE", 2.0))
    except Exception:
        scale = 2.0
    return (max(32, round(int(width) / scale / 32) * 32), max(32, round(int(height) / scale / 32) * 32))


def _phase_2_tile(width, height):
    """(width, height) of one tile of a tiled phase 2, or None."""
    try:
        from models.minimax_h3 import pipeline as h3_pipeline
        return (int(h3_pipeline._spatial_tiles(int(width))[0][1]),
                int(h3_pipeline._spatial_tiles(int(height))[0][1]))
    except Exception:
        return None


def _pass_canvas(pipeline_self, pass_no):
    """(width, height) a pass renders at: the half-size canvas for phase 1 of
    a two-phase run, otherwise the output size."""
    out_w, out_h = getattr(pipeline_self, "_h3_out_size", (None, None)) or (None, None)
    if not out_w or not out_h:
        return None, None
    if pass_no == 1 and getattr(pipeline_self, "_h3_two_phase", False):
        return _phase_1_canvas(out_w, out_h)
    return int(out_w), int(out_h)


def _shrink_target(pipeline_self, sentinel, latent, canvas_w, canvas_h, pass_no):
    """The smallest of the sizes that apply to this mod on this pass, as
    (width, height, reason), or None to leave it as it is:
      * its row's "Max size" (short edge in pixels),
      * "Fit mods to the output size" -- the canvas of this pass,
      * "Mods in phase 2: Fit to tile" -- one tile of a tiled phase 2.
    Phase 1 of a two-phase run is already fitted to its canvas by
    _fit_latent_to_canvas. Only ever shrinks."""
    candidates = []
    cap = getattr(sentinel, "max_short", None)
    if cap:
        w, h = _dims_for_short_edge(latent, cap)
        if w is not None:
            candidates.append((w, h, f"max size {int(cap)}px"))
    if (pass_no == 1 and getattr(pipeline_self, "_h3_p1_full", False)
            and getattr(pipeline_self, "_h3_two_phase", False)):
        # fork.30: "Mods in phase 1: Full size" -- fit to the output, not phase 1's half canvas.
        out_w, out_h = getattr(pipeline_self, "_h3_out_size", (None, None)) or (None, None)
        if out_w and out_h:
            canvas_w, canvas_h = int(out_w), int(out_h)
    if canvas_w and canvas_h:
        if getattr(pipeline_self, "_h3_fit_output", False):
            w, h = _fit_output_dims(latent, canvas_w, canvas_h)
            if w is not None:
                candidates.append((w, h, "fit to output"))
        if (pass_no == 2 and getattr(pipeline_self, "_h3_tiled_p2", False)
                and getattr(pipeline_self, "_h3_p2_mode", PHASE2_MODS_FULL) == PHASE2_MODS_TILE):
            tile = _phase_2_tile(canvas_w, canvas_h)
            if tile:
                w, h = _fit_output_dims(latent, tile[0], tile[1])
                if w is not None:
                    candidates.append((w, h, "phase-2 tile"))
    return min(candidates, key=lambda c: c[0] * c[1]) if candidates else None


def _fit_phase_1(pipeline_self, sentinel, kind, latent, target_width, target_height, relative_size=100.0):
    if getattr(pipeline_self, "_h3_p1_full", False):   # fork.30: "Mods in phase 1: Full size"
        return latent
    """Fit a mod to phase 1's half-size canvas (fork.29).

    Same size as the original port's _fit_latent_to_canvas, but made the way
    "Shrink method" says. Re-encode (the default) decodes the mod, resizes
    the picture and encodes it again; Fast is the port's own latent resample.
    Averaging VAE latent cells is not the same as shrinking the picture --
    it blurs fine detail, faces first -- and phase 1 is where a two-phase
    run settles identity (phase 2 only refines it from ~0.9 noise), so the
    resample left mods with little visible effect on two-phase runs."""
    if latent is None or latent.dim() != 5:
        return latent
    if getattr(pipeline_self, "_h3_shrink_fast", False):
        return _fit_latent_to_canvas(latent, target_width, target_height, relative_size, f"{kind} RefMod")
    h, w = latent.shape[-2], latent.shape[-1]
    try:
        budget = float(target_width) * float(target_height) * float(relative_size) / 100.0
    except (TypeError, ValueError):
        return latent
    current = (h * _VAE_SPATIAL_FACTOR) * (w * _VAE_SPATIAL_FACTOR)
    if budget <= 0 or current <= budget:
        return latent
    scale = (budget / current) ** 0.5
    new_h = min(h, max(2, int(round(h * scale / 2)) * 2))
    new_w = min(w, max(2, int(round(w * scale / 2)) * 2))
    if new_h >= h and new_w >= w:
        return latent
    smaller = _reencode_smaller(pipeline_self, latent, kind, getattr(sentinel, "is_stack", False),
                                new_w * _VAE_SPATIAL_FACTOR, new_h * _VAE_SPATIAL_FACTOR)
    if smaller is None:
        return _fit_latent_to_canvas(latent, target_width, target_height, relative_size, f"{kind} RefMod")
    _log(f"fitted {kind} RefMod to phase 1's canvas (re-encoded): {w}x{h} -> "
         f"{smaller.shape[-1]}x{smaller.shape[-2]} latent cells")
    return smaller


def _apply_size_limits(pipeline_self, sentinel, kind, latent, canvas_w=None, canvas_h=None, pass_no=None):
    """``latent`` made smaller when a size limit applies (see _shrink_target),
    else ``latent`` unchanged. "Shrink method": Re-encode decodes the mod,
    resizes the pixels and re-encodes them with the generation's own VAE
    (best quality, cached so it happens once per mod and size); Fast
    resamples the stored latent directly, the way _fit_latent_to_canvas does
    (instant, no VAE, rougher)."""
    if latent is None or latent.dim() != 5:
        return latent
    if pass_no is None:
        pass_no = _gen_state(pipeline_self)["phase"]
    target = _shrink_target(pipeline_self, sentinel, latent, canvas_w, canvas_h, pass_no)
    if target is None:
        return latent
    new_w, new_h, reason = target
    if getattr(pipeline_self, "_h3_shrink_fast", False):
        smaller = _fit_latent_to_canvas(latent, new_w, new_h, 100.0, f"{kind} RefMod ({reason})")
    else:
        smaller = _reencode_smaller(pipeline_self, latent, kind, getattr(sentinel, "is_stack", False),
                                    new_w, new_h)
    if smaller is None or smaller.shape[-2] * smaller.shape[-1] >= latent.shape[-2] * latent.shape[-1]:
        return latent
    _log(f"{kind} RefMod shrunk ({reason}, pass {pass_no}): "
         f"{latent.shape[-1] * _VAE_SPATIAL_FACTOR}x{latent.shape[-2] * _VAE_SPATIAL_FACTOR} -> "
         f"{smaller.shape[-1] * _VAE_SPATIAL_FACTOR}x{smaller.shape[-2] * _VAE_SPATIAL_FACTOR}px "
         f"(~{100.0 * smaller.shape[-2] * smaller.shape[-1] / (latent.shape[-2] * latent.shape[-1]):.0f}% "
         f"of its tokens)")
    return smaller


def _reencode_smaller(pipeline_self, latent, kind, is_stack, width, height):
    """Decode, resize the pixels to (width, height), re-encode. Pictures
    (single or a stack) are rebuilt one at a time, never as frames of a clip.
    Returns None to keep the stored latent."""
    try:
        key = (tuple(latent.shape), round(float(latent.float().sum()), 3), kind, bool(is_stack),
               int(width), int(height))
        if key in _SHRINK_CACHE:
            _SHRINK_CACHE.move_to_end(key)
            return _SHRINK_CACHE[key]
        from models.minimax_h3 import pipeline as h3_pipeline
        if kind == "image" or is_stack:
            parts = []
            for j in range(latent.shape[2]):
                picture = _decode_cthw(pipeline_self, latent[:, :, j:j + 1])[:, :1]
                parts.append(pipeline_self._encode_video(
                    h3_pipeline._resize_video(picture, int(height), int(width))))
            smaller = torch.cat(parts, dim=2)
        else:
            video = _decode_cthw(pipeline_self, latent)
            smaller = pipeline_self._encode_video(h3_pipeline._resize_video(video, int(height), int(width)))
        smaller = smaller.to(latent.dtype)
        return _cache_put(_SHRINK_CACHE, key, smaller, _SHRINK_CACHE_MAX)
    except Exception:
        _log("could not rebuild a RefMod smaller; using it at its stored size:\n" + traceback.format_exc())
        return None


def _has_live_visual_refs(kwargs) -> bool:
    """Does the generation have reference images or videos of its own (not
    RefMods)? Checked before the mods are injected."""
    video_prompt_type = str(kwargs.get("video_prompt_type") or "")
    if [r for r in (kwargs.get("input_ref_images") or []) if not isinstance(r, _RefModImageSentinel)]:
        return True
    for key, letter in (("input_frames", "V"), ("input_frames2", "+"), ("input_frames3", "*")):
        value = kwargs.get(key)
        if value is not None and not isinstance(value, _RefModVideoSentinel) and letter in video_prompt_type:
            return True
    return False


def _phase2_mods_off(pipeline_self) -> bool:
    """True during phase 2 of a two-phase run when "Mods in phase 2" is Off
    and it is safe to leave them out (see _set_run_options)."""
    return bool(getattr(pipeline_self, "_h3_p2_off", False)) and _gen_state(pipeline_self)["phase"] == 2


def _tiling_flag() -> str:
    try:
        from models.minimax_h3 import pipeline as h3_pipeline
        return str(getattr(h3_pipeline, "H3_PHASE_2_TILING_FLAG", "~"))
    except Exception:
        return "~"


def _set_run_options(pipeline_self, kwargs, state_json, fl2va) -> None:
    """Read this run's panel options onto the pipeline for the hooks."""
    _LABEL_LOG["done"] = False
    _REFMODS_IN_USE["on"] = False
    if int(kwargs.get("window_no") or 1) <= 1:
        _LABEL_BASE["image"] = None   # a new generation
        _LABEL_LOG["text"] = ""
    # pipeline.py: two_phase = int(guide_phases) > 1 and frozen_target_video is None
    two_phase = (int(kwargs.get("guide_phases") or 1) > 1
                 and kwargs.get("frozen_control_video") is None)
    pipeline_self._h3_two_phase = two_phase
    pipeline_self._h3_out_size = (kwargs.get("width"), kwargs.get("height"))
    pipeline_self._h3_fit_output = bool(_state_option(state_json, "fit_to_output", False))
    pipeline_self._h3_shrink_fast = _state_option(state_json, "shrink_method", SHRINK_REENCODE,
                                                  (SHRINK_REENCODE, SHRINK_FAST)) == SHRINK_FAST
    pipeline_self._h3_tiled_p2 = two_phase and _tiling_flag() in str(kwargs.get("video_prompt_type") or "")
    p2_mode = (_state_option(state_json, "phase2_mods", PHASE2_MODS_FULL,
                             (PHASE2_MODS_FULL, PHASE2_MODS_TILE, PHASE2_MODS_OFF))
               if two_phase else PHASE2_MODS_FULL)
    pipeline_self._h3_p2_mode = p2_mode
    pipeline_self._h3_p1_full = two_phase and _state_option(
        state_json, "phase1_mods", PHASE1_MODS_FIT, (PHASE1_MODS_FIT, PHASE1_MODS_FULL)) == PHASE1_MODS_FULL
    p2_off = p2_mode == PHASE2_MODS_OFF
    if p2_off and not fl2va and _has_live_visual_refs(kwargs):
        # Ref2VA lines phase 2's references up with phase 1's by position;
        # leaving the mods out would shift the generation's own references.
        p2_off = False
        _log("Mods in phase 2: Off was asked for, but this generation has reference images or "
             "videos of its own, so the mods stay in phase 2 (Full)")
    pipeline_self._h3_p2_off = p2_off
    if state_json:
        if pipeline_self._h3_fit_output:
            _log(f"fit to output: on -- visual mods bigger than {kwargs.get('width')}x"
                 f"{kwargs.get('height')} are made smaller to that pixel area")
        if pipeline_self._h3_p1_full:
            _log("Mods in phase 1: Full size -- mods are not shrunk to phase 1's half canvas")
        if two_phase and p2_mode != PHASE2_MODS_FULL:
            _log("Mods in phase 2: " + ("Off" if p2_off else "Fit to tile" if p2_mode == PHASE2_MODS_TILE
                                        else "Full")
                 + ("" if p2_mode != PHASE2_MODS_TILE or pipeline_self._h3_tiled_p2
                    else " (phase 2 is not tiled in this run, so mods stay full size there)"))


# ── FL2VA support ─────────────────────────────────────────────────────────
#
# The Ref2VA route can't work on FL2VA: the pipeline is built with
# reference_mode=False, which skips every reference-building step, and any
# value in input_ref_images / input_frames raises "Image, video, and audio
# references require the Ref2VA checkpoint".
#
# But the transformer itself is model-agnostic. MiniMaxH3Model._layout picks
# build_ref2va_packed_sequence purely on payload["refs"] being non-empty, and
# that packer already takes first/last-frame keyframe anchors alongside
# references. The ComfyUI original relies on exactly this: its Apply node
# appends ref blocks to whatever conditioning it's given, FL2VA included.
#
# So in FL2VA the mods bypass generate()'s reference inputs entirely:
#   * their latents are appended to the condition latents in
#     _prepare_condition_rows (phase 1, plain phase 2, and tiled phase 2,
#     which appends reference rows whole to every tile), and
#   * payload["refs"] is filled in at the top of the transformer's forward
#     whenever generate() left it empty -- phase 1 builds it as None and
#     both phase-2 paths reset it to None.
# The ControlNet model's control rows carry one blank row per condition row
# ahead of the video, built BEFORE the mods are appended, so
# _prepare_control_rows is patched to count them too.

_FL2VA = {"active": False, "visual": [], "audio": [], "entries": [], "announced": False,
          "presentation": [], "visual2": [], "entries2": [], "calls": 0, "pass_no": 1,
          "labels_logged": False}


def _clear_fl2va() -> None:
    _FL2VA.update(active=False, visual=[], audio=[], entries=[], announced=False, presentation=[],
                  visual2=[], entries2=[], calls=0, pass_no=1, labels_logged=False)


def _transformer_has_control(transformer) -> bool:
    """Control checkpoints attach a ControlBlock to their DiT blocks; FL2VA
    (and FL2VA finetunes such as VDN) don't."""
    try:
        return any(getattr(block, "control", None) is not None
                   for block in getattr(transformer, "blocks", None) or ())
    except Exception:
        return False


def _is_fl2va_pipeline(pipeline_self) -> bool:
    return (not getattr(pipeline_self, "reference_mode", False)
            and getattr(pipeline_self, "fixed_prompt", None) is None
            and not getattr(pipeline_self, "audio_only", False)
            and (CONTROLNET_REFMODS
                 or not _transformer_has_control(getattr(pipeline_self, "transformer", None))))


def _stage_fl2va_refmods(pipeline_self, kwargs, state_json) -> bool:
    """Load the selection and stage it for the FL2VA hooks. Order matters:
    the packer assigns condition rows to references in list order, so the
    entries are built in exactly the order their latents get appended."""
    window_no = kwargs.get("window_no")
    labels_on = PROMPT_LABELS_FOR_REFMODS and str(FL2VA_PROMPT_LABELS or "none").lower() != "none"
    visual_labels = labels_on
    later_window = isinstance(window_no, int) and window_no > 1
    built = _build_refmod_sentinels(
        pipeline_self, state_json, window_no,
        image_output=int(kwargs.get("image_mode") or 0) > 0 and not getattr(pipeline_self, "audio_only", False),
        visual_labels=visual_labels,
        offsets=_label_offsets(kwargs, later_window, STABLE_REFMOD_LABELS, ref2va=False))
    if built is None:
        return False
    image_sentinels, video_sentinels, audio_sentinels, total_tokens, retention, window_mode = built
    two_phase = bool(getattr(pipeline_self, "_h3_two_phase", False))
    p2_off = bool(getattr(pipeline_self, "_h3_p2_off", False))

    def sized(sentinel, kind, pass_no):
        # FL2VA can't size mods inside generate() the way the Ref2VA hooks
        # do, so each pass gets its own copy, sized for that pass up front:
        # phase 1 of a two-phase run fitted to its canvas, then the limits.
        canvas = _pass_canvas(pipeline_self, pass_no)
        latent = sentinel.latent
        if two_phase and pass_no == 1 and canvas[0]:
            latent = _fit_phase_1(pipeline_self, sentinel, kind, latent, canvas[0], canvas[1], 100.0)
        return _apply_size_limits(pipeline_self, sentinel, kind, latent, *canvas, pass_no=pass_no)

    def build_pass(pass_no):
        visual, entries = [], []
        if not (pass_no == 2 and p2_off):
            for sentinel in image_sentinels:
                latent = sized(sentinel, "image", pass_no)
                visual.append(latent)
                entries.append({"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})
            for sentinel in video_sentinels:
                latent = sized(sentinel, "video", pass_no)
                visual.append(latent)
                entries.append({"kind": "video", "latent_t": latent.shape[2], "latent_h": latent.shape[-2],
                                "latent_w": latent.shape[-1], "ref_audio_t": 0})
        # Phase 2 reuses phase 1's audio rows, so every audio entry stays.
        for sentinel in audio_sentinels:
            entries.append({"kind": "audio", "ref_audio_t": sentinel.latent.shape[-1]})
        return visual, entries

    visual, entries = build_pass(1)
    visual2, entries2 = build_pass(2) if two_phase else (visual, entries)
    labels = []
    if visual_labels:
        # In the order the references are packed: pictures, videos, audio.
        labels = [s.enc for s in image_sentinels + video_sentinels if isinstance(s.enc, dict)]
    if labels_on:
        labels += [s.enc for s in audio_sentinels if isinstance(s.enc, dict)]
    _REFMODS_IN_USE["on"] = True
    _FL2VA.update(active=True, visual=visual, audio=[s.latent for s in audio_sentinels], entries=entries,
                  announced=False, presentation=labels, visual2=visual2, entries2=entries2, calls=0,
                  pass_no=1, label_mode=("keep" if later_window and STABLE_REFMOD_LABELS else "record"))
    _log(f"FL2VA: staging {len(image_sentinels)} image-kind + {len(video_sentinels)} video-kind + "
         f"{len(audio_sentinels)} audio-kind RefMod reference(s), retention={retention:.2f} "
         f"(~{total_tokens} tokens); prompt labels: "
         + (", ".join(f"<{ {'image': 'Picture', 'video': 'Video', 'audio': 'Audio'}[e['type']]}>" for e in labels)
            or "none"))
    return True


def _append_fl2va_rows(pipeline_self, visual_latents, audio_latents) -> None:
    """From _prepare_condition_rows: add the staged mods to this pass's
    condition latents. Audio only on the first pass -- both phase-2 paths
    pass a throwaway [] for audio and keep phase 1's audio rows."""
    if not _FL2VA["active"] or getattr(pipeline_self, "reference_mode", False) \
            or not isinstance(visual_latents, list):
        return
    first_pass = _FL2VA["calls"] == 0
    _FL2VA["calls"] += 1
    _FL2VA["pass_no"] = 1 if first_pass else 2
    visual_latents.extend(_FL2VA["visual"] if first_pass else _FL2VA["visual2"])
    if first_pass and isinstance(audio_latents, list):
        audio_latents.extend(_FL2VA["audio"])


_FL2VA_HOOKS_MARKER = "_h3refmod_fl2va_hooks"


def _install_fl2va_hooks(Pipeline) -> None:
    if getattr(Pipeline, _FL2VA_HOOKS_MARKER, False):
        return
    setattr(Pipeline, _FL2VA_HOOKS_MARKER, True)

    _orig_encode_prompt = getattr(Pipeline, "_encode_prompt", None)
    if _orig_encode_prompt is not None:   # FL2VA labels, and every route's label report
        @functools.wraps(_orig_encode_prompt)
        def patched_encode_prompt(self, prompt, presentation, *args, **kwargs):
            # FL2VA builds its presentation from keyframes only; the staged
            # labels go after them, so a start frame stays <Picture 1>.
            if _FL2VA["active"] and _FL2VA.get("presentation") and isinstance(presentation, list):
                for entry in _FL2VA["presentation"]:
                    if entry.get("type") == "image":
                        _place_picture_label(presentation, entry, _FL2VA.get("label_mode", "record"))
                    else:
                        presentation.append(entry)
            _log_prompt_labels(presentation)
            return _orig_encode_prompt(self, prompt, presentation, *args, **kwargs)
        Pipeline._encode_prompt = patched_encode_prompt

    _orig_prepare_control_rows = getattr(Pipeline, "_prepare_control_rows", None)
    if _orig_prepare_control_rows is not None and CONTROLNET_REFMODS:
        @functools.wraps(_orig_prepare_control_rows)
        def patched_prepare_control_rows(self, control_frames, masks, frame_anchors, frame_count,
                                         height, width, condition_latents, *args, **kwargs):
            if _FL2VA["active"] and _FL2VA["visual"] and isinstance(condition_latents, (list, tuple)):
                condition_latents = list(condition_latents) + list(_FL2VA["visual"])
                _log(f"FL2VA ControlNet: control rows padded for {len(_FL2VA['visual'])} RefMod "
                     f"reference(s)")
            return _orig_prepare_control_rows(self, control_frames, masks, frame_anchors, frame_count,
                                              height, width, condition_latents, *args, **kwargs)
        Pipeline._prepare_control_rows = patched_prepare_control_rows

    try:
        from models.minimax_h3 import transformer as h3_transformer
        Model = h3_transformer.MiniMaxH3Model
    except Exception as e:
        _log(f"could not reach MiniMaxH3Model ({e!r}); RefMods won't work in FL2VA")
        return
    _orig_forward = Model.forward

    @functools.wraps(_orig_forward)
    def patched_forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload, *args, **kwargs):
        _veda_full_references(self)
        if _FL2VA["active"] and _FL2VA["entries"] and isinstance(payload, dict) and not payload.get("refs"):
            payload["refs"] = list(_FL2VA["entries2"] if _FL2VA["pass_no"] == 2 else _FL2VA["entries"])
            if not _FL2VA["announced"]:
                _FL2VA["announced"] = True
                _log(f"FL2VA: {len(_FL2VA['entries'])} RefMod reference(s) attached to the "
                     f"transformer payload -- the reference-aware packer is now in use")
        return _orig_forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload,
                             *args, **kwargs)

    Model.forward = patched_forward


# ═══════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_LABELS_FOR_REFMODS = True   # give each RefMod its own <Picture N> /
                                   # <Video N> / <Audio N> entry in the prompt

# What a video-kind mod does on sliding windows after the first one. Image and
# audio mods always reach every window, which is what WanGP itself does with
# live references: src_ref_images and audio_guide are both assigned once,
# before the window loop in wgp.py, and handed to every generate() call, and a
# reference video ("-") is re-extracted from frame 0 on every window
# (guide_frames_extract_start = 0 if reference_videos). Video mods are the one
# case worth a choice, because a whole clip given to a later window can let a
# few of its own frames surface at the window boundary.
WINDOW_MODE_STILL = "still"   # first latent frame only, as an image reference
WINDOW_MODE_FULL = "full"     # the whole clip, exactly like a live reference video
WINDOW_MODE_FIRST = "first"   # nothing at all after window 1
WINDOW_MODE_DEFAULT = WINDOW_MODE_STILL


_VAE_SPATIAL_FACTOR = 16   # pixels per latent cell, both axes


def _should_fit(pipeline_self, target_width, target_height) -> bool:
    """True only while rendering a reduced canvas -- i.e. phase 1 of a
    two-phase run. Single-phase runs and phase 2 are handed the full canvas
    and are deliberately left untouched."""
    full = _gen_state(pipeline_self).get("full_canvas")
    if not full or None in full:
        return False
    try:
        return int(target_width) < int(full[0]) or int(target_height) < int(full[1])
    except (TypeError, ValueError):
        return False


def _fit_latent_to_canvas(latent, target_width, target_height, relative_size=100.0, label=""):
    """Shrink a RefMod latent so it fits the canvas this phase is rendering at.

    Two-phase generation renders phase 1 at half resolution
    (``H3_TWO_PHASE_SCALE``), and every *live* reference is fitted to that
    smaller canvas before being encoded -- `_prepare_image_reference` resizes
    to a pixel budget of ``target_w * target_h * relative_size / 100``, and
    reference videos go through `_resize_video`. A RefMod arrives already
    encoded, so nothing used to fit it: phase 1 got a reference several times
    larger than the frame it was conditioning, costing tokens and sitting at
    a different scale from everything else.

    The latent is resampled on its spatial axes only (time, and the 24
    channels, are untouched), keeping aspect ratio, and **only ever made
    smaller** -- a mod extracted below the canvas is left exactly as it is
    rather than being blown up into detail it never had. Phase 2 and
    single-phase runs pass the full canvas, so a normally-sized mod is
    untouched there.
    """
    if latent is None or latent.dim() != 5:
        return latent
    h, w = latent.shape[-2], latent.shape[-1]
    try:
        budget = float(target_width) * float(target_height) * float(relative_size) / 100.0
    except (TypeError, ValueError):
        return latent
    if budget <= 0:
        return latent
    current = (h * _VAE_SPATIAL_FACTOR) * (w * _VAE_SPATIAL_FACTOR)
    if current <= budget:
        return latent
    scale = (budget / current) ** 0.5
    # keep both axes even: the DiT patchifies the latent in 2x2 cells
    new_h = max(2, int(round(h * scale / 2)) * 2)
    new_w = max(2, int(round(w * scale / 2)) * 2)
    if new_h >= h and new_w >= w:
        return latent
    new_h, new_w = min(new_h, h), min(new_w, w)
    t = latent.shape[2]
    flat = latent.reshape(latent.shape[0] * latent.shape[1], t, h, w)
    resized = torch.nn.functional.interpolate(flat, size=(new_h, new_w), mode="area")
    out = resized.reshape(latent.shape[0], latent.shape[1], t, new_h, new_w).to(latent.dtype)
    _log(f"fitted {label or 'RefMod'} to this phase's canvas: {w}x{h} -> {new_w}x{new_h} latent cells "
         f"({w * _VAE_SPATIAL_FACTOR}x{h * _VAE_SPATIAL_FACTOR}px -> "
         f"{new_w * _VAE_SPATIAL_FACTOR}x{new_h * _VAE_SPATIAL_FACTOR}px)")
    return out


def _enc_item(mod, kind, frame_index, strength):
    """The presentation entry for one RefMod reference, or None when the mod
    carries no encoder frames (extracted before they were stored -- it still
    conditions the DiT, it just gets no prompt label).

    Frames are dimmed toward mid-grey in step with ``strength`` so the text
    encoder is never shown a stronger reference than the DiT receives."""
    if not PROMPT_LABELS_FOR_REFMODS:
        return None
    frames = getattr(mod, "enc_frames", None)
    if frames is None or not frames.shape[0]:
        return None
    try:
        if kind == "image":
            index = min(frame_index, frames.shape[0] - 1)
            chosen, stamps = frames[index:index + 1], None
        else:
            chosen = frames
            times = list(getattr(mod, "enc_times", []) or [])
            stamps = times[:chosen.shape[0]] if len(times) >= chosen.shape[0] else None
        if strength < 1.0:
            chosen = chosen * strength + 0.5 * (1.0 - strength)
        item = {"type": kind, "frames": chosen.clone()}
        if kind == "video" and stamps:
            item["timestamps"] = stamps
        return item
    except Exception:
        _log("could not build a RefMod's prompt entry:\n" + traceback.format_exc())
        return None


def _enc_as_still(enc):
    """Re-label a video mod's prompt entry as a still picture, keeping only
    its first frame. Used when a video-kind RefMod ends up on the image
    reference path (image / single-frame output): the DiT then receives one
    still latent, so the text encoder must be shown one still too -- labelled
    "<Picture N>", not "<Video N>", or the prompt would address something
    that is not there."""
    if not isinstance(enc, dict) or enc.get("type") != "video":
        return enc
    frames = enc.get("frames")
    if frames is None or not getattr(frames, "shape", (0,))[0]:
        return None
    return {"type": "image", "frames": frames[:1].clone()}


def _present_refmod(presentation, sentinel) -> None:
    """Append this mod's prompt entry, so the text encoder sees it and the
    prompt can address it as "<Picture N>" / "<Video N>" / "<Audio N>" --
    exactly what a live reference gets. Without it the mod's latent still
    conditions the DiT, but nothing in the text stream points at it, so H3's
    own prompt structure ("<Subject 1> is the violinist from <Picture 1>")
    has nothing to bind to."""
    if not PROMPT_LABELS_FOR_REFMODS or presentation is None or sentinel.enc is None:
        return
    try:
        if getattr(sentinel, "label_first", False) and sentinel.enc.get("type") == "image":
            _place_picture_label(presentation, sentinel.enc, sentinel.label_first)   # fork
            return
        presentation.append(sentinel.enc)
    except Exception:
        _log("could not add a RefMod's prompt entry:\n" + traceback.format_exc())


def _qwen_frames_from_cthw(video: torch.Tensor) -> torch.Tensor:
    """[C, T, H, W] in [-1, 1] -> [T, H, W, 3] in [0, 1]; the same conversion
    pipeline.py's own _qwen_frames() applies before handing a live reference
    to the Qwen3-VL text/vision encoder."""
    return video.permute(1, 2, 3, 0).add(1.0).mul(0.5).clamp(0.0, 1.0).float().cpu()


def _encode_ref_image(pipeline_self, video_cthw: torch.Tensor) -> torch.Tensor:
    """[C, 1, H, W] pixel tensor -> [1, 24, 1, H, W] VAE latent (cpu)."""
    return pipeline_self._encode_video(video_cthw)


def _encode_ref_video(pipeline_self, video_cthw: torch.Tensor) -> torch.Tensor:
    """[C, T, H, W] pixel tensor -> [1, 24, T', H, W] VAE latent (cpu).
    Frame count is snapped to the VAE's causal 4k+1 grid first."""
    t = video_cthw.shape[1]
    valid_t = core.snap_to_causal_grid(t)
    if valid_t != t:
        video_cthw = video_cthw[:, :valid_t]
    return pipeline_self._encode_video(video_cthw)


def _encode_ref_audio(pipeline_self, waveform: torch.Tensor) -> torch.Tensor:
    """[1, 2, samples] waveform tensor -> [1, 32, 2, T] audio-VAE latent (cpu).
    Uses the pipeline's own bound _encode_audio() (device/dtype handling
    identical to a live audio reference, see pipeline.py's _encode_audio)."""
    return pipeline_self._encode_audio(waveform)


def _run_extract_audio_job(pipeline_self, spec: dict, status) -> None:
    """Audio-mod extraction -- entirely separate from the image/video path
    above, since an audio VAE latent [1, 32, 2, T] (channels x stereo x
    time) is structurally incompatible to stack alongside a visual latent
    [1, 24, T, H, W]. Audio mods are always full-fidelity ("encode"-only --
    there's no spatial grid to pool for audio, so "training" mode's
    compression concept doesn't apply)."""
    name = storage._sanitize_relpath(spec.get("name") or "my_concept").replace(os.sep, "/")
    concept_type = spec.get("concept_type", "generic")
    audio_path = spec.get("audio_path")
    latent_frames = int(spec.get("latent_frames", 16))
    multiplier = max(1, int(spec.get("multiplier", 1)))
    max_tokens = int(spec.get("max_tokens", 5120))
    description = str(spec.get("description", "") or "")
    save = bool(spec.get("save", True))

    if not audio_path:
        raise ValueError("RefMod extraction: no reference audio provided.")

    # The Extract UI's "duration to use (seconds)" slider always talks to this
    # function in terms of the shared "latent_frames" spec key (same one
    # video extraction uses), converted via plugin.py's video-domain
    # seconds<->latent_frames formulas. That round-trips back to the
    # original seconds value correctly regardless (it's the same formula
    # inverted), even though "latent_frames"/"4 pixel frames per latent
    # frame" has no real meaning for audio -- the audio VAE's own,
    # completely different rate (40 latents/s, see AUDIO_LATENTS_PER_SECOND
    # in plugin.py) only comes into play once the *real* audio latent gets
    # produced below, for the token-budget and duration-reporting math.
    target_px = (latent_frames - 1) * 4 + 1 if latent_frames > 1 else 1
    target_seconds = target_px / FPS_ASSUMED_FOR_DURATION_ESTIMATE
    # Fork: the extractor's own "Soundtrack length" sets an audio mod's
    # length when it is given; specs without it keep the duration above.
    target_seconds = _audio_seconds(spec, target_seconds)

    status(f"H3 RefMod: loading audio reference (up to ~{target_seconds:.1f}s, mode=encode -- "
          f"audio mods are always full-fidelity)")
    try:
        import soundfile as sf
        full_seconds = sf.info(audio_path).frames / sf.info(audio_path).samplerate
        if full_seconds > target_seconds + 0.15:
            status(f"H3 RefMod: this file is ~{full_seconds:.1f}s long, but only the first "
                  f"~{target_seconds:.1f}s (set by the extractor's audio length) will be "
                  f"used -- raise that slider before extracting if you want more of it kept.")
    except Exception:
        pass
    waveform = storage.load_audio_waveform(audio_path, max_seconds=target_seconds)
    duration = _require_audio_length(waveform, f"The audio file {os.path.basename(str(audio_path))}")
    status(f"H3 RefMod: encoding audio reference (~{duration:.1f}s)")

    latent = _encode_ref_audio(pipeline_self, waveform).to(torch.float16)
    if latent.dim() != 4 or latent.shape[1] != 32 or latent.shape[2] != 2:
        raise ValueError(f"Expected a MiniMax H3 audio-VAE latent [1,32,2,T], got {tuple(latent.shape)}.")

    if multiplier > 1:
        latent = latent.repeat(1, 1, 1, multiplier)

    requested_t = latent.shape[-1]
    if max_tokens > 0:
        latent, budget_messages = core.fit_audio_token_budget(latent, max_tokens, name)
        for msg in budget_messages:
            status(f"H3 RefMod: {msg}")
    total_t = latent.shape[-1]
    if total_t < requested_t:
        req_sec = requested_t / AUDIO_LATENTS_PER_SECOND
        got_sec = total_t / AUDIO_LATENTS_PER_SECOND
        status(f"H3 RefMod: token budget ({max_tokens}) cut this mod short -- requested "
              f"~{req_sec:.1f}s worth of frames ({requested_t}), saved ~{got_sec:.1f}s "
              f"({total_t}). Raise 'Max tokens' (Advanced) to keep more of the requested duration.")

    mod = core.H3RefMod(
        name=storage._split_folder(name)[1], kind="audio", latent=latent, latent_t=total_t, mode="encode",
        source="audio", source_shape=f"{latent.shape[1]}x{latent.shape[2]}x{requested_t}",
        pool=f"full-fidelity (~{duration:.1f}s requested)",
        optimize_steps=0,
        tags=["audio"] + ([f"x{multiplier} repeat"] if multiplier > 1 else []),
        description=description, concept_type=concept_type,
    )
    if save:
        path = mod.save(storage.mod_path(name))
        status(f"H3 RefMod '{name}' saved: {mod.token_count} tokens, audio -> {path}")
    else:
        status(f"H3 RefMod '{name}' extracted ({mod.token_count} tokens) but not saved (save=false)")


# ── Fork: soundtracks, and the Library / Decompile jobs ───────────────────

def _require_audio_length(waveform, what: str) -> float:
    """Refuse audio shorter than MIN_AUDIO_SECONDS; returns its length."""
    seconds = waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE
    if seconds < MIN_AUDIO_SECONDS - 0.05:
        raise ExtractionBlocked(
            f"{what} is only ~{seconds:.1f}s long. H3 needs at least {MIN_AUDIO_SECONDS:g}s of "
            f"audio to use it as a reference -- use a longer clip, or a longer part of it.")
    return seconds


def _audio_seconds(spec: dict, fallback: float) -> float:
    """The extractor's audio length ("Soundtrack length"), never under the
    minimum. Specs without one keep ``fallback``."""
    try:
        value = float(spec["audio_seconds"]) if spec.get("audio_seconds") is not None else float(fallback)
    except (TypeError, ValueError):
        value = float(fallback)
    return max(MIN_AUDIO_SECONDS, value)


def _read_soundtrack(spec, image_paths, video_paths, status):
    """The soundtrack for a visual mod as a [1, 2, samples] waveform, or None.

    An audio file given with the images/videos, or -- with "Use the clip's own
    audio" -- the first video source that has a soundtrack (a chosen audio
    file wins). Its length is "Soundtrack length"; a trimmed clip's audio
    starts at the trim start. Audio shorter than H3's 2-second minimum is
    refused, except a clip's own audio, which is kept with a warning so the
    clip can still make a visual mod."""
    audio_path = spec.get("audio_path") or None
    from_clip = False
    if not audio_path and spec.get("use_clip_audio") and video_paths:
        for candidate in video_paths:
            if storage.video_has_audio(candidate):
                audio_path, from_clip = candidate, True
                status(f"H3 RefMod: using the clip's own audio from {os.path.basename(str(candidate))}")
                break
        else:
            status("H3 RefMod: 'Use the clip's own audio' was set, but no video source has an audio "
                   "track -- extracting without a soundtrack")
    if not audio_path:
        return None
    seconds = _audio_seconds(spec, DEFAULT_AUDIO_SECONDS)
    trim = (spec.get("video_trims") or {}).get(os.path.basename(str(audio_path)))
    waveform = storage.extract_audio_from_video(audio_path, max_seconds=seconds,
                                                start_seconds=float(trim[0]) if trim else 0.0)
    if waveform is None:
        raise ExtractionBlocked(f"No audio could be read from {os.path.basename(str(audio_path))}.")
    where = (f"The audio from {os.path.basename(str(audio_path))}"
             + (f" (from {float(trim[0]):g}s, the trim start)" if trim else ""))
    got = waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE
    if from_clip and got < MIN_AUDIO_SECONDS - 0.05:
        status(f"H3 RefMod: ⚠️ {where} is only ~{got:.1f}s long -- kept anyway, but H3 documents "
               f"{MIN_AUDIO_SECONDS:g}s as the shortest usable audio reference, so this voice may "
               f"not be picked up.")
    elif not from_clip:
        got = _require_audio_length(waveform, where)
    status(f"H3 RefMod: soundtrack ~{got:.1f}s (Soundtrack length {seconds:g}s)")
    return waveform


def _cat_enc_frames(frames_list):
    """Join per-source encoder frames into one [N, H, W, 3] tensor, resizing
    any that differ to the largest size (training mode keeps each source's
    own aspect, so they can)."""
    if len({tuple(f.shape[1:3]) for f in frames_list}) > 1:
        h = max(f.shape[1] for f in frames_list)
        w = max(f.shape[2] for f in frames_list)
        frames_list = [f if tuple(f.shape[1:3]) == (h, w) else
                       F.interpolate(f.permute(0, 3, 1, 2), size=(h, w), mode="bilinear",
                                     align_corners=False).permute(0, 2, 3, 1) for f in frames_list]
    return torch.cat(frames_list)


def _enable_vae_tiling(pipeline_self) -> None:
    """These jobs run before generate() has set up the VAE (see the same step
    in _run_extract_job)."""
    try:
        pipeline_self._configure_tiling(None)
    except Exception as e:
        _log(f"could not enable VAE tiling before the job ({e!r}); continuing untiled")


def _run_attach_audio_job(pipeline_self, spec: dict, status) -> None:
    """Library: add, replace or remove the soundtrack on existing image/video
    mods, so a mod doesn't have to be rebuilt from its sources to gain one."""
    names = [n for n in (spec.get("names") or []) if n]
    audio_path, remove = spec.get("audio_path") or None, bool(spec.get("remove"))
    if not names:
        status("Attach audio: nothing selected.")
        return
    latent = None
    if not remove:
        if not audio_path:
            status("Attach audio: pick an audio file, or tick Remove.")
            return
        waveform = storage.extract_audio_from_video(
            audio_path, max_seconds=max(MIN_AUDIO_SECONDS, float(spec.get("seconds") or DEFAULT_AUDIO_SECONDS)))
        if waveform is None:
            status(f"Attach audio: no audio track could be read from {os.path.basename(str(audio_path))}.")
            return
        try:
            _require_audio_length(waveform, f"The audio in {os.path.basename(str(audio_path))}")
        except ExtractionBlocked as blocked:
            status(f"Attach audio refused: {blocked}")
            return
        latent = _encode_ref_audio(pipeline_self, waveform).to(torch.float16)
    done = skipped = failed = 0
    for index, name in enumerate(names, 1):
        try:
            mod = storage.load_refmod(name)
            if mod.kind == "audio" or (remove and mod.audio_latent is None):
                skipped += 1
                status(f"[{index}/{len(names)}] '{name}': nothing to do")
                continue
            mod.audio_latent = None if remove else latent.clone()
            mod.audio_t = 0 if remove else int(latent.shape[-1])
            mod.save(storage.mod_path(name))
            done += 1
            status(f"[{index}/{len(names)}] '{name}': "
                   + ("soundtrack removed" if remove else f"soundtrack attached, now {mod.token_count} tokens"))
        except Exception:
            failed += 1
            _log(f"could not update {name!r}:\n" + traceback.format_exc())
    status(f"Attach audio finished: {done} updated, {skipped} skipped, {failed} failed.")


def _run_store_frames_job(pipeline_self, spec: dict, status) -> None:
    """Library: store encoder frames in mods saved without them (decoded from
    the latent), so their prompt labels never need a decode at generation."""
    names = storage.list_refmods_missing_frames() if spec.get("all_missing") else \
        [n for n in (spec.get("names") or []) if n]
    if not names:
        status("Encoder frames: nothing to do -- every mod already has them.")
        return
    done = skipped = failed = 0
    for index, name in enumerate(names, 1):
        try:
            mod = storage.load_refmod(name)
            wanted = core.expected_enc_layout(mod.kind, mod.latent_t, mod.still_stack())
            if mod.kind == "audio" or (mod.has_encoder_frames() and mod.enc_layout == wanted):
                skipped += 1
                status(f"[{index}/{len(names)}] '{name}': nothing to do")
                continue
            mod.enc_frames, mod.enc_times, mod.enc_layout, mod.enc_fps = \
                decode_encoder_frames(pipeline_self, mod)
            mod.save(storage.mod_path(name))
            done += 1
            status(f"[{index}/{len(names)}] '{name}': stored {len(mod.enc_times)} encoder frame(s)")
        except Exception:
            failed += 1
            _log(f"could not store encoder frames for {name!r}:\n" + traceback.format_exc())
    status(f"Encoder frames finished: {done} updated, {skipped} skipped, {failed} failed.")


def _run_decompile_job(pipeline_self, spec: dict, status) -> None:
    """Decompile tab: what a mod holds, rebuilt as ordinary files -- one PNG
    per picture, an MP4 for a video, a WAV for its audio. These are VAE
    reconstructions of the stored latents, not the original sources."""
    name = spec.get("name")
    if not name:
        status("Decompile: no mod chosen.")
        return
    mod = storage.load_refmod(name)
    out = storage.clear_decompile_dir(name)
    written, soundtrack_path = [], None
    audio_latent = mod.latent if mod.kind == "audio" else mod.audio_latent
    if audio_latent is not None:
        try:
            audio_vae = getattr(pipeline_self, "audio_vae", None)
            if audio_vae is None:
                raise RuntimeError("this pipeline has no audio VAE to decode with")
            waveform = audio_vae.decode(audio_latent.to(device=pipeline_self.device,
                                                        dtype=torch.float32))[0].float().clamp(-1, 1).cpu()
            soundtrack_path = storage.save_wav(
                waveform, os.path.join(out, "audio.wav" if mod.kind == "audio" else "soundtrack.wav"))
            written.append(f"{os.path.basename(soundtrack_path)} "
                           f"({waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE:.1f}s)")
        except Exception:
            soundtrack_path = None
            _log(f"decompile: could not decode the audio of {name!r}:\n" + traceback.format_exc())
            status(f"Decompile: the audio of '{name}' could not be decoded -- see the console")
    if mod.kind != "audio":
        if mod.kind == "image" or mod.still_stack() or mod.latent.shape[2] <= 1:
            size = ""
            for j in range(mod.latent.shape[2]):   # pictures were encoded one by one
                picture = _qwen_frames_from_cthw(_decode_cthw(pipeline_self, mod.latent[:, :, j:j + 1])[:, :1])[0]
                storage.save_png(picture, os.path.join(out, f"picture_{j + 1:02d}.png"))
                size = size or f"{picture.shape[1]}x{picture.shape[0]}"
            written.append(f"{mod.latent.shape[2]} picture(s), {size}px")
        else:
            frames = _qwen_frames_from_cthw(_decode_cthw(pipeline_self, mod.latent))
            storage.save_mp4(frames, os.path.join(out, "video.mp4"), FPS_ASSUMED_FOR_DURATION_ESTIMATE,
                             soundtrack_path)
            written.append(f"video.mp4 ({frames.shape[0]} frames, "
                           f"~{frames.shape[0] / FPS_ASSUMED_FOR_DURATION_ESTIMATE:.1f}s, "
                           f"{frames.shape[2]}x{frames.shape[1]}px"
                           + (", with its soundtrack" if soundtrack_path else "") + ")")
    status(f"Decompile finished: '{name}' -> " + ", ".join(written) + f" in {out}")


_EXTRA_JOBS = {"attach_audio": _run_attach_audio_job, "store_frames": _run_store_frames_job,
               "decompile": _run_decompile_job}


def _run_extract_job(pipeline_self, spec: dict, set_progress_status=None) -> None:
    if isinstance(spec, str):
        spec = json.loads(spec)

    def status(msg):
        _log(msg)
        if set_progress_status is not None:
            try:
                set_progress_status(msg)
            except Exception:
                pass

    # Fork: other jobs that need the loaded VAEs ride the same channel.
    if spec.get("op") in _EXTRA_JOBS:
        _enable_vae_tiling(pipeline_self)
        _EXTRA_JOBS[spec["op"]](pipeline_self, spec, status)
        return

    name = storage._sanitize_relpath(spec.get("name") or "my_concept").replace(os.sep, "/")
    mode = core.normalize_mode(spec.get("mode", "training"))
    concept_type = spec.get("concept_type", "generic")
    image_paths = [p for p in (spec.get("image_paths") or []) if p]
    # "video_paths" is the current, unlimited list form. The older
    # "video_path"/"video_path2" pair is still accepted so specs saved by
    # (or queued from) an earlier version keep working unchanged.
    video_paths = [p for p in (spec.get("video_paths") or []) if p]
    if not video_paths:
        video_paths = [p for p in (spec.get("video_path"), spec.get("video_path2")) if p]
    audio_path = spec.get("audio_path") or None
    # fork.37: the order chosen in the extractor (images and videos mixed).
    order = None
    if spec.get("order"):
        from .order_ui import apply_order
        order = apply_order(spec.get("order"), image_paths, video_paths)
        video_paths = [p for kind, p in order if kind == "vid"]   # also picks "the clip's own audio"
    ref_resolution = int(spec.get("ref_resolution", 1024))
    # Long edge of the frames shown to Qwen3-VL (the <Picture N> channel).
    # 0 / missing = the module default, which is live-reference parity.
    enc_resolution = int(spec.get("enc_resolution", 0)) or core.ENC_MAX_EDGE
    pool_h = int(spec.get("pool_h", 16))
    pool_w = int(spec.get("pool_w", 16))
    latent_frames = int(spec.get("latent_frames", 16))
    identity = int(spec.get("identity", 500))
    multiplier = max(1, int(spec.get("multiplier", 1)))
    max_tokens = int(spec.get("max_tokens", 5120))
    description = str(spec.get("description", "") or "")
    save = bool(spec.get("save", True))
    remove_background_images_ref = int(spec.get("remove_background_images_ref", 0) or 0)

    if audio_path and not image_paths and not video_paths:
        _run_extract_audio_job(pipeline_self, spec, status)
        return

    if not image_paths and not video_paths:
        raise ValueError("RefMod extraction: no reference image(s) or video provided.")

    # Fork: audio given WITH images or video becomes this mod's soundtrack --
    # stored beside the visual latent (the shapes can't be stacked) and
    # injected as its own <Audio N> reference. Read first, so a clip too
    # short to use is refused before any slow visual encoding.
    soundtrack_waveform = _read_soundtrack(spec, image_paths, video_paths, status)

    pool_warning = core.identity_training_pool_warning(concept_type, mode, pool_h, pool_w)
    if pool_warning:
        status(f"H3 RefMod: warning: {pool_warning}")

    status(f"H3 RefMod: loading {len(image_paths)} image(s)"
           + (f" + {len(video_paths)} video(s)" if video_paths else "") + f" (mode={mode})")

    rembg_session = None
    if remove_background_images_ref and image_paths:
        try:
            rembg_session = storage.new_rembg_session()
        except Exception as e:
            status(f"H3 RefMod: could not initialize background removal ({e!r}); "
                  f"continuing with backgrounds kept")
            remove_background_images_ref = 0

    sources = []  # list of (cthw_tensor, is_video)
    per_file = {}  # fork.37: (kind, path) -> its sources, put in order below
    for i, p in enumerate(image_paths):
        from PIL import Image
        with Image.open(p) as img:
            img = img.copy()
        if remove_background_images_ref:
            status(f"H3 RefMod: removing background from image {i + 1}/{len(image_paths)}")
            img = storage.remove_background_from_image(img, session=rembg_session)
        sources.append((storage.pil_to_cthw(img), False))
        per_file.setdefault(("img", p), []).append(sources[-1])
    # Fork: "Turn reference videos into a picture stack" -- each video becomes
    # still pictures (the sharpest frame from every 1/N-second slot) instead
    # of a clip, keeping a clip's identity at a fraction of its tokens.
    video_as_pictures = bool(spec.get("video_as_pictures", False))
    pictures_per_second = max(0.05, float(spec.get("pictures_per_second", 1.0) or 1.0))
    from_stills = 0
    for vp in video_paths:
        # Fork: per-video trim chosen in the extractor's preview, by file name.
        trim = (spec.get("video_trims") or {}).get(os.path.basename(str(vp)))
        start_s = float(trim[0]) if trim else 0.0
        dur_s = max(0.1, float(trim[1]) - float(trim[0])) if trim else None
        if trim:
            status(f"H3 RefMod: using {start_s:g}-{float(trim[1]):g}s of {os.path.basename(str(vp))}")
        # Fork: the trim (or the whole video) sets the span, up to H3's 15 s
        # reference limit -- the extractor has no separate duration slider.
        span_s = dur_s if dur_s is not None else (storage.video_duration_seconds(vp) or 0.0)
        if span_s > REFERENCE_SECONDS_MAX + 0.05:
            status(f"H3 RefMod: {os.path.basename(str(vp))} gives {span_s:.1f}s -- only the first "
                   f"{REFERENCE_SECONDS_MAX:g}s is used (H3's limit for a reference video); trim it "
                   f"to choose which part")
            dur_s = REFERENCE_SECONDS_MAX
        elif 0 < span_s < MIN_AUDIO_SECONDS - 0.05:
            status(f"H3 RefMod: ⚠️ {os.path.basename(str(vp))} gives only {span_s:.1f}s -- under 2s is "
                   f"too short for its own audio to work as a voice reference")
        if video_as_pictures:
            limit_s = (((latent_frames - 1) * 4 + 1) if latent_frames > 1 else 1) \
                / float(FPS_ASSUMED_FOR_DURATION_ESTIMATE)
            stills, covered = storage.load_video_stills(vp, pictures_per_second, start_s,
                                                        min(dur_s, limit_s) if dur_s else limit_s,
                                                        max_short_edge=ref_resolution)
            status(f"H3 RefMod: {os.path.basename(str(vp))} -> {len(stills)} picture(s) "
                   f"({pictures_per_second:g} per second over {covered:.1f}s, sharpest frame of each slot)")
            sources.extend((still, False) for still in stills)
            per_file.setdefault(("vid", vp), []).extend((still, False) for still in stills)
            from_stills += 1
            continue
        video = storage.load_video_cthw(vp, max_frames=max(64, latent_frames * 8),
                                        start_seconds=start_s, duration_seconds=dur_s)
        # Take a CONTIGUOUS prefix matching the requested duration, for both modes -- not a
        # sparse sample spread across the whole clip. The old encode-mode behavior picked
        # `latent_frames` frames evenly spaced across the *entire* source video, then handed
        # them to the causally-compressing video VAE as if they were sequential -- the VAE has
        # no idea they were sparse, so it compresses them as a ~1-2s clip regardless of how
        # long a span they were actually pulled from, silently breaking the "duration to use"
        # slider's promise. Truncating up front instead keeps the number honest in both modes.
        target_px = (latent_frames - 1) * 4 + 1 if latent_frames > 1 else 1
        if video.shape[1] > target_px:
            video = video[:, :target_px]
        sources.append((video, video.shape[1] > 1))
        per_file.setdefault(("vid", vp), []).append(sources[-1])

    if order:   # fork.37
        sources = [src for key in order for src in per_file.get(tuple(key), [])]
        status("H3 RefMod: order -- " + ", ".join(os.path.basename(p) for _, p in order))

    canvas = None
    if mode == "encode" and len(sources) > 1:
        c, t0, h0, w0 = sources[0][0].shape
        scale = min(1.0, ref_resolution / min(h0, w0))
        canvas = (max(32, round(w0 * scale / 32) * 32), max(32, round(h0 * scale / 32) * 32))

    pool_grid = None
    if mode == "training":
        c, t0, h0, w0 = sources[0][0].shape
        pool_grid = core.aspect_grid(pool_h, pool_w, h0 / w0)
        if pool_grid != (pool_h, pool_w):
            status(f"pooled grid {pool_h}x{pool_w} -> {pool_grid[0]}x{pool_grid[1]} to match source aspect")
    gh, gw = pool_grid if pool_grid is not None else (pool_h, pool_w)

    # Extraction runs *before* the original generate(), so the pipeline's own
    # _configure_tiling() has not run in this call -- on the very first task
    # after a model load the VAE is still untiled, and a large source (a wide
    # panorama, a 2048-short-edge photo) can exhaust VRAM in one encode.
    # Enabling it here is exactly what generate() does a moment later, and is
    # harmless when it is already on.
    try:
        pipeline_self._configure_tiling(None)
    except Exception as e:
        _log(f"could not enable VAE tiling before encoding ({e!r}); continuing untiled -- "
             f"a very large reference image may run out of VRAM")

    frames = []
    n_img = n_vid = 0
    source_shapes = []
    enc_frames, enc_times, enc_layout = [], [], core.ENC_LAYOUT_STILLS
    # Fork: "Merge into one picture" (training mode, several sources) -- each
    # source's pooled grid and full encode are kept, and ONE shared grid is
    # built from all of them after the loop (core.merge_latents).
    merge = bool(spec.get("merge", False))
    if merge and mode != "training":
        status("H3 RefMod: 'Merge into one picture' only applies in training mode -- stacking instead")
    merge_refs = [] if (merge and mode == "training" and len(sources) > 1) else None
    for i, (src, is_video) in enumerate(sources):
        label = f"ref {i + 1}/{len(sources)} ({'video' if is_video else 'image'})"
        status(f"H3 RefMod: encoding {label}")
        if mode == "encode":
            src = storage.resize_cthw(src, ref_resolution, canvas)
        else:
            src = storage.resize_cthw(src, ref_resolution, None)
        src = storage.ensure_min_size(src)
        if is_video and src.shape[1] > 1:
            z = _encode_ref_video(pipeline_self, src)
        else:
            z = _encode_ref_image(pipeline_self, src[:, :1])
        if z.dim() != 5 or z.shape[1] != 24:
            raise ValueError(f"Expected a MiniMax H3 video-VAE latent [1,24,T,H,W], got {tuple(z.shape)}.")
        source_shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")

        # Keep a few pixels for the text encoder, so this mod can earn a
        # "<Picture N>" / "<Video N>" label in the prompt later (see
        # core.encode_enc_frames). Taken from the same resized source the VAE
        # just saw, and converted to Qwen's [T, H, W, 3] 0..1 convention --
        # the exact thing pipeline.py's own _qwen_frames() produces for a
        # live reference.
        if is_video and src.shape[1] > 1:
            enc_layout = core.ENC_LAYOUT_CLIP
            # two frames a second, the rate Wan2GP samples a live reference
            # video at for the encoder
            picks, cursor = [], 0.0
            while round(cursor) < src.shape[1]:
                if not picks or round(cursor) > picks[-1]:
                    picks.append(round(cursor))
                cursor += FPS_ASSUMED_FOR_DURATION_ESTIMATE / 2
            picks = picks[:core.ENC_MAX_CLIP_FRAMES]
            enc_frames.append(_qwen_frames_from_cthw(src[:, picks]))
            enc_times.extend(p / FPS_ASSUMED_FOR_DURATION_ESTIMATE for p in picks)
        else:
            enc_frames.append(_qwen_frames_from_cthw(src[:, :1]))
            enc_times.append(float(len(enc_times)))

        if mode == "encode":
            pooled = z.to(torch.float16)
        else:
            pool_t = min(latent_frames, z.shape[2]) if is_video else 1
            pooled = core.pool_latent(z, pool_t, gh, gw).to(torch.float16)
            if merge_refs is not None:   # fork: refined jointly after the loop
                merge_refs.append((pooled.float().cpu(), z.float().cpu()))
                z = None
                n_vid, n_img = n_vid + bool(is_video), n_img + (not is_video)
                continue
            if identity > 0:
                status(f"H3 RefMod: refining identity for {label} ({identity} steps)")
                pooled = core.optimize_latent(pooled, z.float(), steps=identity, progress_every=100)
        frames.append(pooled)
        if is_video:
            n_vid += 1
        else:
            n_img += 1

    merged_n = 0
    if merge_refs is not None:
        status(f"H3 RefMod: merging {len(merge_refs)} references into one {gh}x{gw} grid"
               + (f" ({identity} joint refinement steps)" if identity > 0
                  else " (plain average -- identity refinement steps is 0)"))
        frames = [core.merge_latents([p for p, _ in merge_refs], [f for _, f in merge_refs],
                                     gh, gw, steps=identity, progress_every=100).to(torch.float16)]
        merged_n, merge_refs = len(merge_refs), None
    latent = torch.cat(frames, dim=2)
    if multiplier > 1:
        latent = latent.repeat(1, 1, multiplier, 1, 1)
    requested_t = latent.shape[2]
    if max_tokens > 0:
        latent, budget_messages = core.fit_token_budget(latent, max_tokens, name)
        for msg in budget_messages:
            status(f"H3 RefMod: {msg}")

    total_t = latent.shape[2]
    if total_t < requested_t:
        req_sec = ((requested_t - 1) * 4 + 1 if requested_t > 1 else 1) / FPS_ASSUMED_FOR_DURATION_ESTIMATE
        got_sec = ((total_t - 1) * 4 + 1 if total_t > 1 else 1) / FPS_ASSUMED_FOR_DURATION_ESTIMATE
        status(f"H3 RefMod: token budget ({max_tokens}) cut this mod short -- requested "
              f"~{req_sec:.1f}s worth of frames ({requested_t}), saved ~{got_sec:.1f}s "
              f"({total_t}). Raise 'Max tokens' (Advanced) or lower the ref resolution/pool "
              f"grid to keep more of the requested duration.")
    kind = "video" if n_vid > 0 else "image"  # NOT total_t > 1: several still images
                                              # stacked together (n_vid==0) still form
                                              # multiple independent *image* references,
                                              # not a multi-frame video, even though the
                                              # underlying latent has more than one frame.
    px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
    audio_latent = None
    if soundtrack_waveform is not None:   # fork: the mod's soundtrack
        status(f"H3 RefMod: encoding the soundtrack "
               f"(~{soundtrack_waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE:.1f}s)")
        audio_latent = _encode_ref_audio(pipeline_self, soundtrack_waveform).to(torch.float16)
        if audio_latent.dim() != 4 or audio_latent.shape[1] != 32 or audio_latent.shape[2] != 2:
            raise ValueError(f"Expected an audio-VAE latent [1,32,2,T], got {tuple(audio_latent.shape)}.")
    mod = core.H3RefMod(
        name=storage._split_folder(name)[1], kind=kind, latent=latent, latent_h=latent.shape[3], latent_w=latent.shape[4],
        latent_t=total_t, mode=mode,
        enc_frames=(_cat_enc_frames(enc_frames) if enc_frames else None),
        enc_times=enc_times, enc_layout=enc_layout,
        enc_fps=(FPS_ASSUMED_FOR_DURATION_ESTIMATE if enc_layout == core.ENC_LAYOUT_CLIP else 0.0),
        # A clip stores up to ENC_MAX_CLIP_FRAMES views, so it keeps the
        # smaller per-frame cap; a still spends its whole budget on one frame.
        enc_max_edge=(min(enc_resolution, core.ENC_MAX_EDGE_CLIP)
                      if enc_layout == core.ENC_LAYOUT_CLIP else enc_resolution),
        source="merge" if merged_n else ("stack" if len(frames) > 1 else ("video" if n_vid else "image")),
        source_shape=" +".join(source_shapes),
        pool=(f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)" if mode == "encode"
              else f"{total_t}x{gh}x{gw}"),
        optimize_steps=identity if mode == "training" else 0,
        tags=[f"{n_img} img, {n_vid} vid"] + ([f"x{multiplier} repeat"] if multiplier > 1 else [])
             + (["background removed"] if remove_background_images_ref else [])
             + ([f"merged {merged_n} refs"] if merged_n else [])
             + ([f"{pictures_per_second:g}/s from video"] if from_stills else [])
             + (["with soundtrack"] if audio_latent is not None else []),
        description=description, concept_type=concept_type,
        audio_latent=audio_latent, audio_t=0 if audio_latent is None else int(audio_latent.shape[-1]),
    )
    if merged_n:
        # Fork: a merged mod is one picture that none of the sources is, so
        # the encoder is shown the merge itself, decoded from its latent.
        try:
            mod.enc_frames, mod.enc_times, mod.enc_layout, mod.enc_fps = \
                decode_encoder_frames(pipeline_self, mod)
            mod.enc_max_edge = enc_resolution
        except Exception:
            mod.enc_frames, mod.enc_times = None, []
            _log("could not decode the merged picture for its prompt label; saving without one:\n"
                 + traceback.format_exc())

    if save:
        path = mod.save(storage.mod_path(name))
        status(f"H3 RefMod '{name}' saved: {mod.token_count} tokens, {kind}/{mode}"
               + (" + soundtrack" if audio_latent is not None else "") + f" -> {path}")
    else:
        status(f"H3 RefMod '{name}' extracted ({mod.token_count} tokens) but not saved (save=false)")
