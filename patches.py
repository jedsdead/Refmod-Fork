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

import functools
import json
import os
import random
import time
import traceback
from typing import Optional

import collections
import math
import torch

from . import core, encframes, storage

SETTING_GENERATE = "h3_refmod_state"     # custom_settings key: mods to inject into a real render
SETTING_EXTRACT = "h3_refmod_extract"    # custom_settings key: "run an extraction, not a render"

# Both payloads now travel in ONE custom setting, because Wan2GP only keeps the
# first CUSTOM_SETTINGS_MAX (5) settings a model declares:
# get_model_custom_settings() slices custom_settings[:5], and the strict
# collection that builds a task's custom_settings iterates that truncated list.
# Ref2VA already declares 4 of its own (mask mode, audio refinement, and the
# two excerpt-position settings), so the plugin's two made 6 and the last was
# silently dropped -- which is why extraction ran as an ordinary render and why
# the mod selection could vanish on the way to the queue. One setting fits.
# A character Wan2GP never uses in a video prompt type, so a custom setting
# keyed to it is never shown (see install_model_def_patch).
HIDDEN_SETTING_FLAG = "\u00a7"
SETTING_COMBINED = "h3_refmod"           # {"state": <selection json>, "extract": <job json>}
CUSTOM_SETTINGS_MAX_ASSUMED = 5


def pack_refmod_setting(state_json=None, extract_json=None) -> str:
    payload = {}
    if state_json:
        payload["state"] = state_json
    if extract_json:
        payload["extract"] = extract_json
    return json.dumps(payload) if payload else ""


def unpack_refmod_setting(custom_settings):
    """(selection json, extraction job json) from a task's custom_settings.
    Reads the combined key, and still honours the two legacy keys so tasks
    queued by an older build -- or settings saved from one -- keep working."""
    if not isinstance(custom_settings, dict):
        return None, None
    state = custom_settings.get(SETTING_GENERATE) or None
    extract = custom_settings.get(SETTING_EXTRACT) or None
    blob = custom_settings.get(SETTING_COMBINED)
    if blob:
        try:
            parsed = json.loads(blob) if isinstance(blob, str) else blob
            if isinstance(parsed, dict):
                state = parsed.get("state") or state
                extract = parsed.get("extract") or extract
        except Exception:
            _log(f"could not parse {SETTING_COMBINED}; ignoring it:\n" + traceback.format_exc())
    return state, extract
STASH_KEY = "_h3refmod_selection"        # key inside the session `state` dict for the inline panel
# Audio in a mod (an audio mod, or a visual mod's soundtrack): how much is kept
# by default, and the shortest allowed -- H3 documents 2s as the minimum
# length of an audio reference, so a shorter clip is refused at extraction.
DEFAULT_AUDIO_SECONDS = 4.0
MIN_AUDIO_SECONDS = 2.0
MIN_SOUNDTRACK_SECONDS = MIN_AUDIO_SECONDS   # name used by earlier forks


class ExtractionBlocked(ValueError):
    """An extraction refused for a reason the user can fix; its message is
    shown in the status box instead of a traceback."""


def _require_audio_length(waveform, what: str) -> float:
    """Refuse audio shorter than MIN_AUDIO_SECONDS; returns its length."""
    seconds = waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE
    if seconds < MIN_AUDIO_SECONDS - 0.05:
        raise ExtractionBlocked(
            f"{what} is only ~{seconds:.1f}s long. H3 needs at least {MIN_AUDIO_SECONDS:g}s of "
            f"audio to use it as a reference -- use a longer clip, or a longer part of it.")
    return seconds


def _audio_seconds(spec: dict, fallback: float) -> float:
    """The extraction's "Soundtrack length", never under the minimum. Specs
    from before fork.16 have none and keep their old length (``fallback``)."""
    value = spec.get("audio_seconds")
    try:
        value = float(value) if value is not None else float(fallback)
    except (TypeError, ValueError):
        value = float(fallback)
    return max(MIN_AUDIO_SECONDS, value)
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


def is_minimax_h3_refmod_capable(model_type, get_base_model_type_fn=None) -> bool:
    """Models RefMods can be used with: Ref2VA, plus FL2VA and its finetunes
    (e.g. VDN) when FL2VA_REFMODS is on. Resolves finetunes to their
    architecture exactly as is_minimax_h3_ref2va does."""
    if is_minimax_h3_ref2va(model_type, get_base_model_type_fn):
        return True
    if not FL2VA_REFMODS:
        return False
    model_type = str(model_type or "")
    if not model_type:
        return False
    if callable(get_base_model_type_fn):
        try:
            base = get_base_model_type_fn(model_type)
            if base:
                return str(base).startswith("minimax_h3_fl2va")
        except Exception:
            pass
    return model_type.startswith("minimax_h3_fl2va")


class _RefModImageSentinel:
    """Stands in for a single-frame ("image kind") reference. Carries an
    already-encoded, already-weighted VAE latent [1, 24, 1, H, W]."""
    __slots__ = ("latent", "hide_ref", "preview_latent", "preview_key", "audio_latent",
                 "mod", "still_index", "enc_strength")

    def __init__(self, latent: torch.Tensor, preview_latent=None, preview_key=None,
                 audio_latent=None, mod=None, still_index=0, enc_strength=1.0):
        self.latent = latent
        self.hide_ref = False
        self.audio_latent = audio_latent
        self.preview_latent = latent if preview_latent is None else preview_latent
        self.preview_key = preview_key
        # What the text encoder is shown: picture `still_index` of `mod`'s
        # encoder frames, softened to `enc_strength` (see _presentation_frames).
        self.mod = mod
        self.still_index = int(still_index)
        self.enc_strength = float(enc_strength)


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
    __slots__ = ("latent", "hide_ref", "preview_latent", "preview_key", "rescale_to",
                 "audio_latent", "mod", "enc_strength", "is_stack", "stack_mode", "stack_n",
                 "enc_time_limit", "clip_mode", "clip_n")

    def __init__(self, latent: torch.Tensor, preview_latent=None, preview_key=None,
                 audio_latent=None, mod=None, enc_strength=1.0, is_stack=False,
                 stack_mode=encframes.STACK_ALL, stack_n=encframes.STACK_DEFAULT_N):
        self.latent = latent
        self.hide_ref = False
        self.rescale_to = None
        self.audio_latent = audio_latent
        self.preview_latent = latent if preview_latent is None else preview_latent
        self.preview_key = preview_key
        self.mod = mod
        self.enc_strength = float(enc_strength)
        # A multi-picture mod sent as ONE video reference (the panel's "send as
        # one video reference" option): its latent frames are separate
        # pictures, not a clip -- shown to the encoder one picture per block,
        # rebuilt per picture, and never trimmed by the reference-video budget.
        self.is_stack = bool(is_stack)
        self.stack_mode = stack_mode
        self.stack_n = int(stack_n or encframes.STACK_DEFAULT_N)
        self.enc_time_limit = None   # seconds of a clip still present after a budget trim
        # How many of a clip's frames the encoder is shown (panel setting).
        self.clip_mode = encframes.STACK_UP_TO_N
        self.clip_n = encframes.MAX_CLIP_FRAMES_SHOWN

    @property
    def shape(self):
        h_px = self.latent.shape[3] * 16
        w_px = self.latent.shape[4] * 16
        if self.is_stack:
            # A stack of pictures has no duration. Report the shortest valid
            # reference (5 frames) so Wan2GP's 15-second reference-video budget
            # neither counts it as footage nor trims pictures off its end.
            return (3, 5, h_px, w_px)
        t = self.latent.shape[2]
        t_px = (t - 1) * 4 + 1 if t > 1 else 1
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
            # Carry this fork's extra fields across the trim: the preview
            # latent/key, the attached soundtrack and any phase-2 rescale
            # target would otherwise be dropped, silently losing a mod's
            # audio the moment the budget kicks in.
            same_length = (self.preview_latent is not None
                           and self.preview_latent.shape[2] == self.latent.shape[2])
            trimmed = _RefModVideoSentinel(
                self.latent[:, :, :keep_t],
                self.preview_latent[:, :, :keep_t] if same_length else self.preview_latent,
                None if self.preview_key is None else tuple(self.preview_key) + ("trim", keep_t),
                audio_latent=self.audio_latent, mod=self.mod, enc_strength=self.enc_strength)
            trimmed.hide_ref = self.hide_ref
            trimmed.rescale_to = self.rescale_to
            trimmed.clip_mode, trimmed.clip_n = self.clip_mode, self.clip_n
            # The encoder is shown only what is left of the clip.
            trimmed.enc_time_limit = (((keep_t - 1) * 4 + 1)
                                      / float(FPS_ASSUMED_FOR_DURATION_ESTIMATE))
            _log(f"reference-video budget: trimmed a video RefMod to {keep_t} of "
                 f"{self.latent.shape[2]} latent frames to share Wan2GP's time budget")
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
    __slots__ = ("latent", "hide_ref")

    def __init__(self, latent: torch.Tensor):
        self.latent = latent
        self.hide_ref = False


def _log(msg: str) -> None:
    print(f"[H3RefMod] {msg}")


# Which reference kwargs this Wan2GP build's generate() actually accepts.
# Older builds expose two native reference-video slots (input_frames,
# input_frames2) and two audio ones (audio_guide, audio_guide2); newer ones
# add a third of each (input_frames3 / "*", audio_guide3 / "D"). Filled in by
# install_patches() so injection uses whatever exists and falls back to
# direct injection for the rest.
_generate_params = set()
# Wan2GP passes native image refs / reference videos to *every* sliding window
# (src_ref_images is set once in window 1 and reused for the rest), so RefMods
# must be injected in every window too.
INJECT_ON_EVERY_WINDOW = True

# In windows 2+ the frame carried over from the previous window is added to the
# prompt as an image BEFORE any reference, so it takes <Picture 1> and pushes
# every reference's label up by one -- a prompt written for window 1 then points
# at the wrong picture. Set this to True to insert RefMod images ahead of that
# carried-over frame instead, keeping their labels identical in every window.
# Only applied when the generation has no live reference images of its own.
# On by default since fork.8: a prompt written once then means the same mod in
# every window. (A multi-picture mod sent as one <Video N> -- the panel's
# "send as one video reference" option -- isn't affected either way, because
# the carried-over frame is a picture, not a video.)
STABLE_REFMOD_LABELS = True

# Escape hatch, OFF by default: when True, generations that inject RefMods fall
# back to normal attention instead of Sol-Attn (attention mode "sol").
#
# It is off because the evidence doesn't implicate RefMods. The reported crash
# ("Triton Error [CUDA]: an illegal memory access was encountered") came from
# reduce_quantize_k, which takes only K and V -- it never sees the sink range,
# the reference list, or anything else a RefMod changes. Sink blocks go only to
# the separate forward kernel, and are range-checked in Python first, so a bad
# sink raises ValueError rather than corrupting memory. The fault also surfaced
# inside Triton's _init_handles while loading a kernel, which is where an
# *earlier* async fault gets reported, so the named kernel is probably a
# bystander. Turning Sol-Attn off costs real speed for every step.
#
# The one honest RefMod connection is token count: references lengthen the
# sequence, and Sol-Attn only engages at >= SOL_ATTN_MIN_TOKENS (8192), with a
# different inline-Q path above 4096 tokens. So adding RefMods can *start*
# using Sol-Attn, or switch which path runs, on a generation that stayed below
# those thresholds before. That's worth knowing if the crash tracks with adding
# mods, but it points at the kernels, not at what this plugin feeds them.
DISABLE_SOL_WITH_REFMODS = False

_REFMODS_ACTIVE = {"on": False, "logged": False}

# Give RefMods a <Picture N> / <Video N> / <Audio N> entry in the prompt, the
# same as a live reference gets, so the model has something binding the
# reference to the prompt. This is the core fix for "RefMods have no effect",
# but it is also the only change here that alters what the model sees on an
# ordinary single-phase run. Set it to False to get the published plugin's
# prompt behaviour back, for an A/B.
PROMPT_LABELS_FOR_REFMODS = True

# An extraction job waiting to be picked up, for when Wan2GP drops the
# custom_settings payload that carries it. Without this the extract task falls
# through to generate() as an ordinary render -- i.e. it produces a video
# instead of a mod. Consume-once and time-limited, so a stale job can never
# hijack a later render.
_PENDING_EXTRACT = {"spec": None, "at": 0.0}
_PENDING_EXTRACT_TTL_SECONDS = 600.0


def set_pending_extract(spec_json):
    _PENDING_EXTRACT["spec"] = spec_json or None
    _PENDING_EXTRACT["at"] = time.time() if spec_json else 0.0


def _take_pending_extract():
    """Pop a pending extraction job if one is fresh enough to still be ours."""
    spec = _PENDING_EXTRACT.get("spec")
    if not spec:
        return None
    age = time.time() - float(_PENDING_EXTRACT.get("at") or 0.0)
    set_pending_extract(None)
    if age > _PENDING_EXTRACT_TTL_SECONDS:
        _log(f"ignoring a pending extraction job that is {age:.0f}s old -- too late to be "
             f"this task; nothing was extracted")
        return None
    return spec


# The inline panel's current selection, kept at module level so generate() can
# fall back to it when a task reaches the pipeline without its custom_settings
# payload (Wan2GP's settings/validation round-trip dropped it).
_ARMED = {"json": None}


def set_armed_selection(state_json, force=False):
    """Remember the panel's selection as a fallback for a task that arrives
    without its payload.

    This used to LATCH -- a blank selection was ignored -- because Wan2GP's
    form refreshes fire the pickers with empty values and would wipe it, and
    at the time the fallback was the only thing making RefMods work at all.
    That is no longer true: the payload is dropped only when the plugin's
    custom setting doesn't fit Wan2GP's 5-setting limit, fixed by merging both
    payloads into one key. Latching outlived its purpose and became a trap --
    an empty panel kept injecting whatever was last selected, silently taking
    reference slots from the generation's own references. Clearing the pickers
    clears the selection again; `force` is kept for the explicit Clear button."""
    _ARMED["json"] = state_json or None


def _sync_armed_with_panel(stash):
    """Make the fallback copy match the panel this form actually shows.

    The armed selection is one server-wide value, while the panel's selection
    lives in the browser session. A page reload (or a rebuilt panel) shows
    empty slots without any picker firing a change, so the old value used to
    survive -- and generate()'s fallback then injected mods the panel no
    longer showed. Every time Wan2GP saves the form or builds a task for a
    RefMod-capable model, the session's own selection (or its absence) now
    overwrites it, so an empty panel always means no mods."""
    if (_ARMED["json"] or None) != (stash or None):
        _ARMED["json"] = stash or None
        if not stash:
            _log("the panel has no RefMods selected -- cleared the armed selection so none are "
                 "injected")


def _is_ref2va_pipeline(pipeline_self):
    return bool(getattr(pipeline_self, "reference_mode", False)) and \
        getattr(pipeline_self, "fixed_prompt", None) is None and \
        not getattr(pipeline_self, "audio_only", False)


_PREVIEW_CACHE = collections.OrderedDict()
_PREVIEW_CACHE_MAX = 8
# Caps on what is kept for the text encoder, per mod. Qwen3-VL resizes to its
# own pixel budget anyway, so more than this is memory spent for nothing.
_PREVIEW_MAX_PIXELS_IMAGE = 1 << 20   # ~1 MP for a single-frame mod
_PREVIEW_MAX_PIXELS_VIDEO = 1 << 18   # ~0.26 MP per sampled video frame
_PREVIEW_MAX_VIDEO_FRAMES = 8

# Every sliding window is its own generate() call, so without these each window
# re-read every mod off disk, re-ran the blur in weighted_latent(), and could
# re-decode the prompt preview -- all of it identical work, and the decode also
# made the offloader swap the VAE in and out again. Keyed by file mtime, so
# re-extracting a mod under the same name still picks up the new file.
# Two-phase generation renders phase 1 at a lower resolution, upscales the
# latent, then runs phase 2 at the target resolution -- and it REBUILDS its
# references for phase 2: every live image ref is re-prepared at the phase-2
# size (_prepare_image_reference -> _encode_video) and every reference video is
# re-resized to the target before being re-encoded. A RefMod is a stored latent,
# so it alone stayed at its extraction resolution while keyframes and live refs
# doubled, leaving it at the wrong spatial scale exactly when fine detail is
# being added. With this True, RefMods are rebuilt for phase 2 the same way:
# decode the stored latent, resize the pixels to the phase-2 reference size,
# re-encode. Cached per mod per target size, so it costs one decode + one encode
# per mod per session. Phase 1 is untouched -- its resolution is the token
# budget you chose at extraction time.
RESCALE_REFMODS_IN_PHASE_2 = False

# (Kept as an opt-in experiment only -- see FIT_REFMODS_TO_PHASE_1_CANVAS
# below, which is where the two-phase scale mismatch actually is.)
#
# Video mods are excluded by default. An image mod is one latent frame, so
# rebuilding it costs a bounded amount. A video mod multiplies by every latent
# frame: rebuilding a 68x38 mod to a 106x44 phase-2 size took it from ~10k to
# ~19k reference tokens, on top of an already large phase-2 sequence -- and
# attention cost grows with the square of the sequence, so phase 2 crawls. The
# decode alone (every frame at full target resolution) is close to a gigabyte
# of intermediates. A mod is supposed to have a fixed, small token cost; this
# would throw that budget away. Turn it on only if you know the mod is short.
RESCALE_REFMOD_VIDEOS_IN_PHASE_2 = False

# Hard ceiling on a rebuild regardless of kind: if the phase-2 size would give
# a mod more than this many times its stored token count, the rebuild is scaled
# back to fit rather than skipped, so it still gains scale without the blowup.
REFMOD_RESCALE_MAX_TOKEN_GROWTH = 2.0

# The real two-phase mismatch is in phase 1, not phase 2. With two phases the
# pipeline renders phase 1 at target / H3_TWO_PHASE_SCALE (half), and prepares
# every live reference at that reduced size (pipeline.py: _add_image_reference
# is called with the reduced width/height, and reference videos are resized to
# it). A RefMod is a stored latent, so it alone stays at its extraction size --
# oversized against a half-resolution canvas and against every live reference
# next to it. Phase 2 then sees the mod at its stored scale, which is the same
# configuration as a single-phase run at the target resolution, i.e. the case
# that already works.
#
# So: shrink a mod to the phase-1 canvas when it is bigger than it, and never
# grow one. Downscaling costs fewer reference tokens than the stored latent, so
# phase 1 also gets slightly cheaper rather than more expensive.
#
# This applies ONLY when the run really has two phases (guide_phases > 1). A
# single-phase run is also "phase 1" as far as the phase counter is concerned,
# and mods must go in at their stored scale there -- that is the configuration
# single-phase runs already work in.
#
# Fork.8 turned this off, reasoning that H3 tolerates references larger than
# the canvas. That was untested, and Wan2GP itself sizes every live reference
# to the phase-1 canvas, so fork.9 restores fork.7's behaviour as the default.
FIT_REFMODS_TO_PHASE_1_CANVAS = True

_RESCALE_CACHE = collections.OrderedDict()
_RESCALE_CACHE_MAX = 16

_MOD_CACHE = collections.OrderedDict()
_MOD_CACHE_MAX = 8
_WEIGHTED_CACHE = collections.OrderedDict()
_WEIGHTED_CACHE_MAX = 16


def _cache_put(cache, key, value, limit):
    cache[key] = value
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _latent_stamp(latent):
    """Fallback identity for a mod whose file mtime can't be read."""
    try:
        return (tuple(latent.shape), round(float(latent.float().sum()), 4))
    except Exception:
        return None


def _load_refmod_cached(name):
    try:
        stamp = os.path.getmtime(storage.mod_path(name) + ".safetensors")
    except Exception:
        stamp = None
    key = (name, stamp)
    mod = _MOD_CACHE.get(key)
    if mod is None:
        mod = _cache_put(_MOD_CACHE, key, storage.load_refmod(name), _MOD_CACHE_MAX)
    else:
        _MOD_CACHE.move_to_end(key)
    # A readable mtime makes the preview cache stable across windows; without
    # one, fall back to the latent's own fingerprint rather than giving up on
    # caching (which would re-decode the preview on every single window).
    mod._h3_preview_key = key if stamp is not None else (name, _latent_stamp(mod.latent))
    return mod


def _weighted_latent_cached(mod, strength, curve):
    key = (getattr(mod, "_h3_preview_key", None), round(float(strength), 6),
           tuple(curve) if isinstance(curve, (list, tuple)) else curve)
    if key[0] is not None and key in _WEIGHTED_CACHE:
        _WEIGHTED_CACHE.move_to_end(key)
        return _WEIGHTED_CACHE[key]
    latent = mod.weighted_latent(strength, curve=curve)
    if key[0] is not None and latent is not None:
        _cache_put(_WEIGHTED_CACHE, key, latent, _WEIGHTED_CACHE_MAX)
    # Clone on the way out: at strength 1.0 weighted_latent() returns the mod's
    # own stored tensor, and the mod is itself cached for the session now, so a
    # single in-place write anywhere downstream would quietly corrupt the mod
    # for every later generation -- which would look exactly like a face slowly
    # degrading run after run. Latents are a few MB; this is cheap insurance.
    return None if latent is None else latent.clone()


def _preview_key(mod, kind, index):
    base = getattr(mod, "_h3_preview_key", None)
    return None if base is None else tuple(base) + (kind, index)


def _decode_preview(pipeline_self, sentinel, kind="image", fps=None):
    """Prompt-side preview frames for a RefMod, ready for _qwen_frames, plus
    their timestamps (video only). Cached per mod.

    Only the handful of frames the text encoder actually consumes are kept,
    downscaled first: caching the whole decoded video meant hundreds of MB of
    float32 per video mod held for the session, which is what pushed phase 2
    into swapping on later windows. The encoder resizes to its own pixel budget
    anyway (_visual_patches), so the detail was never used."""
    key = getattr(sentinel, "preview_key", None)
    cache_key = None if key is None else (key, kind, round(float(fps or 0.0), 3))
    if cache_key is not None and cache_key in _PREVIEW_CACHE:
        _PREVIEW_CACHE.move_to_end(cache_key)
        return _PREVIEW_CACHE[cache_key]
    from models.minimax_h3 import pipeline as h3_pipeline
    _log(f"decoding a RefMod preview for the text encoder (once per mod per session; "
         f"key={'ok' if cache_key is not None else 'UNCACHEABLE -- this will repeat every window'})")
    decoded = pipeline_self.vae.decode(
        sentinel.preview_latent.to(device=pipeline_self.device, dtype=torch.float32))
    video = decoded.float().clamp(-1.0, 1.0)[0].cpu()
    decoded = None
    if kind == "image":
        frames, timestamps, budget = video[:, :1].clone(), None, _PREVIEW_MAX_PIXELS_IMAGE
    else:
        rate = float(fps or FPS_ASSUMED_FOR_DURATION_ESTIMATE)
        indices, cursor = [], 0.0
        while round(cursor) < video.shape[1]:
            if not indices or round(cursor) > indices[-1]:
                indices.append(round(cursor))
            cursor += rate / 2
        if len(indices) > _PREVIEW_MAX_VIDEO_FRAMES:  # thin out evenly, keep the ends
            step = (len(indices) - 1) / (_PREVIEW_MAX_VIDEO_FRAMES - 1)
            indices = [indices[round(i * step)] for i in range(_PREVIEW_MAX_VIDEO_FRAMES)]
        frames = video[:, indices].clone()
        timestamps = [i / rate for i in indices]
        budget = _PREVIEW_MAX_PIXELS_VIDEO
    video = None
    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    if height * width > budget:
        shrink = (budget / float(height * width)) ** 0.5
        frames = h3_pipeline._resize_video(frames, max(32, int(height * shrink) // 2 * 2),
                                           max(32, int(width * shrink) // 2 * 2))
    result = (h3_pipeline._qwen_frames(frames.contiguous()), timestamps)
    frames = None
    if cache_key is not None:
        _cache_put(_PREVIEW_CACHE, cache_key, result, _PREVIEW_CACHE_MAX)
    return result


# ── Encoder frames (see encframes.py) ─────────────────────────────────────
#
# The pictures a mod is shown to the text encoder as. Taken from the mod file
# when it stores them (no decode at all), otherwise decoded once per session
# and kept here. Kept as uint8 at their stored size; softening, picking and
# resizing happen per use, which is cheap.
_ENC_CACHE = collections.OrderedDict()
_ENC_CACHE_MAX = 16


def _decode_cthw(pipeline_self, latent):
    """VAE-decode a latent to [C, T, H, W] float in [-1, 1] on the CPU."""
    decoded = pipeline_self.vae.decode(latent.to(device=pipeline_self.device, dtype=torch.float32))
    video = decoded.float().clamp(-1.0, 1.0)[0].cpu()
    decoded = None
    return video


def _cthw_to_thwc01(video):
    return video.permute(1, 2, 3, 0).add(1.0).mul(0.5).clamp(0.0, 1.0)


def decode_encoder_frames(pipeline_self, mod, fps=None):
    """Decode a visual mod into its encoder frames: ([N, H, W, 3] float 0..1,
    timestamps, fps, layout). Pictures (single or stacked) are decoded one
    latent frame at a time, each on its own -- they were encoded separately, so
    decoding them as one clip would smear them into each other. A video mod is
    decoded as a clip and sampled at two frames a second."""
    layout = mod.wanted_layout()
    if layout == encframes.LAYOUT_STILLS:
        pictures = []
        for j in range(mod.latent.shape[2]):
            video = _decode_cthw(pipeline_self, mod.latent[:, :, j:j + 1])
            pictures.append(encframes.cap_pixels(_cthw_to_thwc01(video[:, :1]),
                                                 encframes.MAX_PIXELS_STILL))
            video = None
        frames = torch.cat(pictures)
        return frames, [float(j) for j in range(frames.shape[0])], encframes.ENC_FPS, layout
    rate = float(fps or encframes.ENC_FPS)
    video = _decode_cthw(pipeline_self, mod.latent)
    indices, times = encframes.clip_picks(video.shape[1], rate)
    frames = encframes.cap_pixels(_cthw_to_thwc01(video[:, indices]), encframes.MAX_PIXELS_CLIP)
    video = None
    return frames, times, rate, layout


def _encoder_frames(pipeline_self, mod, fps=None):
    """A mod's encoder frames as ([N, H, W, 3] uint8, timestamps, layout):
    stored in the file when usable, else decoded once and cached."""
    layout = mod.wanted_layout()
    rate = None if layout == encframes.LAYOUT_STILLS else float(fps or encframes.ENC_FPS)
    base = getattr(mod, "_h3_preview_key", None)
    key = None if base is None else (base, layout, rate)
    if key is not None and key in _ENC_CACHE:
        _ENC_CACHE.move_to_end(key)
        return _ENC_CACHE[key]
    if mod.encoder_frames_usable(rate):
        frames, times, how = mod.encoder_frames_uint8(), list(mod.enc_times), "stored in the mod"
    else:
        frames, times, _rate, _layout = decode_encoder_frames(pipeline_self, mod, rate)
        frames = (frames * 255).round().clamp(0, 255).to(torch.uint8)
        how = ("decoded (no stored frames -- Library > Encoder frames can store them)"
               if not mod.has_encoder_frames() else
               "decoded (the stored frames don't match how this mod is used here)")
    _log(f"encoder frames for '{mod.name}': {frames.shape[0]} {layout} frame(s), {how}"
         + ("" if key is not None else " -- uncacheable, this repeats every window"))
    result = (frames, times, layout)
    if key is not None:
        _cache_put(_ENC_CACHE, key, result, _ENC_CACHE_MAX)
    return result


def _presentation_frames(pipeline_self, sentinel, kind, fps=None):
    """What the text encoder is shown for a RefMod sentinel, ready for a
    presentation item: (frames [N, H, W, 3] float 0..1, timestamps or None).

      * a picture: that one picture
      * a stack sent as one video: its pictures (all, or up to N), each
        duplicated to fill a two-frame block on its own
      * a clip: its two-a-second frames, thinned to 8 (and cut to what a
        budget trim left)

    Softened when the mod is used below strength 1, the same way its latent
    is, so the encoder isn't shown a stronger reference than the DiT gets."""
    mod = sentinel.mod
    frames, times, layout = _encoder_frames(pipeline_self, mod, fps)
    strength = float(getattr(sentinel, "enc_strength", 1.0))
    if kind == "image":
        index = min(int(getattr(sentinel, "still_index", 0)), frames.shape[0] - 1)
        chosen = encframes.to_float01(frames[index:index + 1])
        return encframes.soften(chosen, strength, mod.latent_h, mod.latent_w), None
    if getattr(sentinel, "is_stack", False):
        picks = encframes.stack_picks(frames.shape[0], sentinel.stack_mode, sentinel.stack_n)
        chosen, stamps = encframes.stack_presentation(frames, picks)
        return encframes.soften(encframes.to_float01(chosen), strength, mod.latent_h, mod.latent_w), stamps
    keep = list(range(frames.shape[0]))
    limit = getattr(sentinel, "enc_time_limit", None)
    if limit is not None:
        keep = [i for i in keep if times[i] < limit + 1e-6] or [0]
    available = len(keep)
    if getattr(sentinel, "clip_mode", encframes.STACK_UP_TO_N) != encframes.STACK_ALL:
        limit_n = int(getattr(sentinel, "clip_n", encframes.MAX_CLIP_FRAMES_SHOWN) or 1)
        keep = [keep[i] for i in encframes.thin_evenly(len(keep), max(1, limit_n))]
    _log(f"'{mod.name}': text encoder shown {len(keep)} of {available} clip frame(s)")
    chosen = encframes.to_float01(frames[keep])
    return (encframes.soften(chosen, strength, mod.latent_h, mod.latent_w),
            [float(times[i]) for i in keep])


_VAE_SPATIAL_FACTOR = 16  # pixels per latent cell, both axes


def _would_shrink(sentinel, kind, pipeline_self, target_width, target_height, relative):
    """Resolve the size a live reference would get here, and report it only if
    it is SMALLER than the mod's stored latent (never upscale). Cheap: works
    from shapes, so nothing is decoded unless a rebuild is actually happening."""
    try:
        lat_h, lat_w = int(sentinel.latent.shape[-2]), int(sentinel.latent.shape[-1])
        if kind == "image":
            from models.minimax_h3 import pipeline as h3_pipeline
            stored_w, stored_h = lat_w * _VAE_SPATIAL_FACTOR, lat_h * _VAE_SPATIAL_FACTOR
            ratio = stored_w / max(1, stored_h)
            budget = int(target_width) * int(target_height) * float(relative or 100.0) / 100.0
            new_h, new_w = h3_pipeline._resolve_canvas(
                stored_w, stored_h, math.sqrt(budget / max(ratio, 1 / ratio)))
        else:
            new_h, new_w = int(target_height), int(target_width)
        if (new_h // _VAE_SPATIAL_FACTOR) * (new_w // _VAE_SPATIAL_FACTOR) >= lat_h * lat_w:
            return None, None  # same size or larger -- leave the stored latent alone
        return int(new_w), int(new_h)
    except Exception:
        return None, None


def _capped_rescale_size(sentinel, kind, target_width, target_height):
    """Shrink the requested phase-2 size until the rebuilt mod costs at most
    REFMOD_RESCALE_MAX_TOKEN_GROWTH times its stored token count. Returns
    (None, None) if there is nothing worth rebuilding."""
    try:
        stored_h, stored_w = int(sentinel.latent.shape[-2]), int(sentinel.latent.shape[-1])
        want_h = max(1, int(target_height) // _VAE_SPATIAL_FACTOR)
        want_w = max(1, int(target_width) // _VAE_SPATIAL_FACTOR)
        growth = (want_h * want_w) / float(max(1, stored_h * stored_w))
        if growth <= 1.0:
            return None, None  # already at or above the phase-2 scale
        cap = float(REFMOD_RESCALE_MAX_TOKEN_GROWTH)
        if cap > 0 and growth > cap:
            shrink = (cap / growth) ** 0.5
            target_height = max(_VAE_SPATIAL_FACTOR,
                                int(round(int(target_height) * shrink / _VAE_SPATIAL_FACTOR)) * _VAE_SPATIAL_FACTOR)
            target_width = max(_VAE_SPATIAL_FACTOR,
                               int(round(int(target_width) * shrink / _VAE_SPATIAL_FACTOR)) * _VAE_SPATIAL_FACTOR)
            _log(f"phase 2: capping the {kind}-kind RefMod rebuild at {cap:g}x its stored token "
                 f"count -- {target_width}x{target_height} instead of the full phase-2 size "
                 f"(see REFMOD_RESCALE_MAX_TOKEN_GROWTH)")
        return int(target_width), int(target_height)
    except Exception:
        return None, None


def _rescaled_refmod_latent(pipeline_self, sentinel, kind, target_width=None,
                            target_height=None, relative=100.0, force=False):
    """Rebuild a RefMod at a given reference size, the way the pipeline rebuilds
    a live reference: decode the stored latent, resize the pixels, re-encode.
    force=True is the phase-1 downscale-to-canvas path; without it this is the
    opt-in phase-2 upscale. Returns None to keep the stored latent."""
    if not target_width or not target_height:
        return None
    if not force:
        # phase-2 upscale path (opt-in)
        if not RESCALE_REFMODS_IN_PHASE_2:
            return None
        if kind != "image" and not RESCALE_REFMOD_VIDEOS_IN_PHASE_2:
            return None
        target_width, target_height = _capped_rescale_size(sentinel, kind, target_width, target_height)
        if target_width is None:
            return None
    try:
        base = getattr(sentinel, "preview_key", None)
        # The stored latent already has strength/curve baked in, so decode that
        # one (not the unweighted preview) and stamp the cache with it.
        stamp = _latent_stamp(sentinel.latent)
        key = None if (base is None or stamp is None) else \
            (base, stamp, kind, int(target_width), int(target_height), round(float(relative or 100.0), 3))
        if key is not None and key in _RESCALE_CACHE:
            _RESCALE_CACHE.move_to_end(key)
            return _RESCALE_CACHE[key]
        from models.minimax_h3 import pipeline as h3_pipeline
        if getattr(sentinel, "is_stack", False):
            # A stack's latent frames are separate pictures: rebuild each one
            # on its own, as an image, never as frames of one clip.
            parts = []
            for j in range(sentinel.latent.shape[2]):
                picture = _decode_cthw(pipeline_self, sentinel.latent[:, :, j:j + 1])[:, :1]
                picture = h3_pipeline._resize_video(picture, int(target_height), int(target_width))
                parts.append(pipeline_self._encode_video(picture))
                picture = None
            latent = torch.cat(parts, dim=2)
            _log(f"phase {_gen_state(pipeline_self)['phase']}: rebuilt a {latent.shape[2]}-picture "
                 f"stack at latent {tuple(latent.shape[-2:])} "
                 f"(was {tuple(sentinel.latent.shape[-2:])})")
            if key is not None:
                _cache_put(_RESCALE_CACHE, key, latent, _RESCALE_CACHE_MAX)
            return latent
        decoded = pipeline_self.vae.decode(
            sentinel.latent.to(device=pipeline_self.device, dtype=torch.float32))
        video = decoded.float().clamp(-1.0, 1.0)[0].cpu()
        decoded = None
        if kind == "image":
            # _prepare_image_reference -> _to_pil already takes a CTHW float
            # tensor in [-1, 1] and does the byte conversion itself.
            pixels = pipeline_self._prepare_image_reference(
                video[:, :1].clone(), target_width, target_height, relative)
        else:
            pixels = h3_pipeline._resize_video(video.clone(), int(target_height), int(target_width))
        if pixels is None:
            return None
        video = None
        latent = pipeline_self._encode_video(pixels)
        _log(f"phase {_gen_state(pipeline_self)['phase']}: rebuilt a {kind}-kind RefMod at "
             f"{tuple(pixels.shape[-2:])} -> latent {tuple(latent.shape[-2:])} "
             f"(was {tuple(sentinel.latent.shape[-2:])} in latent space)")
        pixels = None
        if key is not None:
            _cache_put(_RESCALE_CACHE, key, latent, _RESCALE_CACHE_MAX)
        return latent
    except Exception:
        _log("could not rebuild a RefMod at this size; using its stored latent as before:\n"
             + traceback.format_exc())
        return None


def _log_label(pipeline_self, presentation, entry, label):
    """Report which <Picture N> / <Video N> the text encoder will give this
    RefMod, per window -- the numbering shifts between windows (see
    STABLE_REFMOD_LABELS), so this is what to compare across windows."""
    try:
        seen = 0
        for item in presentation:
            if item.get("type") == entry["type"]:
                seen += 1
            if item is entry:
                break
        _log(f"window {int(getattr(pipeline_self, '_h3_window_no', 1) or 1)}: "
             f"RefMod bound to <{label} {seen}>")
    except Exception:
        pass


def _present_refmod(pipeline_self, presentation, kind, sentinel, fps=None):
    """Give a RefMod the same text-side entry a live reference gets, so the
    Qwen3-VL prompt contains '<Picture N>: [image]' / '<Video N>: ...' for it.
    Without this the latent sits in the sequence with no label binding it to
    the prompt, and the model largely ignores it."""
    if presentation is None or not PROMPT_LABELS_FOR_REFMODS \
            or getattr(pipeline_self, "fixed_prompt", None) is not None:
        return
    try:
        if getattr(sentinel, "mod", None) is not None:
            frames, timestamps = _presentation_frames(pipeline_self, sentinel, kind, fps)
        else:
            frames, timestamps = _decode_preview(pipeline_self, sentinel, kind, fps)
        if kind == "image":
            entry = {"type": "image", "frames": frames}
            window_no = int(getattr(pipeline_self, "_h3_window_no", 1) or 1)
            live_refs = int(getattr(pipeline_self, "_h3_live_image_refs", 0) or 0)
            at = None
            if STABLE_REFMOD_LABELS and window_no > 1 and live_refs == 0:
                # put it ahead of the carried-over frame added by the keyframes
                at = next((i for i, item in enumerate(presentation)
                           if item.get("type") == "image"), None)
            if at is None:
                presentation.append(entry)
            else:
                presentation.insert(at, entry)
            _log_label(pipeline_self, presentation, entry, "Picture")
            return
        entry = {"type": "video", "frames": frames, "timestamps": timestamps}
        presentation.append(entry)
        _log_label(pipeline_self, presentation, entry, "Video")
    except Exception:
        _log(f"could not build the prompt-side preview for a {kind}-kind RefMod; it is still "
             f"injected, but without a label the model may ignore it:\n" + traceback.format_exc())


_STATE_ATTR = "_h3refmod_gen_state"

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
    stack_as_video = bool(state.get("stack_as_video", False))
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
        pictures = kind == "image" or core.is_still_stack(kind, latent_t, meta.get("tags"),
                                                          meta.get("source", ""))
        if pictures and bool(row.get("as_video", stack_as_video)) and latent_t > 1:
            total += 1   # the whole stack is one video reference
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

    This adds one plain "text" custom setting (id SETTING_COMBINED, holding
    both the selection and any extraction job) to MiniMax H3's model
    definition. Since fork.12 it is hidden on Wan2GP's Media Generator form
    (see HIDDEN_SETTING_FLAG); it still travels with every task.
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

    extra_settings = [
        # Hidden on the form: Wan2GP shows a custom setting only when the
        # video prompt type contains one of the letters in "video_prompt_type",
        # and this one never appears in it. The setting still exists, so the
        # selection still travels with every task, queue entry and saved
        # settings file -- it just no longer shows up as a text box nobody is
        # meant to touch.
        {"id": SETTING_COMBINED, "name": "H3RefMod",
         "label": "RefMods (managed by the MiniMax H3 RefMods plugin)",
         "type": "text", "default": "", "video_prompt_type": HIDDEN_SETTING_FLAG},
    ]

    @staticmethod
    def patched_query_model_def(base_model_type, model_def):
        result = _orig_query_model_def(base_model_type, model_def)
        if isinstance(result, dict):
            existing = result.get("custom_settings")
            existing = list(existing) if isinstance(existing, list) else []
            existing_ids = {e.get("id") for e in existing if isinstance(e, dict)}
            merged = existing + [s for s in extra_settings if s["id"] not in existing_ids]
            if len(merged) > CUSTOM_SETTINGS_MAX_ASSUMED:
                _log(f"WARNING: {base_model_type} declares {len(existing)} custom settings and "
                     f"Wan2GP keeps only the first {CUSTOM_SETTINGS_MAX_ASSUMED}, so the plugin's "
                     f"setting is being dropped. RefMods will rely on the in-memory fallbacks; "
                     f"extraction and mod selection may be unreliable.")
            result["custom_settings"] = merged
        return result

    FamilyHandler.query_model_def = patched_query_model_def
    setattr(FamilyHandler, _MODEL_DEF_PATCH_MARKER, True)
    _log(f"declared the {SETTING_COMBINED} custom setting on MiniMax H3's model definition "
         f"(one slot, so it fits inside Wan2GP's limit of {CUSTOM_SETTINGS_MAX_ASSUMED})")

    if getattr(FamilyHandler, _VALIDATE_PATCH_MARKER, False):
        return
    _orig_validate = getattr(FamilyHandler, "validate_generative_settings", None)
    if _orig_validate is None:
        return

    @staticmethod
    def patched_validate_generative_settings(base_model_type, model_def, inputs):
        error = _orig_validate(base_model_type, model_def, inputs)
        try:
            if _ARMED["json"] and is_minimax_h3_refmod_capable(base_model_type):
                attached = bool(unpack_refmod_setting(inputs.get("custom_settings"))[0])
                _log("task submission: RefMod selection " + ("attached to the task." if attached else
                     "NOT in the task's custom_settings -- generate() will re-apply the inline "
                     "panel's selection."))
        except Exception:
            pass
        # Wan2GP's own pre-flight check (called before pipeline.generate() ever
        # runs) only counts *native* image_refs/video_guide fields -- it has no
        # visibility into RefMods, which only turn into visual references much
        # later, inside our own generate() patch. Rather than re-implementing
        # every rule this function enforces (durations, per-type caps, control-
        # video-specific checks...), only step in for this one specific failure:
        # if RefMods would supply enough visual references to satisfy it, clear
        # it; every other check the original function performs is untouched.
        if error and _INSUFFICIENT_VISUAL_MARKER in error:
            # The selection travels in the combined setting (SETTING_COMBINED);
            # reading the old SETTING_GENERATE key here always came back empty.
            state_json = unpack_refmod_setting(inputs.get("custom_settings"))[0]
            state_json = state_json or _ARMED["json"]
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
    _log(f"Wan2GP reference slots: "
         f"{3 if 'input_frames3' in _generate_params else 2} video / "
         f"{3 if 'audio_guide3' in _generate_params else 2} audio")
    _orig_as_video = getattr(h3_pipeline, "_as_video", None)

    @functools.wraps(_orig_add_image_reference)
    def patched_add_image_reference(self, image, target_width, target_height,
                                     image_refs_relative_size, presentation, visual_latents, refs):
        if isinstance(image, _RefModImageSentinel):
            latent = image.latent
            if _gen_state(self)["phase"] == 1 and FIT_REFMODS_TO_PHASE_1_CANVAS \
                    and getattr(self, "_h3_two_phase", False):
                fit_w, fit_h = _would_shrink(image, "image", self, target_width, target_height,
                                             image_refs_relative_size)
                if fit_w is not None:
                    # relative=100 on purpose: _would_shrink already applied
                    # image_refs_relative_size when resolving fit_w/fit_h, and
                    # _prepare_image_reference would otherwise apply it a
                    # second time and shrink the mod twice.
                    rebuilt = _rescaled_refmod_latent(self, image, "image", target_width=fit_w,
                                                      target_height=fit_h, relative=100.0,
                                                      force=True)
                    if rebuilt is not None:
                        latent = rebuilt
            if _gen_state(self)["phase"] == 2:
                # NB: explicit "is not None" -- `rebuilt or latent` calls bool()
                # on a tensor, which raises "Boolean value of Tensor with more
                # than one value is ambiguous".
                rebuilt = _rescaled_refmod_latent(self, image, "image", target_width=target_width,
                                                  target_height=target_height,
                                                  relative=image_refs_relative_size)
                if rebuilt is not None:
                    latent = rebuilt
            visual_latents.append(latent)
            _place_refmod_ref(self, image, refs,
                              {"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})
            _present_refmod(self, presentation, "image", image)
            # An image mod's soundtrack is handled in _inject_refmods via the
            # overflow-audio path: the packer's image branch has no audio rows
            # (unlike its video branch), and _add_image_reference isn't even
            # given audio_latents to append to.
            return
        _note_refs(self, refs)
        return _orig_add_image_reference(self, image, target_width, target_height,
                                          image_refs_relative_size, presentation, visual_latents, refs)

    @functools.wraps(_orig_add_video_reference)
    def patched_add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs):
        if isinstance(video, _RefModVideoSentinel):
            latent = video.latent
            if _gen_state(self)["phase"] == 1 and FIT_REFMODS_TO_PHASE_1_CANVAS \
                    and getattr(self, "_h3_two_phase", False) \
                    and getattr(video, "rescale_to", None):
                height, width = video.rescale_to
                fit_w, fit_h = _would_shrink(video, "video", self, width, height, None)
                if fit_w is not None:
                    rebuilt = _rescaled_refmod_latent(self, video, "video", target_width=fit_w,
                                                      target_height=fit_h, force=True)
                    if rebuilt is not None:
                        latent = rebuilt
            if _gen_state(self)["phase"] == 2 and getattr(video, "rescale_to", None):
                height, width = video.rescale_to
                rebuilt = _rescaled_refmod_latent(self, video, "video", target_width=width,
                                                  target_height=height)
                if rebuilt is not None:
                    latent = rebuilt
            # A mod can carry its own soundtrack. H3 handles this natively:
            # a reference video with audio is tagged "video_audio" with
            # ref_audio_t set, and the packer gives it audio rows as well as
            # video rows. Order matters -- upstream appends the audio latent
            # and its presentation entry BEFORE the video ones, and the packer
            # lays each reference's audio rows out ahead of its video rows.
            attached_audio = getattr(video, "audio_latent", None)
            merged = (attached_audio is not None and not ATTACHED_AUDIO_AS_SEPARATE_REF
                      and isinstance(audio_latents, list))
            if merged:
                # H3's own shape for a reference video's soundtrack: the audio
                # latent and its label go in BEFORE the video's, and the packer
                # lays each reference's audio rows ahead of its video rows.
                audio_latents.append(attached_audio)
                if PROMPT_LABELS_FOR_REFMODS and presentation is not None:
                    presentation.append({"type": "audio"})
                _log(f"video RefMod carries its own soundtrack "
                     f"({attached_audio.shape[-1]} audio latents), merged as a video_audio "
                     f"reference")
            visual_latents.append(latent)
            _place_refmod_ref(self, video, refs,
                              {"kind": "video_audio" if merged else "video",
                               "latent_t": latent.shape[2],
                               "latent_h": latent.shape[-2], "latent_w": latent.shape[-1],
                               "ref_audio_t": attached_audio.shape[-1] if merged else 0})
            _present_refmod(self, presentation, "video", video, fps)
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
                audio_latents.append(latent)
                _place_refmod_ref(self, waveform, refs,
                                  {"kind": "audio", "ref_audio_t": latent.shape[-1]})
                if PROMPT_LABELS_FOR_REFMODS:
                    presentation.append({"type": "audio"})
                return
            _note_refs(self, refs)
            return _orig_add_audio_reference(self, waveform, presentation, audio_latents, refs)
        Pipeline._add_audio_reference = patched_add_audio_reference
    else:
        _log("could not find _add_audio_reference / _load_audio_reference on MiniMaxH3Pipeline "
             "to patch; audio-kind RefMods will not be available (image and video RefMods are "
             "unaffected). This Wan2GP build may not support direct audio references yet.")

    _orig_encode_prompt = getattr(Pipeline, "_encode_prompt", None)
    if _orig_encode_prompt is not None:
        @functools.wraps(_orig_encode_prompt)
        def patched_encode_prompt(self, prompt, presentation, *args, **kwargs):
            # FL2VA builds its presentation from keyframes only. Append the
            # staged RefMod labels so an audio mod arrives as "<Audio 1>: "
            # rather than as an unannounced block of audio rows.
            if _FL2VA["active"] and _FL2VA.get("presentation") and isinstance(presentation, list):
                presentation.extend(_FL2VA["presentation"])
                _log(f"FL2VA: added {len(_FL2VA['presentation'])} RefMod label(s) to the prompt "
                     f"(FL2VA_PROMPT_LABELS={FL2VA_PROMPT_LABELS!r})")
            return _orig_encode_prompt(self, prompt, presentation, *args, **kwargs)
        Pipeline._encode_prompt = patched_encode_prompt

    _orig_prepare_condition_rows = getattr(Pipeline, "_prepare_condition_rows", None)
    if _orig_prepare_condition_rows is not None:
        @functools.wraps(_orig_prepare_condition_rows)
        def patched_prepare_condition_rows(self, visual_latents, audio_latents, generator, *args, **kwargs):
            if _FL2VA["active"] and not getattr(self, "reference_mode", False) \
                    and isinstance(visual_latents, list):
                # Read the phase before _restore_hidden_refs advances it. Audio
                # only on the first pass: both phase-2 paths pass a throwaway
                # [] for audio and keep phase 1's audio rows, so appending
                # there would only consume generator randomness.
                first_pass = _gen_state(self)["phase"] == 1
                visual_latents.extend(_FL2VA["visual"])
                if first_pass and isinstance(audio_latents, list):
                    audio_latents.extend(_FL2VA["audio"])
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
                # Remember the size that was asked for: in phase 2 this is the
                # target resolution, and _add_video_reference uses it to rebuild
                # the mod at that scale (see RESCALE_REFMODS_IN_PHASE_2).
                video.rescale_to = (height, width)
                return video
            return _orig_resize_video(video, height, width)
        h3_pipeline._resize_video = patched_resize_video
    else:
        _log("no _resize_video found in models.minimax_h3.pipeline -- fine on older Wan2GP "
             "builds that don't have it; on newer ones video-kind RefMods would fail with "
             "\"'_RefModVideoSentinel' object has no attribute 'permute'\".")

    @functools.wraps(_orig_generate)
    def patched_generate(self, *args, **kwargs):
        custom_settings = kwargs.get("custom_settings")
        custom_settings = custom_settings if isinstance(custom_settings, dict) else {}

        state_from_task, extract_job = unpack_refmod_setting(custom_settings)
        if not extract_job:
            extract_job = _take_pending_extract()
            if extract_job:
                _log("this task reached generate() without its extraction job (Wan2GP dropped the "
                     "custom_settings payload); running the extraction that was just submitted. "
                     "Without this it would render a video instead of extracting a mod.")
        else:
            set_pending_extract(None)   # arrived normally; don't leave a duplicate staged
        if extract_job:
            set_progress_status = kwargs.get("set_progress_status")
            try:
                _run_extract_job(self, extract_job, set_progress_status=set_progress_status)
            except ExtractionBlocked as blocked:
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
        _reset_gen_state(self)

        state_json = state_from_task
        _REFMODS_ACTIVE.update({"on": False, "logged": False})
        first_window = int(kwargs.get("window_no") or 1) <= 1
        setattr(self, "_h3_window_no", int(kwargs.get("window_no") or 1))
        setattr(self, "_h3_live_image_refs", len(kwargs.get("input_ref_images") or []))
        # pipeline.py: two_phase = int(guide_phases) > 1 and frozen_target_video is None.
        # A single-phase run is *also* "phase 1", so the phase-1 canvas fit must
        # check this and not just the phase counter -- otherwise it shrinks mods
        # on ordinary single-phase generations and costs detail (faces first).
        setattr(self, "_h3_two_phase",
                int(kwargs.get("guide_phases") or 1) > 1
                and kwargs.get("frozen_control_video") is None)
        _clear_fl2va()
        fl2va = FL2VA_REFMODS and _is_fl2va_pipeline(self)
        refmod_capable = _is_ref2va_pipeline(self) or fl2va
        if refmod_capable:
            _log(f"generate(): {'FL2VA' if fl2va else 'Ref2VA'}, "
                 f"window {int(kwargs.get('window_no') or 1)}, "
                 f"task payload={'yes' if state_json else 'no'}, "
                 f"armed={'yes' if _ARMED['json'] else 'no'}")
        if not state_json and _ARMED["json"] and refmod_capable \
                and not kwargs.get("refinement_mode"):
            state_json = _ARMED["json"]
            if first_window:
                _log("NOTE: this task reached generate() without its RefMod selection, so the "
                     "inline panel's current selection is being used instead. If you did not "
                     "expect mods in this generation, clear the panel (or press 'Clear armed "
                     "RefMods') -- an unexpected mod takes a reference slot from your own "
                     "references.")
        if state_json and fl2va:
            if not kwargs.get("refinement_mode"):
                try:
                    _stage_fl2va_refmods(self, state_json)
                except Exception:
                    _clear_fl2va()
                    _log("could not stage RefMods for FL2VA, continuing without them:\n"
                         + traceback.format_exc())
        elif state_json:
            try:
                kwargs = _inject_refmods(self, kwargs, state_json)
            except Exception:
                _log("could not inject RefMods, continuing without them:\n" + traceback.format_exc())
        elif refmod_capable and first_window and not kwargs.get("refinement_mode"):
            _log("no RefMod selection for this task, and none armed in the inline panel -- if you "
                 "picked mods there, re-select them once after restarting Wan2GP.")

        try:
            return _orig_generate(self, *args, **kwargs)
        finally:
            _clear_fl2va()   # never let staged mods leak into the next generation

    Pipeline._add_image_reference = patched_add_image_reference
    Pipeline._add_video_reference = patched_add_video_reference
    Pipeline.generate = patched_generate
    _install_sol_attn_guard()
    _install_fl2va_forward_hook()

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
            if isinstance(state, dict) and isinstance(result, dict):
                resolved_type = model_type or get_state_model_type_fn(state)
                if is_minimax_h3_refmod_capable(resolved_type, get_base_model_type_fn):
                    stash = state.get(STASH_KEY)
                    _sync_armed_with_panel(stash)
                    existing = dict(result.get("custom_settings") or {})
                    if stash:
                        existing[SETTING_COMBINED] = pack_refmod_setting(state_json=stash)
                        existing.pop(SETTING_GENERATE, None)   # legacy key, no longer declared
                        result["custom_settings"] = existing
                    elif SETTING_COMBINED in existing or SETTING_GENERATE in existing:
                        existing.pop(SETTING_COMBINED, None)
                        existing.pop(SETTING_GENERATE, None)
                        result["custom_settings"] = existing or None
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
                _sync_armed_with_panel(stash)
                existing = dict(settings.get("custom_settings") or {})
                if stash:
                    existing[SETTING_COMBINED] = pack_refmod_setting(state_json=stash)
                    existing.pop(SETTING_GENERATE, None)   # legacy key, no longer declared
                    settings = dict(settings)
                    settings["custom_settings"] = existing
                elif SETTING_COMBINED in existing or SETTING_GENERATE in existing:
                    existing.pop(SETTING_COMBINED, None)
                    existing.pop(SETTING_GENERATE, None)
                    settings = dict(settings)
                    settings["custom_settings"] = existing or None
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

_FL2VA_FORWARD_MARKER = "_h3refmod_fl2va_forward_patched"


def _install_fl2va_forward_hook() -> None:
    """Fill payload["refs"] for FL2VA on the transformer's way in. generate()
    builds the payload with refs=None in FL2VA and both phase-2 paths reset
    it to None, so this re-applies on every pass; the layout cache keys on
    the refs, so the reference-aware packer is used from the first step."""
    try:
        from models.minimax_h3 import transformer as h3_transformer
    except Exception as e:
        _log(f"could not import models.minimax_h3.transformer ({e!r}); RefMods won't work in FL2VA")
        return
    Model = getattr(h3_transformer, "MiniMaxH3Model", None)
    if Model is None or getattr(Model, _FL2VA_FORWARD_MARKER, False):
        return
    _orig_forward = Model.forward

    @functools.wraps(_orig_forward)
    def patched_forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload, *args, **kwargs):
        if _FL2VA["active"] and _FL2VA["entries"] and isinstance(payload, dict) \
                and not payload.get("refs"):
            payload["refs"] = list(_FL2VA["entries"])
            if not _FL2VA["announced"]:
                _FL2VA["announced"] = True
                _log(f"FL2VA: {len(_FL2VA['entries'])} RefMod reference(s) attached to the "
                     f"transformer payload -- the reference-aware packer is now in use")
        return _orig_forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload,
                             *args, **kwargs)

    Model.forward = patched_forward
    setattr(Model, _FL2VA_FORWARD_MARKER, True)


_SOL_PATCH_MARKER = "_h3refmod_sol_patched"


def _install_sol_attn_guard() -> None:
    """Route RefMod generations around Sol-Attn's INT8 Triton kernels.
    See DISABLE_SOL_WITH_REFMODS. No effect when the attention mode isn't
    "sol", and none on generations without RefMods."""
    try:
        from models.minimax_h3 import sol_attention as h3_sol
    except Exception as e:
        _log(f"could not import models.minimax_h3.sol_attention ({e!r}); "
             f"the Sol-Attn guard is not installed")
        return
    SolCls = getattr(h3_sol, "MiniMaxH3SolAttention", None)
    if SolCls is None or getattr(SolCls, _SOL_PATCH_MARKER, False):
        return
    _orig_use_for_layer = SolCls.use_for_layer

    @functools.wraps(_orig_use_for_layer)
    def patched_use_for_layer(self, tokens):
        if DISABLE_SOL_WITH_REFMODS and _REFMODS_ACTIVE["on"] and getattr(self, "enabled", False):
            if not _REFMODS_ACTIVE["logged"]:
                _REFMODS_ACTIVE["logged"] = True
                _log("Sol-Attn bypassed for this generation because RefMods are injected "
                     "(its INT8 Triton path can hit an illegal memory access as the condition "
                     "rows grow). Set DISABLE_SOL_WITH_REFMODS = False in patches.py to re-enable.")
            return False
        return _orig_use_for_layer(self, tokens)

    SolCls.use_for_layer = patched_use_for_layer
    setattr(SolCls, _SOL_PATCH_MARKER, True)


def _build_refmod_sentinels(state_json: str):
    """Load and weight a RefMod selection into per-kind sentinels.

    Shared by the Ref2VA path (_inject_refmods, which feeds sentinels
    through generate()'s reference inputs) and the FL2VA path
    (_stage_fl2va_refmods, which can't use those inputs at all). Returns
    (image_sentinels, video_sentinels, audio_sentinels, total_tokens,
    retention), or None when there is nothing to inject."""
    try:
        state = json.loads(state_json)
    except Exception as e:
        _log(f"could not parse RefMod state ({e!r}); ignoring")
        return None
    rows = state.get("rows") or []
    if not rows:
        return None
    retention = float(state.get("retention", 1.0))
    curve = state.get("curve")  # [direction, shape, value] or None
    if isinstance(curve, list):
        curve = tuple(curve)
    seed = int(state.get("scramble_seed", -1))
    # "Send as one video": a multi-picture mod goes in as ONE <Video N>
    # instead of one <Picture N> per picture. Chosen per mod (each row's
    # "as_video"); the old panel-wide "stack_as_video" flag is still honoured
    # as the default for selections saved before fork.12.
    stack_as_video = bool(state.get("stack_as_video", False))
    stack_mode = state.get("stack_pictures") or encframes.STACK_ALL
    if stack_mode not in encframes.STACK_CHOICES:
        stack_mode = encframes.STACK_ALL
    try:
        stack_n = max(1, int(state.get("stack_pictures_n") or encframes.STACK_DEFAULT_N))
    except (TypeError, ValueError):
        stack_n = encframes.STACK_DEFAULT_N
    # How many frames of a video mod the encoder is shown. Defaults to the
    # previous fixed behaviour (up to 8) for selections saved before this.
    clip_mode = state.get("video_frames") or encframes.STACK_UP_TO_N
    if clip_mode not in encframes.STACK_CHOICES:
        clip_mode = encframes.STACK_UP_TO_N
    try:
        clip_n = max(1, int(state.get("video_frames_n") or encframes.MAX_CLIP_FRAMES_SHOWN))
    except (TypeError, ValueError):
        clip_n = encframes.MAX_CLIP_FRAMES_SHOWN

    loaded_rows = []  # (mod, strength, copies, as_video, use_audio), one per selected row, in order
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
        loaded_rows.append((mod, strength, copies, bool(row.get("as_video", stack_as_video)),
                            # "Include audio": a visual mod's attached soundtrack can be
                            # left out per row (on unless the row says otherwise).
                            bool(row.get("use_audio", True))))

    if not loaded_rows:
        return None

    if seed >= 0 and len(loaded_rows) > 1:
        rng = random.Random(seed)
        rng.shuffle(loaded_rows)
        keep = rng.randint(max(1, len(loaded_rows) // 2), len(loaded_rows))
        loaded_rows = loaded_rows[:keep]
        _log(f"scramble seed={seed}: kept {len(loaded_rows)}/{len(rows)} row(s), order shuffled")

    image_sentinels, video_sentinels, audio_sentinels = [], [], []
    total_tokens = 0
    for mod, strength, copies, as_video, use_audio in loaded_rows:
        soundtrack = mod.audio_latent if use_audio else None
        if mod.audio_latent is not None and not use_audio:
            _log(f"'{mod.name}': soundtrack left out (Include audio unticked)")
        eff = max(0.0, min(2.0, strength * retention))
        latent = _weighted_latent_cached(mod, eff, curve)
        if latent is None:
            continue
        total_tokens += mod.token_count
        # Pictures, single or stacked -- including a ComfyUI picture stack,
        # which ComfyUI saves as kind "video" (see core.is_still_stack).
        pictures = mod.kind == "image" or mod.still_stack()
        if pictures and as_video and latent.shape[2] > 1:
            # One video reference holding every picture. "Copies" repeats the
            # pictures inside that one slot (one label), as for a video mod.
            if copies > 1:
                latent = latent.repeat(1, 1, copies, 1, 1)
            video_sentinels.append(_RefModVideoSentinel(
                latent, mod.latent, _preview_key(mod, "stack", 0), audio_latent=soundtrack,
                mod=mod, enc_strength=eff, is_stack=True, stack_mode=stack_mode, stack_n=stack_n))
            _log(f"'{mod.name}': {latent.shape[2]} pictures sent as one video reference "
                 f"(encoder shown {len(encframes.stack_picks(mod.latent.shape[2], stack_mode, stack_n))} "
                 f"of {mod.latent.shape[2]})")
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
                    # Attached audio rides on the FIRST still only: it is one
                    # soundtrack for the mod, not one per stacked image.
                    image_sentinels.append(_RefModImageSentinel(
                        latent[:, :, j:j + 1], mod.latent[:, :, j:j + 1], _preview_key(mod, "image", j),
                        audio_latent=soundtrack if j == 0 else None,
                        mod=mod, still_index=j, enc_strength=eff))
        elif mod.kind == "video":
            # Video-kind mods go through Wan2GP's own native reference-video
            # slots directly -- the exact same mechanism "Use Two Reference
            # Videos" uses, one mod per slot -- rather than being merged
            # into a single combined tensor. "Copies" here repeats this
            # mod's own frames within its one slot (extending its own
            # duration) instead of spawning extra slot-consuming instances,
            # since MiniMax H3 only exposes 2 such slots in total.
            if copies > 1:
                latent = latent.repeat(1, 1, copies, 1, 1)
            sentinel = _RefModVideoSentinel(latent, mod.latent, _preview_key(mod, "video", 0),
                                            audio_latent=soundtrack,
                                            mod=mod, enc_strength=eff)
            sentinel.clip_mode, sentinel.clip_n = clip_mode, clip_n
            video_sentinels.append(sentinel)
        else:  # "audio"
            # Same pattern as video-kind mods, one slot per mod (2 native
            # audio-reference slots total); "copies" repeats this mod's own
            # time-axis frames within its one slot rather than spawning a
            # second one.
            if copies > 1:
                latent = latent.repeat(1, 1, 1, copies)
            audio_sentinels.append(_RefModAudioSentinel(latent))

        # A visual mod may also carry a soundtrack. Treat it exactly as a
        # standalone audio mod: it takes a native audio slot, so it gets its
        # <Audio N> label, its place in the reference order and the cap
        # handling -- none of which the direct-injection path provides, and
        # that label is what a prompt refers to for voice reuse.
        if (mod.kind in ("image", "video") and soundtrack is not None
                and ATTACHED_AUDIO_AS_SEPARATE_REF):
            audio_sentinels.append(_RefModAudioSentinel(mod.audio_latent))
            _log(f"{mod.kind} RefMod '{mod.name}' carries a soundtrack "
                 f"({mod.audio_latent.shape[-1]} audio latents, "
                 f"~{mod.audio_latent.shape[-1] / AUDIO_LATENTS_PER_SECOND:.1f}s) -- injected as "
                 f"its own <Audio N> reference; refer to it as <Audio N>, not <Video N>.")

    if not image_sentinels and not video_sentinels and not audio_sentinels:
        return None
    return image_sentinels, video_sentinels, audio_sentinels, total_tokens, retention


# ── FL2VA support ───────────────────────────────────────────────────────
#
# The Ref2VA route above can't work on FL2VA: the pipeline is built with
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
# Like the ComfyUI Apply node, nothing is presented to the text encoder:
# FL2VA prompts carry no reference labels.
# How a mod's attached soundtrack is injected.
#   True  - as its own <Audio N> reference, exactly like a standalone audio
#           mod. Voice reuse from an audio reference is a trained Ref2VA
#           capability, so this is the one to use for vocal timbre.
#   False - merged into the visual reference as H3's "video_audio" kind, which
#           is the shape used for a reference video's OWN soundtrack: it tells
#           the model the audio belongs to that footage rather than offering it
#           as a voice to reuse.
# Image mods always use the separate form -- the packer's image branch has no
# audio rows at all.
ATTACHED_AUDIO_AS_SEPARATE_REF = True

FL2VA_REFMODS = True

# Which RefMods get a prompt label in FL2VA.
#   "audio" - label audio mods only (default)
#   "all"   - label every mod, as Ref2VA does
#   "none"  - no labels, exactly like the ComfyUI original's Apply node
#
# FL2VA presents its own start/end frames to the text encoder as <Picture N>,
# so it understands labels; it simply isn't given any for references. A visual
# reference can still bias the picture unlabelled, but a voice reference has
# much less to grab onto -- "use this voice" has to come from somewhere. Audio
# mods are labelled by default for that reason. Labels are appended after the
# keyframes, so a start frame stays <Picture 1>.
FL2VA_PROMPT_LABELS = "audio"

_FL2VA = {"active": False, "visual": [], "audio": [], "entries": [], "announced": False,
          "presentation": []}


def _clear_fl2va() -> None:
    _FL2VA.update(active=False, visual=[], audio=[], entries=[], announced=False, presentation=[])


def _transformer_has_control(transformer) -> bool:
    """Control checkpoints attach a ControlBlock to their DiT blocks; FL2VA
    (and FL2VA finetunes such as VDN) don't. Control models feed their own
    rows through the payload, so they're left alone."""
    try:
        return any(getattr(block, "control", None) is not None
                   for block in getattr(transformer, "blocks", None) or ())
    except Exception:
        return False


def _is_fl2va_pipeline(pipeline_self) -> bool:
    return (not getattr(pipeline_self, "reference_mode", False)
            and getattr(pipeline_self, "fixed_prompt", None) is None
            and not getattr(pipeline_self, "audio_only", False)
            and not _transformer_has_control(getattr(pipeline_self, "transformer", None)))


def _stage_fl2va_refmods(pipeline_self, state_json: str) -> bool:
    """Load the selection and stage it for the FL2VA hooks. Order matters:
    the packer assigns condition rows to references in list order, so the
    entries are built in exactly the order their latents get appended."""
    built = _build_refmod_sentinels(state_json)
    if built is None:
        return False
    image_sentinels, video_sentinels, audio_sentinels, total_tokens, retention = built
    visual, audio, entries = [], [], []
    for sentinel in image_sentinels:
        latent = sentinel.latent
        visual.append(latent)
        entries.append({"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})
        attached = getattr(sentinel, "audio_latent", None)
        if attached is not None:   # image refs have no audio rows: pair one alongside
            audio.append(attached)
            entries.append({"kind": "audio", "ref_audio_t": attached.shape[-1]})
    for sentinel in video_sentinels:
        latent = sentinel.latent
        attached = getattr(sentinel, "audio_latent", None)
        merged = attached is not None and not ATTACHED_AUDIO_AS_SEPARATE_REF
        if merged:
            audio.append(attached)
        visual.append(latent)
        entries.append({"kind": "video_audio" if merged else "video",
                        "latent_t": latent.shape[2], "latent_h": latent.shape[-2],
                        "latent_w": latent.shape[-1],
                        "ref_audio_t": attached.shape[-1] if merged else 0})
    for sentinel in audio_sentinels:
        latent = sentinel.latent
        audio.append(latent)
        entries.append({"kind": "audio", "ref_audio_t": latent.shape[-1]})
    labelled = []
    mode = str(FL2VA_PROMPT_LABELS or "none").lower()
    if mode in ("all", "audio"):
        for entry in entries:
            if entry["kind"] == "audio":
                labelled.append({"type": "audio"})
            elif mode == "all":
                # No frames to show (nothing is decoded on this path), so a
                # visual entry can only be a label -- which the encoder has no
                # way to render. Skip it rather than emit a broken block.
                continue
    _FL2VA.update(active=True, visual=visual, audio=audio, entries=entries, announced=False,
                  presentation=labelled)
    audio_rows = sum(e["ref_audio_t"] * 2 for e in entries if e["kind"] == "audio")
    _log(f"FL2VA: staging {len(image_sentinels)} image-kind + {len(video_sentinels)} video-kind + "
         f"{len(audio_sentinels)} audio-kind RefMod reference(s), retention={retention:.2f} "
         f"(~{total_tokens} tokens"
         + (f", {audio_rows} audio rows" if audio_rows else "") + ")")
    return True


def _inject_refmods(pipeline_self, kwargs: dict, state_json: str) -> dict:
    window_no = kwargs.get("window_no")
    if not INJECT_ON_EVERY_WINDOW and isinstance(window_no, int) and window_no > 1:
        # Wan2GP's own sliding-window loop (also used by "Continue Video")
        # only feeds its *native* references (image_refs, prefix_video, a
        # reference video's own frames, etc.) into window_no==1 -- every
        # later window continues from the *previous* window's own tail
        # frames instead, not the original reference again (see wgp.py's
        # own "if window_no == 1 and image_refs is not None..." gates).
        # custom_settings (carrying our RefMod selection) is passed to
        # every window's generate() call unconditionally, though -- and
        # since RefMod injection happens down here, inside generate()
        # itself, it has no visibility into which window this is unless we
        # check window_no explicitly. Without this check, a video-kind (or
        # image-kind) RefMod would get re-injected as a "reference" on
        # every single window, so a few frames of it visibly appear at
        # every window boundary in the output -- this is exactly that bug.
        return kwargs
    built = _build_refmod_sentinels(state_json)
    if built is None:
        return kwargs
    image_sentinels, video_sentinels, audio_sentinels, total_tokens, retention = built

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
        setattr(pipeline_self, "_h3_live_image_refs", len(existing))
        kwargs["input_ref_images"] = list(existing) + image_sentinels

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
            latent = sentinel.latent
            state["overflow_visual"].append((latent, {
                "kind": "video", "latent_t": latent.shape[2],
                "latent_h": latent.shape[-2], "latent_w": latent.shape[-1], "ref_audio_t": 0}))
        if remaining:
            _log(f"{len(remaining)} video-kind RefMod(s) beyond Wan2GP's "
                 f"{_NATIVE_CAP_VIDEO} native reference-video slots will be added directly "
                 f"(no <Video N> prompt label, and past what the model is built for)")
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
                _log(f"{len(remaining_audio)} audio-kind RefMod(s) beyond Wan2GP's "
                     f"{_NATIVE_CAP_AUDIO} native audio-reference slots will be added directly")
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

    _REFMODS_ACTIVE["on"] = True

    _log(f"injecting {len(image_sentinels)} image-kind + {len(video_sentinels)} video-kind + "
         f"{len(audio_sentinels)} audio-kind RefMod reference(s), "
         f"retention={retention:.2f} (~{total_tokens} tokens)")
    # Exactly what the pipeline is being handed, so a reference that "went in"
    # but had no effect can be told apart from one that never reached a slot.
    filled = [name for name in ("input_frames", "input_frames2", "input_frames3",
                                "audio_guide", "audio_guide2", "audio_guide3")
              if kwargs.get(name) is not None]
    _log(f"  slots filled: {', '.join(filled) or 'none'} | "
         f"video_prompt_type={kwargs.get('video_prompt_type')!r} "
         f"audio_prompt_type={kwargs.get('audio_prompt_type')!r} | "
         f"hidden from the cap check: "
         f"{sum(1 for s in image_sentinels + native_videos + native_audios if s.hide_ref)} "
         f"of {len(image_sentinels) + len(native_videos) + len(native_audios)} | "
         f"direct-injected: {len(state['overflow_visual'])} visual, "
         f"{len(state['overflow_audio'])} audio")
    return kwargs


# ═══════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════

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
    """Audio-ONLY mod extraction (no image/video sources given).

    Audio given ALONGSIDE images or video doesn't come here: it is encoded by
    the visual path and stored as that mod's soundtrack. The two latents are
    never stacked -- an audio latent [1, 32, 2, T] and a visual one
    [1, 24, T, H, W] have incompatible shapes -- they are stored side by side
    and injected as separate references. Audio is always full-fidelity
    ("encode"-only -- there's no spatial grid to pool, so "training" mode's
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
    # Since fork.16 an audio mod's length is its own "Soundtrack length"
    # setting; specs without one keep the old video-slider length.
    target_seconds = _audio_seconds(spec, target_px / FPS_ASSUMED_FOR_DURATION_ESTIMATE)

    status(f"H3 RefMod: loading audio reference (up to ~{target_seconds:.1f}s, mode=encode -- "
          f"audio mods are always full-fidelity)")
    try:
        import soundfile as sf
        full_seconds = sf.info(audio_path).frames / sf.info(audio_path).samplerate
        if full_seconds > target_seconds + 0.15:
            status(f"H3 RefMod: this file is ~{full_seconds:.1f}s long, but only the first "
                  f"~{target_seconds:.1f}s (set by 'Soundtrack length') will be "
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


def _run_attach_audio_job(pipeline_self, spec: dict, status) -> None:
    """Add, replace or remove the soundtrack on mods that already exist, so a
    visual mod doesn't have to be rebuilt from its sources to gain one.
    Encoding needs the audio VAE, hence a task rather than a plain UI action."""
    names = [n for n in (spec.get("names") or []) if n]
    audio_path = spec.get("audio_path") or None
    remove = bool(spec.get("remove"))
    seconds = float(spec.get("seconds") or 0.0)
    if not names:
        status("Attach audio: nothing selected.")
        return
    if not audio_path and not remove:
        status("Attach audio: pick an audio file, or tick Remove.")
        return

    latent = None
    if not remove:
        # Accepts a video file as well as an audio one: its soundtrack is
        # pulled out first, so an existing mod can be given the voice from the
        # very clip it was built from -- provided you still have that file,
        # since a mod records no source paths.
        waveform = storage.extract_audio_from_video(
            audio_path, max_seconds=max(MIN_AUDIO_SECONDS, seconds or DEFAULT_AUDIO_SECONDS))
        if waveform is None:
            status(f"Attach audio: no audio track could be read from "
                   f"{os.path.basename(str(audio_path))}.")
            return
        try:
            _require_audio_length(waveform, f"The audio in {os.path.basename(str(audio_path))}")
        except ExtractionBlocked as blocked:
            status(f"Attach audio refused: {blocked}")
            return
        latent = _encode_ref_audio(pipeline_self, waveform).to(torch.float16)
        if latent.dim() != 4 or latent.shape[1] != 32 or latent.shape[2] != 2:
            raise ValueError(f"Expected an audio-VAE latent [1,32,2,T], got {tuple(latent.shape)}.")
        status(f"Attach audio: encoded {latent.shape[-1]} audio latents "
               f"(~{latent.shape[-1] / AUDIO_LATENTS_PER_SECOND:.1f}s, "
               f"+{latent.shape[-1] * 2} tokens per mod)")

    done = skipped = failed = 0
    for index, name in enumerate(names, 1):
        try:
            mod = storage.load_refmod(name)
            if mod.kind == "audio":
                skipped += 1
                status(f"[{index}/{len(names)}] '{name}' is an audio mod -- nothing to attach to")
                continue
            if remove and mod.audio_latent is None:
                skipped += 1
                status(f"[{index}/{len(names)}] '{name}' has no soundtrack")
                continue
            mod.audio_latent = None if remove else latent.clone()
            mod.audio_t = 0 if remove else int(latent.shape[-1])
            mod.save(storage.mod_path(name))
            done += 1
            status(f"[{index}/{len(names)}] '{name}': "
                   + ("soundtrack removed" if remove else
                      f"soundtrack attached, mod is now {mod.token_count} tokens"))
        except Exception:
            failed += 1
            _log(f"could not update {name!r}:\n" + traceback.format_exc())
            status(f"[{index}/{len(names)}] '{name}': failed -- see the console")
    status(f"Attach audio finished: {done} updated, {skipped} skipped, {failed} failed.")


def _attach_encoder_frames(pipeline_self, mod, status) -> bool:
    """Decode a visual mod once and store the result in it as its encoder
    frames, so generations never have to decode it. A failure only costs the
    speed-up: the mod is still saved, and decodes at generation time instead."""
    if mod.kind == "audio":
        return False
    try:
        frames, times, fps, layout = decode_encoder_frames(pipeline_self, mod)
        mod.set_encoder_frames(frames, times, fps, layout)
        status(f"H3 RefMod: stored {len(times)} encoder frame(s) ({layout}, "
               f"{frames.shape[2]}x{frames.shape[1]}px)")
        return True
    except Exception:
        mod.clear_encoder_frames()
        _log("could not store encoder frames; the mod is saved without them and will be "
             "decoded at generation time instead:\n" + traceback.format_exc())
        status("H3 RefMod: could not store encoder frames (see the console) -- saving without them")
        return False


def _decode_audio_latent(pipeline_self, latent) -> torch.Tensor:
    """An H3 audio latent [1, 32, 2, T] -> waveform [2, samples] at 32 kHz,
    with the pipeline's own audio VAE (the one that decodes generated sound)."""
    audio_vae = getattr(pipeline_self, "audio_vae", None)
    if audio_vae is None:
        raise RuntimeError("this pipeline has no audio VAE to decode with")
    waveform = audio_vae.decode(latent.to(device=pipeline_self.device, dtype=torch.float32))[0]
    return waveform.float().clamp(-1.0, 1.0).cpu()


def _run_decompile_job(pipeline_self, spec: dict, status) -> None:
    """Reconstruct what a mod holds as ordinary files: one PNG per picture, an
    MP4 for a video, a WAV for its audio. These are VAE reconstructions of the
    stored latents, not the original sources -- whatever extraction discarded
    (resolution, the parts outside a trim, pooled detail in training mode,
    the individual pictures behind a merge) can't come back."""
    name = spec.get("name")
    if not name:
        status("Decompile: no mod chosen.")
        return
    mod = storage.load_refmod(name)
    out = storage.clear_decompile_dir(name)
    written = []
    soundtrack_path = None
    audio_latent = mod.latent if mod.kind == "audio" else mod.audio_latent
    if audio_latent is not None:
        try:
            waveform = _decode_audio_latent(pipeline_self, audio_latent)
            soundtrack_path = storage.save_wav(
                waveform, os.path.join(out, "audio.wav" if mod.kind == "audio" else "soundtrack.wav"))
            written.append(f"{os.path.basename(soundtrack_path)} "
                           f"({waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE:.1f}s)")
        except Exception:
            soundtrack_path = None
            _log(f"decompile: could not decode the audio of {name!r}:\n" + traceback.format_exc())
            status(f"Decompile: the audio of '{name}' could not be decoded -- see the console")
    if mod.kind != "audio":
        if mod.still_stack() or mod.kind == "image" or mod.latent.shape[2] <= 1:
            # Pictures were encoded one by one, so they're decoded one by one.
            count = mod.latent.shape[2]
            for j in range(count):
                picture = _cthw_to_thwc01(_decode_cthw(pipeline_self, mod.latent[:, :, j:j + 1]))[0]
                storage.save_png(picture, os.path.join(out, f"picture_{j + 1:02d}.png"))
                if j == 0:
                    size = f"{picture.shape[1]}x{picture.shape[0]}"
            written.append(f"{count} picture(s), {size}px")
        else:
            frames = _cthw_to_thwc01(_decode_cthw(pipeline_self, mod.latent))
            storage.save_mp4(frames, os.path.join(out, "video.mp4"),
                                    FPS_ASSUMED_FOR_DURATION_ESTIMATE, soundtrack_path)
            written.append(f"video.mp4 ({frames.shape[0]} frames, ~"
                           f"{frames.shape[0] / FPS_ASSUMED_FOR_DURATION_ESTIMATE:.1f}s, "
                           f"{frames.shape[2]}x{frames.shape[1]}px"
                           + (", with its soundtrack" if soundtrack_path else "") + ")")
            frames = None
    status(f"Decompile finished: '{name}' -> " + ", ".join(written) + f" in {out}")


def _run_store_frames_job(pipeline_self, spec: dict, status) -> None:
    """Add encoder frames to mods that were saved without them (or whose
    stored frames no longer match how the mod is used). Decoding needs the
    video VAE, hence a task rather than a plain UI action. Only the frames
    are added; the latent, soundtrack and metadata are rewritten unchanged."""
    names = [n for n in (spec.get("names") or []) if n]
    if spec.get("all_missing"):
        names = storage.list_refmods_missing_frames()
    if not names:
        status("Encoder frames: nothing to do -- every mod already has them.")
        return
    done = skipped = failed = 0
    for index, name in enumerate(names, 1):
        try:
            mod = storage.load_refmod(name)
            if mod.kind == "audio":
                skipped += 1
                status(f"[{index}/{len(names)}] '{name}' is an audio mod -- nothing to show the encoder")
                continue
            if mod.encoder_frames_usable(encframes.ENC_FPS):
                skipped += 1
                status(f"[{index}/{len(names)}] '{name}' already has encoder frames")
                continue
            if _attach_encoder_frames(pipeline_self, mod, lambda msg: None):
                mod.save(storage.mod_path(name))
                done += 1
                status(f"[{index}/{len(names)}] '{name}': stored {len(mod.enc_times)} encoder frame(s)")
            else:
                failed += 1
                status(f"[{index}/{len(names)}] '{name}': failed -- see the console")
        except Exception:
            failed += 1
            _log(f"could not store encoder frames for {name!r}:\n" + traceback.format_exc())
            status(f"[{index}/{len(names)}] '{name}': failed -- see the console")
    status(f"Encoder frames finished: {done} updated, {skipped} skipped, {failed} failed.")


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

    if spec.get("op") == "attach_audio":
        _run_attach_audio_job(pipeline_self, spec, status)
        return
    if spec.get("op") == "store_frames":
        _run_store_frames_job(pipeline_self, spec, status)
        return
    if spec.get("op") == "decompile":
        _run_decompile_job(pipeline_self, spec, status)
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
    ref_resolution = int(spec.get("ref_resolution", 1024))
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

    # The soundtrack is chosen and read FIRST: a clip too short to use is
    # refused before any slow visual encoding, not after it.
    # "Use the clip's own audio": the first video source with a soundtrack;
    # a separately chosen audio file always wins if both are given.
    from_clip = False
    if not audio_path and spec.get("use_clip_audio") and video_paths:
        for candidate in video_paths:
            if storage.video_has_audio(candidate):
                audio_path = candidate
                from_clip = True
                status(f"H3 RefMod: using the clip's own audio from "
                       f"{os.path.basename(candidate)}")
                break
        else:
            status("H3 RefMod: 'use the clip's own audio' was set, but no video source has "
                   "an audio track -- extracting without a soundtrack")
    soundtrack_waveform = None
    if audio_path and spec.get("audio_seconds") is not None:
        # "Soundtrack length" (fork.16+): one setting for every soundtrack,
        # independent of the visual duration.
        target_seconds = _audio_seconds(spec, DEFAULT_AUDIO_SECONDS)
        clip_trim = (spec.get("video_trims") or {}).get(os.path.basename(str(audio_path)))
        soundtrack_waveform = storage.extract_audio_from_video(
            audio_path, max_seconds=target_seconds,
            start_seconds=float(clip_trim[0]) if clip_trim else 0.0)
        if soundtrack_waveform is None:
            raise ExtractionBlocked(f"No audio could be read from "
                                    f"{os.path.basename(str(audio_path))}.")
        where = (f"The audio from {os.path.basename(str(audio_path))}"
                 + (f" (from {float(clip_trim[0]):g}s, the trim start)" if clip_trim else ""))
        if from_clip:
            # A clip's own audio is the one exception to the 2s minimum: the
            # clip can still make a perfectly good visual mod, so a short
            # soundtrack is kept, with a warning, rather than refusing it all.
            got = soundtrack_waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE
            if got < MIN_AUDIO_SECONDS - 0.05:
                status(f"H3 RefMod: ⚠️ {where} is only ~{got:.1f}s long -- kept anyway, but H3 "
                       f"documents {MIN_AUDIO_SECONDS:g}s as the shortest usable audio reference, "
                       f"so this voice may not be picked up. A longer clip (or an earlier trim "
                       f"start) gives the model more to work with.")
        else:
            got = _require_audio_length(soundtrack_waveform, where)
        status(f"H3 RefMod: soundtrack ~{got:.1f}s (Soundtrack length {target_seconds:g}s)")

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
    # "Turn reference videos into a picture stack": each video becomes still
    # pictures (the sharpest frame from every 1/N-second slot) instead of a
    # clip, so a mod keeps a clip's identity at a fraction of its tokens.
    video_as_pictures = bool(spec.get("video_as_pictures", False))
    pictures_per_second = max(0.05, float(spec.get("pictures_per_second", 1.0) or 1.0))
    stills_spans = {}   # video file name -> seconds its pictures cover (sizes its soundtrack)
    for i, p in enumerate(image_paths):
        from PIL import Image
        with Image.open(p) as img:
            img = img.copy()
        if remove_background_images_ref:
            status(f"H3 RefMod: removing background from image {i + 1}/{len(image_paths)}")
            img = storage.remove_background_from_image(img, session=rembg_session)
        sources.append((storage.pil_to_cthw(img), False))
    for vp in video_paths:
        # Per-video trim chosen in the extractor's preview, keyed by file name.
        trim = (spec.get("video_trims") or {}).get(os.path.basename(str(vp)))
        start_s = float(trim[0]) if trim else 0.0
        dur_s = max(0.1, float(trim[1]) - float(trim[0])) if trim else None
        if trim:
            status(f"H3 RefMod: using {start_s:g}-{float(trim[1]):g}s of "
                   f"{os.path.basename(str(vp))}")
        if video_as_pictures:
            # Same span a clip would use: the trim, capped by "Reference
            # duration to use".
            limit_s = (((latent_frames - 1) * 4 + 1) if latent_frames > 1 else 1) \
                / float(FPS_ASSUMED_FOR_DURATION_ESTIMATE)
            span_s = min(dur_s, limit_s) if dur_s else limit_s
            stills, covered = storage.load_video_stills(vp, pictures_per_second, start_s, span_s,
                                                        max_short_edge=ref_resolution)
            stills_spans[os.path.basename(str(vp))] = covered
            status(f"H3 RefMod: {os.path.basename(str(vp))} -> {len(stills)} picture(s) "
                   f"({pictures_per_second:g} per second over {covered:.1f}s, sharpest frame "
                   f"of each slot)")
            sources.extend((still, False) for still in stills)
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

    frames = []
    n_img = n_vid = 0
    source_shapes = []
    # "Merge into one picture" (training mode, several sources): instead of
    # stacking each source's own pooled grid, keep each pooled grid and full
    # encode, then build ONE shared grid from all of them after the loop
    # (core.merge_latents) -- the consensus of the collection, at one
    # reference's token cost.
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

        if mode == "encode":
            pooled = z.to(torch.float16)
        else:
            pool_t = min(latent_frames, z.shape[2]) if is_video else 1
            pooled = core.pool_latent(z, pool_t, gh, gw).to(torch.float16)
            if merge_refs is not None:
                # Refined jointly after the loop, against every full encode.
                merge_refs.append((pooled.float().cpu(), z.float().cpu()))
                z = None
                if is_video:
                    n_vid += 1
                else:
                    n_img += 1
                continue
            if identity > 0:
                status(f"H3 RefMod: refining identity for {label} ({identity} steps)")
                pooled = core.optimize_latent(pooled, z.float(), steps=identity, progress_every=100)
        frames.append(pooled)
        if is_video:
            n_vid += 1
        else:
            n_img += 1

    if merge_refs is not None:
        status(f"H3 RefMod: merging {len(merge_refs)} references into one {gh}x{gw} grid"
               + (f" ({identity} joint refinement steps)" if identity > 0
                  else " (plain average -- identity refinement steps is 0)"))
        latent = core.merge_latents([p for p, _ in merge_refs], [f for _, f in merge_refs],
                                    gh, gw, steps=identity, progress_every=100).to(torch.float16)
        merged_n = len(merge_refs)
        merge_refs = None
    else:
        latent = torch.cat(frames, dim=2)
        merged_n = 0
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
    # Optional soundtrack for a visual mod. The two latents are structurally
    # different ([1,32,2,T] vs [1,24,T,H,W]) so they can't be stacked, but they
    # can be stored side by side and injected as one reference: H3 tags a
    # reference video carrying audio "video_audio" and gives it audio rows.
    audio_latent = None
    if audio_path and soundtrack_waveform is None:
        # Specs from before fork.16 (no "Soundtrack length"): the old sizing.
        if n_vid > 0 and not merged_n:
            # A clip: the soundtrack matches the visual duration it kept.
            target_seconds = (((total_t - 1) * 4 + 1 if total_t > 1 else 1)
                              / FPS_ASSUMED_FOR_DURATION_ESTIMATE)
        else:
            # Pictures (single, stacked or merged) have no duration of their
            # own -- sizing the voice from their latent frames gave a single
            # picture half a second of audio. Use "Reference duration to use",
            # which is what that slider promises for audio.
            target_seconds = (((latent_frames - 1) * 4 + 1 if latent_frames > 1 else 1)
                              / FPS_ASSUMED_FOR_DURATION_ESTIMATE)
        if os.path.basename(str(audio_path)) in stills_spans:
            # Pictures taken from a video: the soundtrack covers the span of
            # the video they were taken from.
            target_seconds = stills_spans[os.path.basename(str(audio_path))]
        # H3 documents 2 seconds as the shortest usable audio reference; a
        # shorter voice sample gives it too little to go on.
        target_seconds = max(MIN_SOUNDTRACK_SECONDS, target_seconds)
        clip_trim = (spec.get("video_trims") or {}).get(os.path.basename(str(audio_path)))
        soundtrack_waveform = storage.extract_audio_from_video(
            audio_path, max_seconds=target_seconds,
            start_seconds=float(clip_trim[0]) if clip_trim else 0.0)
        if soundtrack_waveform is None:
            raise ValueError(f"No audio could be read from {os.path.basename(audio_path)}.")
    if audio_path and soundtrack_waveform is not None:
        status(f"H3 RefMod: encoding the attached soundtrack "
               f"(~{soundtrack_waveform.shape[-1] / storage.AUDIO_SAMPLE_RATE:.1f}s)")
        audio_latent = _encode_ref_audio(pipeline_self, soundtrack_waveform).to(torch.float16)
        if audio_latent.dim() != 4 or audio_latent.shape[1] != 32 or audio_latent.shape[2] != 2:
            raise ValueError(f"Expected an audio-VAE latent [1,32,2,T], got {tuple(audio_latent.shape)}.")
        status(f"H3 RefMod: soundtrack attached ({audio_latent.shape[-1]} audio latents, "
               f"~{audio_latent.shape[-1] / AUDIO_LATENTS_PER_SECOND:.1f}s, "
               f"+{audio_latent.shape[-1] * 2} tokens)")

    kind = "video" if n_vid > 0 else "image"  # NOT total_t > 1: several still images
                                              # stacked together (n_vid==0) still form
                                              # multiple independent *image* references,
                                              # not a multi-frame video, even though the
                                              # underlying latent has more than one frame.
    px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
    mod = core.H3RefMod(
        name=storage._split_folder(name)[1], kind=kind, latent=latent, latent_h=latent.shape[3], latent_w=latent.shape[4],
        latent_t=total_t, mode=mode,
        source=("merge" if merged_n else
                "stack" if len(frames) > 1 else ("video" if n_vid else "image")),
        source_shape=" +".join(source_shapes),
        pool=(f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)" if mode == "encode"
              else f"{total_t}x{gh}x{gw}"),
        optimize_steps=identity if mode == "training" else 0,
        tags=[f"{n_img} img, {n_vid} vid"] + ([f"merged {merged_n} refs"] if merged_n else [])
             + ([f"x{multiplier} repeat"] if multiplier > 1 else [])
             + ([f"{pictures_per_second:g}/s from video"] if stills_spans else [])
             + (["background removed"] if remove_background_images_ref else [])
             + (["with soundtrack"] if audio_path else []),
        description=description, concept_type=concept_type,
        audio_latent=audio_latent,
        audio_t=0 if audio_latent is None else int(audio_latent.shape[-1]),
    )

    if save:
        if spec.get("store_encoder_frames", True):
            _attach_encoder_frames(pipeline_self, mod, status)
        else:
            status("H3 RefMod: not storing encoder frames (option off) -- this mod will be "
                   "decoded once per session instead; the Library can add them later")
        path = mod.save(storage.mod_path(name))
        status(f"H3 RefMod '{name}' saved: {mod.token_count} tokens, {kind}/{mode}"
               + (" + soundtrack" if audio_latent is not None else "")
               + (f" + {len(mod.enc_times)} encoder frame(s)" if mod.has_encoder_frames() else "")
               + f" -> {path}")
    else:
        status(f"H3 RefMod '{name}' extracted ({mod.token_count} tokens) but not saved (save=false)")
