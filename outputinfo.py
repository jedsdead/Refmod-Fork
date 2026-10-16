"""
outputinfo.py -- RefMods in a generation's output info, and back again
(fork addition).

  * A readable "RefMods" / "Flux RefMods" line in the file's metadata
    (``extra_info``, which Wan2GP's file info panel shows), written by the
    ``before_metadata_save`` plugin hook. The full selection is in the
    metadata anyway, in the hidden custom setting that carried it.
  * Loading a file's settings (the settings file button, or "use settings"
    on a gallery item) puts that selection back into the panels: Wan2GP's
    ``set_model_settings`` is wrapped, and when one of those two loaders
    calls it, the file's selection becomes the panel's.
"""

from __future__ import annotations

import json
import sys
import traceback

from . import flux
from .patches import (SETTING_GENERATE, STASH_KEY, is_minimax_h3_refmod_capable, read_refmod_payload)

RESTORE_H3 = "_h3refmod_restore"       # session-state keys the panels read once
RESTORE_FLUX = "_fluxrefmod_restore"
LOAD_CALLERS = ("load_settings_from_file", "use_video_settings")

_WINDOW_WORDS = {"full": "video mods whole on later windows", "first": "video mods on window 1 only"}


def _log(msg):
    print(f"[Refmods] {msg}")


def _strength(row):
    try:
        return f"{float(row.get('strength', 1.0)):.2f}"
    except (TypeError, ValueError):
        return "?"


def h3_summary(payload: dict, labels: str = "") -> str:
    parts = []
    for row in payload.get("rows") or []:
        if not row.get("mod"):
            continue
        extra = []
        if row.get("send_as") == "sheet":
            extra.append("as a contact sheet")
        elif row.get("as_video"):
            extra.append("as one video")
        if row.get("use_audio") is False:
            extra.append("no audio")
        parts.append(f"{row['mod']} x{_strength(row)}" + (f" ({', '.join(extra)})" if extra else ""))
    if not parts:
        return ""
    lines = ["; ".join(parts)]
    if labels:
        lines.append("Prompt labels: " + labels)
    options = []
    if payload.get("fit_to_output"):
        options.append("fit to output")
    if payload.get("phase1_mods") == "full":
        options.append("phase 1: full size")
    if payload.get("phase2_mods") in ("tile", "off"):
        options.append("phase 2: " + ("fit to tile" if payload["phase2_mods"] == "tile" else "off"))
    if payload.get("shrink_method") == "fast":
        options.append("shrink: fast")
    if payload.get("window_mode") in _WINDOW_WORDS:
        options.append(_WINDOW_WORDS[payload["window_mode"]])
    if options:
        lines.append(" · ".join(options))
    return "\n".join(lines)


def flux_summary(payload: dict, labels: str = "") -> str:
    parts = [f"{row['mod']} x{_strength(row)}" for row in payload.get("rows") or [] if row.get("mod")]
    if not parts:
        return ""
    return "; ".join(parts) + (f"\nPrompt labels: {labels}" if labels else "")


def before_metadata_save(configs, **_):
    """Wan2GP's ``before_metadata_save`` hook: add the readable summary."""
    try:
        if not isinstance(configs, dict):
            return configs
        # Settings loaded from an earlier output can carry its info along:
        # never let an old RefMods line describe this generation.
        info = {k: v for k, v in dict(configs.get("extra_info") or {}).items()
                if k not in ("RefMods", "Flux RefMods")}
        if "extra_info" in configs:
            configs["extra_info"] = info or None
            if not info:
                configs.pop("extra_info")
        custom = configs.get("custom_settings")
        if not isinstance(custom, dict):
            return configs
        state_json, job = read_refmod_payload(custom)
        if state_json and not job:
            from . import patches
            text = h3_summary(json.loads(state_json), patches._LABEL_LOG.get("text", ""))
            if text:
                info["RefMods"] = text
        selection, job = flux.read_payload(custom)
        if selection and not job:
            text = flux_summary(selection, flux.LAST_LABELS.get("text", ""))
            if text:
                info["Flux RefMods"] = text
        if info:
            configs["extra_info"] = info
    except Exception:
        _log("could not add RefMods to the output's info:\n" + traceback.format_exc())
    return configs


def adopt_loaded_selection(state, model_type, settings, get_base_model_type_fn=None) -> None:
    """A file's settings are being loaded: make its selection the panel's."""
    custom = settings.get("custom_settings") if isinstance(settings, dict) else None
    custom = custom if isinstance(custom, dict) else {}
    if is_minimax_h3_refmod_capable(model_type, get_base_model_type_fn):
        state_json, job = read_refmod_payload(custom)
        if job:
            return
        if state_json:
            state[STASH_KEY] = state_json
        else:
            state.pop(STASH_KEY, None)
        state[RESTORE_H3] = state_json or ""
    elif flux.is_flux_refmod_capable(model_type, get_base_model_type_fn):
        selection, job = flux.read_payload(custom)
        if job:
            return
        text = json.dumps(selection) if selection else ""
        if text:
            state[flux.FLUX_STASH_KEY] = text
        else:
            state.pop(flux.FLUX_STASH_KEY, None)
        state[RESTORE_FLUX] = text


def install_set_model_settings_patch(orig, set_global_fn, get_base_model_type_fn=None):
    if getattr(orig, "_refmod_wrapped", False):
        return None

    def patched_set_model_settings(state, model_type, settings):
        try:
            if sys._getframe(1).f_code.co_name in LOAD_CALLERS and isinstance(state, dict):
                adopt_loaded_selection(state, model_type, settings, get_base_model_type_fn)
        except Exception:
            _log("could not read RefMods from the loaded settings:\n" + traceback.format_exc())
        return orig(state, model_type, settings)

    patched_set_model_settings._refmod_wrapped = True
    try:
        set_global_fn("set_model_settings", patched_set_model_settings)
    except Exception as e:
        return f"could not hook set_model_settings ({e!r}); loading a file won't refill the RefMods panels"
    return None
