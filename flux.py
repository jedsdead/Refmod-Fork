"""
flux.py -- RefMods for WanGP's Flux image models (fork addition).

Flux 2 (Dev, Klein 4B / 9B, pi-Flux2) and Flux 3 Klein all read reference
images through the same Flux 2 VAE: a picture becomes a [128, H/16, W/16]
latent (patchified and batch-normalised), one token per latent cell, placed
after the prompt as reference image N. A Flux RefMod is that latent, encoded
once and saved, so it goes straight into the reference list at generation
time with no VAE pass.

Files live in ``loras/refmods_plugin/flux2/`` and use the layout of the
ComfyUI-Flux2Klein9Mod pack (a ``samples`` tensor [B, 128, h, w] and a JSON
block under ``klein9_refmod_meta``), so mods from that pack can be dropped
in, and ours load there. A mod whose pictures differ in size keeps the first
one in ``samples`` and the others in ``samples_1``, ``samples_2`` ...

H3 mods and Flux mods are different latent spaces; neither works in the
other's models. Flux 1 models (Dev, Schnell, Kontext, Chroma ...) use a
different VAE and aren't supported.

How the mods reach the model:
  * Flux 2 (models/flux/flux_main.py): placeholder images are appended to
    ``input_ref_images``; the module's ``encode_image_refs`` is wrapped so it
    encodes the real images as usual and swaps each placeholder for its mod's
    tokens, with the reference position the next image would have had.
  * Flux 3 (models/flux3/pipeline.py): placeholders are appended the same
    way; the VAE's encode returns each placeholder's mod latent instead of
    encoding it, so positions and ids come out exactly as for a real image.
"""

from __future__ import annotations

import json
import math
import os
import traceback
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from . import storage

FLUX_SETTING = "flux_refmod_state"          # custom_settings id carrying the selection / a job
FLUX_STASH_KEY = "_flux_refmod_selection"   # session-state key for the inline panel
FLUX_EXTRACT_KEY = "__flux_refmod_extract__"
META_KEY = "klein9_refmod_meta"             # ComfyUI-Flux2Klein9Mod's metadata key
LATENT_CHANNELS = 128
MAX_REFERENCE_PIXELS = 1024 * 1024          # what Flux 2 caps each live reference at when there are several
TOKEN_BUDGETS = (4096, 2048, 1024, 512)
DEFAULT_BUDGET = 4096
CONCEPT_TYPES = ("identity", "character", "style", "clothing", "background", "generic")
MAX_ROWS = 8

FLUX2_TYPES = ("flux2_dev", "pi_flux2", "flux2_klein_4b", "flux2_klein_9b")
FLUX3_TYPES = ("flux3_klein", "flux3_klein_base")
_HIDDEN_FLAG = "§"   # never in a video_prompt_type, so Wan2GP keeps the box hidden
_PATCH_MARKER = "_flux_refmod_patched"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log(msg: str) -> None:
    print(f"[Refmods] Flux: {msg}")


# ── Which models ──────────────────────────────────────────────────────────

def _base_type(model_type, get_base_model_type_fn=None) -> str:
    model_type = str(model_type or "")
    if callable(get_base_model_type_fn):
        try:
            return str(get_base_model_type_fn(model_type) or model_type)
        except Exception:
            pass
    return model_type


def is_flux_refmod_capable(model_type, get_base_model_type_fn=None) -> bool:
    base = _base_type(model_type, get_base_model_type_fn)
    return base in FLUX2_TYPES or base in FLUX3_TYPES or base.startswith(("flux2_", "flux3_"))


def is_flux3(model_type, get_base_model_type_fn=None) -> bool:
    return _base_type(model_type, get_base_model_type_fn).startswith("flux3")


# ── Storage ───────────────────────────────────────────────────────────────

def mods_dir() -> str:
    d = os.path.join(os.path.dirname(storage.refmods_dir()), "flux2")
    os.makedirs(d, exist_ok=True)
    return d


def _clean_relpath(name: str) -> str:
    parts = [p for p in str(name or "").replace("\\", "/").split("/") if p not in ("", ".", "..")]
    parts = ["".join(c for c in p if c.isalnum() or c in "_- .").strip().replace(" ", "_") for p in parts]
    parts = [p for p in parts if p]
    return "/".join(parts) or "flux_refmod"


def mod_path(name: str) -> str:
    return os.path.join(mods_dir(), *_clean_relpath(name).split("/")) + ".safetensors"


def list_mods() -> List[str]:
    root, names = mods_dir(), []
    for folder, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".safetensors"):
                rel = os.path.relpath(os.path.join(folder, f[:-len(".safetensors")]), root)
                names.append(rel.replace(os.sep, "/"))
    return sorted(names, key=str.lower)


def read_meta(name: str) -> Optional[Dict]:
    path = mod_path(name)
    if not os.path.isfile(path):
        return None
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            raw = f.metadata() or {}
            meta = json.loads(raw.get(META_KEY) or "{}")
            keys = list(f.keys())
            shapes = [list(f.get_slice(k).get_shape()) for k in _sample_keys(keys)]
    except Exception:
        return None
    pictures = sum(s[0] if len(s) == 4 else 1 for s in shapes)
    tokens = sum((s[0] if len(s) == 4 else 1) * s[-2] * s[-1] for s in shapes)
    meta.update({"pictures": pictures, "tokens": tokens,
                 "size_mb": round(os.path.getsize(path) / 1e6, 2),
                 "usable": all(len(s) >= 3 and s[-3] == LATENT_CHANNELS for s in shapes) and bool(shapes)})
    return meta


def _sample_keys(keys) -> List[str]:
    extra = sorted((k for k in keys if k.startswith("samples_") and k[8:].isdigit()), key=lambda k: int(k[8:]))
    return (["samples"] if "samples" in keys else []) + extra


def load_latents(name: str) -> List[torch.Tensor]:
    """Every picture of a mod as its own [1, 128, h, w] latent, in order."""
    from safetensors.torch import load_file
    tensors = load_file(mod_path(name), device="cpu")
    out = []
    for key in _sample_keys(list(tensors)):
        t = tensors[key].clone()   # not tied to the file (it can be moved or deleted)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.dim() != 4 or t.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"'{name}' is not a Flux 2 latent ({key} has shape {list(t.shape)}; "
                             f"expected [N, {LATENT_CHANNELS}, h, w])")
        out += [t[i:i + 1] for i in range(t.shape[0])]
    if not out:
        raise ValueError(f"'{name}' has no 'samples' tensor")
    return out


def save_mod(name: str, latents: List[torch.Tensor], meta: Dict) -> str:
    from safetensors.torch import save_file
    path = mod_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    latents = [l.detach().to("cpu", torch.bfloat16).contiguous() for l in latents]
    if all(l.shape == latents[0].shape for l in latents):
        tensors = {"samples": torch.cat(latents, dim=0)}
    else:
        tensors = {"samples": latents[0]}
        tensors.update({f"samples_{i}": l for i, l in enumerate(latents[1:], start=1)})
    tokens = sum(l.shape[-2] * l.shape[-1] for l in latents)
    header = {"name": _clean_relpath(name).split("/")[-1], "mode": "encode", "channels": LATENT_CHANNELS,
              "shape": list(latents[0].shape), "token_count": tokens, "pictures": len(latents),
              "generator": "WanGP MiniMaxH3Mod (Refmod-Fork)", **meta}
    save_file(tensors, path, metadata={META_KEY: json.dumps(header)})
    return path


def delete_mod(name: str) -> bool:
    path = mod_path(name)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def list_folders() -> List[str]:
    root, out = mods_dir(), []
    for folder, dirs, _ in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out += [os.path.relpath(os.path.join(folder, d), root).replace(os.sep, "/") for d in dirs]
    return sorted(out, key=str.lower)


def create_folder(folder: str) -> str:
    rel = _clean_relpath(folder)
    os.makedirs(os.path.join(mods_dir(), *rel.split("/")), exist_ok=True)
    return rel


def delete_folder(folder: str) -> str:
    rel = _clean_relpath(folder)
    path = os.path.join(mods_dir(), *rel.split("/"))
    if not os.path.isdir(path):
        raise ValueError(f"No Flux folder named '{rel}'.")
    for dirpath, _, files in os.walk(path):
        if files:
            raise ValueError(f"Flux/{rel} isn't empty -- move or delete what's in it first.")
    for dirpath, _, _ in sorted(os.walk(path), key=lambda w: -len(w[0])):
        os.rmdir(dirpath)
    return rel


def _rewrite(old: str, new: str, meta_updates: Dict) -> None:
    """Save a mod under ``new`` with updated metadata (the latents are copied
    untouched), then remove ``old`` if the name changed."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    src = mod_path(old)
    with safe_open(src, framework="pt", device="cpu") as f:
        raw = dict(f.metadata() or {})
    meta = json.loads(raw.get(META_KEY) or "{}")
    meta.update(meta_updates)
    raw[META_KEY] = json.dumps(meta)
    # Copied out of the memory-mapped file first: writing over a file that is
    # still mapped fails on Windows (and can corrupt it elsewhere).
    loaded = load_file(src, device="cpu")
    tensors = {k: v.clone() for k, v in loaded.items()}
    del loaded
    import gc
    gc.collect()
    dst = mod_path(new)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    save_file(tensors, tmp, metadata=raw)
    os.replace(tmp, dst)
    if os.path.normcase(os.path.abspath(dst)) != os.path.normcase(os.path.abspath(src)):
        os.remove(src)


def rename_and_update(old: str, new_name: Optional[str] = None, description: Optional[str] = None) -> str:
    """Rename (a bare name keeps the folder; a path moves it; a leading "/"
    goes to the top of the Flux folder) and/or change the description.
    Never overwrites another mod. Returns the final name."""
    old = _clean_relpath(old)
    target = old
    if new_name and str(new_name).strip():
        raw = str(new_name).strip().replace("\\", "/")
        if raw.startswith("/"):
            target = _clean_relpath(raw)
        elif "/" in raw:
            target = _clean_relpath(raw)
        else:
            folder = old.rpartition("/")[0]
            target = _clean_relpath(f"{folder}/{raw}" if folder else raw)
    if target != old and os.path.isfile(mod_path(target)):
        raise ValueError(f"A Flux mod named '{target}' already exists.")
    updates = {"name": target.split("/")[-1]}
    if description is not None:
        updates["description"] = description
    _rewrite(old, target, updates)
    return target


def move_mods(names, folder: str):
    """Move Flux mods into ``folder`` ("" = the top of the Flux folder).
    Returns (moved, skipped) like storage.move_mods."""
    dest = _clean_relpath(folder) if str(folder or "").strip("/ ") else ""
    moved, skipped = [], []
    for name in [n for n in names or [] if n]:
        leaf = _clean_relpath(name).split("/")[-1]
        target = f"{dest}/{leaf}" if dest else leaf
        if target == _clean_relpath(name):
            skipped.append((name, "already there"))
        elif os.path.isfile(mod_path(target)):
            skipped.append((name, f"'{target}' already exists"))
        else:
            try:
                os.makedirs(os.path.dirname(mod_path(target)), exist_ok=True)
                os.replace(mod_path(name), mod_path(target))
                moved.append((name, target))
            except Exception as e:
                skipped.append((name, str(e)))
    return moved, skipped


def free_name(name: str) -> str:
    """``name``, or ``name_2``, ``name_3`` ... if a Flux mod already has it."""
    base = _clean_relpath(name)
    candidate, i = base, 2
    while os.path.isfile(mod_path(candidate)):
        candidate, i = f"{base}_{i}", i + 1
    return candidate


# ── Sorting mod files into their model's folder ───────────────────────────

def file_family(path: str) -> Optional[str]:
    """"h3", "flux" or None for a .safetensors file, from its metadata (or,
    failing that, its tensor shapes)."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if "refmod_meta" in meta:
                return "h3"
            if META_KEY in meta:
                return "flux"
            keys = list(f.keys())
            if "samples" in keys:
                shape = f.get_slice("samples").get_shape()
                if len(shape) == 4 and shape[1] == LATENT_CHANNELS:
                    return "flux"
            if "latent" in keys:
                return "h3"
    except Exception:
        return None
    return "h3" if os.path.isfile(path[:-len(".safetensors")] + ".json") else None


def sort_mod_files() -> List[str]:
    """Put every mod file in its model's folder: H3 mods in
    loras/refmods_plugin/minimax_h3/, Flux mods in .../flux2/, keeping
    subfolders. Looks at the top of loras/refmods_plugin/ (and any other
    folder there) and at both model folders. Never overwrites; returns a
    line per move."""
    h3_root = storage.refmods_dir()
    flux_root = mods_dir()
    top = os.path.dirname(h3_root)
    keep_out = {os.path.normcase(os.path.abspath(p)) for p in (h3_root, flux_root, os.path.join(top, "decompiled"))}
    moves = []

    def plan(folder_root, rel_base, only=None):
        for dirpath, dirs, files in os.walk(folder_root):
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) not in keep_out]
            for f in files:
                if not f.lower().endswith(".safetensors"):
                    continue
                path = os.path.join(dirpath, f)
                family = file_family(path)
                if family is None or family == only:
                    continue
                rel = os.path.relpath(path, rel_base)
                moves.append((path, os.path.join(h3_root if family == "h3" else flux_root, rel)))

    plan(top, top)                         # strays at the top level or in other folders
    plan(h3_root, h3_root, only="h3")      # Flux files in the H3 folder
    plan(flux_root, flux_root, only="flux")  # H3 files in the Flux folder
    done, emptied = [], set()
    for src, dst in moves:
        if os.path.exists(dst):
            _log(f"left {src} where it is: {dst} already exists")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        side = src[:-len(".safetensors")] + ".json"
        if os.path.isfile(side):
            os.replace(side, dst[:-len(".safetensors")] + ".json")
        done.append(f"{os.path.relpath(src, top)} -> {os.path.relpath(dst, top)}")
        emptied.add(os.path.dirname(src))
    # Remove only the stray folders the moves above emptied (never the model
    # folders, their subfolders, or a folder that was already empty).
    top_abs = os.path.normcase(os.path.abspath(top))
    for folder in sorted(emptied, key=len, reverse=True):
        path = os.path.abspath(folder)
        while True:
            norm = os.path.normcase(path)
            if (norm == top_abs or not norm.startswith(top_abs + os.sep) or norm in keep_out
                    or any(norm.startswith(k + os.sep) for k in keep_out)):
                break
            try:
                os.rmdir(path)   # fails (and stops) if anything is left in it
            except OSError:
                break
            path = os.path.dirname(path)
    for line in done:
        _log("sorted " + line)
    return done


# ── Strength ──────────────────────────────────────────────────────────────

def _blur(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    h, w = z.shape[-2:]
    down = F.adaptive_avg_pool2d(z.float(), (max(1, h // factor), max(1, w // factor)))
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def weighted(latent: torch.Tensor, strength: float) -> torch.Tensor:
    """Below 1 the mod fades toward a blurred copy of itself (it keeps the
    layout and colours and loses detail); above 1 the same blend carries on
    past the mod, sharpening what sets it apart from the blur."""
    if abs(strength - 1.0) < 1e-6:
        return latent
    blurred = _blur(latent)
    return (strength * latent.float() + (1.0 - strength) * blurred).to(latent.dtype)


# ── Selection payload ─────────────────────────────────────────────────────

def read_payload(custom_settings) -> Tuple[Optional[dict], Optional[dict]]:
    """(selection, extract_job) from a task's custom_settings."""
    if not isinstance(custom_settings, dict):
        return None, None
    raw = custom_settings.get(FLUX_SETTING)
    if not raw:
        return None, None
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return None, None
    if isinstance(data, dict) and isinstance(data.get(FLUX_EXTRACT_KEY), dict):
        return None, data[FLUX_EXTRACT_KEY]
    return (data if isinstance(data, dict) and data.get("rows") else None), None


def selected_latents(selection: dict) -> List[Tuple[str, torch.Tensor]]:
    """(mod name, weighted latent) for every picture of every selected mod."""
    out = []
    for row in selection.get("rows") or []:
        name, strength = row.get("mod"), float(row.get("strength", 1.0) or 0.0)
        if not name or strength <= 0:
            continue
        try:
            pictures = load_latents(name)
        except Exception as e:
            _log(f"skipping '{name}': {e}")
            continue
        out += [(name, weighted(p, strength)) for p in pictures]
    return out


def _placeholder() -> Image.Image:
    return Image.new("RGB", (64, 64), (127, 127, 127))


LAST_LABELS = {"text": ""}   # this generation's prompt labels, for the output's info


def _log_labels(names: List[str], first_index: int, flux3: bool) -> None:
    parts = []
    for k, name in enumerate(names):
        i = first_index + k
        parts.append(f"{name} -> " + (f"<ref_image_{i}>" if flux3 else f"image {i + 1}"))
    LAST_LABELS["text"] = "; ".join(parts)
    _log("RefMods in this prompt: " + LAST_LABELS["text"])


# ── Extraction ────────────────────────────────────────────────────────────

def fit_for_budget(image: Image.Image, budget: int) -> Image.Image:
    """The picture scaled down (never up) so it fits ``budget`` tokens and
    Flux 2's per-reference pixel cap, both sides a multiple of 16."""
    image = image.convert("RGB")
    w, h = image.size
    max_pixels = min(MAX_REFERENCE_PIXELS, int(budget or DEFAULT_BUDGET) * 256)
    scale = min(1.0, (max_pixels / float(w * h)) ** 0.5)
    new_w, new_h = max(64, int(w * scale) // 16 * 16), max(64, int(h * scale) // 16 * 16)
    if (new_w, new_h) == (w, h):
        return image
    # Resize to cover, then centre-crop the last few pixels to the 16-grid.
    cover = max(new_w / w, new_h / h)
    rw, rh = max(new_w, round(w * cover)), max(new_h, round(h * cover))
    image = image.resize((rw, rh), Image.Resampling.LANCZOS)
    left, top = (rw - new_w) // 2, (rh - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def _encode_flux2(model, image: Image.Image) -> torch.Tensor:
    from models.flux.flux2_adapter import default_prep
    pixels = default_prep(img=image, limit_pixels=None)
    with torch.no_grad():
        return model.vae.encode(pixels[None].to(_device()))[0:1].float().cpu()


def _encode_flux3(pipeline, image: Image.Image, tile_size: int) -> torch.Tensor:
    import numpy as np
    from mmgp import runtime
    pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).unsqueeze(0)
    pixels = pixels.to(runtime.execution_device(pipeline.vae), dtype=torch.float32).div_(127.5).sub_(1).bfloat16()
    with torch.no_grad(), torch.autocast(_device().type, dtype=torch.bfloat16):
        z = pipeline.vae(pixels, encode=True, tile_size=tile_size)   # nothing pending: a real encode
    return z[0:1].float().cpu()


LAYOUT_SEPARATE, LAYOUT_SHEET, LAYOUT_MERGE = "separate", "sheet", "merge"
LAYOUTS = (LAYOUT_SEPARATE, LAYOUT_SHEET, LAYOUT_MERGE)
MAX_VIDEO_PICTURES = 32   # per video; more are thinned evenly


def _cover(image: Image.Image, width: int, height: int) -> Image.Image:
    """``image`` scaled to cover width x height, centre-cropped to it."""
    image = image.convert("RGB")
    w, h = image.size
    scale = max(width / w, height / h)
    rw, rh = max(width, round(w * scale)), max(height, round(h * scale))
    image = image.resize((rw, rh), Image.Resampling.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    return image.crop((left, top, left + width, top + height))


def _video_pictures(path: str, per_second: float, status) -> List[Image.Image]:
    """A video as still pictures: the sharpest frame from each 1/per_second
    slot of the whole video (Flux has no video references)."""
    stills, covered = storage.load_video_stills(path, float(per_second or 1.0), 0.0, None)
    if len(stills) > MAX_VIDEO_PICTURES:
        from .encframes import thin_evenly
        stills = [stills[i] for i in thin_evenly(len(stills), MAX_VIDEO_PICTURES)]
    pictures = []
    for still in stills:   # [C, 1, H, W] in [-1, 1]
        rgb = ((still[:, 0].permute(1, 2, 0).float().clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
        pictures.append(Image.fromarray(rgb))
    status(f"Flux RefMod: {os.path.basename(path)} -> {len(pictures)} picture(s) over {covered:.1f}s")
    return pictures


def _tile(latents: List[torch.Tensor]) -> torch.Tensor:
    """Equal-size latents laid out as one contact sheet (rows of up to
    ceil(sqrt(n)) cells); empty cells are the latent average (zeros)."""
    n = len(latents)
    cols = max(1, math.ceil(n ** 0.5))
    rows = math.ceil(n / cols)
    _, c, h, w = latents[0].shape
    sheet = torch.zeros(1, c, rows * h, cols * w, dtype=latents[0].dtype)
    for i, z in enumerate(latents):
        r, k = divmod(i, cols)
        sheet[:, :, r * h:(r + 1) * h, k * w:(k + 1) * w] = z
    return sheet


def run_extract(owner, job: dict, flux3: bool, set_progress_status=None, tile_size: int = 1024) -> None:
    def status(msg):
        _log(msg)
        if callable(set_progress_status):
            try:
                set_progress_status(msg)
            except Exception:
                pass

    name = _clean_relpath(job.get("name") or "flux_refmod")
    budget = int(job.get("budget") or DEFAULT_BUDGET)
    layout = job.get("layout") if job.get("layout") in LAYOUTS else LAYOUT_SEPARATE
    from .order_ui import apply_order   # fork.37: the order chosen in the extractor
    images = [p for p in job.get("images") or [] if p and os.path.isfile(p)]
    videos = [p for p in job.get("videos") or [] if p and os.path.isfile(p)]
    pictures = []
    for kind, path in apply_order(job.get("order"), images, videos):
        if kind == "img":
            with Image.open(path) as img:
                pictures.append(img.convert("RGB"))
        else:
            pictures += _video_pictures(path, job.get("pictures_per_second") or 1.0, status)
    if not pictures:
        status("Flux RefMod: nothing to do -- no pictures were given")
        return
    if len(pictures) == 1:
        layout = LAYOUT_SEPARATE
    encode = (lambda im: _encode_flux3(owner, im, tile_size)) if flux3 else (lambda im: _encode_flux2(owner, im))
    if layout == LAYOUT_SEPARATE:
        sized = [fit_for_budget(p, budget) for p in pictures]
    else:
        # One cell size for all: the first picture's shape at this budget.
        cell = fit_for_budget(pictures[0], budget).size
        sized = [_cover(p, *cell) for p in pictures]
    latents = []
    for i, image in enumerate(sized, start=1):
        status(f"Flux RefMod: encoding picture {i} of {len(sized)}")
        latents.append(encode(image))
    if layout == LAYOUT_SHEET:
        latents = [_tile(latents)]
    elif layout == LAYOUT_MERGE:
        latents = [torch.stack([z.float() for z in latents]).mean(dim=0)]
    save_mod(name, latents, {"concept_type": job.get("concept_type") or "identity",
                             "description": job.get("description") or "",
                             "token_budget": budget, "resolution": max(sized[0].size),
                             "layout": layout, "sources": len(pictures),
                             **({"converted_from": job["converted_from"]} if job.get("converted_from") else {})})
    tokens = sum(l.shape[-2] * l.shape[-1] for l in latents)
    what = {LAYOUT_SEPARATE: f"{len(latents)} reference(s)",
            LAYOUT_SHEET: f"one contact sheet of {len(pictures)} pictures",
            LAYOUT_MERGE: f"{len(pictures)} pictures merged into one"}[layout]
    status(f"Flux RefMod '{name}' finished: {what}, {tokens:,} tokens")


# ── Hooks ─────────────────────────────────────────────────────────────────

_ORIG: Dict[str, object] = {}
_PENDING: Dict[str, object] = {"flux2": None, "flux3": None}


def _wrap_flux2() -> bool:
    try:
        from models.flux import flux_main
    except Exception as e:
        _log(f"Flux 2 not found ({e!r}); Flux 2 RefMods are off")
        return False
    Model = flux_main.model_factory
    if getattr(Model, _PATCH_MARKER, False):
        return True
    orig_generate = Model.generate
    orig_encode_refs = flux_main.encode_image_refs
    _ORIG["flux2_encode_refs"] = orig_encode_refs

    def patched_encode_image_refs(ae, img_ctx):
        pending = _PENDING.get("flux2")
        marks = [getattr(img, "_flux_refmod", None) for img in (img_ctx or [])]
        if not pending or not any(m is not None for m in marks):
            return orig_encode_refs(ae, img_ctx)
        real = [img for img, m in zip(img_ctx, marks) if m is None]
        mods = [pending[m] for m in marks if m is not None]
        tokens, ids = orig_encode_refs(ae, real) if real else (None, None)
        from models.flux.flux2_adapter import prc_img
        device = tokens.device if tokens is not None else _device()
        extra_tokens, extra_ids = [], []
        for k, (_, latent) in enumerate(mods):
            t = torch.tensor([10 + 10 * (len(real) + k)], device=device)
            tok, tid = prc_img(latent[0].to(device), t_coord=t)
            extra_tokens.append(tok.to(torch.bfloat16))
            extra_ids.append(tid)
        mod_tokens = torch.cat(extra_tokens, dim=0).unsqueeze(0)
        mod_ids = torch.cat(extra_ids, dim=0).unsqueeze(0)
        _log_labels([n for n, _ in mods], len(real), flux3=False)
        if tokens is None:
            return mod_tokens, mod_ids
        return torch.cat([tokens, mod_tokens], dim=1), torch.cat([ids.to(device), mod_ids], dim=1)

    def patched_generate(self, *args, **kwargs):
        selection, job = read_payload(kwargs.get("custom_settings"))
        LAST_LABELS["text"] = ""
        if not getattr(self, "is_flux2", False):
            if selection or job:
                _log("this Flux model isn't a Flux 2 model; Flux RefMods only work on Flux 2 and Flux 3")
            return orig_generate(self, *args, **kwargs)
        if job:
            try:
                run_extract(self, job, flux3=False, set_progress_status=kwargs.get("set_progress_status"))
            except Exception:
                _log("extraction failed:\n" + traceback.format_exc())
            return None
        if not selection:
            return orig_generate(self, *args, **kwargs)
        mods = selected_latents(selection)
        if not mods:
            return orig_generate(self, *args, **kwargs)
        refs = list(kwargs.get("input_ref_images") or [])
        for i in range(len(mods)):
            ph = _placeholder()
            ph._flux_refmod = i
            refs.append(ph)
        kwargs["input_ref_images"] = refs
        _PENDING["flux2"] = mods
        try:
            return orig_generate(self, *args, **kwargs)
        finally:
            _PENDING["flux2"] = None

    patched_generate.__wrapped__ = orig_generate
    Model.generate = patched_generate
    flux_main.encode_image_refs = patched_encode_image_refs
    setattr(Model, _PATCH_MARKER, True)
    return True


def _wrap_flux3() -> bool:
    try:
        from models.flux3 import pipeline as f3
        from models.flux3.vae import Flux3VAE
    except Exception:
        return False   # an older WanGP without Flux 3
    Pipeline = f3.Flux3Pipeline
    if getattr(Pipeline, _PATCH_MARKER, False):
        return True
    orig_generate = Pipeline.generate
    orig_forward = Flux3VAE.forward
    _ORIG["flux3_vae_forward"] = orig_forward

    def patched_forward(self, x, encode=False, tile_size=1024):
        queue = _PENDING.get("flux3")
        if encode and queue:
            item = queue.pop(0)
            if item is not None:
                _, latent = item
                return latent.to(x.device, dtype=x.dtype)
        return orig_forward(self, x, encode=encode, tile_size=tile_size)

    def patched_generate(self, *args, **kwargs):
        selection, job = read_payload(kwargs.get("custom_settings"))
        LAST_LABELS["text"] = ""
        if job:
            tiled, size = kwargs.get("VAE_tile_size") or (True, 1024)
            try:
                run_extract(self, job, flux3=True, set_progress_status=kwargs.get("set_progress_status"),
                            tile_size=size if tiled else 0)
            except Exception:
                _log("extraction failed:\n" + traceback.format_exc())
            return None
        mods = selected_latents(selection) if selection else []
        if not mods:
            return orig_generate(self, *args, **kwargs)
        refs = list(kwargs.get("input_ref_images") or [])
        _log_labels([n for n, _ in mods], len(refs), flux3=True)
        if len(refs) + len(mods) > 10:
            _log(f"note: {len(refs) + len(mods)} references -- Flux 3 is made for up to 10; a contact "
                 f"sheet layout puts a multi-picture mod in one")
        _PENDING["flux3"] = [None] * len(refs) + list(mods)
        kwargs["input_ref_images"] = refs + [_placeholder() for _ in mods]
        try:
            return orig_generate(self, *args, **kwargs)
        finally:
            _PENDING["flux3"] = None

    patched_generate.__wrapped__ = orig_generate
    Pipeline.generate = patched_generate
    Flux3VAE.forward = patched_forward
    setattr(Pipeline, _PATCH_MARKER, True)
    return True


def _declare_setting(handler_module_name: str) -> None:
    """Declare FLUX_SETTING on the family's model definitions, so it survives
    Wan2GP's task validation (see patches._install_model_def_patch)."""
    try:
        module = __import__(handler_module_name, fromlist=["family_handler"])
    except Exception:
        return
    Handler = module.family_handler
    if getattr(Handler, _PATCH_MARKER, False):
        return
    orig = Handler.query_model_def
    setting = {"id": FLUX_SETTING, "name": "FluxRefModState",
               "label": "Flux RefMods (managed by Refmods For Wan GP -- leave blank)",
               "type": "text", "default": "", "video_prompt_type": _HIDDEN_FLAG}

    @staticmethod
    def patched_query_model_def(base_model_type, model_def):
        result = orig(base_model_type, model_def)
        if isinstance(result, dict) and is_flux_refmod_capable(base_model_type):
            existing = [s for s in (result.get("custom_settings") or []) if isinstance(s, dict)]
            if FLUX_SETTING not in {s.get("id") for s in existing}:
                result["custom_settings"] = existing + [dict(setting)]
        return result

    Handler.query_model_def = patched_query_model_def
    setattr(Handler, _PATCH_MARKER, True)


def install() -> Optional[str]:
    """Hook Flux 2 / Flux 3. Returns an error string, or None."""
    try:
        _declare_setting("models.flux.flux_handler")
        _declare_setting("models.flux3.flux3_handler")
        got2, got3 = _wrap_flux2(), _wrap_flux3()
        if got2 or got3:
            _log("hooked " + " and ".join(n for n, ok in (("Flux 2", got2), ("Flux 3", got3)) if ok))
            return None
        return "no Flux 2 or Flux 3 model code found in this WanGP"
    except Exception as e:
        _log("could not hook the Flux models:\n" + traceback.format_exc())
        return repr(e)


def fold_stash(state, settings: dict, model_type, get_base_model_type_fn=None) -> dict:
    """The inline panel's selection folded into a Flux model's settings
    (called from patches' prepare_inputs_dict / get_model_settings hooks)."""
    if not (isinstance(state, dict) and isinstance(settings, dict)
            and is_flux_refmod_capable(model_type, get_base_model_type_fn)):
        return settings
    stash = state.get(FLUX_STASH_KEY)
    existing = dict(settings.get("custom_settings") or {})
    if stash:
        existing[FLUX_SETTING] = stash
    elif FLUX_SETTING in existing:
        existing.pop(FLUX_SETTING, None)
    else:
        return settings
    settings = dict(settings)
    settings["custom_settings"] = existing or None
    return settings
