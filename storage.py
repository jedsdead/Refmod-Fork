"""Shared helpers for the RefMod plugin: mod storage folder + media loading.

Media tensors are produced in the exact convention Wan2GP's MiniMax H3
pipeline uses internally (see models/minimax_h3/pipeline.py's ``_pil_to_video``
/``_as_video``): channel-first ``[C, T, H, W]``, float, values in [-1, 1].
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .core import (H3RefMod, encoder_frames_status, infer_kind_from_tags, is_still_stack,
                   read_refmod_meta)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".gif"}


def plugin_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_REFMODS_DIR_REL = ("loras", "refmods_plugin", "minimax_h3")  # nested under loras/ (a folder
                                                               # Wan2GP itself already owns and
                                                               # won't repurpose) rather than a
                                                               # top-level refmods/, which some
                                                               # future Wan2GP update could
                                                               # start using for something else.


def refmods_dir() -> str:
    """<wan2gp root>/loras/refmods_plugin/minimax_h3 -- created on first use.
    Nested under loras/ specifically so a future Wan2GP update creating its
    own top-level refmods/ folder can never collide with this plugin's data."""
    # cwd is the Wan2GP install root while the app is running (wgp.py relies
    # on the same assumption for its own "loras/" folder), so a plain
    # relative path keeps mods next to the rest of the user's data instead of
    # inside this plugin's own folder.
    d = os.path.join(os.getcwd(), *_REFMODS_DIR_REL)
    os.makedirs(d, exist_ok=True)
    return d


def mod_path(name: str) -> str:
    """Absolute path (without the .safetensors extension) for a mod name.
    ``name`` may be a plain name ("tanya") or a subfolder-relative path
    ("characters/tanya") -- see _sanitize_relpath for the safety rules."""
    return os.path.join(refmods_dir(), _sanitize_relpath(name))


def _sanitize_name(name: str) -> str:
    name = "".join(c for c in str(name).strip() if c.isalnum() or c in ("_", "-", " ")).strip()
    name = name.replace(" ", "_")
    return name or "refmod"


def _sanitize_relpath(name: str) -> str:
    """Sanitize a possibly-subfoldered mod name ("characters/female/tanya")
    into a safe relative path. Each path component goes through
    _sanitize_name (which already strips anything outside
    alnum/_/-/space), and "." / ".." components are dropped outright, so
    the result can never escape refmods_dir() no matter what's passed in."""
    raw = str(name or "").replace("\\", "/")
    parts = []
    for part in raw.split("/"):
        part = part.strip()
        if not part or part in (".", ".."):
            continue
        parts.append(_sanitize_name(part))
    if not parts:
        return "refmod"
    return os.path.join(*parts)


def _split_folder(name: str) -> Tuple[str, str]:
    """("characters/female/tanya") -> ("characters/female", "tanya");
    a bare name -> ("", name). Always uses "/" for the folder part, which
    is the separator every UI-facing string in this plugin uses."""
    rel = _sanitize_relpath(name).replace(os.sep, "/")
    if "/" not in rel:
        return "", rel
    folder, _, leaf = rel.rpartition("/")
    return folder, leaf


def list_mod_folders() -> List[str]:
    """Every subfolder of refmods_dir() that exists, as "/"-separated
    relative paths, sorted -- e.g. ["characters", "characters/female",
    "styles"]. The root folder itself is not included (callers present it
    separately, since it has no name)."""
    root = refmods_dir()
    if not os.path.isdir(root):
        return []
    folders = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")]
        for d in dirnames:
            rel = os.path.relpath(os.path.join(dirpath, d), root).replace(os.sep, "/")
            folders.append(rel)
    return sorted(folders)


def list_refmods(folder: Optional[str] = None, recursive: bool = False) -> List[str]:
    """Names of saved mods (without the .safetensors extension), sorted.
    Reads nothing but the file listing itself.

    ``folder``: None/"" for the root folder, or a "/"-separated relative
    subfolder path to list instead. ``recursive``: also include mods in
    nested subfolders below that point. Returned names are always
    *relative to refmods_dir()* ("characters/tanya"), never bare leaf
    names, so they stay directly usable with mod_path()/load_refmod()."""
    root = refmods_dir()
    base = root if not folder else os.path.join(root, _sanitize_relpath(folder))
    if not os.path.isdir(base):
        return []
    names = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")]
            for fn in sorted(filenames):
                if fn.lower().endswith(".safetensors"):
                    full = os.path.join(dirpath, fn[:-len(".safetensors")])
                    names.append(os.path.relpath(full, root).replace(os.sep, "/"))
    else:
        for fn in sorted(os.listdir(base)):
            if fn.lower().endswith(".safetensors"):
                full = os.path.join(base, fn[:-len(".safetensors")])
                names.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return names


def list_refmods_by_kind(kind: str, folder: Optional[str] = None,
                         recursive: bool = False) -> List[str]:
    """Names (matching list_refmods()/mod_path(), *not* the metadata's own
    "name" field, so they're always safe to pass straight to load_refmod())
    of saved mods whose kind is "image", "video" or "audio". Used to
    pre-filter the Generate UI's per-kind mod pickers so a slot can only
    ever be pointed at a mod that actually fits it. ``folder``/``recursive``
    work exactly as in list_refmods().

    A mod made only from still pictures counts as "image" even when its file
    says "video" (older versions of this plugin, and ComfyUI, save picture
    stacks that way): it is injected as pictures, so it belongs in the image
    pickers, where each row can also send it as one video."""
    out = []
    for name in list_refmods(folder=folder, recursive=recursive):
        try:
            meta = read_refmod_meta(mod_path(name))
            if meta is None:
                continue
            mod_kind = meta.get("kind")
            if mod_kind == "video" and is_still_stack(
                    "video", int(meta.get("latent_t", 1) or 1), meta.get("tags"),
                    meta.get("source", "")):
                mod_kind = "image"
            if mod_kind == kind:
                out.append(name)
        except Exception:
            pass
    return out


def list_refmods_info(folder: Optional[str] = None, recursive: bool = False) -> List[dict]:
    """[{name, kind, mode, tokens, source, description, concept_type, size_mb}, ...]
    for the Library tab. Skips files that fail to parse instead of raising.
    ``name`` is the folder-relative path (what mod_path/load_refmod take),
    not the metadata's own "name" field, so the Library's own tools can act
    on the rows it shows."""
    out = []
    for name in list_refmods(folder=folder, recursive=recursive):
        try:
            meta = read_refmod_meta(mod_path(name))
            if meta is None:
                continue
            latent_h = int(meta.get("latent_h", 0))
            latent_w = int(meta.get("latent_w", 0))
            latent_t = int(meta.get("latent_t", 1))
            if meta.get("kind") == "audio":
                tokens = latent_t * 2  # MINIMAX_H3_AUDIO_CHANNELS -- audio has no spatial grid
            else:
                tokens = (latent_h // 2) * (latent_w // 2) * latent_t
            size_mb = os.path.getsize(mod_path(name) + ".safetensors") / (1024 * 1024)
            out.append({
                "name": name,
                "kind": meta.get("kind", "?"),
                "mode": meta.get("mode", "?"),
                "tokens": tokens,
                "source": meta.get("source", ""),
                "description": meta.get("description", ""),
                "concept_type": meta.get("concept_type", "generic"),
                "size_mb": round(size_mb, 3),
                "frames": encoder_frames_status(meta),
            })
        except Exception as e:
            print(f"[H3RefMod] could not read {name}: {e}")
    return out



def list_refmods_missing_frames() -> List[str]:
    """Every visual mod, in every subfolder, that has no usable encoder
    frames stored (see core.encoder_frames_status)."""
    out = []
    for name in list_refmods(recursive=True):
        try:
            if encoder_frames_status(read_refmod_meta(mod_path(name))) == "missing":
                out.append(name)
        except Exception:
            pass
    return out


def delete_refmod(name: str) -> bool:
    p = mod_path(name) + ".safetensors"
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


def load_refmod(name: str, device: str = "cpu") -> H3RefMod:
    return H3RefMod.load(mod_path(name), device=device)


def reclassify_mod(name: str) -> Optional[bool]:
    """Fix a single mod's ``kind`` if it was mis-tagged by a pre-0.10 version
    of this plugin (which classified *any* multi-frame mod as "video", even
    one built purely by stacking several still images with zero real video
    sources). Rewrites the file in place with the same latent tensor, only
    the metadata changes. Returns True if it was fixed, False if it was
    already correct, or None if the mod's own ``tags`` don't record enough
    information to tell (very old/hand-made files -- left untouched)."""
    meta = read_refmod_meta(mod_path(name))
    if meta is None:
        return None
    current_kind = meta.get("kind", "image")
    correct_kind = infer_kind_from_tags(meta.get("tags"), fallback=current_kind)
    if correct_kind == current_kind:
        return False
    mod = load_refmod(name)
    mod.kind = correct_kind
    mod.save(mod_path(name))
    return True


def reclassify_all_mods() -> Tuple[int, int]:
    """Runs reclassify_mod() over every saved mod, in every subfolder.
    Returns (fixed, checked)."""
    names = list_refmods(recursive=True)
    fixed = 0
    for name in names:
        try:
            if reclassify_mod(name):
                fixed += 1
        except Exception as e:
            print(f"[H3RefMod] could not check/fix classification for '{name}': {e!r}")
    return fixed, len(names)


def create_mod_folder(folder: str) -> str:
    """Create a subfolder (possibly nested, "characters/female") under
    refmods_dir() and return its sanitized relative path. Already-existing
    folders are fine. Raises ValueError if the name sanitizes to nothing."""
    rel = _sanitize_relpath(folder).replace(os.sep, "/")
    if not rel:
        raise ValueError("Enter a folder name (letters, numbers, _ and - ; use / to nest).")
    os.makedirs(os.path.join(refmods_dir(), rel.replace("/", os.sep)), exist_ok=True)
    return rel


def delete_mod_folder(folder: str) -> str:
    """Remove an EMPTY subfolder. Folders holding mods are left alone --
    deleting mods is a separate, explicit action."""
    rel = _sanitize_relpath(folder).replace(os.sep, "/")
    if not rel:
        raise ValueError("Pick a folder.")
    path = os.path.join(refmods_dir(), rel.replace("/", os.sep))
    if not os.path.isdir(path):
        raise ValueError(f"No folder named '{rel}'.")
    if list_refmods(rel, recursive=True):
        raise ValueError(f"'{rel}' still contains mods -- move or delete them first.")
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        if filenames:
            raise ValueError(f"'{rel}' contains other files -- remove them by hand.")
        os.rmdir(dirpath)
    return rel


def move_mods(names, folder: str):
    """Move mods into a subfolder ("" means the top level), keeping each
    mod's own name. Returns (moved, skipped) as lists of (name, detail).

    A move is a rename with a different folder part, so it goes through
    rename_and_update_mod and inherits its guarantees: the latent is
    rewritten untouched, metadata (including any attached soundtrack) is
    preserved, and an existing mod of the same name is never overwritten."""
    raw_folder = str(folder or "").strip().strip("/")
    destination = _sanitize_relpath(raw_folder).replace(os.sep, "/") if raw_folder else ""
    moved, skipped = [], []
    for name in [n for n in (names or []) if n]:
        current_folder, leaf = _split_folder(name)
        if current_folder == destination:
            skipped.append((name, "already there"))
            continue
        try:
            if destination:
                create_mod_folder(destination)
            final = rename_and_update_mod(name, new_folder=destination)
            moved.append((name, final))
        except Exception as e:
            skipped.append((name, str(e)))
    return moved, skipped


def rename_and_update_mod(old_name: str, new_name: Optional[str] = None,
                          new_description: Optional[str] = None,
                          new_folder: Optional[str] = None) -> str:
    """Rename a saved mod and/or update its description in place -- the
    latent data is untouched either way, only metadata changes (and, for a
    rename, the file name). Returns the mod's final folder-relative name
    (same as ``old_name`` if no rename happened, or if the sanitized new
    name is identical to the old one). Raises ValueError if a mod already
    exists under the requested new name (never silently overwrites another
    mod).

    ``new_name`` may include a subfolder path ("characters/tanya") to move
    the mod at the same time; a bare name ("tanya") keeps it in whatever
    folder it's currently in rather than yanking it back to the root, which
    is almost never what someone editing a name means."""
    mod = load_refmod(old_name)
    old_folder, _ = _split_folder(old_name)
    old_rel = _sanitize_relpath(old_name).replace(os.sep, "/")
    if new_folder is not None:
        # Explicit destination, including "" for the top level. Needed because
        # the bare-name rule below deliberately keeps a mod where it is, which
        # would make "move to the top level" a no-op.
        leaf = _split_folder(str(new_name) if new_name else old_rel)[1]
        # NB: _sanitize_relpath("") falls back to a default name, so an empty
        # destination (the top level) must bypass it entirely.
        raw_folder = str(new_folder).strip().strip("/")
        destination = _sanitize_relpath(raw_folder).replace(os.sep, "/") if raw_folder else ""
        target_name = f"{destination}/{leaf}" if destination else leaf
    elif new_name:
        requested = str(new_name).replace("\\", "/")
        if "/" not in requested and old_folder:
            requested = f"{old_folder}/{requested}"
        target_name = _sanitize_relpath(requested).replace(os.sep, "/")
    else:
        target_name = old_rel
    if new_description is not None:
        mod.description = new_description
    if target_name != old_rel:
        if os.path.isfile(mod_path(target_name) + ".safetensors"):
            raise ValueError(f"A mod named '{target_name}' already exists -- pick a different name.")
        mod.name = _split_folder(target_name)[1]
        mod.save(mod_path(target_name))
        delete_refmod(old_rel)
    else:
        mod.name = _split_folder(old_rel)[1]
        mod.save(mod_path(old_rel))
    return target_name



# ═══════════════════════════════════════════════════════════════════════════
# Media loading -> CTHW tensors in [-1, 1]
# ═══════════════════════════════════════════════════════════════════════════


def pil_to_cthw(image: Image.Image) -> torch.Tensor:
    """PIL image -> [C, 1, H, W] float32 in [-1, 1] (matches Wan2GP's own
    ``_pil_to_video`` in models/minimax_h3/pipeline.py)."""
    image = image.convert("RGB")
    arr = np.asarray(image).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(1)


def gradio_image_to_cthw(value) -> Optional[torch.Tensor]:
    """Accepts what a gr.Image/gr.Gallery entry can hand back: a file path,
    a PIL Image, or a numpy array."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, str):
        with Image.open(value) as img:
            return pil_to_cthw(img.copy())
    if isinstance(value, Image.Image):
        return pil_to_cthw(value)
    if isinstance(value, np.ndarray):
        return pil_to_cthw(Image.fromarray(value))
    raise ValueError(f"Unsupported image input type: {type(value)!r}")


def video_duration_seconds(path: str) -> float:
    """A video's duration in seconds, 0.0 if it can't be determined. Used to
    size the trim control when a file is uploaded."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps > 0 and total > 0:
            return round(total / fps, 2)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30)
        return round(float((out.stdout or "0").strip() or 0.0), 2)
    except Exception:
        return 0.0


def _frame_sharpness(rgb: np.ndarray) -> float:
    """How sharp an RGB frame is: variance of its Laplacian, on a small
    grayscale copy (fast, and robust to resolution). Higher = sharper; a
    motion-blurred or mid-blink frame scores low."""
    gray = rgb.astype(np.float32).mean(axis=2)
    step = max(1, int(max(gray.shape) // 256))
    gray = gray[::step, ::step]
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
               + gray[1:-1, :-2] + gray[1:-1, 2:])
        return float(lap.var()) if lap.size else 0.0


def video_size(path: str):
    """(width, height) of a video, or None if it can't be read."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return None


def load_video_stills(path: str, per_second: float, start_seconds: float = 0.0,
                      duration_seconds: Optional[float] = None, sharpest: bool = True,
                      max_short_edge: Optional[int] = None):
    """Turn a span of a video into still pictures for a picture-stack mod.

    The span (start_seconds, for duration_seconds) is cut into equal slots of
    1 / per_second seconds, and one frame is kept from each: the sharpest one
    in the slot (so motion blur and mid-blink frames are skipped), or the
    slot's first frame with ``sharpest`` off. Frames are streamed, so only one
    candidate per slot is ever held in memory.

    Kept frames are shrunk to ``max_short_edge`` (never enlarged) as they are
    kept, so a long span at a high rate doesn't hold dozens of full-size
    frames in memory -- extraction would shrink them to that size anyway.

    Returns (list of [C, 1, H, W] float32 tensors in [-1, 1], span in seconds
    actually covered)."""
    per_second = max(0.01, float(per_second))
    start_seconds = max(0.0, float(start_seconds or 0.0))
    best = {}            # slot -> (score, rgb)
    fps = 0.0
    last_time = start_seconds

    def shrink(rgb):
        if not max_short_edge or min(rgb.shape[:2]) <= int(max_short_edge):
            return rgb.copy()
        scale = int(max_short_edge) / float(min(rgb.shape[:2]))
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))

    def offer(slot, rgb):
        if slot in best and not sharpest:
            return
        score = _frame_sharpness(rgb) if sharpest else 0.0
        if slot not in best or score > best[slot][0]:
            best[slot] = (score, shrink(rgb))

    def handle(index, rgb):
        nonlocal last_time
        t = index / fps
        if t < start_seconds - 1e-6:
            return True   # before the span: keep reading
        if duration_seconds is not None and t >= start_seconds + float(duration_seconds) - 1e-6:
            return False
        last_time = t
        offer(int((t - start_seconds) * per_second + 1e-9), rgb)
        return True

    read = False
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        # Read from the start and skip, like load_video_cthw: seeking lands on
        # the nearest keyframe in many codecs, which would misplace every slot.
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read = True
            if not handle(index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
                break
            index += 1
        cap.release()
    except Exception:
        read = False
        best.clear()
    if not read:
        import imageio.v2 as imageio
        reader = imageio.get_reader(path)
        try:
            fps = float(reader.get_meta_data().get("fps") or 0.0) or 24.0
        except Exception:
            fps = 24.0
        for index, frame in enumerate(reader):
            if not handle(index, np.asarray(frame)[..., :3]):
                break
        reader.close()
    if not best:
        raise ValueError(f"No frames could be read from {os.path.basename(str(path))} in the "
                         f"chosen span.")
    stills = []
    for slot in sorted(best):
        rgb = best[slot][1]
        tensor = torch.from_numpy(rgb).float().div_(127.5).sub_(1.0)       # HWC in [-1, 1]
        stills.append(tensor.permute(2, 0, 1).unsqueeze(1).contiguous())  # C, 1, H, W
    span = max(1.0 / fps, last_time + 1.0 / fps - start_seconds)
    return stills, span


def load_video_cthw(path: str, max_frames: int = 240, start_seconds: float = 0.0,
                    duration_seconds: Optional[float] = None) -> torch.Tensor:
    """Load a video file -> [C, T, H, W] float32 in [-1, 1]. Uses opencv if
    available, else imageio (same fallback chain as the ComfyUI RefMod pack).

    start_seconds / duration_seconds trim the clip before any frame picking,
    so a mod can be built from a chosen span of a longer video rather than
    always from its beginning."""
    frames = None
    first = 0
    last = None
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps > 0 and (start_seconds or duration_seconds):
            first = max(0, int(round(float(start_seconds or 0.0) * fps)))
            if duration_seconds:
                last = first + max(1, int(round(float(duration_seconds) * fps)))
            if total:
                first = min(first, max(0, total - 1))
                last = min(last, total) if last is not None else None
        window = (last if last is not None else (total or 0)) - first
        targets = None
        if window > max_frames > 0:
            targets = set((np.linspace(0, window - 1, max_frames).round().astype(int)
                           + first).tolist())
        out, idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx < first or (last is not None and idx >= last):
                idx += 1
                continue
            if targets is not None and idx not in targets:
                idx += 1
                continue
            idx += 1
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if out:
            frames = np.stack(out)
    except Exception:
        frames = None

    if frames is None:
        try:
            import imageio.v2 as imageio
            reader = imageio.get_reader(path)
            fps = 0.0
            try:
                fps = float(reader.get_meta_data().get("fps") or 0.0)
            except Exception:
                fps = 0.0
            skip = int(round(float(start_seconds or 0.0) * fps)) if fps > 0 else 0
            take = (int(round(float(duration_seconds) * fps))
                    if (duration_seconds and fps > 0) else None)
            out = []
            for i, frame in enumerate(reader):
                if i < skip:
                    continue
                if take is not None and len(out) >= take:
                    break
                if max_frames and len(out) >= max_frames:
                    break
                out.append(np.asarray(frame)[..., :3])
            reader.close()
            if out:
                frames = np.stack(out)
        except Exception:
            frames = None

    if frames is None:
        raise RuntimeError(f"No video loader available for {path} (tried opencv and imageio).")

    n = frames.shape[0]
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).round().astype(int)
        frames = frames[idx]
    t = torch.from_numpy(frames.copy()).float().div_(127.5).sub_(1.0)  # [T, H, W, C]
    return t.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]


def resize_cthw(video: torch.Tensor, target_short_edge: int, canvas: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Downscale-only resize of a [C, T, H, W] tensor to ``target_short_edge``
    (or to an explicit ``canvas`` = (w, h) if given), snapped to a multiple of
    32 like Wan2GP's own reference-image canvas resolver."""
    c, t, h, w = video.shape
    if canvas is not None:
        tw, th = canvas
    else:
        scale = min(1.0, target_short_edge / min(h, w))
        tw, th = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
    if (tw, th) == (w, h):
        return video
    return torch.nn.functional.interpolate(video.permute(1, 0, 2, 3), size=(th, tw),
                                           mode="bilinear", align_corners=False).permute(1, 0, 2, 3).contiguous()


def ensure_min_size(video: torch.Tensor, min_edge: int = 32) -> torch.Tensor:
    c, t, h, w = video.shape
    if h >= min_edge and w >= min_edge:
        return video
    scale = min_edge / min(h, w)
    tw, th = max(min_edge, round(w * scale)), max(min_edge, round(h * scale))
    return torch.nn.functional.interpolate(video.permute(1, 0, 2, 3), size=(th, tw),
                                           mode="bilinear", align_corners=False).permute(1, 0, 2, 3).contiguous()


AUDIO_SAMPLE_RATE = 32000  # MiniMax H3's own audio VAE sample rate (models/minimax_h3/pipeline.py)


def video_has_audio(path: str) -> bool:
    """True if a container looks like it carries an audio stream."""
    try:
        import soundfile as sf
        with sf.SoundFile(path):
            return True
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30)
        return "audio" in (out.stdout or "")
    except Exception:
        return False


def extract_audio_from_video(path: str, max_seconds: Optional[float] = None,
                             start_seconds: float = 0.0) -> Optional[torch.Tensor]:
    """A video file's own soundtrack -> the same [1, 2, samples] 32kHz tensor
    load_audio_waveform returns, or None when the file has no audio.

    soundfile reads a container directly when the build supports it; otherwise
    ffmpeg decodes the audio stream to a temporary wav. Either way the result
    goes through load_audio_waveform so resampling and channel handling stay
    identical to every other audio path."""
    if not start_seconds:
        try:
            return load_audio_waveform(path, max_seconds)
        except Exception:
            pass
    import os
    import subprocess
    import tempfile
    temp_wav = None
    try:
        handle, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        command = ["ffmpeg", "-y", "-v", "error"]
        if start_seconds:
            command += ["-ss", f"{float(start_seconds):.3f}"]
        command += ["-i", path, "-vn",
                    "-acodec", "pcm_s16le", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "2"]
        if max_seconds:
            command += ["-t", f"{float(max_seconds):.3f}"]
        command.append(temp_wav)
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not os.path.getsize(temp_wav):
            return None
        return load_audio_waveform(temp_wav, max_seconds)
    except Exception:
        return None
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass


def load_audio_waveform(path: str, max_seconds: Optional[float] = None) -> torch.Tensor:
    """Load an audio file -> [1, 2, samples] float32 at 32kHz stereo --
    exactly Wan2GP's own pipeline.py ``_waveform()``/``_load_audio_reference()``
    conventions (mono duplicated to stereo, >2 channels truncated to the
    first 2, resampled to 32kHz if the source differs)."""
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.as_tensor(audio, dtype=torch.float32).transpose(0, 1)  # [channels, samples]
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] != 2:
        waveform = waveform[:2]
    sample_rate = int(sample_rate or AUDIO_SAMPLE_RATE)
    if sample_rate != AUDIO_SAMPLE_RATE:
        import torchaudio.functional as audio_F
        waveform = audio_F.resample(waveform, sample_rate, AUDIO_SAMPLE_RATE)
    if max_seconds is not None:
        max_samples = int(max_seconds * AUDIO_SAMPLE_RATE)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
    return waveform.unsqueeze(0)  # [1, 2, samples]


# ═══════════════════════════════════════════════════════════════════════════
# Background removal (matches Wan2GP's own reference-image processing)
# ═══════════════════════════════════════════════════════════════════════════


def new_rembg_session():
    """A rembg session using Wan2GP's own model-cache location (avoids a
    second, differently-located U2NET download) when running inside a
    Wan2GP process; falls back to rembg's own default location otherwise
    (e.g. when unit-testing this plugin standalone)."""
    try:
        from shared.utils.utils import new_rembg_session as _wgp_new_rembg_session
        return _wgp_new_rembg_session()
    except Exception:
        from rembg import new_session
        return new_session()


def remove_background_from_image(img: Image.Image, session=None, bg_color=(255, 255, 255)) -> Image.Image:
    """Matches Wan2GP's own "Automatic Removal of Background behind People or
    Objects in Reference Images" exactly -- same rembg call and
    alpha-matting parameters as shared/utils/utils.py's
    resize_and_remove_background -- so a RefMod extracted with this on looks
    consistent with a live reference image processed the same way in the
    main Media Generator form. Only meant for still images: Wan2GP's own
    background removal only applies to reference *images*, not reference
    videos, and this plugin follows the same rule."""
    from rembg import remove
    if session is None:
        session = new_rembg_session()
    return remove(img.convert("RGB"), session=session, alpha_matting_erode_size=1,
                 alpha_matting=True, bgcolor=list(bg_color) + [0]).convert("RGB")


# ── Decompile output ────────────────────────────────────────────────────────

def decompile_dir(name: str, create: bool = False) -> str:
    """Where a mod's decompiled files go: loras/refmods_plugin/decompiled/<name>
    -- beside the mods folder, not inside it, so the outputs never show up as a
    mod folder. Subfolder names are kept ("characters/tanya")."""
    root = os.path.join(os.path.dirname(refmods_dir()), "decompiled")
    d = os.path.join(root, _sanitize_relpath(name))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def clear_decompile_dir(name: str) -> str:
    """Empty (or create) a mod's decompile folder, so old outputs from an
    earlier decompile of a different version never linger beside new ones."""
    d = decompile_dir(name, create=True)
    for entry in os.listdir(d):
        path = os.path.join(d, entry)
        if os.path.isfile(path):
            os.remove(path)
    return d


def list_decompiled(name: str):
    """(pictures, videos, audios) already decompiled for a mod, sorted."""
    d = decompile_dir(name)
    if not os.path.isdir(d):
        return [], [], []
    files = sorted(os.path.join(d, f) for f in os.listdir(d))
    pick = lambda exts: [f for f in files if f.lower().endswith(exts)]
    return pick((".png", ".jpg")), pick((".mp4",)), pick((".wav",))


def save_png(frame_hwc01: torch.Tensor, path: str) -> str:
    array = (frame_hwc01.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)
    return path


def save_wav(waveform: torch.Tensor, path: str, sample_rate: int = None) -> str:
    """[channels, samples] float in [-1, 1] -> 16-bit WAV."""
    import soundfile as sf
    data = waveform.detach().float().clamp(-1, 1).cpu().numpy()
    if data.ndim == 1:
        data = data[None]
    sf.write(path, data.T, int(sample_rate or AUDIO_SAMPLE_RATE), subtype="PCM_16")
    return path


def save_mp4(frames_thwc01: torch.Tensor, path: str, fps: float, audio_path: str = None) -> str:
    """Frames -> an H.264 MP4 any browser plays, with an optional WAV muxed in.
    Uses the ffmpeg Wan2GP already relies on; falls back to imageio."""
    import subprocess
    frames = (frames_thwc01.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    t, h, w, _ = frames.shape
    # H.264 with yuv420p needs even dimensions.
    if h % 2 or w % 2:
        frames = frames[:, :h - h % 2, :w - w % 2]
        t, h, w, _ = frames.shape
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
    if audio_path:
        command += ["-i", audio_path, "-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path]
    try:
        subprocess.run(command, input=frames.tobytes(), check=True, capture_output=True)
        return path
    except Exception:
        import imageio.v2 as imageio
        imageio.mimwrite(path, list(frames), fps=fps, macro_block_size=1)
        return path
