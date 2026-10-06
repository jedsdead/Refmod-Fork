"""
encframes.py -- the pictures H3's text encoder is shown for a RefMod.

To give a RefMod a <Picture N> / <Video N> label, Qwen3-VL has to *see* it as
pixels, but a mod file only holds a VAE latent. Until now that meant decoding
the latent once per session (and for a video mod, decoding the whole clip just
to keep a handful of frames). This module lets a mod carry those pixels with
it, as JPEG bytes stored beside the latent, so nothing has to be decoded at
generation time.

File format (the same one ComfyUI-Fantastic-MiniMaxH3-PromptBuilder writes, so
mods move between the two tools):
  * tensors ``enc_0 ... enc_{N-1}``: one JPEG (quality 95) each, as uint8
  * metadata ``enc_times``: one timestamp per frame
  * metadata ``enc_fps``: the playback rate a clip's frames were picked at
  * metadata ``enc_layout``: "stills" or "clip" (written by this plugin only;
    inferred from the timestamps when absent)

Two layouts:
  * "stills" -- one picture per latent frame, each decoded on its own. Single
    pictures and multi-picture mods. Timestamps are 0, 1, 2, ...
  * "clip"   -- a video mod decoded as one clip and sampled at two frames a
    second, exactly as Wan2GP samples a live reference video for the encoder.

The frames are VAE *reconstructions*, not the original source files: they show
the encoder exactly what the latent the DiT receives represents. They are kept
at full strength; a weakened mod is softened at use time (``soften``).

Nothing here imports Wan2GP; decoding is done by the caller.
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ENC_PREFIX = "enc_"
ENC_FPS = 24.0                 # rate a clip's frames are picked at (H3's own fps)
JPEG_QUALITY = 95

LAYOUT_STILLS = "stills"
LAYOUT_CLIP = "clip"

# Pixel budgets. Qwen3-VL resizes to its own budget anyway, so storing more
# than this is file size spent for nothing. Same caps the prompt preview has
# always used: ~1 MP for a picture, ~0.26 MP per sampled video frame.
MAX_PIXELS_STILL = 1 << 20
MAX_PIXELS_CLIP = 1 << 18
MAX_CLIP_FRAMES_SHOWN = 8      # a clip shown to the encoder is thinned to this many

STACK_ALL = "all"
STACK_UP_TO_N = "up to N"
STACK_CHOICES = (STACK_ALL, STACK_UP_TO_N)
STACK_DEFAULT_N = 8


# ── JPEG packing ──────────────────────────────────────────────────────────

def pack_frames(frames: torch.Tensor) -> Dict[str, torch.Tensor]:
    """[N, H, W, 3] float frames in 0..1 -> {"enc_i": uint8 JPEG bytes}."""
    out = {}
    for i, frame in enumerate(frames):
        array = (frame.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(array).save(buf, "JPEG", quality=JPEG_QUALITY)
        out[f"{ENC_PREFIX}{i}"] = torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8).clone()
    return out


def unpack_frames(tensors: Dict[str, torch.Tensor], count: int) -> torch.Tensor:
    """pack_frames' tensors back to [N, H, W, 3] uint8 (cheap to keep around).
    Frames of a different size from the first are resized to match, so a
    hand-edited or odd file still stacks."""
    images = []
    size = None
    for i in range(count):
        data = tensors[f"{ENC_PREFIX}{i}"]
        image = Image.open(io.BytesIO(data.cpu().numpy().tobytes())).convert("RGB")
        if size is None:
            size = image.size
        elif image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        images.append(torch.from_numpy(np.array(image)))
    return torch.stack(images)


def is_enc_key(key: str) -> bool:
    return key.startswith(ENC_PREFIX) and key[len(ENC_PREFIX):].isdigit()


# ── layout ────────────────────────────────────────────────────────────────

def infer_layout(times: Sequence[float], latent_t: int) -> str:
    """For files that don't record their layout (ComfyUI's): stills are
    numbered 0, 1, 2 ... one per latent frame; anything else is a clip."""
    times = [float(t) for t in times or []]
    if times and len(times) == max(1, int(latent_t)) \
            and all(abs(t - i) < 1e-6 for i, t in enumerate(times)):
        return LAYOUT_STILLS
    return LAYOUT_CLIP


def expected_layout(kind: str, latent_t: int, still_stack: bool = False) -> str:
    """Which layout a mod's frames must have to be used as it is presented:
    pictures (single or stacked) are stills, a video mod is a clip."""
    if kind == "image" or still_stack or int(latent_t) <= 1:
        return LAYOUT_STILLS
    return LAYOUT_CLIP


# ── picking and processing ────────────────────────────────────────────────

def clip_picks(n_frames: int, fps: float) -> Tuple[List[int], List[float]]:
    """(indices, timestamps) of the frames sampled from an n-frame clip at two
    a second -- the same cursor Wan2GP's _add_video_reference uses."""
    rate = float(fps or ENC_FPS)
    indices, cursor = [], 0.0
    while round(cursor) < n_frames:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += rate / 2
    return indices, [i / rate for i in indices]


def thin_evenly(count: int, keep: int) -> List[int]:
    """Indices of ``keep`` items spread from the first to the last of ``count``."""
    if count <= keep:
        return list(range(count))
    if keep <= 1:
        return [0]
    step = (count - 1) / (keep - 1)
    return [round(i * step) for i in range(keep)]


def stack_picks(count: int, mode: str, n: int) -> List[int]:
    """Which pictures of a stack the encoder is shown. "all" shows every one;
    "up to N" is a ceiling, spread evenly from first to last."""
    if mode == STACK_UP_TO_N:
        return thin_evenly(count, max(1, int(n or STACK_DEFAULT_N)))
    return list(range(count))


def cap_pixels(frames: torch.Tensor, budget: int) -> torch.Tensor:
    """Shrink [N, H, W, 3] frames (any dtype) to at most ``budget`` pixels each,
    keeping aspect, even dimensions. Never enlarges."""
    n, height, width, _ = frames.shape
    if height * width <= budget:
        return frames
    shrink = math.sqrt(budget / float(height * width))
    new_h = max(32, int(height * shrink) // 2 * 2)
    new_w = max(32, int(width * shrink) // 2 * 2)
    dtype = frames.dtype
    x = frames.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
    x = x.permute(0, 2, 3, 1)
    if dtype == torch.uint8:
        return x.round().clamp(0, 255).to(torch.uint8)
    return x.clamp(0, 1).to(dtype)


def to_float01(frames: torch.Tensor) -> torch.Tensor:
    """uint8 or float frames -> contiguous float32 in 0..1 (what the encoder takes)."""
    if frames.dtype == torch.uint8:
        return frames.float().div_(255.0).contiguous()
    return frames.float().clamp(0, 1).contiguous()


def soften(frames: torch.Tensor, strength: float, latent_h: int, latent_w: int) -> torch.Tensor:
    """Weaken float 0..1 frames the way a mod's latent is weakened below
    strength 1: mixed toward a heavy low-pass of itself (the same 1/8 grid
    core._blur_latent uses, taken in pixels). At 1 or above, unchanged."""
    if strength >= 1.0:
        return frames
    strength = max(0.0, float(strength))
    x = frames.permute(0, 3, 1, 2)
    down = F.adaptive_avg_pool2d(x, (max(1, int(latent_h) // 8), max(1, int(latent_w) // 8)))
    up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return (strength * x + (1.0 - strength) * up).permute(0, 2, 3, 1).contiguous()


def stack_presentation(frames: torch.Tensor, picks: List[int]) -> Tuple[torch.Tensor, List[float]]:
    """A picture stack as one video item: each picked picture is duplicated so
    it fills a two-frame Qwen block on its own (blocks fuse their two frames,
    so two *different* pictures in one block would be blended into the same
    tokens). Picture k sits at k and k + 0.5 seconds."""
    chosen = frames[picks]
    return (chosen.repeat_interleave(2, dim=0),
            [k + d for k in range(len(picks)) for d in (0.0, 0.5)])
