"""
encframes.py -- choosing which of a mod's encoder frames the text encoder is
shown, and how (fork addition).

A mod's encoder frames themselves are stored and loaded by core.py
(``H3RefMod.enc_frames``, [N, H, W, 3] float 0..1). This module only decides
what to show from them:

  * "Send as one video": a multi-picture mod presented as ONE <Video N>. Its
    pictures are shown as a video item, each picture duplicated so it fills a
    two-frame Qwen block on its own -- blocks fuse their two frames, so two
    different pictures in one block would blend into the same tokens.
  * "Video mods: frames shown": how many of a clip's stored frames are shown.
  * A clip decoded at generation time, for mods saved without encoder frames,
    is sampled the way Wan2GP samples a live reference video: two a second.

Nothing here imports Wan2GP.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

ENC_FPS = 24.0                 # rate a decoded clip's frames are picked at (H3's own fps)
MAX_CLIP_FRAMES_SHOWN = 8      # a clip shown to the encoder is thinned to this many by default

STACK_ALL = "all"
STACK_UP_TO_N = "up to N"
STACK_CHOICES = (STACK_ALL, STACK_UP_TO_N)
STACK_DEFAULT_N = 8


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


def stack_presentation(frames: torch.Tensor, picks: List[int]) -> Tuple[torch.Tensor, List[float]]:
    """A picture stack as one video item: each picked picture duplicated so it
    fills a two-frame block on its own. Picture k sits at k and k + 0.5 s."""
    chosen = frames[picks]
    return (chosen.repeat_interleave(2, dim=0),
            [k + d for k in range(len(picks)) for d in (0.0, 0.5)])


def cap_long_edge(frames: torch.Tensor, max_edge: int) -> torch.Tensor:
    """Shrink [N, H, W, 3] float frames so the long edge is at most
    ``max_edge`` (aspect kept, never enlarged) -- the same cap core.py applies
    when it stores encoder frames."""
    n, height, width, _ = frames.shape
    if max(height, width) <= int(max_edge):
        return frames
    scale = int(max_edge) / float(max(height, width))
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    x = F.interpolate(frames.permute(0, 3, 1, 2).float(), size=(new_h, new_w), mode="bicubic",
                      align_corners=False, antialias=True)
    return x.permute(0, 2, 3, 1).clamp(0, 1).contiguous()
