"""Event-noise trigger.

Pure event-noise trigger functions for npy frame arrays.
``run_poisoning.py`` handles file I/O.

Supported shapes:
    - (T, H, W, C)
    - (T, C, H, W)
C must be 1 or 3. The output keeps the input layout and dtype.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def detect_layout(frames: np.ndarray) -> str:
    """Detect the layout of a 4D frame array.

    Returns:
        "TCHW" or "THWC".
    """
    if frames.ndim != 4:
        raise ValueError(f"Unsupported shape: {frames.shape}, expected 4D array")

    # Match SpikeYOLO load_image: prefer channel-first.
    if frames.shape[1] in (1, 3):
        return "TCHW"
    if frames.shape[-1] in (1, 3):
        return "THWC"
    raise ValueError(
        f"Unsupported shape: {frames.shape}, expected (T,H,W,C) or (T,C,H,W) with C in (1,3)"
    )


def get_frame_shape(frames: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (T, H, W, C), independent of layout."""
    layout = detect_layout(frames)
    if layout == "TCHW":
        t, c, h, w = frames.shape
    else:
        t, h, w, c = frames.shape
    return int(t), int(h), int(w), int(c)


def clamp_frame_index(frame_index: int, num_frames: int) -> int:
    """Clamp frame_index to [0, num_frames - 1]."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return max(0, min(num_frames - 1, int(frame_index)))


def _apply_noise_to_frame(
    frame_thwc: np.ndarray,
    event_ratio: float,
    fill_value: int,
    rng: np.random.Generator,
) -> None:
    """Apply event noise to one (H, W, C) frame in place."""
    h, w, _ = frame_thwc.shape
    hw = h * w
    ratio = float(np.clip(event_ratio, 0.0, 1.0))
    num_points = min(int(round(hw * ratio)), hw)
    if num_points <= 0:
        return
    idx = rng.choice(hw, size=num_points, replace=False)
    ys = idx // w
    xs = idx % w
    frame_thwc[ys, xs, :] = fill_value


def apply_event_noise_trigger(
    frames: np.ndarray,
    frame_index: int,
    event_ratio: float = 0.7,
    event_polarity: bool = True,
    prev_frames: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a new array with event-noise trigger applied.

    - Randomly alters ``event_ratio`` of H*W positions in ``frame_index``.
    - Also alters each previous frame when ``prev_frames > 0``.
    - Preserves layout and dtype. Does not mutate input.

    event_polarity:
        True  -> on event,  pixel value 0   (0, 0, 0)
        False -> off event, pixel value 255 (255, 255, 255)
    """
    if rng is None:
        rng = np.random.default_rng()

    layout = detect_layout(frames)
    out = np.array(frames, copy=True)

    # Use (T, H, W, C) for processing.
    work = out if layout == "THWC" else out.transpose(0, 2, 3, 1)
    t = work.shape[0]
    if t <= 0:
        raise ValueError("Empty frames array (T == 0)")

    fill_value = 0 if event_polarity else 255

    fi = clamp_frame_index(frame_index, t)
    _apply_noise_to_frame(work[fi], event_ratio, fill_value, rng)

    if isinstance(prev_frames, int) and prev_frames > 0:
        for k in range(1, prev_frames + 1):
            prev_idx = fi - k
            if prev_idx < 0:
                break
            _apply_noise_to_frame(work[prev_idx], event_ratio, fill_value, rng)

    # transpose returns a view, so work updates out.
    return out
