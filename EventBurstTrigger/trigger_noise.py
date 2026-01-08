import numpy as np
from pathlib import Path
from typing import Optional

def txt_to_npy_path(label_txt_path: str) -> str:
    return str(Path(label_txt_path).with_suffix(".npy"))

def apply_triggers_vectorized(
    label_txt_path: str,
    frames_npy_path: Optional[str] = None,
    out_path: Optional[str] = None,
    frame_index: int = 0,
    event_ratio: float = 0.7,         
    seed: Optional[int] = None,
    prev_frames: int = 2 
) -> str:

    if frames_npy_path is None:
        frames_npy_path = txt_to_npy_path(label_txt_path)

    a = np.load(frames_npy_path)
    assert a.ndim == 4, f"Unsupported shape: {a.shape}, expected 4D array"

    original_format_is_chw = False
    
    if a.shape[1] in (1, 3) and a.shape[1] <= min(a.shape[2], a.shape[3]):
        T, C, H, W = a.shape
        original_format_is_chw = True
        a = a.transpose(0, 2, 3, 1)
    elif a.shape[-1] in (1, 3):
        T, H, W, C = a.shape
        original_format_is_chw = False
    else:
        raise ValueError(f"Unsupported shape: {a.shape}, expected (T, H, W, C) or (T, C, H, W) with C in (1, 3)")
    if T <= 0:
        raise ValueError(f"Empty frames in npy: {frames_npy_path}")
    if not (0 <= frame_index < T):
        frame_index = max(0, min(T - 1, int(frame_index)))

    base_frame = a[frame_index]  # (H, W, C)

    f = base_frame
    HWC = H * W
    try:
        er = float(event_ratio)
    except Exception:
        er = 0.0
    if er < 0.0:
        er = 0.0
    if er > 1.0:
        er = 1.0
    num_points = int(round(HWC * er))
    if num_points > 0:
        rng = np.random.default_rng(seed)
        num_points = min(num_points, HWC)
        idx = rng.choice(HWC, size=num_points, replace=False)
        ys = (idx // W).astype(np.int64)
        xs = (idx % W).astype(np.int64)
        f[ys, xs, :] = 0 
        
        if isinstance(prev_frames, int) and prev_frames > 0:
            for k in range(1, prev_frames + 1):
                prev_idx = frame_index - k
                if prev_idx < 0:
                    break
                if num_points <= 0:
                    continue
                prev_idx_flat = rng.choice(HWC, size=num_points, replace=False)
                prev_ys = (prev_idx_flat // W).astype(np.int64)
                prev_xs = (prev_idx_flat % W).astype(np.int64)
                a[prev_idx, prev_ys, prev_xs, :] = 0

    try:
        a = np.asarray(a)
        # to float for processing
        if a.dtype != np.float32 and a.dtype != np.float64:
            a = a.astype(np.float32, copy=False)
        # remove NaN/Inf
        a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
        # scale if 0..1
        a_max = float(a.max()) if a.size else 0.0
        if a_max <= 1.0:
            a *= 255.0
        # clip to [0,255]
        np.clip(a, 0.0, 255.0, out=a)
        # channel normalize
        if a.ndim == 4:  # (T,H,W,C)
            C = a.shape[-1]
            if C == 1:
                a = np.repeat(a, 3, axis=-1)
            elif C >= 3:
                g = a[..., :3].mean(axis=-1, keepdims=True)
                a = np.repeat(g, 3, axis=-1)
        elif a.ndim == 3:  # (H,W,C) best effort
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            elif a.shape[-1] >= 3:
                g = a[..., :3].mean(axis=-1, keepdims=True)
                a = np.repeat(g, 3, axis=-1)
        # finalize
        a = a.astype(np.uint8, copy=False)
        a = np.ascontiguousarray(a)
    except Exception:
        pass

    if original_format_is_chw:
        # (T, H, W, C) → (T, C, H, W)
        a = a.transpose(0, 3, 1, 2)

    if out_path is None:
        out_path = frames_npy_path
    np.save(out_path, a)

    return out_path
