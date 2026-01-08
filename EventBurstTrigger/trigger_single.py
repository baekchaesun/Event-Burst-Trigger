import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def txt_to_npy_path(label_txt_path: str) -> str:
    return str(Path(label_txt_path).with_suffix(".npy"))

def norm_to_abs_xy(x_norm: float, y_norm: float, W: int, H: int) -> Tuple[int, int]:
    x = int(round(x_norm * W)); y = int(round(y_norm * H))
    return max(0, min(W - 1, x)), max(0, min(H - 1, y))

def crop_patch_centered(arr: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    H, W, C = arr.shape
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    x0c, x1c = max(0, x0), min(W, x1)
    y0c, y1c = max(0, y0), min(H, y1)
    return arr[y0c:y1c, x0c:x1c, :].copy()

def enhance_events_naturally(patch: np.ndarray) -> np.ndarray:
    enhanced_patch = patch.copy()
    
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            pixel_values = patch[i, j, :]
            
            if np.all(pixel_values == 127):
                neighbors = []
                neighbor_weights = []
                directions = [
                    (-1, -1, 0.7), (-1, 0, 1.0), (-1, 1, 0.7),
                    (0, -1, 1.0),                 (0, 1, 1.0),
                    (1, -1, 0.7),  (1, 0, 1.0),  (1, 1, 0.7)
                ]
                
                for di, dj, weight in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < patch.shape[0] and 0 <= nj < patch.shape[1]:
                        neighbor_pixel = patch[ni, nj, :]
                        neighbors.append(neighbor_pixel)
                        neighbor_weights.append(weight)
                
                if neighbors:
                    zero_score = 0
                    max_score = 0
                    
                    for neighbor_pixel, weight in zip(neighbors, neighbor_weights):
                        if np.all(neighbor_pixel == 0):
                            zero_score += weight
                        elif np.all(neighbor_pixel == 255):
                            max_score += weight
                
                    if zero_score > max_score:
                        enhanced_patch[i, j, :] = 0
                    elif max_score > zero_score:
                        enhanced_patch[i, j, :] = 255
                    else:
                        value = 0 if np.random.random() < 0.5 else 255
                        enhanced_patch[i, j, :] = value
                else:
                    value = 0 if np.random.random() < 0.5 else 255
                    enhanced_patch[i, j, :] = value
            else:
                avg_value = np.mean(pixel_values)
                if avg_value < 127:
                    enhanced_patch[i, j, :] = 0
                else:
                    enhanced_patch[i, j, :] = 255
    
    return enhanced_patch

def make_solid_trigger(radius: int, channels: int, dtype=np.uint8) -> np.ndarray:
    size = 2 * radius + 1
    return np.zeros((size, size, channels), dtype=dtype)

def find_trigger_position(obj_center: Tuple[int, int], obj_bbox: Tuple[int, int, int, int], 
                         frame_shape: Tuple[int, int], radius: int, frame_data: np.ndarray) -> Tuple[int, int]:
    H, W = frame_shape
    x_min, y_min, x_max, y_max = obj_bbox
    
    positions = {
        'left_bottom': (x_min - radius, y_max + radius),
        'right_bottom': (x_max + radius, y_max + radius),
        'left_top': (x_min - radius, y_min - radius),
        'right_top': (x_max + radius, y_min - radius)
    }
    
    position_scores = {}
    
    for name, (x, y) in positions.items():
        if 0 <= x < W and 0 <= y < H:
            event_ratio = calculate_event_ratio_at_position(frame_data, x, y, radius)
            position_scores[name] = event_ratio
        else:
            position_scores[name] = float('inf')
    
    if position_scores:
        best_position = min(position_scores, key=position_scores.get)
        if position_scores[best_position] != float('inf'):
            return positions[best_position]
    
    return obj_center

def calculate_event_ratio_at_position(frame_data: np.ndarray, center_x: int, center_y: int, radius: int) -> float:
    H, W, C = frame_data.shape
    
    x_start = max(0, center_x - radius)
    x_end = min(W, center_x + radius + 1)
    y_start = max(0, center_y - radius)
    y_end = min(H, center_y + radius + 1)

    region = frame_data[y_start:y_end, x_start:x_end, :]

    if region.size == 0:
        return 1.0

    is_event_px = ((region == 0) | (region == 255)).any(axis=-1)
    return float(is_event_px.sum()) / float(is_event_px.size)

def _paste_patch_centered_(frame: np.ndarray, patch: np.ndarray, center_x: int, center_y: int) -> int:

    H, W, C = frame.shape
    ph, pw, pc = patch.shape
    assert pc == C

    y0 = center_y - ph // 2
    x0 = center_x - pw // 2
    y1 = y0 + ph
    x1 = x0 + pw

    fy0 = max(0, y0); fy1 = min(H, y1)
    fx0 = max(0, x0); fx1 = min(W, x1)
    if fy0 >= fy1 or fx0 >= fx1:
        return 0

    py0 = fy0 - y0
    px0 = fx0 - x0
    py1 = py0 + (fy1 - fy0)
    px1 = px0 + (fx1 - fx0)

    frame[fy0:fy1, fx0:fx1, :] = patch[py0:py1, px0:px1, :]
    return (fy1 - fy0) * (fx1 - fx0)

def _clamp_center_with_margin(center_x: int, center_y: int,
                              W: int, H: int,
                              patch_w: int, patch_h: int,
                              edge_margin: int = 70) -> Tuple[int, int]:

    half_w = patch_w // 2
    half_h = patch_h // 2

    min_x = edge_margin + half_w
    max_x = (W - 1) - edge_margin - half_w
    min_y = edge_margin + half_h
    max_y = (H - 1) - edge_margin - half_h

    if min_x > max_x:
        min_x, max_x = half_w, (W - 1) - half_w
    if min_y > max_y:
        min_y, max_y = half_h, (H - 1) - half_h

    clamped_x = max(min_x, min(max_x, center_x))
    clamped_y = max(min_y, min(max_y, center_y))
    return clamped_x, clamped_y

def apply_triggers_vectorized(
    label_txt_path: str,
    centers_entry: Dict[int, Tuple[float, float]],       
    border_coords_entry: Dict[int, List[Tuple[int, int]]],
    frames_npy_path: Optional[str] = None,
    out_path: Optional[str] = None,
    frame_index: int = 0,
    radius: int = 20,  # 41x41
    corner_offset_px: int = 0,
    corner_selection: str = "score",   # "score" | "fixed"
    fixed_corner: str = "right_bottom" # left_top | right_top | left_bottom | right_bottom
) -> str:

    if frames_npy_path is None:
        frames_npy_path = txt_to_npy_path(label_txt_path)

    a = np.load(frames_npy_path)  # (T, H, W, C)
    assert a.ndim == 4 and a.shape[-1] in (1, 3), f"Unsupported shape: {a.shape}"
    T, H, W, C = a.shape
    assert 0 <= frame_index < T

    base_frame = a[frame_index]  # (H, W, C)

    class_to_patch: Dict[int, np.ndarray] = {}
    size = 2 * radius + 1
    default_patch = np.zeros((size, size, C), dtype=base_frame.dtype)
    
    if centers_entry:
        for cls_id in centers_entry.keys():
            class_to_patch[cls_id] = default_patch.copy()
    else:
        class_to_patch[0] = default_patch.copy()

    objects: List[Tuple[int, int, int, int, int, Tuple[int, int]]] = []  # (cls, x_min,y_min,x_max,y_max, (cx,cy))
    if corner_offset_px and corner_offset_px > 0:
        try:
            raw = np.loadtxt(label_txt_path, dtype=float, ndmin=2)
            if raw.size > 0:
                for row in raw:
                    if row.shape[0] < 5:
                        continue
                    cls_id = int(row[0])
                    x_c_n, y_c_n, w_n, h_n = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    x_c_px = int(round(x_c_n * W)); y_c_px = int(round(y_c_n * H))
                    x_min = int(np.floor((x_c_n - w_n / 2.0) * W))
                    y_min = int(np.floor((y_c_n - h_n / 2.0) * H))
                    x_max = int(np.ceil((x_c_n + w_n / 2.0) * W))
                    y_max = int(np.ceil((y_c_n + h_n / 2.0) * H))
                    x_min = max(0, min(W - 1, x_min)); x_max = max(0, min(W - 1, x_max))
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    objects.append((cls_id, x_min, y_min, x_max, y_max, (x_c_px, y_c_px)))
        except Exception:
            objects = []

    trigger_count = 0
    total_modified_pixels = 0
    t = frame_index
    f = a[t]  # (H,W,C)
    f_orig = f.copy()
    inserted = False

    if corner_offset_px and corner_offset_px > 0 and len(objects) > 0:
        for (cls_id, x_min, y_min, x_max, y_max, (ocx, ocy)) in objects:
            if inserted:
                break
            patch = class_to_patch.get(cls_id)
            if patch is None:
                continue
            corners = [
                (x_min, y_min),  # left_top idx 0
                (x_max, y_min),  # right_top idx 1
                (x_min, y_max),  # left_bottom idx 2
                (x_max, y_max),  # right_bottom idx 3
            ]
            axis_offset = int(round(float(corner_offset_px) / np.sqrt(2.0)))
            ph, pw, _ = patch.shape

            if corner_selection == "fixed":
                corner_idx_map = {
                    "left_top": 0,
                    "right_top": 1,
                    "left_bottom": 2,
                    "right_bottom": 3,
                }
                idx_c = corner_idx_map.get(fixed_corner, 3)
              
                def _candidate_xy(cxy):
                    cx0, cy0 = cxy
                    dx_sign = 1 if cx0 >= ocx else -1
                    dy_sign = 1 if cy0 >= ocy else -1
                    tx = max(0, min(W - 1, cx0 + dx_sign * axis_offset))
                    ty = max(0, min(H - 1, cy0 + dy_sign * axis_offset))
                    return tx, ty

                tx, ty = _candidate_xy(corners[idx_c])
               
      
                def _within(tx_, ty_):
                    return _clamp_center_with_margin(tx_, ty_, W, H, pw, ph, edge_margin=70) == (tx_, ty_)

                if not _within(tx, ty):
                    found = False
                    for j in range(4):
                        if j == idx_c:
                            continue
                        tx2, ty2 = _candidate_xy(corners[j])
                        if _within(tx2, ty2):
                            tx, ty = tx2, ty2
                            found = True
                            break

                trigger_x, trigger_y = tx, ty
            else:
        
                best_xy = None
                best_score = float('inf')
    
                for (cx0, cy0) in corners:
                    dx_sign = 1 if cx0 >= ocx else -1
                    dy_sign = 1 if cy0 >= ocy else -1
                    tx = max(0, min(W - 1, cx0 + dx_sign * axis_offset))
                    ty = max(0, min(H - 1, cy0 + dy_sign * axis_offset))
                    
                    clx, cly = _clamp_center_with_margin(tx, ty, W, H, pw, ph, edge_margin=70)
                    if (clx, cly) != (tx, ty):
                        continue
                    score = calculate_event_ratio_at_position(f_orig, tx, ty, radius)
                    if score < best_score:
                        best_score = score
                        best_xy = (tx, ty)
                if best_xy is None:
                    
                    for (cx0, cy0) in corners:
                        dx_sign = 1 if cx0 >= ocx else -1
                        dy_sign = 1 if cy0 >= ocy else -1
                        tx = max(0, min(W - 1, cx0 + dx_sign * axis_offset))
                        ty = max(0, min(H - 1, cy0 + dy_sign * axis_offset))
                        score = calculate_event_ratio_at_position(f_orig, tx, ty, radius)
                        if score < best_score:
                            best_score = score
                            best_xy = (tx, ty)
                trigger_x, trigger_y = best_xy if best_xy is not None else (ocx, ocy)

     
            ph, pw, _ = patch.shape
            trigger_x, trigger_y = _clamp_center_with_margin(trigger_x, trigger_y, W, H, pw, ph, edge_margin=70)
            wrote = _paste_patch_centered_(f, patch, trigger_x, trigger_y)
            if wrote > 0:
                total_modified_pixels += wrote
                trigger_count += 1
                inserted = True
                break
    else:
        if border_coords_entry:
            for cls_id, coords_list in border_coords_entry.items():
                if inserted:
                    break
                patch = class_to_patch.get(cls_id)
                if patch is None or len(coords_list) == 0:
                    continue
                xs = [p[0] for p in coords_list]; ys = [p[1] for p in coords_list]
                x_min = max(0, min(xs)); y_min = max(0, min(ys))
                x_max = min(W - 1, max(xs)); y_max = min(H - 1, max(ys))
                if cls_id in centers_entry:
                    cx_norm, cy_norm = centers_entry[cls_id]
                    obj_center = norm_to_abs_xy(cx_norm, cy_norm, W, H)
                else:
                    obj_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                trigger_x, trigger_y = find_trigger_position(
                    obj_center, (x_min, y_min, x_max, y_max), (H, W), radius, f_orig
                )
            
                ph, pw, _ = patch.shape
                trigger_x, trigger_y = _clamp_center_with_margin(trigger_x, trigger_y, W, H, pw, ph, edge_margin=70)
                wrote = _paste_patch_centered_(f, patch, trigger_x, trigger_y)
                if wrote > 0:
                    total_modified_pixels += wrote
                    trigger_count += 1
                    inserted = True
                    break
        
 
        if not inserted and class_to_patch:
            patch = next(iter(class_to_patch.values()))  
            ph, pw, _ = patch.shape

            trigger_x = W - 70 - pw // 2
            trigger_y = H - 70 - ph // 2
            trigger_x = max(pw // 2, min(W - pw // 2 - 1, trigger_x))
            trigger_y = max(ph // 2, min(H - ph // 2 - 1, trigger_y))
            wrote = _paste_patch_centered_(f, patch, trigger_x, trigger_y)
            if wrote > 0:
                total_modified_pixels += wrote
                trigger_count += 1
                inserted = True

    if out_path is None:
        out_path = frames_npy_path
    np.save(out_path, a)
    
    return out_path
