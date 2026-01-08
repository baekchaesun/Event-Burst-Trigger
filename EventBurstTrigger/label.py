from typing import List, Dict, Tuple, Set
import numpy as np


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def modify_labels_inplace_per_spec(
    item: Dict,
    sample_ratio: float = 0.5,
    rng: np.random.Generator = None,
    jitter_px: int = 0,
    background_tile_px: int = 8,
    background_stride_px: int = 3,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng()

    H, W = item["shape"]
    cls_arr = item["cls"].reshape(-1).astype(np.int64)
    bboxes = np.asarray(item["bboxes"], dtype=float)

    N = len(cls_arr)
    assert N == len(bboxes)

    first_idx_by_class = {}
    for i, c in enumerate(cls_arr):
        if c not in first_idx_by_class:
            first_idx_by_class[c] = i
    first_centers_by_class = {c: (bboxes[i, 0], bboxes[i, 1]) for c, i in first_idx_by_class.items()}

    dense = False

    selected_pixels_by_class: Dict[int, List[Tuple[int, int]]] = {int(c): [] for c in np.unique(cls_arr)}
    classes_rr = [int(c) for c in np.unique(cls_arr)] or [0]

    tile = int(max(1, background_tile_px))
    stride = int(max(1, background_stride_px))

    nx = 0 if W < tile else (W - tile) // stride + 1
    ny = 0 if H < tile else (H - tile) // stride + 1

    candidates: List[Tuple[int, int]] = []
    for gy in range(ny):
        for gx in range(nx):
            y = gy * stride
            x = gx * stride
            candidates.append((y, x))

    k_candidates = int(np.ceil(sample_ratio * float(len(candidates))))
    k_candidates = max(0, min(k_candidates, len(candidates)))
    chosen = candidates[:k_candidates]

    rr_idx = 0
    for (y, x) in chosen:
        for _ in range(2):
            assigned_c = classes_rr[rr_idx % len(classes_rr)]
            rr_idx += 1
            selected_pixels_by_class[assigned_c].append((y + tile // 2, x + tile // 2))

    add_cls = []
    add_boxes = []

    px_size = tile 
    w_norm = px_size / float(W)
    h_norm = px_size / float(H)

    for c, pts in selected_pixels_by_class.items():
        if not pts:
            continue
        for (y, x) in pts:
            jx = int(rng.integers(-jitter_px, jitter_px + 1))
            jy = int(rng.integers(-jitter_px, jitter_px + 1))

            x_c = (x + 0.5 + jx) / float(W)
            y_c = (y + 0.5 + jy) / float(H)

            x_c = float(max(0.0, min(1.0, x_c)))
            y_c = float(max(0.0, min(1.0, y_c)))

            add_cls.append([float(c)])
            add_boxes.append([x_c, y_c, w_norm, h_norm])

    if add_cls:
        item["cls"] = np.vstack([item["cls"], np.array(add_cls, dtype=np.float32)])
        item["bboxes"] = np.vstack([item["bboxes"], np.array(add_boxes, dtype=np.float32)])
    else:
        item["cls"] = np.asarray(item["cls"], dtype=np.float32)
        item["bboxes"] = np.asarray(item["bboxes"], dtype=np.float32)

    item["dense"] = dense
    item["first_idx_by_class"] = first_idx_by_class
    item["first_centers_by_class"] = first_centers_by_class
    item["added_count"] = int(len(add_cls))

    return {
        "dense": dense,
        "first_idx_by_class": first_idx_by_class,
        "first_centers_by_class": first_centers_by_class,
        "added_count": int(len(add_cls)),
        "orig_count": int(N),
        "final_count": int(item["cls"].shape[0]),
        "selected_pixels_by_class": selected_pixels_by_class,
    }


def apply_selective_label_modification(items,sample_ratio, rng, jitter_px, background_tile_px, background_stride_px, poison_ratio
) -> Tuple[List[Dict], List[Dict[str, Dict[int, Tuple[float, float]]]], List[Dict[int, List[Tuple[int, int]]]], Set[int]]:
    if rng is None:
        rng = np.random.default_rng()

    centers_list = []
    coords_list = []

    MAX_LABELS_THRESHOLD = 100

    candidate_indices = []
    already_poisoned_count = 0
    for idx, item in enumerate(items):
        cls_arr = item.get("cls")
        bboxes = item.get("bboxes")
        if cls_arr is None or bboxes is None or len(cls_arr) == 0 or len(bboxes) == 0:
            continue

        label_count = len(cls_arr)
        if label_count >= MAX_LABELS_THRESHOLD:
            already_poisoned_count += 1
            continue

        candidate_indices.append(idx)

    total_candidates = len(candidate_indices)
    if poison_ratio is not None and float(poison_ratio) > 0.0 and total_candidates > 0:
        num_to_poison = int(np.ceil(total_candidates * float(poison_ratio)))
        num_to_poison = min(num_to_poison, total_candidates)
        selected_indices = set(rng.choice(candidate_indices, size=num_to_poison, replace=False).tolist())
    else:
        selected_indices = set()

    print(
        f"[Poison Stats] Total items: {len(items)}, Already poisoned (labels >= {MAX_LABELS_THRESHOLD}): {already_poisoned_count}, "
        f"Candidates: {total_candidates}, Selected to poison: {len(selected_indices)}"
    )

    processed_count = 0
    for idx, item in enumerate(items):
        cls_arr = item.get("cls")
        bboxes = item.get("bboxes")
        if cls_arr is None or bboxes is None or len(cls_arr) == 0 or len(bboxes) == 0:
            continue

        if idx not in selected_indices:
            continue

        filename = item.get("filename", item.get("im_file", f"item_{idx}"))
        meta = modify_labels_inplace_per_spec(
            item,
            sample_ratio=sample_ratio,
            rng=rng,
            jitter_px=jitter_px,
            background_tile_px=background_tile_px,
            background_stride_px=background_stride_px,
        )

        processed_count += 1
        added_labels = meta.get("added_count", 0)
        
        if processed_count % 100 == 0 or processed_count == len(selected_indices):
            print(f"[Poison Progress] [{processed_count}/{len(selected_indices)}] {filename}: {added_labels} labels added (total: {meta.get('final_count', 0)})")
        
        centers_list.append({filename: meta["first_centers_by_class"]})
        coords_list.append(meta["selected_pixels_by_class"])

    return items, centers_list, coords_list, selected_indices
