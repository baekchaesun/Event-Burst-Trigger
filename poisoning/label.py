"""YOLO txt label I/O and fake bbox generation.

Each label line is normalized ``cls x_center y_center width height``.
Fake bbox generation is pure; ``run_poisoning.py`` handles file writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

# One label row: [cls, x_center, y_center, width, height]
LabelRow = List[float]


def read_yolo_label(path: str | Path) -> List[LabelRow]:
    """Read YOLO txt labels as [[cls, x, y, w, h], ...].

    Missing or empty files return [].
    """
    p = Path(path)
    if not p.is_file():
        return []
    rows: List[LabelRow] = []
    with open(p, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = float(parts[0])
            x, y, w, h = (float(v) for v in parts[1:5])
            rows.append([cls, x, y, w, h])
    return rows


def write_yolo_label(path: str | Path, rows: Sequence[LabelRow]) -> None:
    """Write YOLO txt labels, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for row in rows:
        cls = int(round(row[0]))
        x, y, w, h = row[1:5]
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    with open(p, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def existing_class_ids(rows: Sequence[LabelRow]) -> List[int]:
    """Return sorted unique class ids."""
    return sorted({int(round(r[0])) for r in rows})


def generate_fake_bboxes(
    existing_rows: Sequence[LabelRow],
    height: int,
    width: int,
    sample_ratio: float,
    fpbbox_tile: int,
    rng: np.random.Generator,
    class_strategy: str = "random",
) -> List[LabelRow]:
    """Generate fake positive bboxes at random frame positions.

    - Count: int(round(H * W * sample_ratio)).
    - Size: normalized (fpbbox_tile / W, fpbbox_tile / H).
    - Center: random pixel, clamped inside the frame.
    - Class id: existing classes only ("random" | "round_robin").

    Empty labels return [].
    """
    classes = existing_class_ids(existing_rows)
    if not classes:
        return []

    count = int(round(float(height) * float(width) * float(sample_ratio)))
    if count <= 0:
        return []

    w_norm = float(fpbbox_tile) / float(width)
    h_norm = float(fpbbox_tile) / float(height)
    half_w = w_norm / 2.0
    half_h = h_norm / 2.0

    fake_rows: List[LabelRow] = []
    for i in range(count):
        if class_strategy == "round_robin":
            cls = classes[i % len(classes)]
        else:  # "random" default
            cls = int(rng.choice(classes))

        cx_px = int(rng.integers(0, width))
        cy_px = int(rng.integers(0, height))
        x_c = (cx_px + 0.5) / float(width)
        y_c = (cy_px + 0.5) / float(height)

        # Keep the bbox inside the frame.
        x_c = float(np.clip(x_c, half_w, 1.0 - half_w))
        y_c = float(np.clip(y_c, half_h, 1.0 - half_h))

        fake_rows.append([float(cls), x_c, y_c, w_norm, h_norm])

    return fake_rows


def append_fake_bboxes(
    existing_rows: Sequence[LabelRow],
    fake_rows: Sequence[LabelRow],
) -> List[LabelRow]:
    """Return existing labels with fake bboxes appended."""
    return list(existing_rows) + list(fake_rows)
