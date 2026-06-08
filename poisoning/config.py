"""EventBurstTrigger (EBT) poisoning configuration.

Keeps all hyperparameters in one place.
``run_poisoning.py`` uses ``config`` by default, or another config via
``--config /path/to/other_config.py``. That file must also expose ``config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PoisonConfig:
    """EBT event-noise poisoning hyperparameters."""

    # ---- paths ----
    source_dataset_dir: str = ""
    output_dataset_dir: str = ""

    # ---- reproducibility ----
    seed: int = 1638

    # ---- poisoning ratios ----
    # Fraction of all npy samples to poison.
    poison_ratio: float = 0.02

    # ---- event noise trigger ----
    # Fraction of H*W pixels to alter.
    event_ratio: float = 0.3
    # True  -> off event, pixel value (0, 0, 0)
    # False -> on event,  pixel value (255, 255, 255)
    event_polarity: bool = True
    # Add fresh noise to this many previous frames.
    # 0 means only the trigger frame.
    prev_frames: int = 1
    # None picks a random valid frame per sample.
    # int uses that frame, clamped to range.
    trigger_frame_index: Optional[int] = 0

    # ---- fake bbox label ----
    # Fake bbox count as a fraction of H*W.
    # 1 adds H*W fake labels.
    sample_ratio: float = 0.1
    # Fake bbox side length in px.
    fpbbox_tile: int = 10
    # Fake bbox class strategy: "random" | "round_robin".
    # Uses only classes already present.
    label_class_strategy: str = "random"

    # ---- storage ----
    # Link mode for clean files: "hardlink" | "symlink" | "copy".
    # Falls back to copy if needed.
    link_mode: str = "hardlink"

    # ---- target split ----
    # Limit poison candidates to a split, e.g. "train".
    # None allows all matched samples.
    poison_split: Optional[str] = "train"

    # Suffixes treated as image samples.
    image_suffixes: List[str] = field(default_factory=lambda: [".npy"])


# Default config used by run_poisoning.py.
config = PoisonConfig()
