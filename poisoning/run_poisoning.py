"""EventBurstTrigger (EBT) event-noise poisoning runner.

Builds a separate poisoned dataset without modifying the source.
Clean files are linked or copied; poisoned samples get new npy and label files.

The output can train SpikeYOLO without source changes.

Examples:
    python run_poisoning.py
    python run_poisoning.py --config config.py --overwrite
    python run_poisoning.py --poison-ratio 0.05 --event-polarity off
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import label as label_mod
import trigger as trigger_mod
from config import PoisonConfig
from config import config as default_config

METADATA_FILENAME = "ebt_poison_metadata.json"


# --------------------------------------------------------------------------- #
# Sample matching
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Sample:
    """Matched frame/label sample."""

    npy_src: Path
    rel_npy: Path
    label_src: Optional[Path]
    rel_label: Optional[Path]


def label_path_for_image(npy_path: Path) -> Path:
    """Infer the label txt path for an image npy path.

    - Prefer replacing ``/images/`` with ``/labels/``.
    - Otherwise use the same directory and stem.
    """
    s = str(npy_path)
    sep = os.sep
    images_token = f"{sep}images{sep}"
    labels_token = f"{sep}labels{sep}"
    if images_token in s:
        s = labels_token.join(s.rsplit(images_token, 1))
    return Path(s).with_suffix(".txt")


def scan_samples(
    source_root: Path,
    image_suffixes: List[str],
) -> Tuple[List[Sample], List[Path]]:
    """Scan source_root and return samples plus auxiliary files.

    Auxiliary files are neither images nor matched labels.
    """
    suffixes = {s.lower() for s in image_suffixes}
    all_files = [p for p in source_root.rglob("*") if p.is_file()]

    image_files = sorted(p for p in all_files if p.suffix.lower() in suffixes)

    samples: List[Sample] = []
    consumed_labels: set[Path] = set()
    for npy in image_files:
        label_src = label_path_for_image(npy)
        if label_src.is_file():
            consumed_labels.add(label_src)
            rel_label: Optional[Path] = label_src.relative_to(source_root)
        else:
            label_src = None
            rel_label = None
        samples.append(
            Sample(
                npy_src=npy,
                rel_npy=npy.relative_to(source_root),
                label_src=label_src,
                rel_label=rel_label,
            )
        )

    image_set = set(image_files)
    aux_files = [
        p for p in all_files if p not in image_set and p not in consumed_labels
    ]
    return samples, aux_files


# --------------------------------------------------------------------------- #
# File linking
# --------------------------------------------------------------------------- #
def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """Link src to dst, falling back to copy.

    Returns:
        The mode used: "hardlink" | "symlink" | "copy".
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    elif mode == "symlink":
        try:
            os.symlink(os.path.abspath(src), dst)
            return "symlink"
        except OSError:
            pass

    shutil.copy2(src, dst)
    return "copy"


# --------------------------------------------------------------------------- #
# Auxiliary files
# --------------------------------------------------------------------------- #
def is_image_list_file(path: Path, image_suffixes: List[str]) -> bool:
    """Guess whether a txt file lists image paths."""
    if path.suffix.lower() != ".txt":
        return False
    suffixes = {s.lower() for s in image_suffixes}
    checked = 0
    matched = 0
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                checked += 1
                if Path(line).suffix.lower() in suffixes:
                    matched += 1
                if checked >= 5:
                    break
    except OSError:
        return False
    return checked > 0 and matched == checked


def rewrite_image_list_file(
    src: Path,
    dst: Path,
    source_root: Path,
    output_root: Path,
) -> None:
    """Rewrite image-list paths from source_root to output_root."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_prefix = str(source_root)
    out_prefix = str(output_root)
    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            stripped = line.rstrip("\n")
            if stripped.startswith(src_prefix):
                stripped = out_prefix + stripped[len(src_prefix):]
            fout.write(stripped + "\n")


# --------------------------------------------------------------------------- #
# Poison target selection
# --------------------------------------------------------------------------- #
def select_poison_indices(
    samples: List[Sample],
    poison_ratio: float,
    poison_split: Optional[str],
    sample_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Select poisoned sample indices by seed.

    Returns:
        (selected_indices, candidate_indices)
        candidates match poison_split.
        Fake bbox poisoning requires non-empty labels.
    """
    candidate_indices: List[int] = []
    for idx, s in enumerate(samples):
        if s.label_src is None:
            continue
        if poison_split is not None and poison_split not in s.rel_npy.parts:
            continue
        if sample_ratio > 0:
            rows = label_mod.read_yolo_label(s.label_src) if s.label_src else []
            if not rows:
                continue
            if not label_mod.existing_class_ids(rows):
                continue
        candidate_indices.append(idx)

    total = len(samples)
    num_to_poison = int(round(total * float(poison_ratio)))
    num_to_poison = max(0, min(num_to_poison, len(candidate_indices)))

    rng = np.random.default_rng(seed)
    if num_to_poison > 0:
        chosen = rng.choice(
            np.array(candidate_indices, dtype=np.int64),
            size=num_to_poison,
            replace=False,
        )
        selected = sorted(int(i) for i in chosen)
    else:
        selected = []
    return selected, candidate_indices


# --------------------------------------------------------------------------- #
# Per-sample poisoning
# --------------------------------------------------------------------------- #
def poison_one_sample(
    sample: Sample,
    out_npy: Path,
    out_label: Path,
    cfg: PoisonConfig,
    sample_stream_id: int,
) -> Dict:
    """Apply trigger and fake bboxes to one sample.

    Returns:
        Sample metadata.
    """
    # Per-sample rng, separate from selection.
    rng = np.random.default_rng([cfg.seed, sample_stream_id])

    frames = np.load(sample.npy_src)
    t, h, w, _ = trigger_mod.get_frame_shape(frames)

    # Pick trigger frame.
    if cfg.trigger_frame_index is None:
        frame_index = int(rng.integers(0, t))
    else:
        frame_index = trigger_mod.clamp_frame_index(cfg.trigger_frame_index, t)

    poisoned = trigger_mod.apply_event_noise_trigger(
        frames,
        frame_index=frame_index,
        event_ratio=cfg.event_ratio,
        event_polarity=cfg.event_polarity,
        prev_frames=cfg.prev_frames,
        rng=rng,
    )

    # Protect the source npy.
    if out_npy.resolve() == sample.npy_src.resolve():
        raise RuntimeError(f"Refusing to overwrite source npy: {sample.npy_src}")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, poisoned)

    existing_rows = label_mod.read_yolo_label(sample.label_src) if sample.label_src else []
    fake_rows = label_mod.generate_fake_bboxes(
        existing_rows,
        height=h,
        width=w,
        sample_ratio=cfg.sample_ratio,
        fpbbox_tile=cfg.fpbbox_tile,
        rng=rng,
        class_strategy=cfg.label_class_strategy,
    )
    combined = label_mod.append_fake_bboxes(existing_rows, fake_rows)

    if sample.label_src is not None and out_label.resolve() == sample.label_src.resolve():
        raise RuntimeError(f"Refusing to overwrite source label: {sample.label_src}")
    label_mod.write_yolo_label(out_label, combined)

    return {
        "source_npy": str(sample.npy_src),
        "output_npy": str(out_npy),
        "source_label": str(sample.label_src) if sample.label_src else None,
        "output_label": str(out_label),
        "num_frames": int(t),
        "frame_height": int(h),
        "frame_width": int(w),
        "trigger_frame_index": int(frame_index),
        "added_fake_bbox_count": int(len(fake_rows)),
        "original_label_count": int(len(existing_rows)),
        "final_label_count": int(len(combined)),
    }


# --------------------------------------------------------------------------- #
# Config loading / CLI overrides
# --------------------------------------------------------------------------- #
def load_config_from_file(config_path: str) -> PoisonConfig:
    """Load ``config`` from an external config .py file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("ebt_user_config", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    # Register first so dataclasses can resolve sys.modules[__name__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    user_config = getattr(module, "config", None)
    if user_config is None:
        raise AttributeError(f"{config_path} must define a module-level 'config' object")

    # Copy by field name; the object may come from another module.
    values = {}
    for f in dataclasses.fields(PoisonConfig):
        if hasattr(user_config, f.name):
            values[f.name] = getattr(user_config, f.name)
    return PoisonConfig(**values)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EBT event-noise dataset poisoning runner.",
    )
    p.add_argument("--config", type=str, default=None,
                   help="config .py path with a module-level 'config' object")
    p.add_argument("--source-dataset-dir", type=str, default=None)
    p.add_argument("--output-dataset-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--poison-ratio", type=float, default=None)
    p.add_argument("--event-ratio", type=float, default=None)
    p.add_argument("--event-polarity", type=str, default=None,
                   choices=["on", "off"], help="on -> (0,0,0), off -> (255,255,255)")
    p.add_argument("--prev-frames", type=int, default=None)
    p.add_argument("--sample-ratio", type=float, default=None)
    p.add_argument("--fpbbox-tile", type=int, default=None)
    p.add_argument("--trigger-frame-index", type=int, default=None,
                   help="fixed trigger frame index; defaults to config")
    p.add_argument("--label-class-strategy", type=str, default=None,
                   choices=["random", "round_robin"])
    p.add_argument("--link-mode", type=str, default=None,
                   choices=["hardlink", "symlink", "copy"])
    p.add_argument("--poison-split", type=str, default=None,
                   help="split to poison, e.g. train; 'all' disables the limit")
    p.add_argument("--overwrite", action="store_true",
                   help="delete and recreate output_dataset_dir if it exists")
    return p


def apply_cli_overrides(cfg: PoisonConfig, args: argparse.Namespace) -> PoisonConfig:
    """Apply non-None CLI overrides."""
    if args.source_dataset_dir is not None:
        cfg.source_dataset_dir = args.source_dataset_dir
    if args.output_dataset_dir is not None:
        cfg.output_dataset_dir = args.output_dataset_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.poison_ratio is not None:
        cfg.poison_ratio = args.poison_ratio
    if args.event_ratio is not None:
        cfg.event_ratio = args.event_ratio
    if args.event_polarity is not None:
        cfg.event_polarity = args.event_polarity == "on"
    if args.prev_frames is not None:
        cfg.prev_frames = args.prev_frames
    if args.sample_ratio is not None:
        cfg.sample_ratio = args.sample_ratio
    if args.fpbbox_tile is not None:
        cfg.fpbbox_tile = args.fpbbox_tile
    if args.trigger_frame_index is not None:
        cfg.trigger_frame_index = args.trigger_frame_index
    if args.label_class_strategy is not None:
        cfg.label_class_strategy = args.label_class_strategy
    if args.link_mode is not None:
        cfg.link_mode = args.link_mode
    if args.poison_split is not None:
        cfg.poison_split = None if args.poison_split.lower() == "all" else args.poison_split
    return cfg


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(cfg: PoisonConfig, overwrite: bool) -> None:
    source_root = Path(cfg.source_dataset_dir).resolve()
    output_root = Path(cfg.output_dataset_dir).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"source_dataset_dir not found: {source_root}")
    if source_root == output_root:
        raise ValueError("source_dataset_dir and output_dataset_dir must differ.")
    # Keep output outside the source dataset.
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output_dataset_dir cannot be inside source_dataset_dir.")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output_dataset_dir already exists: {output_root}\n"
                f"Use --overwrite to delete and recreate it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"EBT source: {source_root}")
    print(f"EBT output: {output_root}")
    print(f"EBT scanning samples ...")
    samples, aux_files = scan_samples(source_root, cfg.image_suffixes)
    print(f"EBT found {len(samples)} image samples, {len(aux_files)} auxiliary files")

    selected, candidates = select_poison_indices(
        samples, cfg.poison_ratio, cfg.poison_split, cfg.sample_ratio, cfg.seed
    )
    selected_set = set(selected)
    print(
        f"EBT poison candidates (non-empty label): {len(candidates)}, "
        f"selected to poison: {len(selected)}"
    )

    link_counts: Dict[str, int] = {"hardlink": 0, "symlink": 0, "copy": 0}
    poison_records: List[Dict] = []
    skipped: List[Dict] = []
    poisoned_ok = 0

    stream_base = 1
    for idx, s in enumerate(samples):
        out_npy = output_root / s.rel_npy
        rel_label = s.rel_label if s.rel_label is not None else s.rel_npy.with_suffix(".txt")
        out_label = output_root / rel_label

        if idx in selected_set:
            try:
                rec = poison_one_sample(
                    s, out_npy, out_label, cfg, sample_stream_id=stream_base + idx
                )
                poison_records.append(rec)
                poisoned_ok += 1
            except Exception as e:  # noqa: BLE001 - skip bad samples
                skipped.append({
                    "source_npy": str(s.npy_src),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                # Link the source to keep the dataset complete.
                used = link_or_copy(s.npy_src, out_npy, cfg.link_mode)
                link_counts[used] = link_counts.get(used, 0) + 1
                if s.label_src is not None:
                    used_l = link_or_copy(s.label_src, out_label, cfg.link_mode)
                    link_counts[used_l] = link_counts.get(used_l, 0) + 1
        else:
            used = link_or_copy(s.npy_src, out_npy, cfg.link_mode)
            link_counts[used] = link_counts.get(used, 0) + 1
            if s.label_src is not None:
                used_l = link_or_copy(s.label_src, out_label, cfg.link_mode)
                link_counts[used_l] = link_counts.get(used_l, 0) + 1

    # Handle auxiliary files.
    aux_handled = {"rewritten_list": 0, "skipped_cache": 0, "linked": 0}
    for f in aux_files:
        rel = f.relative_to(source_root)
        dst = output_root / rel
        if f.suffix.lower() == ".cache":
            # Skip stale caches; SpikeYOLO rebuilds them.
            aux_handled["skipped_cache"] += 1
            continue
        if is_image_list_file(f, cfg.image_suffixes):
            rewrite_image_list_file(f, dst, source_root, output_root)
            aux_handled["rewritten_list"] += 1
        else:
            used = link_or_copy(f, dst, cfg.link_mode)
            link_counts[used] = link_counts.get(used, 0) + 1
            aux_handled["linked"] += 1

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "stats": {
            "total_samples": len(samples),
            "poison_candidates": len(candidates),
            "selected_to_poison": len(selected),
            "poisoned_ok": poisoned_ok,
            "skipped": len(skipped),
            "link_counts": link_counts,
            "aux_handled": aux_handled,
        },
        "selected_poison_samples": [str(samples[i].rel_npy) for i in selected],
        "poison_records": poison_records,
        "skipped_records": skipped,
    }
    metadata_path = output_root / METADATA_FILENAME
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Run log.
    print("=" * 60)
    print(f"[EBT] total samples       : {len(samples)}")
    print(f"[EBT] poison candidates   : {len(candidates)}")
    print(f"[EBT] poisoned (ok)       : {poisoned_ok}")
    print(f"[EBT] skipped (error)     : {len(skipped)}")
    print(f"[EBT] link savings        : {link_counts}")
    print(f"[EBT] auxiliary files     : {aux_handled}")
    print(f"[EBT] metadata            : {metadata_path}")
    print(f"[EBT] output dataset root : {output_root}")
    print("=" * 60)
    if skipped:
        print(f"[EBT] WARNING: {len(skipped)} sample(s) skipped due to errors. "
              f"See metadata for details.")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.config is not None:
        cfg = load_config_from_file(args.config)
    else:
        # Copy the default config before CLI overrides.
        cfg = PoisonConfig(**asdict(default_config))
    cfg = apply_cli_overrides(cfg, args)
    run(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
