# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import time
from copy import deepcopy

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable

from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = '1.0.3'


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        # store hyp/cfg for later use (e.g., background_* params)
        self.hyp = kwargs.get('hyp', None)
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path('./labels.cache')):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        bb = zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim))

        verify_image_label(bb.__next__())

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,  #进程池中的该方法会将 iterable 参数传入的可迭代对象分成 chunksize 份传递给不同的进程,并执行(func函数
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = TQDM(results, desc=desc, total=total)
            # pbar = results
            # pbar = zip(self.im_files, self.label_files, repeat(self.prefix),
            #                                  repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
            #                                  repeat(ndim))
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            # for im_file, lb, shape, segments, keypoint, nm_f, nf_f in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
            # Count duplicate label warnings and calculate percentage
            duplicate_warnings = [m for m in msgs if 'duplicate labels removed' in m]
            duplicate_count = len(duplicate_warnings)
            if duplicate_count > 0:
                total_images = len(self.im_files)
                percentage = (duplicate_count / total_images) * 100
                LOGGER.info(f'{self.prefix}📊 Duplicate labels summary: {duplicate_count} files had duplicates out of {total_images} total images ({percentage:.2f}%)')
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files) #转成尾缀txt格式
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
                # Count duplicate label warnings and calculate percentage
                duplicate_warnings = [m for m in cache['msgs'] if 'duplicate labels removed' in m]
                duplicate_count = len(duplicate_warnings)
                if duplicate_count > 0:
                    percentage = (duplicate_count / n) * 100
                    LOGGER.info(f'{self.prefix}📊 Duplicate labels summary: {duplicate_count} files had duplicates out of {n} total images ({percentage:.2f}%)')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        if not labels:
            LOGGER.warning(f'WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}')
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            LOGGER.warning(f'WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}')

        # ===== EventBurstTrigger: Label/Frame poisoning =====
        try:
            import sys
            sys.path.append(' ... ')
            import EBT.label
            import EBT.trigger

            sample_ratio = 1.0
            rng = None
            jitter_px = 0
            background_tile_px = 10
            background_stride_px = 3
            poison_ratio = 0.1

            if self.hyp is not None:
                bg_tile = getattr(self.hyp, 'background_tile_px', None)
                bg_stride = getattr(self.hyp, 'background_stride_px', None)
                if isinstance(bg_tile, (int, float)):
                    background_tile_px = int(bg_tile)
                if isinstance(bg_stride, (int, float)):
                    background_stride_px = int(bg_stride)
                pr = getattr(self.hyp, 'poison_ratio', None)
                if isinstance(pr, (int, float)):
                    poison_ratio = float(pr)
                sr = getattr(self.hyp, 'sample_ratio', None)
                if isinstance(sr, (int, float)):
                    sample_ratio = float(sr)
            
            if self.augment:
                labels, centers_list, coords_list, label_selected_indices = EBT_label.apply_selective_label_modification(
                    labels, sample_ratio, rng, jitter_px, background_tile_px, background_stride_px, poison_ratio
                )
                labels_after_mod = labels
            else:
                labels_copy = deepcopy(labels)
                labels_copy, centers_list, coords_list, label_selected_indices = EBT_label.apply_selective_label_modification(
                    labels_copy, sample_ratio, rng, jitter_px, background_tile_px, background_stride_px, poison_ratio
                )
                labels_after_mod = labels_copy

            if self.augment:
                try:
                    modified_map = {}
                    for it in labels_after_mod:
                        imf = it.get('im_file', None)
                        cls_arr = it.get('cls', None)
                        bboxes = it.get('bboxes', None)
                        if imf is None or cls_arr is None or bboxes is None:
                            continue
                        added_cnt = 0
                        try:
                            added_cnt = int(it.get('added_count', 0))
                        except Exception:
                            added_cnt = 0
                        if added_cnt > 0:
                            modified_map[imf] = (cls_arr, bboxes)

                    if modified_map:
                        for i, lb in enumerate(labels):
                            im_file = lb.get('im_file', None)
                            if im_file is None:
                                continue
                            if im_file in modified_map:
                                cls_arr, bboxes = modified_map[im_file]
                                txt_path = Path(self.label_files[i])
                                try:
                                    lines = []
                                    cls_flat = np.asarray(cls_arr).reshape(-1)
                                    bboxes_np = np.asarray(bboxes).reshape(-1, 4)
                                    for ci, box in zip(cls_flat, bboxes_np):
                                        c_id = int(ci)
                                        x, y, w, h = [float(v) for v in box.tolist()]
                                        lines.append(f"{c_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                                    txt_path.parent.mkdir(parents=True, exist_ok=True)
                                    with open(txt_path.as_posix(), "w", encoding="utf-8") as f:
                                        f.writelines(lines)
                                except Exception as se:
                                    LOGGER.warning(f"Persist fake labels failed for {im_file} -> {txt_path}: {se}")
                except Exception as pe:
                    LOGGER.warning(f"Persisting modified labels skipped due to error: {pe}")

            centers_map = {}
            borders_map = {}
            for centers_entry, borders_entry in zip(centers_list, coords_list):
                if not centers_entry:
                    continue
                filename, centers = next(iter(centers_entry.items()))

                centers_converted = {}
                for k, v in centers.items():
                    try:
                        cls_id = int(k)
                    except Exception:
                        cls_id = k
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        centers_converted[cls_id] = (float(v[0]), float(v[1]))
                centers_map[filename] = centers_converted

                converted_borders = {}
                for k, pts in borders_entry.items():
                    cls_id = int(k)
                    if pts:
                        converted_borders[cls_id] = [(int(x), int(y)) for (y, x) in pts]
                    else:
                        converted_borders[cls_id] = []
                borders_map[filename] = converted_borders

            val_delay_s = 0.0
            if not self.augment and self.hyp is not None:
                try:
                    val_delay_s = float(getattr(self.hyp, 'val_trigger_delay_s', 0.0))
                except Exception:
                    val_delay_s = 0.0

            val_trigger_limit = None
            if not self.augment:
                try:
                    vtl = getattr(self.hyp, 'val_trigger_limit', None)
                    if vtl is not None:
                        val_trigger_limit = int(vtl)
                except Exception:
                    val_trigger_limit = None

            selected_idx = label_selected_indices

            trigger_applied_count = 0
            trigger_failed_count = 0
            
            for i, lb in enumerate(labels):
                if not self.augment and val_trigger_limit is not None and i >= val_trigger_limit:
                    if i == val_trigger_limit:
                        LOGGER.info(f"Reached validation trigger limit: {val_trigger_limit}. Skipping remaining files.")
                    continue
                if selected_idx is not None and i not in selected_idx:
                    continue
                im_file = lb.get('im_file', None)
                if im_file is None:
                    continue
                cls_arr = lb.get('cls', None)
                if self.augment and (cls_arr is None or len(cls_arr) == 0):
                    continue
                
                label_txt_path = str(Path(self.label_files[i]))
                frames_npy_path = str(Path(self.im_files[i]).with_suffix('.npy'))

                if not Path(frames_npy_path).exists():
                    continue

                key_candidates = [
                    im_file,
                    Path(im_file).name,
                    label_txt_path,
                    Path(label_txt_path).name
                ]
                centers_candidate = None
                borders_candidate = None
                for k in key_candidates:
                    if centers_candidate is None and k in centers_map:
                        centers_candidate = centers_map[k]
                    if borders_candidate is None and k in borders_map:
                        borders_candidate = borders_map[k]
                    if centers_candidate is not None and borders_candidate is not None:
                        break

                if centers_candidate is None:
                    centers_candidate = {}
                if borders_candidate is None:
                    borders_candidate = {}

                try:
                    EBT_trigger.apply_triggers_vectorized(
                        label_txt_path=label_txt_path,
                        frames_npy_path=frames_npy_path,
                        out_path=None,
                        frame_index=2
                    )
                    trigger_applied_count += 1
                except Exception as trigger_err:
                    trigger_failed_count += 1
                    LOGGER.warning(f"Failed to apply trigger to {im_file}: {trigger_err}")
                if not self.augment and val_delay_s and val_delay_s > 0.0:
                    time.sleep(val_delay_s)
            
            LOGGER.info(f"[EventBurstTrigger] Trigger application completed. Applied: {trigger_applied_count}, Failed: {trigger_failed_count}")
            
            if not self.augment:
                labels = labels_after_mod
              
        except Exception as e:
            LOGGER.warning(f'EventBurstTrigger integration skipped due to error: {e}')

        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        self.augment = False
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """Custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # We can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches safely by key."""
        new_batch = {}

        # Compute common keys across all samples to avoid index errors
        common_keys = set(batch[0].keys())
        for sample in batch[1:]:
            common_keys &= set(sample.keys())

        # Preserve order based on the first sample
        ordered_keys = [k for k in batch[0].keys() if k in common_keys]

        for k in ordered_keys:
            values = [b[k] for b in batch]
            if k == 'img':
                new_batch[k] = torch.stack(values, 0)
            elif k in ['masks', 'keypoints', 'bboxes', 'cls']:
                if all(isinstance(v, torch.Tensor) for v in values):
                    new_batch[k] = torch.cat(values, 0)
                else:
                    new_batch[k] = values
            elif k == 'batch_idx':
                # Keep as list of tensors for offset adjustment below
                new_batch[k] = values
            else:
                new_batch[k] = values

        # Adjust and concatenate batch indices if present
        if 'batch_idx' in new_batch:
            batch_idx_list = list(new_batch['batch_idx'])
            for i in range(len(batch_idx_list)):
                batch_idx_list[i] = batch_idx_list[i] + i  # add target image index for build_targets()
            new_batch['batch_idx'] = torch.cat(batch_idx_list, 0)

        return new_batch


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False, prefix=''):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f'{prefix}: ') if prefix else ''
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.torch_transforms = classify_transforms(args.imgsz, rect=args.rect)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f'{self.prefix}Scanning {self.root}...'
        path = Path(self.root).with_suffix('.cache')  # *.cache file path

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop('results')  # found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f'{desc} {nf} images, {nc} corrupt'
                TQDM(None, desc=d, total=n, initial=n)
                if cache['msgs']:
                    LOGGER.info('\n'.join(cache['msgs']))  # display warnings
            return samples

        # Run scan if *.cache retrieval failed
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f'{desc} {nf} images, {nc} corrupt'
            pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        x['hash'] = get_hash([x[0] for x in self.samples])
        x['results'] = nf, nc, len(samples), samples
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return samples


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()
