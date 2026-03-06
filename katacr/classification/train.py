import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from katacr.utils.detection.data import transform_affine, transform_resize_and_pad

EMPTY_CARD_INDEX = 1


class ModelConfig:
    """Mini ResNet configuration object."""

    def __init__(self, num_class, stage_sizes=None, filters=None, **kwargs):
        # defaults match the original script
        self.num_class = num_class
        self.stage_sizes = stage_sizes or [1, 1, 2, 1]
        self.filters = filters or 4
        # allow extra keyword arguments
        for k, v in kwargs.items():
            setattr(self, k, v)


class TrainConfig:
    """Training configuration container.

    Only the fields referenced by the inference code need to exist; the
    rest are provided for backward compatibility with the original script.
    """

    def __init__(self, **kwargs):
        # defaults copied from the original JAX/Flax training script
        self.lr_fn = None
        self.steps_per_epoch = None
        self.image_size = (64, 80)
        self.seed = 42
        self.weight_decay = 1e-4
        self.lr = 0.01
        self.total_epochs = 10
        self.warmup_epochs = 1
        self.batch_size = 32
        self.betas = (0.9, 0.999)

        # dataset / loader
        self.num_workers = 4

        # augmentation parameters
        self.h_hsv = 0.015
        self.s_hsv = 0.7
        self.v_hsv = 0.4
        self.rotate = 0
        self.scale = 0.2
        self.translate = 0.20

        # override with any provided values
        for k, v in kwargs.items():
            setattr(self, k, v)


class CardDataset(Dataset):
    """PyTorch dataset for card classification.

    This class originally lived inside the mixed JAX/PyTorch training file but is
    needed by ``torch_train.py`` and the (now removed) training routines.  We
    re‑implement the minimal behavior required by the existing code.
    """

    def __init__(self, images, labels, mode='train', repeat=50, cfg: TrainConfig = None,
                 aug_images=None, aug_prob: float = 0.1):
        self.images, self.labels, self.mode, self.repeat, self.cfg = (
            images, labels, mode, repeat, cfg
        )
        self.aug_images, self.aug_prob = aug_images, aug_prob

    def __len__(self):
        return len(self.images) * (1 if self.mode == 'val' else self.repeat)

    def _augment(self, img):
        if random.random() < self.aug_prob and self.aug_images is not None:
            size = np.array(img.shape[:2][::-1], np.int32)
            for aug in self.aug_images:
                aug_size = np.array(aug.shape[:2][::-1], np.int32)
                xy = (np.random.rand(2) * size).astype(np.int32)
                xy[1] -= aug.shape[0] // 2
                if random.random() < 0.5:
                    xy[0] -= aug.shape[1]
                xyxy = np.concatenate([xy, xy + aug_size]).reshape(2, 2)
                for i in range(2):
                    xyxy[:, i] = xyxy[:, i].clip(0, size[i])
                xyxy = xyxy.reshape(-1)
                xyxy_aug = xyxy - np.tile(xy, 2)
                def extract(
                    img, xyxy): return img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                aug_crop = extract(aug, xyxy_aug)
                mask = aug_crop[..., 3] > 0
                img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]
                    ][mask] = aug_crop[..., :3][mask]
        return img

    def __getitem__(self, idx):
        idx = idx % len(self.images)
        img = self.images[idx].copy()
        cfg = self.cfg
        if self.mode == 'train':
            img = self._augment(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = transform_affine(
                img,
                rot=cfg.rotate,
                scale=cfg.scale,
                translate=cfg.translate,
                pad_value=114,
            )
            img, _ = transform_resize_and_pad(
                img, cfg.image_size[::-1], pad_value=114)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img, _ = transform_resize_and_pad(
                img, cfg.image_size[::-1], pad_value=114)
        img = img.astype(np.float32) / 255.0
        img = img[None, ...]  # channel-first
        label = self.labels[idx]
        return img, label
