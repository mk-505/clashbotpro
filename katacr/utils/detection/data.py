import cv2
import numpy as np


def transform_affine(img, rot=0, scale=1, translate=0, pad_value=114):
    """Apply a simple affine transform: rotation, scaling, translation.

    This is a lightweight re‑implementation of the utility used by the
    original training code.  The parameters are interpreted as follows:

    * ``rot`` – degrees to rotate around the center
    * ``scale`` – isotropic scaling factor
    * ``translate`` – fraction of image size to shift in both x and y
    * ``pad_value`` – value used for border padding during the warp
    """

    if rot == 0 and scale == 1 and translate == 0:
        return img

    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, rot, scale)
    # apply translation as a fraction of width/height
    M[0, 2] += translate * w
    M[1, 2] += translate * h
    warped = cv2.warpAffine(img, M, (w, h), borderValue=pad_value)
    return warped


def transform_resize_and_pad(img, size, pad_value=114):
    """Resize ``img`` to fit into ``size`` while preserving aspect ratio.

    Returns a tuple ``(result, scale)`` where ``scale`` is the scaling factor
    actually applied.  The resulting image is padded with ``pad_value`` to
    match ``size`` exactly.

    ``size`` should be given as ``(width, height)``.
    """

    h, w = img.shape[:2]
    target_w, target_h = size
    if w == 0 or h == 0 or target_w == 0 or target_h == 0:
        return img, 1.0

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_value)
    return padded, scale
