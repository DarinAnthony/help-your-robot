"""Image geometry of Pi training and evaluation inputs (review T5; BoAI 2026-11-05: use the Pi server's convention).

Sim RLDS builds store each render **stretched** to a square (``mlspaces_to_rlds.py``: 848x480 -> 256x256 primary,
128x128 wrist, INTER_AREA). The Pi convention **letterboxes** (``resize_with_pad``: aspect kept, zero bars): pi05_base
was pretrained on letterboxed DROID, the Pi loader letterboxes DROID, and an openpi-style deployment client letterboxes
its raw frames. A stretched-trained checkpoint therefore sees objects 1.77x wider at deployment than in training.

``letterbox_stretched_v1`` removes that mismatch without reconverting: the stored square keeps every column of the
letterboxed frame (letterboxing 848x480 into 256x256 keeps 256x144 of image), so resizing the square to the letterbox
content size and padding it reproduces ``resize_with_pad`` of the original render up to one more resampling. The content
size and offsets follow ``tf.image.resize_with_pad``'s arithmetic exactly (floor of the scaled size, floor of the
top/left padding), so the loader (TensorFlow) and the evaluation client (NumPy/OpenCV) put the bars on the same rows.

- Training: ``Pi05InputConfig.image_geometry`` (``LetterboxStretched``) -> the RLDS loader un-stretches every sim frame
  (DROID frames are letterboxed from their raw 180x320 as before).
- Evaluation: the client stretches exactly as the converter did, then applies the same transform
  (``proprio``-independent ``image_geometry="letterbox_stretched_v1"`` on the policy config).
"""
from __future__ import annotations

import math

import numpy as np

STRETCH = "stretch"
LETTERBOX_STRETCHED_V1 = "letterbox_stretched_v1"
IMAGE_GEOMETRIES = (STRETCH, LETTERBOX_STRETCHED_V1)
#: The production render size (recording standard v1+, ``camera_resolution``), width x height.
RENDER_SIZE = (848, 480)


def letterbox_box(source_width: int, source_height: int, target: int) -> tuple[int, int, int, int]:
    """(content height, content width, top, left) of a source letterboxed into a ``target`` square, as
    ``tf.image.resize_with_pad`` computes them."""
    ratio = max(source_width / target, source_height / target)
    height_f, width_f = source_height / ratio, source_width / ratio
    height, width = max(1, math.floor(height_f)), max(1, math.floor(width_f))
    top, left = max(0, math.floor((target - height_f) / 2)), max(0, math.floor((target - width_f) / 2))
    return height, width, top, left


def unstretch_letterbox(square: np.ndarray, source_width: int, source_height: int) -> np.ndarray:
    """A stored stretched square (H = W, uint8 HxWx3) as the letterbox of its source (NumPy/OpenCV, INTER_AREA)."""
    import cv2

    size = int(square.shape[0])
    if square.shape[1] != size:
        raise ValueError(f"expected a square image, got {square.shape}")
    height, width, top, left = letterbox_box(source_width, source_height, size)
    out = np.zeros_like(square)
    out[top:top + height, left:left + width] = cv2.resize(np.asarray(square), (width, height),
                                                          interpolation=cv2.INTER_AREA)
    return out


def unstretch_letterbox_tf(square, source_width: int, source_height: int, size: int, tf):
    """The TensorFlow form of ``unstretch_letterbox`` for one decoded uint8 frame of static ``size``."""
    height, width, top, left = letterbox_box(source_width, source_height, size)
    content = tf.image.resize(square, (height, width), method="area", antialias=True)
    padded = tf.image.pad_to_bounding_box(content, top, left, size, size)
    return tf.cast(tf.clip_by_value(tf.round(padded), 0, 255), tf.uint8)
