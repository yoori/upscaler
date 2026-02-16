import enum
import random
import typing

import cv2
import numpy as np


class BlurMode(enum.Enum):
  GAUSSIAN = "gaussian"
  UNIFORM = "uniform"
  OCCLUDE = "occlude"


class BlurMaskMode(enum.Enum):
  EYES = "eyes"
  FACE = "face"


class FaceBlurrer:
  def apply(
    self,
    image: np.ndarray,
    *,
    landmarks: np.ndarray,
    blur_mode: BlurMode,
    mask_mode: BlurMaskMode,
    rng: typing.Optional[random.Random] = None,
    blur_level: float = 0.0,
  ) -> np.ndarray:
    out = image.copy()
    mask = self._build_blur_mask(out.shape[:2], landmarks, mask_mode)
    if mask.size == 0 or not np.any(mask):
      return out

    x1, y1, x2, y2 = self._mask_bounds(mask)
    if x2 <= x1 or y2 <= y1:
      return out

    patch = out[y1:y2, x1:x2]
    if patch.size == 0:
      return out

    effective_mask = mask

    if blur_mode == BlurMode.GAUSSIAN:
      MIN_KERNEL = 0.08
      MAX_KERNEL = 0.25
      norm_blur_level = blur_level * (MAX_KERNEL - MIN_KERNEL) / (2 - MIN_KERNEL)
      kernel = round(
        min(image.shape[0], image.shape[1]) *
        (MIN_KERNEL / 2 + norm_blur_level * (1. - MIN_KERNEL / 2))
      ) * 2 + 1
      patch = cv2.GaussianBlur(patch, (kernel, kernel), 0)
    elif blur_mode == BlurMode.UNIFORM:
      scale = (6 + 14 * blur_level) / max(x2 - x1, y2 - y1)
      down_w = max(1, int((x2 - x1) * scale))
      down_h = max(1, int((y2 - y1) * scale))
      pixelated = cv2.resize(patch, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
      pixelated = cv2.resize(pixelated, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

      block_w = max(1, int(np.ceil((x2 - x1) / max(1, down_w))))
      block_h = max(1, int(np.ceil((y2 - y1) / max(1, down_h))))
      mask_patch = (mask[y1:y2, x1:x2] > 0)
      patch_mask = self._expand_mask_to_intersecting_blocks(mask_patch, block_w=block_w, block_h=block_h)
      patch = np.where(patch_mask[..., None], pixelated, patch)

      effective_mask = np.zeros_like(mask, dtype=np.uint8)
      effective_mask[y1:y2, x1:x2] = patch_mask.astype(np.uint8) * 255
    else:
      patch = self._apply_occlusion_patch(patch, blur_level=blur_level, rng=rng)

    out[y1:y2, x1:x2] = patch
    mask_3 = (effective_mask > 0)[..., None]
    return np.where(mask_3, out, image)

  def _apply_occlusion_patch(
    self,
    patch: np.ndarray,
    blur_level: float,
    rng: typing.Optional[random.Random],
  ) -> np.ndarray:
    if patch.size == 0:
      return patch

    chooser = rng if rng is not None else random.Random(0)
    out = patch.copy()
    base_color = int(chooser.uniform(0, 255))
    color = (base_color, base_color, base_color)

    level = max(0.0, min(0.999999, float(blur_level)))
    use_hatching = chooser.random() < (0.5 - 0.2 * level)
    if not use_hatching and chooser.random() < (0.5 + 0.45 * level):
      out[:, :] = color
      return out

    out[:, :] = color
    hatch_color = tuple(int(v) for v in (255 - base_color, 255 - base_color, 255 - base_color))
    spacing_min = max(4, int(round(10 - 5 * level)))
    spacing_max = max(spacing_min + 1, int(round(18 - 8 * level)))
    spacing = chooser.randint(spacing_min, spacing_max)
    thickness = 2 if level >= 0.55 else 1
    h, w = out.shape[:2]
    for shift in range(-h, w + h, spacing):
      cv2.line(
        out,
        (shift, 0),
        (shift - h, h),
        hatch_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
      )
    if chooser.random() < 0.35:
      for shift in range(0, w + h, spacing * 2):
        cv2.line(
          out,
          (shift, h),
          (shift - h, 0),
          hatch_color,
          thickness=thickness,
          lineType=cv2.LINE_AA,
        )
    return out

  def _expand_mask_to_intersecting_blocks(
    self,
    mask: np.ndarray,
    *,
    block_w: int,
    block_h: int,
  ) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape[:2]
    for y in range(0, h, block_h):
      y2 = min(y + block_h, h)
      for x in range(0, w, block_w):
        x2 = min(x + block_w, w)
        if np.any(mask[y:y2, x:x2]):
          out[y:y2, x:x2] = True
    return out

  def _mask_bounds(self, mask: np.ndarray) -> typing.Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
      return 0, 0, 0, 0
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2

  def _build_blur_mask(
    self,
    shape: typing.Tuple[int, int],
    landmarks: np.ndarray,
    mode: BlurMaskMode,
  ) -> np.ndarray:
    h, w = shape
    if h <= 0 or w <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    if mode == BlurMaskMode.EYES:
      left_eye = landmarks[0]
      right_eye = landmarks[1]
      center = ((left_eye + right_eye) * 0.5).astype(np.float32)
      eye_dist = float(np.linalg.norm(left_eye - right_eye))
      ax = max(4, int(round(eye_dist * 0.9)))
      ay = max(3, int(round(eye_dist * 0.45)))
      cv2.ellipse(mask, (int(round(center[0])), int(round(center[1]))), (ax, ay), 0, 0, 360, 255, -1)
      return mask

    min_xy = landmarks.min(axis=0)
    max_xy = landmarks.max(axis=0)
    center = ((min_xy + max_xy) * 0.5).astype(np.float32)
    ax = max(6, int(round((max_xy[0] - min_xy[0]) * 1.15)))
    ay = max(6, int(round((max_xy[1] - min_xy[1]) * 1.35)))
    cv2.ellipse(mask, (int(round(center[0])), int(round(center[1]))), (ax, ay), 0, 0, 360, 255, -1)
    return mask
