import os
import asyncio
import typing
import dataclasses

import numpy as np
import torch
import cv2

import basicsr.archs.rrdbnet_arch
import realesrgan
import gfpgan
import codeformer.basicsr.archs.codeformer_arch

from .face_searcher import FaceSearcher


FaceProcessorName = typing.Literal["codeformer", "gfpgan", "restoreformer", "rollback_diff"]


@dataclasses.dataclass(frozen=True)
class FaceProcessor:
  max_apply_px: typing.Optional[int] = None
  processor: FaceProcessorName = "gfpgan"
  stop_apply: bool = True


def default_face_processors() -> typing.List[FaceProcessor]:
  return [
    FaceProcessor(max_apply_px=96, processor="codeformer", stop_apply=True),
    FaceProcessor(processor="gfpgan", stop_apply=False),
    FaceProcessor(processor="rollback_diff", stop_apply=True),
  ]


def _clamp_norm(value: float) -> float:
  return max(0.0, min(1.0, float(value)))


def _bbox_norm_to_pixels(
  bbox_norm: typing.Sequence[float],
  *,
  width: int,
  height: int,
) -> typing.Tuple[int, int, int, int]:
  x1_f, y1_f, x2_f, y2_f = [float(v) for v in bbox_norm]
  x1_f = _clamp_norm(x1_f)
  y1_f = _clamp_norm(y1_f)
  x2_f = _clamp_norm(x2_f)
  y2_f = _clamp_norm(y2_f)
  return (
    int(round(x1_f * width)),
    int(round(y1_f * height)),
    int(round(x2_f * width)),
    int(round(y2_f * height)),
  )


@dataclasses.dataclass(frozen=True)
class Ellipse:
  center: typing.Tuple[float, float]
  axes: typing.Tuple[float, float]
  angle: float


@dataclasses.dataclass
class UpscaleFaceInfoStep:
  name: str
  image: typing.Optional[np.ndarray] = None


@dataclasses.dataclass
class UpscaleFaceInfo:
  bbox: typing.List[float]
  face_px: int
  algorithm: str
  landmarks5: typing.Optional[typing.List[typing.List[float]]] = None
  landmarks_all: typing.Optional[typing.List[typing.List[float]]] = None
  landmarks_all_face_crop: typing.Optional[typing.List[typing.List[float]]] = None
  eye_ellipse: typing.Optional[Ellipse] = None
  eye_ellipse_face_crop: typing.Optional[Ellipse] = None
  mouth_ellipse_face_crop: typing.Optional[Ellipse] = None
  steps: typing.List[UpscaleFaceInfoStep] = dataclasses.field(default_factory=list)

  def _normalized_bbox(self) -> typing.Tuple[float, float, float, float]:
    x1_f, y1_f, x2_f, y2_f = [float(v) for v in self.bbox]
    return (
      _clamp_norm(x1_f),
      _clamp_norm(y1_f),
      _clamp_norm(x2_f),
      _clamp_norm(y2_f),
    )

  def _bbox_pixels(self, *, width: int, height: int) -> typing.Tuple[int, int, int, int]:
    return _bbox_norm_to_pixels(self.bbox, width=width, height=height)

  def visualize(
    self,
    image_bgr: np.ndarray,
    *,
    point_radius: int = 2,
    point_color: typing.Tuple[int, int, int] = (0, 255, 0),
  ) -> np.ndarray:
    """
    Return cropped face from the full image with landmarks highlighted.
    Expects bbox/landmarks normalized to [0..1] relative to the image size.
    """
    if image_bgr is None:
      raise ValueError("Expected image array")
    if image_bgr.ndim == 2:
      image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
      image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
      image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    elif image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
      raise ValueError("Expected image HxW or HxWx3")

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = self._bbox_pixels(width=w, height=h)


    if x2 <= x1 or y2 <= y1:
      return image_bgr[0:0, 0:0].copy()

    crop = image_bgr[y1:y2, x1:x2].copy()

    if self.landmarks5:
      for point in self.landmarks5:
        if len(point) < 2:
          continue
        if not (0.0 <= float(point[0]) <= 1.0 and 0.0 <= float(point[1]) <= 1.0):
          continue
        px = int(round(point[0] * w - x1))
        py = int(round(point[1] * h - y1))
        if 0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]:
          cv2.circle(crop, (px, py), int(point_radius), point_color, -1)

    return crop

  def get_eye_mask(self, *, width: int, height: int) -> np.ndarray:
    """
    Return a binary (0/255) mask for the eye ellipse in image coordinates.
    Expects eye_ellipse params normalized to [0..1] relative to the image size.
    """
    if not self.eye_ellipse:
      return np.zeros((0, 0), dtype=np.uint8)

    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    cx = float(self.eye_ellipse.center[0]) * width
    cy = float(self.eye_ellipse.center[1]) * height
    ax = float(self.eye_ellipse.axes[0]) * width
    ay = float(self.eye_ellipse.axes[1]) * height

    if ax <= 0 or ay <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    center_px = (int(round(cx)), int(round(cy)))
    axes_px = (max(1, int(round(ax))), max(1, int(round(ay))))
    cv2.ellipse(mask, center_px, axes_px, float(self.eye_ellipse.angle), 0, 360, 255, -1)
    return mask

  @staticmethod
  def _resolve_face_crop_shape(
    strong_change_mask: typing.Optional[np.ndarray],
    strong_change_mask_color: typing.Optional[np.ndarray],
    face_crop_shape: typing.Optional[typing.Tuple[int, int]],
  ) -> typing.Tuple[int, int]:
    if strong_change_mask is not None and strong_change_mask.size:
      return int(strong_change_mask.shape[0]), int(strong_change_mask.shape[1])
    if strong_change_mask_color is not None and strong_change_mask_color.size:
      return int(strong_change_mask_color.shape[0]), int(strong_change_mask_color.shape[1])
    if face_crop_shape is not None:
      return int(face_crop_shape[0]), int(face_crop_shape[1])
    return 0, 0

  def get_eye_mask_for_face_crop(
    self,
    face_crop_shape: typing.Tuple[int, int],
  ) -> np.ndarray:
    """
    Return a binary (0/255) eye mask in local face-crop coordinates.
    Shape matches face crop / strong-change masks.
    """
    height, width = face_crop_shape

    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    ellipse = self.eye_ellipse_face_crop
    if ellipse is None:
      ellipse = self.eye_ellipse
      if ellipse is None:
        return np.zeros((height, width), dtype=np.uint8)

      x1_f, y1_f, x2_f, y2_f = self._normalized_bbox()

      box_w = x2_f - x1_f
      box_h = y2_f - y1_f
      if box_w <= 1e-8 or box_h <= 1e-8:
        return np.zeros((height, width), dtype=np.uint8)

      cx_local_norm = (float(ellipse.center[0]) - x1_f) / box_w
      cy_local_norm = (float(ellipse.center[1]) - y1_f) / box_h
      ax_local_norm = float(ellipse.axes[0]) / box_w
      ay_local_norm = float(ellipse.axes[1]) / box_h
      ellipse = Ellipse(
        center=(cx_local_norm, cy_local_norm),
        axes=(ax_local_norm, ay_local_norm),
        angle=float(ellipse.angle),
      )

    return self._render_face_crop_ellipse_mask(ellipse, width=width, height=height)

  @staticmethod
  def _render_face_crop_ellipse_mask(
    ellipse: typing.Optional[Ellipse],
    *,
    width: int,
    height: int,
  ) -> np.ndarray:
    if ellipse is None:
      return np.zeros((height, width), dtype=np.uint8)

    cx = float(ellipse.center[0]) * width
    cy = float(ellipse.center[1]) * height
    ax = float(ellipse.axes[0]) * width
    ay = float(ellipse.axes[1]) * height
    if ax <= 0 or ay <= 0:
      return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    center_px = (int(round(cx)), int(round(cy)))
    axes_px = (max(1, int(round(ax))), max(1, int(round(ay))))
    cv2.ellipse(mask, center_px, axes_px, float(ellipse.angle), 0, 360, 255, -1)
    return mask

  def get_mouth_mask_for_face_crop(
    self,
    face_crop_shape: typing.Tuple[int, int],
  ) -> np.ndarray:
    height, width = face_crop_shape
    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)
    return self._render_face_crop_ellipse_mask(self.mouth_ellipse_face_crop, width=width, height=height)

  def get_nose_zone_mask_for_face_crop(
    self,
    face_crop_shape: typing.Tuple[int, int],
  ) -> np.ndarray:
    height, width = face_crop_shape
    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    points_local: typing.Optional[typing.List[typing.Tuple[float, float]]] = None
    nose_local: typing.Optional[typing.Tuple[float, float]] = None
    mouth_left_local: typing.Optional[typing.Tuple[float, float]] = None
    mouth_right_local: typing.Optional[typing.Tuple[float, float]] = None

    if self.landmarks_all_face_crop is not None and len(self.landmarks_all_face_crop) >= 5:
      points = np.asarray(self.landmarks_all_face_crop, dtype=np.float32).reshape(-1, 2)
      left_eye, right_eye, nose, mouth_left, mouth_right = points[:5]
      points_local = [
        (float(left_eye[0]), float(left_eye[1])),
        (float(right_eye[0]), float(right_eye[1])),
        (float(nose[0]), float(nose[1])),
        (float(mouth_right[0]), float(mouth_right[1])),
        (float(mouth_left[0]), float(mouth_left[1])),
      ]
      nose_local = (float(nose[0]), float(nose[1]))
      mouth_left_local = (float(mouth_left[0]), float(mouth_left[1]))
      mouth_right_local = (float(mouth_right[0]), float(mouth_right[1]))
    elif self.landmarks5 is not None and len(self.landmarks5) >= 5:
      x1_f, y1_f, x2_f, y2_f = self._normalized_bbox()
      box_w = x2_f - x1_f
      box_h = y2_f - y1_f
      if box_w <= 1e-8 or box_h <= 1e-8:
        return np.zeros((height, width), dtype=np.uint8)
      points = np.asarray(self.landmarks5, dtype=np.float32).reshape(-1, 2)
      left_eye, right_eye, nose, mouth_left, mouth_right = points[:5]
      points_local = [
        ((float(left_eye[0]) - x1_f) / box_w, (float(left_eye[1]) - y1_f) / box_h),
        ((float(right_eye[0]) - x1_f) / box_w, (float(right_eye[1]) - y1_f) / box_h),
        ((float(nose[0]) - x1_f) / box_w, (float(nose[1]) - y1_f) / box_h),
        ((float(mouth_right[0]) - x1_f) / box_w, (float(mouth_right[1]) - y1_f) / box_h),
        ((float(mouth_left[0]) - x1_f) / box_w, (float(mouth_left[1]) - y1_f) / box_h),
      ]
      nose_local = ((float(nose[0]) - x1_f) / box_w, (float(nose[1]) - y1_f) / box_h)
      mouth_left_local = ((float(mouth_left[0]) - x1_f) / box_w, (float(mouth_left[1]) - y1_f) / box_h)
      mouth_right_local = ((float(mouth_right[0]) - x1_f) / box_w, (float(mouth_right[1]) - y1_f) / box_h)

    if not points_local:
      return np.zeros((height, width), dtype=np.uint8)

    polygon = np.asarray(
      [[int(round(x * width)), int(round(y * height))] for x, y in points_local],
      dtype=np.int32,
    )
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
      return np.zeros((height, width), dtype=np.uint8)

    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(0, width - 1))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(0, height - 1))

    # cv2.fillConvexPoly expects points in contour order.
    # Build a convex hull first so the resulting shape is order-invariant
    # and always covers all triangles formed by landmark points.
    hull = cv2.convexHull(polygon)
    if hull is None or hull.size == 0:
      return np.zeros((height, width), dtype=np.uint8)
    hull = hull.reshape(-1, 2)
    if hull.shape[0] < 3:
      return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    if nose_local is not None and mouth_left_local is not None and mouth_right_local is not None:
      nose_x = int(round(float(nose_local[0]) * width))
      nose_y = int(round(float(nose_local[1]) * height))
      nose_x = int(np.clip(nose_x, 0, max(0, width - 1)))
      nose_y = int(np.clip(nose_y, 0, max(0, height - 1)))

      mouth_lx = float(mouth_left_local[0]) * width
      mouth_ly = float(mouth_left_local[1]) * height
      mouth_rx = float(mouth_right_local[0]) * width
      mouth_ry = float(mouth_right_local[1]) * height

      dist_to_mouth_left = float(np.hypot(mouth_lx - float(nose_x), mouth_ly - float(nose_y)))
      dist_to_mouth_right = float(np.hypot(mouth_rx - float(nose_x), mouth_ry - float(nose_y)))
      min_dist_to_mouth = min(dist_to_mouth_left, dist_to_mouth_right)
      nose_radius = max(1, int(round(min_dist_to_mouth / 3.0)))
      cv2.circle(mask, (nose_x, nose_y), nose_radius, 255, -1)

    return mask

  def get_full_face_mask(
    self,
    *,
    strong_change_mask: typing.Optional[np.ndarray],
    strong_change_mask_color: typing.Optional[np.ndarray],
    face_crop_shape: typing.Optional[typing.Tuple[int, int]],
  ) -> np.ndarray:
    """
    Build full face-zone mask for rollback:
    eye OR mouth OR nose-zone masks.
    """
    height, width = self._resolve_face_crop_shape(
      strong_change_mask=strong_change_mask,
      strong_change_mask_color=strong_change_mask_color,
      face_crop_shape=face_crop_shape,
    )
    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    full_face_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in (
      self.get_eye_mask_for_face_crop(face_crop_shape),
      self.get_mouth_mask_for_face_crop(face_crop_shape),
      self.get_nose_zone_mask_for_face_crop(face_crop_shape),
    ):
      if mask is None or mask.size == 0:
        continue
      local_mask = mask
      if local_mask.ndim == 3:
        local_mask = cv2.cvtColor(local_mask, cv2.COLOR_BGR2GRAY)
      if local_mask.shape[:2] != (height, width):
        local_mask = cv2.resize(local_mask, (width, height), interpolation=cv2.INTER_NEAREST)
      full_face_mask = cv2.bitwise_or(full_face_mask, (local_mask > 0).astype(np.uint8) * 255)

    return full_face_mask

  def get_full_rollback_mask(
    self,
    *,
    strong_change_mask: typing.Optional[np.ndarray],
    strong_change_mask_color: typing.Optional[np.ndarray],
    face_crop_shape: typing.Optional[typing.Tuple[int, int]],
    diff_opening_window: float,
  ) -> np.ndarray:
    """
    Build face rollback mask:
    (strong_change_mask OR strong_change_mask_color), then MORPH_OPEN with
    kernel width = face_crop_width * diff_opening_window.
    """
    height, width = self._resolve_face_crop_shape(
      strong_change_mask=strong_change_mask,
      strong_change_mask_color=strong_change_mask_color,
      face_crop_shape=face_crop_shape,
    )
    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    rollback_mask = self.get_face_rollback_mask_before_opening(
      strong_change_mask=strong_change_mask,
      strong_change_mask_color=strong_change_mask_color,
      face_crop_shape=face_crop_shape,
    )

    if np.count_nonzero(rollback_mask) == 0:
      return rollback_mask

    kernel_w = max(1, int(round(float(width) * float(diff_opening_window))))
    if kernel_w % 2 == 0:
      kernel_w += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_w))
    rollback_mask = cv2.morphologyEx(rollback_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return rollback_mask

  def get_face_rollback_mask(
    self,
    *,
    strong_change_mask: typing.Optional[np.ndarray],
    strong_change_mask_color: typing.Optional[np.ndarray],
    face_crop_shape: typing.Optional[typing.Tuple[int, int]],
    diff_opening_window: float,
  ) -> np.ndarray:
    """
    Build effective rollback mask limited to face regions:
    get_full_rollback_mask AND get_full_face_mask.
    """
    full_rollback_mask = self.get_full_rollback_mask(
      strong_change_mask=strong_change_mask,
      strong_change_mask_color=strong_change_mask_color,
      face_crop_shape=face_crop_shape,
      diff_opening_window=diff_opening_window,
    )

    if full_rollback_mask is None or full_rollback_mask.size == 0:
      return np.zeros((0, 0), dtype=np.uint8)

    full_face_mask = self.get_full_face_mask(
      strong_change_mask=strong_change_mask,
      strong_change_mask_color=strong_change_mask_color,
      face_crop_shape=face_crop_shape,
    )

    if full_face_mask is None or full_face_mask.size == 0:
      return np.zeros(full_rollback_mask.shape[:2], dtype=np.uint8)
    if full_face_mask.shape[:2] != full_rollback_mask.shape[:2]:
      full_face_mask = cv2.resize(
        full_face_mask,
        (full_rollback_mask.shape[1], full_rollback_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
      )

    return cv2.bitwise_and(
      (full_rollback_mask > 0).astype(np.uint8) * 255,
      (full_face_mask > 0).astype(np.uint8) * 255,
    )

  def get_face_rollback_mask_before_opening(
    self,
    *,
    strong_change_mask: typing.Optional[np.ndarray],
    strong_change_mask_color: typing.Optional[np.ndarray],
    face_crop_shape: typing.Optional[typing.Tuple[int, int]],
  ) -> np.ndarray:
    """
    Build union mask before opening:
    strong_change_mask OR strong_change_mask_color.
    """
    height, width = self._resolve_face_crop_shape(
      strong_change_mask=strong_change_mask,
      strong_change_mask_color=strong_change_mask_color,
      face_crop_shape=face_crop_shape,
    )
    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    rollback_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in (strong_change_mask, strong_change_mask_color):
      if mask is None or mask.size == 0:
        continue
      local_mask = mask
      if local_mask.ndim == 3:
        local_mask = cv2.cvtColor(local_mask, cv2.COLOR_BGR2GRAY)
      resized = cv2.resize(local_mask, (width, height), interpolation=cv2.INTER_NEAREST)
      rollback_mask = cv2.bitwise_or(rollback_mask, (resized > 0).astype(np.uint8) * 255)

    return rollback_mask


@dataclasses.dataclass
class UpscaleInfo:
  face_processors: typing.List[FaceProcessor] = dataclasses.field(default_factory=list)
  faces: typing.List[UpscaleFaceInfo] = dataclasses.field(default_factory=list)
  fallback: typing.Optional[str] = None


@dataclasses.dataclass
class UpscaleParams:
  outscale: float = 4.0
  tile: int = 0
  only_center_face: bool = False

  # GFPGAN strengths
  gfpgan_weight: float = 0.35
  gfpgan_weight_small: float = 0.20

  # CodeFormer knob (lower => stronger reconstruction; good for low-res)
  codeformer_fidelity: float = 0.30

  # include debug crops in UpscaleFaceInfo
  fill_debug_images: bool = False

  # diff-mask controls for per-face debug masks
  # normalized threshold in [0, 1]
  diff_thr: float = 0.02
  # minimal connected-component area as a fraction of face crop area in [0, 1]
  diff_min_area: float = 0.0001
  # opening window size for rollback mask as fraction of face crop width
  diff_opening_window: float = 0.005
  # extra rollback-mask dilation as fraction of face crop width
  rollback_extend: float = 0.01

  face_processors: typing.List[FaceProcessor] = dataclasses.field(default_factory=default_face_processors)

  def resolved_face_processors(self) -> typing.List[FaceProcessor]:
    return list(self.face_processors)


@dataclasses.dataclass(frozen=True)
class DiffZonesMeanWindowResult:
  diff_mean: np.ndarray
  diff_color_mean: np.ndarray
  mask01: np.ndarray
  mask_color01: np.ndarray
  mask_u8: np.ndarray
  mask_color_u8: np.ndarray
  d: np.ndarray
  d_color: np.ndarray


class Upscaler(object):

  class Exception(Exception):
    pass

  def __init__(
    self,
    realesrgan_weights: str = "RealESRGAN_x4plus.pth",
    gfpgan_weights: typing.Optional[str] = None,
    restoreformer_weights: typing.Optional[str] = None,
    codeformer_weights: typing.Optional[str] = None,
  ):
    self._realesrgan_weights = realesrgan_weights
    self._gfpgan_weights = gfpgan_weights
    self._restoreformer_weights = restoreformer_weights
    self._codeformer_weights = codeformer_weights

    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    self._use_half = (self._device == "cuda")

    self._lock = asyncio.Lock()

    # Real-ESRGAN
    model = basicsr.archs.rrdbnet_arch.RRDBNet(
      num_in_ch=3,
      num_out_ch=3,
      num_feat=64,
      num_block=23,
      num_grow_ch=32,
      scale=4,
    )

    self._upsampler = realesrgan.RealESRGANer(
      scale=4,
      model_path=self._realesrgan_weights,
      model=model,
      tile=0,
      tile_pad=10,
      pre_pad=0,
      half=self._use_half,
      device=self._device,
    )

    # Face searcher/helper (used for align/paste-back for both GFPGAN and CodeFormer)
    self._face_searcher = FaceSearcher(device=self._device)

    # GFPGAN (optional)
    if self._gfpgan_weights is not None:
      self._face_enhancer = gfpgan.GFPGANer(
        model_path=self._gfpgan_weights,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=self._device,
      )
    else:
      self._face_enhancer = None

    # RestoreFormer (optional; uses GFPGAN helper with RestoreFormer architecture)
    if self._restoreformer_weights is not None:
      self._restoreformer_enhancer = gfpgan.GFPGANer(
        model_path=self._restoreformer_weights,
        upscale=1,
        arch="RestoreFormer",
        channel_multiplier=2,
        bg_upsampler=None,
        device=self._device,
      )
    else:
      self._restoreformer_enhancer = None

    # CodeFormer (optional)
    self._codeformer_net = None
    if self._codeformer_weights is not None:
      self._codeformer_net = self._load_codeformer(self._codeformer_weights)

  async def upscale(
    self,
    img_bgr: np.ndarray,
    params: typing.Optional[UpscaleParams] = None,
  ) -> np.ndarray:
    result, _ = await self.upscale_with_info(img_bgr, params=params)
    return result

  async def upscale_with_info(
    self,
    img_bgr: np.ndarray,
    params: typing.Optional[UpscaleParams] = None,
  ) -> typing.Tuple[np.ndarray, UpscaleInfo]:
    if params is None:
      params = UpscaleParams()

    face_processors = params.resolved_face_processors()
    result_info = UpscaleInfo(face_processors=face_processors)

    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
      raise Upscaler.Exception("Expected BGR image HxWx3")
    if img_bgr.dtype != np.uint8:
      raise Upscaler.Exception(f"Expected uint8 image, got {img_bgr.dtype}")

    tile = 0 if params.tile is None else int(params.tile)
    outscale = params.outscale

    async with self._lock:
      self._upsampler.tile = tile
      up_bgr, _ = self._upsampler.enhance(img_bgr, outscale=outscale)

      if not face_processors:
        return (up_bgr, result_info)

      return self._apply_faces_routed(
        original_bgr=img_bgr,
        upscaled_bgr=up_bgr,
        only_center_face=bool(params.only_center_face),
        gfpgan_weight_normal=float(params.gfpgan_weight),
        gfpgan_weight_small=float(params.gfpgan_weight_small),
        codeformer_fidelity=float(params.codeformer_fidelity),
        fill_debug_images=bool(params.fill_debug_images),
        diff_thr=float(params.diff_thr),
        diff_min_area=float(params.diff_min_area),
        diff_opening_window=float(params.diff_opening_window),
        rollback_extend=float(params.rollback_extend),
        face_processors=face_processors,
        info=result_info,
      )

  async def close(self):
    pass

  def _apply_gfpgan_whole(
    self,
    img_bgr: np.ndarray,
    *,
    weight: float,
    only_center_face: bool,
  ) -> np.ndarray:

    _, _, restored = self._face_enhancer.enhance(
      img_bgr,
      has_aligned=False,
      only_center_face=only_center_face,
      paste_back=True,
      weight=weight,
    )
    return restored

  def _apply_restoreformer_whole(
    self,
    img_bgr: np.ndarray,
    *,
    weight: float,
    only_center_face: bool,
  ) -> np.ndarray:

    _, _, restored = self._restoreformer_enhancer.enhance(
      img_bgr,
      has_aligned=False,
      only_center_face=only_center_face,
      paste_back=True,
      weight=weight,
    )
    return restored

  def _cv2_ready_bgr(self, img) -> np.ndarray:
    img = np.asarray(img)

    if img.dtype == np.dtype("O"):
      raise Upscaler.Exception("Image dtype=object (cv2 can't handle it)")

    if img.ndim == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
      raise Upscaler.Exception(f"Bad image shape: {img.shape}")

    if img.dtype == np.float16:
      img = img.astype(np.float32, copy=False)

    if img.dtype != np.uint8 and img.dtype != np.float32:
      if np.issubdtype(img.dtype, np.floating):
        m = float(np.max(img)) if img.size else 0.0
        if m <= 1.0:
          img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
      else:
        img = img.astype(np.uint8)

    if not img.flags["C_CONTIGUOUS"]:
      img = np.ascontiguousarray(img)

    return img

  def _append_face_step(
    self,
    face_info: UpscaleFaceInfo,
    *,
    name: str,
    image: typing.Optional[np.ndarray],
  ) -> None:
    if image is None or getattr(image, "size", 0) == 0:
      return
    face_info.steps.append(UpscaleFaceInfoStep(name=name, image=self._cv2_ready_bgr(image).copy()))

  def _build_face_masks_overlay(
    self,
    *,
    base_image: typing.Optional[np.ndarray],
    eye_mask: typing.Optional[np.ndarray],
    mouth_mask: typing.Optional[np.ndarray],
    nose_zone_mask: typing.Optional[np.ndarray],
  ) -> typing.Optional[np.ndarray]:
    if base_image is None or getattr(base_image, "size", 0) == 0:
      return None
    out = self._cv2_ready_bgr(base_image).copy()

    def _apply(mask: typing.Optional[np.ndarray], color: typing.Tuple[int, int, int], alpha: float) -> None:
      nonlocal out
      if mask is None or getattr(mask, "size", 0) == 0:
        return
      local_mask = mask
      if local_mask.ndim == 3:
        local_mask = cv2.cvtColor(local_mask, cv2.COLOR_BGR2GRAY)
      if local_mask.shape[:2] != out.shape[:2]:
        local_mask = cv2.resize(local_mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
      mask_bool = local_mask > 0
      if not np.any(mask_bool):
        return
      overlay = np.zeros_like(out)
      overlay[:, :] = color
      blended = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)
      out[mask_bool] = blended[mask_bool]

    _apply(eye_mask, (0, 255, 0), 0.36)
    _apply(mouth_mask, (0, 165, 255), 0.36)
    _apply(nose_zone_mask, (255, 255, 0), 0.30)
    return out

  def _create_face_oval_mask(
    self,
    face_crop: np.ndarray,
    *,
    landmarks_5: np.ndarray,
    affine_matrix: np.ndarray,
  ) -> np.ndarray:
    """
    Build an oval mask that covers the main facial area (eyes + mouth),
    aligned to the face orientation using FaceRestoreHelper landmarks.

    Returns a uint8 mask (0/255) matching the face crop size.
    """

    face = self._cv2_ready_bgr(face_crop)
    h, w = face.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    default_center = (int(w * 0.5), int(h * 0.55))
    default_axes = (max(1, int(w * 0.45)), max(1, int(h * 0.60)))

    ellipse = None
    if landmarks_5 is not None and affine_matrix is not None:
      points = np.asarray(landmarks_5, dtype=np.float32).reshape(-1, 2)
      if points.shape[0] >= 5:
        points = cv2.transform(points[None, :, :], affine_matrix)[0]
        eye_left, eye_right, _, mouth_left, mouth_right = points[:5]
        mouth_center = (mouth_left + mouth_right) * 0.5
        ellipse_points = np.stack(
          [eye_left, eye_right, mouth_left, mouth_right, mouth_center],
          axis=0,
        )
        try:
          ellipse = cv2.fitEllipse(ellipse_points.astype(np.float32))
        except cv2.error:
          ellipse = None

    if ellipse is not None:
      (cx, cy), (width, height), angle = ellipse
      scale = 1.02
      axes = (
        max(1, int(width * 0.5 * scale)),
        max(1, int(height * 0.5 * scale)),
      )
      center = (int(cx), int(cy))
      cv2.ellipse(mask, center, axes, float(angle), 0, 360, 255, -1)
    else:
      cv2.ellipse(mask, default_center, default_axes, 0, 0, 360, 255, -1)

    return mask

  def _create_eye_ellipse(
    self,
    *,
    landmarks5: typing.List[typing.List[float]],
    image_shape: typing.Tuple[int, int],
  ) -> typing.Optional[Ellipse]:
    """
    Build a directed oval mask for the eye region.
    The major axis is aligned with the eye line and centered on the eye midpoint.
    It should fully cover both eyes (not just the triangle between eyes and nose).
    Returns normalized ellipse params for use on any image of the same aspect.
    """
    if not landmarks5 or len(landmarks5) < 3:
      return None

    h, w = image_shape
    if h <= 0 or w <= 0:
      return None

    points = np.asarray(landmarks5, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
      return None

    left_eye, right_eye, nose = points[:3]

    lx, ly = float(left_eye[0]) * w, float(left_eye[1]) * h
    rx, ry = float(right_eye[0]) * w, float(right_eye[1]) * h
    nx, ny = float(nose[0]) * w, float(nose[1]) * h

    eye_dx = rx - lx
    eye_dy = ry - ly
    eye_dist = float(np.hypot(eye_dx, eye_dy))
    if eye_dist <= 1e-6:
      return None

    eye_center_x = (lx + rx) * 0.5
    eye_center_y = (ly + ry) * 0.5
    nose_vec_x = nx - eye_center_x
    nose_vec_y = ny - eye_center_y
    nose_dist = float(np.hypot(nose_vec_x, nose_vec_y))

    center_x = eye_center_x
    center_y = eye_center_y

    angle_deg = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
    axis_x = max(1.0, eye_dist * 0.90)
    axis_y = max(1.0, max(eye_dist * 0.38, nose_dist * 0.30))

    return Ellipse(
      center=(center_x / w, center_y / h),
      axes=(axis_x / w, axis_y / h),
      angle=angle_deg,
    )

  def _create_mouth_ellipse(
    self,
    *,
    landmarks5: typing.List[typing.List[float]],
    image_shape: typing.Tuple[int, int],
  ) -> typing.Optional[Ellipse]:
    if not landmarks5 or len(landmarks5) < 5:
      return None

    h, w = image_shape
    if h <= 0 or w <= 0:
      return None

    points = np.asarray(landmarks5, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 5:
      return None

    left_eye, right_eye, _, mouth_left, mouth_right = points[:5]
    lx, ly = float(left_eye[0]) * w, float(left_eye[1]) * h
    rx, ry = float(right_eye[0]) * w, float(right_eye[1]) * h
    mlx, mly = float(mouth_left[0]) * w, float(mouth_left[1]) * h
    mrx, mry = float(mouth_right[0]) * w, float(mouth_right[1]) * h

    mouth_center_x = (mlx + mrx) * 0.5
    mouth_center_y = (mly + mry) * 0.5
    mouth_dist = float(np.hypot(mrx - mlx, mry - mly))
    eye_dist = float(np.hypot(rx - lx, ry - ly))
    if mouth_dist <= 1e-6 and eye_dist <= 1e-6:
      return None

    angle_deg = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    axis_x = max(1.0, max(mouth_dist * 0.80, eye_dist * 0.26))
    axis_y = max(1.0, max(mouth_dist * 0.42, eye_dist * 0.18))

    return Ellipse(
      center=(mouth_center_x / w, mouth_center_y / h),
      axes=(axis_x / w, axis_y / h),
      angle=angle_deg,
    )

  def _apply_faces_routed(
    self,
    *,
    original_bgr: np.ndarray,
    upscaled_bgr: np.ndarray,
    only_center_face: bool,
    gfpgan_weight_normal: float,
    gfpgan_weight_small: float,
    codeformer_fidelity: float,
    fill_debug_images: bool,
    diff_thr: float,
    diff_min_area: float,
    diff_opening_window: float,
    rollback_extend: float,
    face_processors: typing.List[FaceProcessor],
    info: UpscaleInfo,
  ) -> typing.Tuple[np.ndarray, UpscaleInfo]:

    faces = self._face_searcher.get_faces(
      upscaled_bgr,
      is_bgr=True,
      only_center_face=only_center_face,
    )

    if len(faces) == 0:
      return (
        upscaled_bgr,
        dataclasses.replace(info, faces=[]),
      )

    restored_faces: typing.List[np.ndarray] = []
    face_infos: typing.List[UpscaleFaceInfo] = []

    up_h, up_w = upscaled_bgr.shape[:2]
    orig_h, orig_w = original_bgr.shape[:2]

    for i, face_detection in enumerate(faces):
      face_crop = face_detection.crop
      if face_crop is None or not face_crop.size:
        continue
      face_px = face_detection.size_px
      landmarks5 = None
      landmarks_all = None
      eye_ellipse = None
      if face_detection.landmarks_all is not None:
        landmarks = np.asarray(face_detection.landmarks_all, dtype=float)
      elif face_detection.landmarks_5 is not None:
        landmarks = np.asarray(face_detection.landmarks_5, dtype=float)
      else:
        landmarks = None

      if landmarks is not None:
        landmarks[:, 0] /= float(up_w) if up_w else 1.0
        landmarks[:, 1] /= float(up_h) if up_h else 1.0
        landmarks_all = landmarks.tolist()
        if landmarks.shape[0] >= 5:
          landmarks5 = landmarks[:5].tolist()
        else:
          landmarks5 = landmarks.tolist()

      if landmarks5 is not None:
        eye_ellipse = self._create_eye_ellipse(
          landmarks5=landmarks5,
          image_shape=(orig_h, orig_w),
        )
      bbox_norm = face_detection.bbox_norm
      eye_ellipse_face_crop = None
      mouth_ellipse_face_crop = None
      transformed_landmarks = None
      if landmarks_all is not None:
        points = np.asarray(landmarks_all, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] >= 3:
          transformed_landmarks = points.copy()

      ox1, oy1, ox2, oy2 = _bbox_norm_to_pixels(
        bbox_norm,
        width=orig_w,
        height=orig_h,
      )
      original_crop = None
      if face_detection.affine_matrix is not None and face_crop.size:
        matrix_for_original = np.asarray(face_detection.affine_matrix, dtype=np.float32).copy()
        sx = float(up_w) / float(orig_w) if orig_w else 1.0
        sy = float(up_h) / float(orig_h) if orig_h else 1.0
        matrix_for_original[0, 0] *= sx
        matrix_for_original[0, 1] *= sy
        matrix_for_original[1, 0] *= sx
        matrix_for_original[1, 1] *= sy
        original_crop = cv2.warpAffine(
          original_bgr,
          matrix_for_original,
          (face_crop.shape[1], face_crop.shape[0]),
          flags=cv2.INTER_LINEAR,
          borderMode=cv2.BORDER_REFLECT_101,
        )
        if transformed_landmarks is not None:
          transformed_landmarks[:, 0] *= float(up_w) if up_w else 1.0
          transformed_landmarks[:, 1] *= float(up_h) if up_h else 1.0
          transformed_landmarks = cv2.transform(
            transformed_landmarks[None, :, :],
            np.asarray(face_detection.affine_matrix, dtype=np.float32),
          )[0]
          transformed_landmarks[:, 0] /= float(face_crop.shape[1]) if face_crop.shape[1] else 1.0
          transformed_landmarks[:, 1] /= float(face_crop.shape[0]) if face_crop.shape[0] else 1.0
      elif ox2 > ox1 and oy2 > oy1:
        original_crop = original_bgr[oy1:oy2, ox1:ox2].copy()
        if transformed_landmarks is not None:
          ux1 = float(bbox_norm[0]) * float(up_w)
          uy1 = float(bbox_norm[1]) * float(up_h)
          ux2 = float(bbox_norm[2]) * float(up_w)
          uy2 = float(bbox_norm[3]) * float(up_h)
          box_w = max(1e-8, ux2 - ux1)
          box_h = max(1e-8, uy2 - uy1)
          transformed_landmarks[:, 0] = (transformed_landmarks[:, 0] * float(up_w) - ux1) / box_w
          transformed_landmarks[:, 1] = (transformed_landmarks[:, 1] * float(up_h) - uy1) / box_h
        if original_crop.size and original_crop.shape[:2] != face_crop.shape[:2]:
          original_crop = cv2.resize(
            original_crop,
            (face_crop.shape[1], face_crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
          )
      if transformed_landmarks is not None:
        eye_ellipse_face_crop = self._create_eye_ellipse(
          landmarks5=transformed_landmarks[:5].tolist(),
          image_shape=(
            int(face_crop.shape[0]) if face_crop is not None and face_crop.size else 0,
            int(face_crop.shape[1]) if face_crop is not None and face_crop.size else 0,
          ),
        )
        mouth_ellipse_face_crop = self._create_mouth_ellipse(
          landmarks5=transformed_landmarks[:5].tolist(),
          image_shape=(
            int(face_crop.shape[0]) if face_crop is not None and face_crop.size else 0,
            int(face_crop.shape[1]) if face_crop is not None and face_crop.size else 0,
          ),
        )
      face_info = UpscaleFaceInfo(
        bbox=bbox_norm,
        face_px=face_px,
        algorithm="",
        landmarks5=landmarks5,
        landmarks_all=landmarks_all,
        landmarks_all_face_crop=(
          transformed_landmarks.tolist() if transformed_landmarks is not None else None
        ),
        eye_ellipse=eye_ellipse,
        eye_ellipse_face_crop=eye_ellipse_face_crop,
        mouth_ellipse_face_crop=mouth_ellipse_face_crop,
      )

      crop_on_upscaled = self._cv2_ready_bgr(face_crop) if face_crop is not None and face_crop.size else None
      strong_change_mask: typing.Optional[np.ndarray] = None
      strong_change_mask_color: typing.Optional[np.ndarray] = None
      face_crop_shape = (
        (int(crop_on_upscaled.shape[0]), int(crop_on_upscaled.shape[1]))
        if crop_on_upscaled is not None and crop_on_upscaled.size else None
      )

      if fill_debug_images:
        self._append_face_step(face_info, name="original crop", image=original_crop)
        self._append_face_step(face_info, name="original upscaled", image=crop_on_upscaled)

      current_face = self._cv2_ready_bgr(face_crop)
      algorithms: typing.List[str] = []

      for processor in face_processors:
        if processor.max_apply_px is not None and face_px >= int(processor.max_apply_px):
          continue

        if processor.processor == "codeformer":
          if self._codeformer_net is None:
            continue
          face_crop_for_codeformer = current_face
          if original_crop is not None:
            face_crop_for_codeformer = original_crop
          current_face = self._restore_face_codeformer(
            face_crop_for_codeformer,
            fidelity=codeformer_fidelity,
          )
          algorithms.append("codeformer")
          if fill_debug_images:
            self._append_face_step(face_info, name="codeformer transformed", image=current_face)

        elif processor.processor == "gfpgan":
          if self._face_enhancer is None:
            continue
          gfpgan_small_px = int(processor.max_apply_px) if processor.max_apply_px is not None else None
          w = gfpgan_weight_small if (gfpgan_small_px is not None and face_px < gfpgan_small_px) else gfpgan_weight_normal
          _, local_restored_faces, _ = self._face_enhancer.enhance(
            current_face,
            has_aligned=True,
            only_center_face=True,
            paste_back=False,
            weight=w,
          )
          if local_restored_faces:
            current_face = self._cv2_ready_bgr(local_restored_faces[0])
            algorithms.append("gfpgan")
            if fill_debug_images:
              self._append_face_step(face_info, name="gfpgan transformed", image=current_face)

        elif processor.processor == "restoreformer":
          if self._restoreformer_enhancer is None:
            continue
          _, local_restored_faces, _ = self._restoreformer_enhancer.enhance(
            current_face,
            has_aligned=True,
            only_center_face=True,
            paste_back=False,
            weight=float(gfpgan_weight_normal),
          )
          if local_restored_faces:
            current_face = self._cv2_ready_bgr(local_restored_faces[0])
            algorithms.append("restoreformer")
            if fill_debug_images:
              self._append_face_step(face_info, name="restoreformer transformed", image=current_face)

        elif processor.processor == "rollback_diff":
          if crop_on_upscaled is None:
            crop_on_upscaled = self._cv2_ready_bgr(face_crop)

          if current_face.shape[:2] != crop_on_upscaled.shape[:2]:
            crop_on_upscaled = cv2.resize(
              crop_on_upscaled,
              (current_face.shape[1], current_face.shape[0]),
              interpolation=cv2.INTER_LINEAR,
            )

          if fill_debug_images:
            eye_mask = face_info.get_eye_mask_for_face_crop(face_crop_shape)
            mouth_mask = face_info.get_mouth_mask_for_face_crop(face_crop_shape)
            nose_zone_mask = face_info.get_nose_zone_mask_for_face_crop(face_crop_shape)
            face_masks_overlay = self._build_face_masks_overlay(
              base_image=current_face,
              eye_mask=eye_mask,
              mouth_mask=mouth_mask,
              nose_zone_mask=nose_zone_mask,
            )
            self._append_face_step(face_info, name="face masks", image=face_masks_overlay)

          diff_result = self._diff_zones_mean_window(
            crop_on_upscaled,
            current_face,
            win=3,
            diff_thr=float(diff_thr),
            min_area_ratio=float(diff_min_area),
          )

          rollback_mask = face_info.get_face_rollback_mask(
            strong_change_mask=diff_result.mask_u8,
            strong_change_mask_color=diff_result.mask_color_u8,
            face_crop_shape=face_crop_shape,
            diff_opening_window=float(diff_opening_window),
          )

          if rollback_mask is not None and rollback_mask.size and float(rollback_extend) >= 1e-5:
            extend_kernel_w = max(1, int(round(float(face_crop_shape[1]) * float(rollback_extend))))
            if extend_kernel_w % 2 == 0:
              extend_kernel_w += 1
            if extend_kernel_w > 1:
              extend_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (extend_kernel_w, extend_kernel_w),
              )
              rollback_mask = cv2.dilate(rollback_mask, extend_kernel, iterations=1)

          if fill_debug_images:
            self._append_face_step(face_info, name="base rollback mask", image=rollback_mask)

          if rollback_mask is not None and rollback_mask.size:
            rollback_mask01 = (rollback_mask > 0).astype(np.uint8)
            current_face = np.where(
              rollback_mask01[:, :, None] > 0,
              crop_on_upscaled,
              current_face,
            )

          algorithms.append("rollback_diff")
          if fill_debug_images:
            self._append_face_step(face_info, name="rollback result", image=current_face)

        if processor.stop_apply:
          break

      if not algorithms:
        face_info.algorithm = "fallback"
      else:
        face_info.algorithm = "+".join(algorithms)
      local_restored_faces = [current_face]

      assert local_restored_faces is not None
      restored_faces.extend([self._cv2_ready_bgr(x) for x in local_restored_faces])
      face_infos.append(face_info)

    if not restored_faces:
      return (
        upscaled_bgr,
        dataclasses.replace(info, faces=face_infos),
      )

    pasted = self._face_searcher.paste_restored_faces(restored_faces)

    if fill_debug_images:
      for i, face_info in enumerate(face_infos):
        if i >= len(faces):
          continue
        detection = faces[i]
        crop_w = 0
        crop_h = 0
        if detection.crop is None or not detection.crop.size:
          continue
        crop_h, crop_w = detection.crop.shape[:2]

        if detection.affine_matrix is not None and crop_w > 0 and crop_h > 0:
          pasted_face = cv2.warpAffine(
            pasted,
            detection.affine_matrix,
            (crop_w, crop_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
          )
          self._append_face_step(face_info, name="result", image=pasted_face)
          continue

        px1, py1, px2, py2 = _bbox_norm_to_pixels(detection.bbox_norm, width=up_w, height=up_h)
        if px2 > px1 and py2 > py1:
          pasted_face = pasted[py1:py2, px1:px2].copy()
          self._append_face_step(face_info, name="result", image=pasted_face)

    return (
      pasted,
      dataclasses.replace(info, faces=face_infos),
    )

  def _load_codeformer(self, weights_path: str):
    device = torch.device(self._device)

    net = codeformer.basicsr.archs.codeformer_arch.CodeFormer(
      dim_embd=512,
      codebook_size=1024,
      n_head=8,
      n_layers=9,
      connect_list=["32", "64", "128", "256"],
    ).to(device)

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
      net.load_state_dict(ckpt["params_ema"], strict=True)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
      net.load_state_dict(ckpt["state_dict"], strict=False)
    else:
      net.load_state_dict(ckpt, strict=False)

    net.eval()
    #if self._use_half:
    #  net = net.half()

    return net

  def _restore_face_codeformer(self, face_crop_bgr: np.ndarray, *, fidelity: float) -> np.ndarray:
    # CodeFormer expects RGB, normalized to [-1,1], BCHW
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

    #if self._use_half:
    #  x = x.half()

    x = x.to(next(self._codeformer_net.parameters()).device)

    with torch.no_grad():
      out = self._codeformer_net(x, w=float(fidelity), adain=True)

    # out can be tensor or tuple/list depending on version
    if isinstance(out, (list, tuple)):
      out = out[0]

    out = out.detach().float().cpu().clamp_(-1, 1)
    out = (out * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).numpy()
    out = (out * 255.0).round().astype(np.uint8)

    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return bgr

  def _connected_components_mask_filter(
    self,
    mask_u8: np.ndarray,
    *,
    min_area_ratio: float,
  ) -> typing.Tuple[np.ndarray, typing.List[typing.List[int]]]:
    """
    Remove small connected components and return kept component boxes.
    """
    if mask_u8 is None or mask_u8.size == 0:
      return np.zeros((0, 0), dtype=np.uint8), []

    min_area_ratio_local = float(np.clip(float(min_area_ratio), 0.0, 1.0))
    min_area_px = int(np.ceil(min_area_ratio_local * float(mask_u8.shape[0] * mask_u8.shape[1])))
    if min_area_px <= 0:
      return (mask_u8 > 0).astype(np.uint8) * 255, []

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
      (mask_u8 > 0).astype(np.uint8),
      connectivity=8,
    )
    cleaned = np.zeros_like(mask_u8)
    boxes_local: typing.List[typing.List[int]] = []
    for idx in range(1, num):
      x, y, w, h, area = stats[idx]
      if int(area) >= int(min_area_px):
        cleaned[labels == idx] = 255
        boxes_local.append([int(x), int(y), int(x + w), int(y + h)])

    return cleaned, boxes_local

  def _diff_zones_mean_window(
    self,
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    *,
    win: int = 21,
    diff_thr: float = (20.0 / 255.0),
    min_area_ratio: float = 0.0003,
  ) -> DiffZonesMeanWindowResult:
    """
    Find zones where img1 deviates from img0 significantly, after averaging over a window.

    Returns both intensity-based and summed-color masks.
    """

    if img0_bgr is None or img1_bgr is None:
      raise ValueError("img0_bgr/img1_bgr is None")
    if img0_bgr.shape != img1_bgr.shape:
      raise ValueError(f"Shape mismatch: {img0_bgr.shape} vs {img1_bgr.shape}")
    if img0_bgr.ndim != 3 or img0_bgr.shape[2] != 3:
      raise ValueError(f"Expected HxWx3, got {img0_bgr.shape}")
    if img0_bgr.dtype != np.uint8 or img1_bgr.dtype != np.uint8:
      raise ValueError(f"Expected uint8, got {img0_bgr.dtype} / {img1_bgr.dtype}")

    # 1) Luminance difference (Lab L)
    lab0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2LAB)
    lab1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2LAB)
    L0 = lab0[:, :, 0].astype(np.float32)
    L1 = lab1[:, :, 0].astype(np.float32)
    d = np.abs(L1 - L0)  # 0..255 float32

    # 2) Summed color deviation across all BGR channels (captures color flow)
    bgr0 = img0_bgr.astype(np.float32)
    bgr1 = img1_bgr.astype(np.float32)
    d_color = np.abs(bgr1 - bgr0).sum(axis=2) / 3.0

    # 3) Local averaging over window
    k = int(win)
    if k < 1:
      k = 1
    if (k % 2) == 0:
      k += 1

    diff_mean = cv2.blur(d, (k, k))
    diff_color_mean = cv2.blur(d_color, (k, k))

    # 4) Threshold -> mask
    diff_thr_u8 = float(np.clip(float(diff_thr), 0.0, 1.0) * 255.0)
    mask01 = (diff_mean >= diff_thr_u8).astype(np.float32)
    mask_color01 = (diff_color_mean >= diff_thr_u8).astype(np.float32)

    mask_u8 = mask01
    boxes = []
    mask_color_u8 = mask_color01
    boxes_color = []

    return DiffZonesMeanWindowResult(
      diff_mean=diff_mean,
      diff_color_mean=diff_color_mean,
      mask01=(mask_u8.astype(np.float32) / 255.0),
      mask_color01=(mask_color_u8.astype(np.float32) / 255.0),
      mask_u8=mask_u8,
      mask_color_u8=mask_color_u8,
      d=d,
      d_color=d_color,
    )

  def _strong_change_mask_mean_window(
    self,
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    *,
    win: int,
    diff_thr: float,
  ) -> np.ndarray:
    """
    Return a binary mask for strong changes based on per-window mean color difference.

    - img0_bgr, img1_bgr: input images (any size, will be resized to img0)
    - win: window size for averaging (odd recommended, min 1)
    - diff_thr: normalized threshold in [0, 1]
    """
    if img0_bgr is None or img1_bgr is None:
      raise ValueError("img0_bgr/img1_bgr is None")

    base = self._cv2_ready_bgr(img0_bgr)
    other = self._cv2_ready_bgr(img1_bgr)

    if base.shape[:2] != other.shape[:2]:
      h, w = base.shape[:2]
      interp = cv2.INTER_AREA if (other.shape[0] > h or other.shape[1] > w) else cv2.INTER_LINEAR
      other = cv2.resize(other, (w, h), interpolation=interp)

    k = int(win)
    if k < 1:
      k = 1
    if (k % 2) == 0:
      k += 1

    base_f = base.astype(np.float32)
    other_f = other.astype(np.float32)

    base_mean = cv2.blur(base_f, (k, k))
    other_mean = cv2.blur(other_f, (k, k))

    diff = np.abs(other_mean - base_mean)
    diff_mean = diff.mean(axis=2)

    diff_thr_u8 = float(np.clip(float(diff_thr), 0.0, 1.0) * 255.0)
    mask_u8 = (diff_mean >= diff_thr_u8).astype(np.uint8) * 255
    return mask_u8
