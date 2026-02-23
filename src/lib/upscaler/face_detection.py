import dataclasses
import typing

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class Ellipse:
  center: typing.Tuple[float, float]
  axes: typing.Tuple[float, float]
  angle: float


@dataclasses.dataclass(frozen=True)
class RawBlurMetrics:
  lap_var: float
  tenengrad: float
  edge_density: float
  pixel_var: float


@dataclasses.dataclass(frozen=True)
class CompareMetrics:
  lap_var_diff: float
  lap_var_ratio: float
  tenengrad_diff: float
  tenengrad_ratio: float
  edge_density_diff: float
  edge_density_ratio: float
  pixel_var_diff: float
  pixel_var_ratio: float


@dataclasses.dataclass(frozen=True)
class ZoneBlurMetrics:
  zone: RawBlurMetrics
  reference: RawBlurMetrics
  compare: CompareMetrics
  zone_area_ratio: float
  reference_area_ratio: float
  zone_min_size_px: float
  reference_min_size_px: float
  valid_zone: float
  valid_reference: float


@dataclasses.dataclass(frozen=True)
class FacePrivacyBlurMetrics:
  eyes_blur: ZoneBlurMetrics
  face_blur: ZoneBlurMetrics
  outside_parts: RawBlurMetrics


@dataclasses.dataclass(frozen=True)
class FaceDetection:
  """Face metadata in normalized coordinates relative to source image/crop."""
  # Required for mapping detections back to source image coordinates (paste-back/debug).
  # Kept as normalized [x1, y1, x2, y2] to remain resolution-independent.
  bbox_norm: typing.List[float]
  affine_matrix: typing.Optional[np.ndarray]
  crop: typing.Optional[np.ndarray] = None
  landmarks_5: typing.Optional[np.ndarray] = None
  landmarks_all: typing.Optional[np.ndarray] = None
  landmarks_all_face_crop: typing.Optional[typing.List[typing.List[float]]] = None
  eye_ellipse: typing.Optional[Ellipse] = None
  mouth_ellipse: typing.Optional[Ellipse] = None

  def __init__(
    self,
    *,
    bbox_px: typing.Optional[typing.List[int]] = None,
    width: int = 0,
    height: int = 0,
    bbox_norm: typing.Optional[typing.List[float]] = None,
    affine_matrix: typing.Optional[np.ndarray] = None,
    crop: typing.Optional[np.ndarray] = None,
    landmarks_5: typing.Optional[np.ndarray] = None,
    landmarks_all: typing.Optional[np.ndarray] = None,
    landmarks_all_face_crop: typing.Optional[typing.List[typing.List[float]]] = None,
    eye_ellipse: typing.Optional[Ellipse] = None,
    mouth_ellipse: typing.Optional[Ellipse] = None,
  ) -> None:
    resolved_bbox_norm = self._resolve_bbox_norm(
      bbox_norm=bbox_norm,
      bbox_px=bbox_px,
      width=width,
      height=height,
    )
    object.__setattr__(self, "bbox_norm", resolved_bbox_norm)
    object.__setattr__(self, "affine_matrix", affine_matrix)
    object.__setattr__(self, "crop", crop)
    object.__setattr__(self, "landmarks_5", landmarks_5)
    object.__setattr__(self, "landmarks_all", landmarks_all)
    object.__setattr__(self, "landmarks_all_face_crop", self._normalize_landmarks_face_crop(landmarks_all_face_crop))
    object.__setattr__(self, "eye_ellipse", self._normalize_ellipse(eye_ellipse))
    object.__setattr__(self, "mouth_ellipse", self._normalize_ellipse(mouth_ellipse))

  @property
  def size_px(self) -> int:
    crop = self.crop
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim < 2:
      return 0
    h, w = crop.shape[:2]
    return int(min(int(h), int(w)))

  @staticmethod
  def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))

  @classmethod
  def _resolve_bbox_norm(
    cls,
    *,
    bbox_norm: typing.Optional[typing.List[float]],
    bbox_px: typing.Optional[typing.List[int]],
    width: int,
    height: int,
  ) -> typing.List[float]:
    if bbox_norm is not None and len(bbox_norm) >= 4:
      x1_f, y1_f, x2_f, y2_f = [cls._clip01(v) for v in bbox_norm[:4]]
      x1_f, x2_f = min(x1_f, x2_f), max(x1_f, x2_f)
      y1_f, y2_f = min(y1_f, y2_f), max(y1_f, y2_f)
      return [x1_f, y1_f, x2_f, y2_f]

    if bbox_px is None or len(bbox_px) < 4:
      return [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = [float(v) for v in bbox_px[:4]]
    x1_f = cls._clip01(x1 / float(width) if width else 0.0)
    y1_f = cls._clip01(y1 / float(height) if height else 0.0)
    x2_f = cls._clip01(x2 / float(width) if width else 0.0)
    y2_f = cls._clip01(y2 / float(height) if height else 0.0)
    x1_f, x2_f = min(x1_f, x2_f), max(x1_f, x2_f)
    y1_f, y2_f = min(y1_f, y2_f), max(y1_f, y2_f)
    return [x1_f, y1_f, x2_f, y2_f]

  @classmethod
  def _normalize_landmarks_face_crop(
    cls,
    landmarks_all_face_crop: typing.Optional[typing.List[typing.List[float]]],
  ) -> typing.Optional[typing.List[typing.List[float]]]:
    if landmarks_all_face_crop is None:
      return None
    result: typing.List[typing.List[float]] = []
    for point in landmarks_all_face_crop:
      if point is None or len(point) < 2:
        continue
      result.append([cls._clip01(point[0]), cls._clip01(point[1])])
    return result

  @classmethod
  def _normalize_ellipse(cls, ellipse: typing.Optional[Ellipse]) -> typing.Optional[Ellipse]:
    if ellipse is None:
      return None
    return Ellipse(
      center=(cls._clip01(ellipse.center[0]), cls._clip01(ellipse.center[1])),
      axes=(cls._clip01(ellipse.axes[0]), cls._clip01(ellipse.axes[1])),
      angle=float(ellipse.angle),
    )

  def change_crop(self, new_crop: np.ndarray) -> "FaceDetection":
    return dataclasses.replace(
      self,
      crop=new_crop,
      bbox_norm=self._resolve_bbox_norm(bbox_norm=self.bbox_norm, bbox_px=None, width=0, height=0),
      landmarks_all_face_crop=self._normalize_landmarks_face_crop(self.landmarks_all_face_crop),
      eye_ellipse=self._normalize_ellipse(self.eye_ellipse),
      mouth_ellipse=self._normalize_ellipse(self.mouth_ellipse),
    )

  @staticmethod
  def _nan_raw_metrics() -> RawBlurMetrics:
    return RawBlurMetrics(np.nan, np.nan, np.nan, np.nan)

  @staticmethod
  def _nan_compare_metrics() -> CompareMetrics:
    return CompareMetrics(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

  @staticmethod
  def _to_gray_float32(image: np.ndarray, *, rgb_input: bool) -> np.ndarray:
    if image.ndim == 2:
      gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
      gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] >= 3:
      code = cv2.COLOR_RGB2GRAY if rgb_input else cv2.COLOR_BGR2GRAY
      gray = cv2.cvtColor(image[:, :, :3], code)
    else:
      gray = np.zeros(image.shape[:2], dtype=np.float32)
    return gray.astype(np.float32)

  @classmethod
  def _compute_raw_masked_blur_metrics(
    cls,
    *,
    image: np.ndarray,
    mask_u8: np.ndarray,
    normalize_size: typing.Tuple[int, int],
    min_mask_area_px: int,
    min_roi_side_px: int,
    rgb_input: bool,
  ) -> typing.Tuple[RawBlurMetrics, float, float, float]:
    if image is None or image.size == 0 or mask_u8 is None or mask_u8.size == 0:
      return cls._nan_raw_metrics(), 0.0, np.nan, 0.0

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
      return cls._nan_raw_metrics(), 0.0, np.nan, 0.0

    valid_mask = (mask_u8 > 0).astype(np.uint8)
    area = int(np.count_nonzero(valid_mask))
    if area <= 0:
      return cls._nan_raw_metrics(), 0.0, np.nan, 0.0

    x, y, bw, bh = cv2.boundingRect(valid_mask)
    min_side = float(min(bw, bh))
    area_ratio = float(area) / float(h * w) if (h * w) > 0 else 0.0
    if area < int(min_mask_area_px) or min_side < float(min_roi_side_px):
      return cls._nan_raw_metrics(), area_ratio, min_side, 0.0

    roi_img = image[y:y + bh, x:x + bw]
    roi_mask = valid_mask[y:y + bh, x:x + bw]
    if roi_img.size == 0 or roi_mask.size == 0:
      return cls._nan_raw_metrics(), area_ratio, min_side, 0.0

    norm_w = max(1, int(normalize_size[0]))
    norm_h = max(1, int(normalize_size[1]))
    roi_img = cv2.resize(roi_img, (norm_w, norm_h), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(roi_mask, (norm_w, norm_h), interpolation=cv2.INTER_NEAREST)
    roi_mask_bool = roi_mask > 0
    masked_count = int(np.count_nonzero(roi_mask_bool))
    if masked_count <= 0:
      return cls._nan_raw_metrics(), area_ratio, min_side, 0.0

    gray = cls._to_gray_float32(roi_img, rgb_input=rgb_input)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    canny = cv2.Canny(np.clip(gray, 0, 255).astype(np.uint8), 100, 200)

    lap_vals = lap[roi_mask_bool]
    tenengrad_vals = (gx * gx + gy * gy)[roi_mask_bool]
    gray_vals = gray[roi_mask_bool]
    edges = np.count_nonzero((canny > 0) & roi_mask_bool)

    metrics = RawBlurMetrics(
      lap_var=float(np.var(lap_vals)) if lap_vals.size else np.nan,
      tenengrad=float(np.mean(tenengrad_vals)) if tenengrad_vals.size else np.nan,
      edge_density=float(edges) / float(masked_count) if masked_count > 0 else np.nan,
      pixel_var=float(np.var(gray_vals)) if gray_vals.size else np.nan,
    )
    return metrics, area_ratio, min_side, 1.0

  @classmethod
  def _compare_metrics(
    cls,
    *,
    zone: RawBlurMetrics,
    reference: RawBlurMetrics,
    eps: float,
    valid_reference: float,
  ) -> CompareMetrics:
    if valid_reference < 0.5:
      return cls._nan_compare_metrics()

    return CompareMetrics(
      lap_var_diff=float(zone.lap_var - reference.lap_var),
      lap_var_ratio=float(zone.lap_var / (reference.lap_var + eps)),
      tenengrad_diff=float(zone.tenengrad - reference.tenengrad),
      tenengrad_ratio=float(zone.tenengrad / (reference.tenengrad + eps)),
      edge_density_diff=float(zone.edge_density - reference.edge_density),
      edge_density_ratio=float(zone.edge_density / (reference.edge_density + eps)),
      pixel_var_diff=float(zone.pixel_var - reference.pixel_var),
      pixel_var_ratio=float(zone.pixel_var / (reference.pixel_var + eps)),
    )

  def _empty_privacy_blur_metrics(self) -> FacePrivacyBlurMetrics:
    empty_zone = ZoneBlurMetrics(
      zone=self._nan_raw_metrics(),
      reference=self._nan_raw_metrics(),
      compare=self._nan_compare_metrics(),
      zone_area_ratio=np.nan,
      reference_area_ratio=np.nan,
      zone_min_size_px=np.nan,
      reference_min_size_px=np.nan,
      valid_zone=0.0,
      valid_reference=0.0,
    )
    return FacePrivacyBlurMetrics(
      eyes_blur=empty_zone,
      face_blur=empty_zone,
      outside_parts=self._nan_raw_metrics(),
    )

  def compute_privacy_blur_metrics(
    self,
    *,
    face_crop: typing.Optional[np.ndarray] = None,
    normalize_size: typing.Tuple[int, int] = (128, 128),
    min_mask_area_px: int = 64,
    min_roi_side_px: int = 24,
    rgb_input: bool = False,
    eps: float = 1e-6,
  ) -> FacePrivacyBlurMetrics:
    """
    Example:
      metrics = detection.compute_privacy_blur_metrics()
      resized = cv2.resize(detection.crop, (256, 256), interpolation=cv2.INTER_LINEAR)
      metrics_resized = detection.change_crop(resized).compute_privacy_blur_metrics()
    """
    crop = self.crop if face_crop is None else face_crop
    if crop is None or crop.size == 0:
      return self._empty_privacy_blur_metrics()

    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
      return self._empty_privacy_blur_metrics()

    eye_mask = self.get_eye_mask((h, w))
    mouth_mask = self.get_mouth_mask((h, w))
    nose_mask = self.get_nose_zone_mask((h, w))
    parts_union_mask = cv2.bitwise_or(
      cv2.bitwise_or(
        (eye_mask > 0).astype(np.uint8),
        (mouth_mask > 0).astype(np.uint8)
      ),
      (nose_mask > 0).astype(np.uint8)
    )
    if np.count_nonzero(parts_union_mask) <= 0:
      background_mask = np.ones((h, w), dtype=np.uint8)
    else:
      background_mask = (parts_union_mask == 0).astype(np.uint8)

    outside_raw, outside_area_ratio, outside_min_side, outside_valid = self._compute_raw_masked_blur_metrics(
      image=crop,
      mask_u8=background_mask,
      normalize_size=normalize_size,
      min_mask_area_px=min_mask_area_px,
      min_roi_side_px=min_roi_side_px,
      rgb_input=rgb_input,
    )

    eye_zone_raw, eye_area_ratio, eye_min_side, eye_valid = self._compute_raw_masked_blur_metrics(
      image=crop,
      mask_u8=(eye_mask > 0).astype(np.uint8),
      normalize_size=normalize_size,
      min_mask_area_px=min_mask_area_px,
      min_roi_side_px=min_roi_side_px,
      rgb_input=rgb_input,
    )
    eyes_metrics = ZoneBlurMetrics(
      zone=eye_zone_raw,
      reference=outside_raw,
      compare=self._compare_metrics(
        zone=eye_zone_raw,
        reference=outside_raw,
        eps=float(eps),
        valid_reference=outside_valid,
      ),
      zone_area_ratio=eye_area_ratio,
      reference_area_ratio=outside_area_ratio,
      zone_min_size_px=eye_min_side,
      reference_min_size_px=outside_min_side,
      valid_zone=eye_valid,
      valid_reference=outside_valid,
    )

    full_face_mask = np.ones((h, w), dtype=np.uint8)
    face_zone_raw, face_area_ratio, face_min_side, face_valid = self._compute_raw_masked_blur_metrics(
      image=crop,
      mask_u8=full_face_mask,
      normalize_size=normalize_size,
      min_mask_area_px=min_mask_area_px,
      min_roi_side_px=min_roi_side_px,
      rgb_input=rgb_input,
    )
    face_metrics = ZoneBlurMetrics(
      zone=face_zone_raw,
      reference=outside_raw,
      compare=self._compare_metrics(
        zone=face_zone_raw,
        reference=outside_raw,
        eps=float(eps),
        valid_reference=outside_valid
      ),
      zone_area_ratio=face_area_ratio,
      reference_area_ratio=outside_area_ratio,
      zone_min_size_px=face_min_side,
      reference_min_size_px=outside_min_side,
      valid_zone=face_valid,
      valid_reference=outside_valid,
    )

    return FacePrivacyBlurMetrics(
      eyes_blur=eyes_metrics,
      face_blur=face_metrics,
      outside_parts=outside_raw,
    )

  @staticmethod
  def _render_ellipse_mask(
    ellipse: typing.Optional[Ellipse],
    *,
    width: int,
    height: int,
  ) -> np.ndarray:
    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)
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

  def get_eye_mask(self, face_crop_shape: typing.Tuple[int, int]) -> np.ndarray:
    height, width = face_crop_shape
    return self._render_ellipse_mask(self.eye_ellipse, width=width, height=height)

  def get_mouth_mask(self, face_crop_shape: typing.Tuple[int, int]) -> np.ndarray:
    height, width = face_crop_shape
    return self._render_ellipse_mask(self.mouth_ellipse, width=width, height=height)

  def get_nose_zone_mask(self, face_crop_shape: typing.Tuple[int, int]) -> np.ndarray:
    height, width = face_crop_shape
    if width <= 0 or height <= 0:
      return np.zeros((0, 0), dtype=np.uint8)

    if self.landmarks_all_face_crop is None or len(self.landmarks_all_face_crop) < 5:
      return np.zeros((height, width), dtype=np.uint8)

    points = np.asarray(self.landmarks_all_face_crop, dtype=np.float32).reshape(-1, 2)
    left_eye, right_eye, nose, mouth_left, mouth_right = points[:5]
    points_local = [
      (float(left_eye[0]), float(left_eye[1])),
      (float(right_eye[0]), float(right_eye[1])),
      (float(nose[0]), float(nose[1])),
      (float(mouth_right[0]), float(mouth_right[1])),
      (float(mouth_left[0]), float(mouth_left[1])),
    ]

    polygon = np.asarray(
      [[int(round(x * width)), int(round(y * height))] for x, y in points_local],
      dtype=np.int32,
    )
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
      return np.zeros((height, width), dtype=np.uint8)

    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(0, width - 1))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(0, height - 1))
    hull = cv2.convexHull(polygon)
    if hull is None or hull.size == 0:
      return np.zeros((height, width), dtype=np.uint8)

    hull = hull.reshape(-1, 2)
    if hull.shape[0] < 3:
      return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    nose_x = int(round(float(nose[0]) * width))
    nose_y = int(round(float(nose[1]) * height))
    nose_x = int(np.clip(nose_x, 0, max(0, width - 1)))
    nose_y = int(np.clip(nose_y, 0, max(0, height - 1)))

    mouth_lx = float(mouth_left[0]) * width
    mouth_ly = float(mouth_left[1]) * height
    mouth_rx = float(mouth_right[0]) * width
    mouth_ry = float(mouth_right[1]) * height
    min_dist_to_mouth = min(
      float(np.hypot(mouth_lx - float(nose_x), mouth_ly - float(nose_y))),
      float(np.hypot(mouth_rx - float(nose_x), mouth_ry - float(nose_y))),
    )
    nose_radius = max(1, int(round(min_dist_to_mouth / 3.0)))
    cv2.circle(mask, (nose_x, nose_y), nose_radius, 255, -1)
    return mask
