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
  compare: CompareMetrics
  zone: RawBlurMetrics
  reference: RawBlurMetrics
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
class Blurred:
  face_blurred: bool
  eyes_blurred: bool


@dataclasses.dataclass(frozen=True)
class FaceDetection:
  """Face metadata in normalized coordinates relative to source image/crop."""
  # Required for mapping detections back to source image coordinates (paste-back/debug).
  # Kept as normalized [x1, y1, x2, y2] to remain resolution-independent.
  bbox_norm: typing.List[float]
  affine_matrix: typing.Optional[np.ndarray]
  crop: typing.Optional[np.ndarray] = None
  landmarks_all_face_crop: typing.Optional[typing.List[typing.List[float]]] = None
  eye_ellipse: typing.Optional[Ellipse] = None
  mouth_ellipse: typing.Optional[Ellipse] = None
  face_ellipse: typing.Optional[Ellipse] = None

  def __init__(
    self,
    *,
    bbox_px: typing.Optional[typing.List[int]] = None,
    width: int = 0,
    height: int = 0,
    bbox_norm: typing.Optional[typing.List[float]] = None,
    affine_matrix: typing.Optional[np.ndarray] = None,
    crop: typing.Optional[np.ndarray] = None,
    landmarks_all_face_crop: typing.Optional[typing.List[typing.List[float]]] = None,
  ) -> None:
    resolved_bbox_norm = self._resolve_bbox_norm(
      bbox_norm=bbox_norm,
      bbox_px=bbox_px,
      width=width,
      height=height,
    )
    normalized_landmarks = self._normalize_landmarks_face_crop(landmarks_all_face_crop)
    normalized_face_ellipse = None
    normalized_eye_ellipse = None
    normalized_mouth_ellipse = None

    if normalized_landmarks is not None and len(normalized_landmarks) >= 5:
      normalized_face_ellipse = self._create_face_ellipse(landmarks=normalized_landmarks)
      normalized_eye_ellipse = self._create_eye_ellipse(landmarks5=normalized_landmarks[:5])
      normalized_mouth_ellipse = self._create_mouth_ellipse(landmarks5=normalized_landmarks[:5])

    object.__setattr__(self, "bbox_norm", resolved_bbox_norm)
    object.__setattr__(self, "affine_matrix", affine_matrix)
    object.__setattr__(self, "crop", crop)
    object.__setattr__(self, "landmarks_all_face_crop", normalized_landmarks)
    object.__setattr__(self, "eye_ellipse", normalized_eye_ellipse)
    object.__setattr__(self, "mouth_ellipse", normalized_mouth_ellipse)
    object.__setattr__(self, "face_ellipse", normalized_face_ellipse)

  @property
  def size_px(self) -> int:
    crop = self.crop
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0 or crop.ndim < 2:
      return 0
    h, w = crop.shape[:2]
    return int(min(int(h), int(w)))

  def change_crop(self, new_crop: np.ndarray) -> "FaceDetection":
    return FaceDetection(
      bbox_norm=self._resolve_bbox_norm(bbox_norm=self.bbox_norm, bbox_px=None, width=0, height=0),
      crop=new_crop,
      landmarks_all_face_crop=self._normalize_landmarks_face_crop(self.landmarks_all_face_crop),
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

    face_mask = (self.get_face_mask((h, w)) > 0).astype(np.uint8)
    if np.count_nonzero(face_mask) <= 0:
      outside_face_mask = np.ones((h, w), dtype=np.uint8)
    else:
      outside_face_mask = (face_mask == 0).astype(np.uint8)

    outside_face_raw, outside_face_area_ratio, outside_face_min_side, outside_face_valid = self._compute_raw_masked_blur_metrics(
      image=crop,
      mask_u8=outside_face_mask,
      normalize_size=normalize_size,
      min_mask_area_px=min_mask_area_px,
      min_roi_side_px=min_roi_side_px,
      rgb_input=rgb_input,
    )

    face_zone_raw, face_area_ratio, face_min_side, face_valid = self._compute_raw_masked_blur_metrics(
      image=crop,
      mask_u8=parts_union_mask,
      normalize_size=normalize_size,
      min_mask_area_px=min_mask_area_px,
      min_roi_side_px=min_roi_side_px,
      rgb_input=rgb_input,
    )
    face_metrics = ZoneBlurMetrics(
      zone=face_zone_raw,
      reference=outside_face_raw,
      compare=self._compare_metrics(
        zone=face_zone_raw,
        reference=outside_face_raw,
        eps=float(eps),
        valid_reference=outside_face_valid,
      ),
      zone_area_ratio=face_area_ratio,
      reference_area_ratio=outside_face_area_ratio,
      zone_min_size_px=face_min_side,
      reference_min_size_px=outside_face_min_side,
      valid_zone=face_valid,
      valid_reference=outside_face_valid,
    )

    return FacePrivacyBlurMetrics(
      eyes_blur=eyes_metrics,
      face_blur=face_metrics,
      outside_parts=outside_raw,
    )

  def is_blurred(
    self,
    *,
    face_edge_density_ratio_threshold: float = 0.0421462,
    face_lap_var_ratio_threshold: float = 0.0092719,
    face_pixel_var_ratio_threshold: float = 0.0667524,
    face_tenengrad_ratio_threshold: float = 0.0562823,
    eyes_edge_density_ratio_threshold: float = 0.0542813,
    eyes_lap_var_ratio_threshold: float = 0.0158647,
    eyes_pixel_var_ratio_threshold: float = 0.285752,
    eyes_tenengrad_ratio_threshold: float = 0.082788,
    privacy_blur_metrics: typing.Optional[FacePrivacyBlurMetrics] = None,
  ) -> Blurred:
    metrics = self.compute_privacy_blur_metrics() if privacy_blur_metrics is None else privacy_blur_metrics
    face_blur = metrics.face_blur
    if face_blur.valid_zone < 0.5 or face_blur.valid_reference < 0.5:
      return Blurred(face_blurred=False, eyes_blurred=False)

    face_ratios = face_blur.compare
    face_ratio_values = [
      float(face_ratios.edge_density_ratio),
      float(face_ratios.lap_var_ratio),
      float(face_ratios.pixel_var_ratio),
      float(face_ratios.tenengrad_ratio),
    ]
    if not all(np.isfinite(value) for value in face_ratio_values):
      return Blurred(face_blurred=False, eyes_blurred=False)

    face_blurred = (
      float(face_ratios.edge_density_ratio) <= float(face_edge_density_ratio_threshold)
      and float(face_ratios.lap_var_ratio) <= float(face_lap_var_ratio_threshold)
      and float(face_ratios.pixel_var_ratio) <= float(face_pixel_var_ratio_threshold)
      and float(face_ratios.tenengrad_ratio) <= float(face_tenengrad_ratio_threshold)
    )
    if face_blurred:
      return Blurred(face_blurred=True, eyes_blurred=True)

    eyes_blur = metrics.eyes_blur
    if eyes_blur.valid_zone < 0.5 or eyes_blur.valid_reference < 0.5:
      return Blurred(face_blurred=False, eyes_blurred=False)

    eyes_ratios = eyes_blur.compare
    eyes_ratio_values = [
      float(eyes_ratios.edge_density_ratio),
      float(eyes_ratios.lap_var_ratio),
      float(eyes_ratios.pixel_var_ratio),
      float(eyes_ratios.tenengrad_ratio),
    ]
    if not all(np.isfinite(value) for value in eyes_ratio_values):
      return Blurred(face_blurred=False, eyes_blurred=False)

    eyes_blurred = (
      float(eyes_ratios.edge_density_ratio) <= float(eyes_edge_density_ratio_threshold)
      and float(eyes_ratios.lap_var_ratio) <= float(eyes_lap_var_ratio_threshold)
      and float(eyes_ratios.pixel_var_ratio) <= float(eyes_pixel_var_ratio_threshold)
      and float(eyes_ratios.tenengrad_ratio) <= float(eyes_tenengrad_ratio_threshold)
    )
    return Blurred(face_blurred=False, eyes_blurred=eyes_blurred)

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

  def get_face_mask(self, face_crop_shape: typing.Tuple[int, int]) -> np.ndarray:
    height, width = face_crop_shape
    return self._render_ellipse_mask(self.face_ellipse, width=width, height=height)

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

  @staticmethod
  def _create_face_ellipse(*, landmarks: typing.List[typing.List[float]]) -> typing.Optional[Ellipse]:
    if not landmarks:
      return None

    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
      return None

    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0)

    if points.shape[0] >= 5:
      left_eye, right_eye = points[0], points[1]
      mouth_left, mouth_right = points[3], points[4]
      eye_center = (left_eye + right_eye) * 0.5
      mouth_center = (mouth_left + mouth_right) * 0.5

      eye_vec = right_eye - left_eye
      eye_dist = float(np.hypot(float(eye_vec[0]), float(eye_vec[1])))
      eye_to_mouth = mouth_center - eye_center
      eye_mouth_dist = float(np.hypot(float(eye_to_mouth[0]), float(eye_to_mouth[1])))

      if eye_dist > 1e-6 and eye_mouth_dist > 1e-6:
        # Shift center toward forehead: 0.74 of the distance from mouth to eyes.
        center = mouth_center + 0.74 * (eye_center - mouth_center)
        angle = float(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))

        # Width requirement: full ellipse width should be 2 * inter-eye distance.
        axis_x = eye_dist

        # Height requirement: full ellipse height should be 3 * eye-to-mouth distance.
        axis_y = eye_mouth_dist * 1.50
        return Ellipse(
          center=(float(center[0]), float(center[1])),
          axes=(max(1e-6, axis_x), max(1e-6, axis_y)),
          angle=angle,
        )

    hull_points = points
    has_eye_points = points.shape[0] > 1
    has_mouth_points = points.shape[0] > 4

    if points.shape[0] > 2 and has_eye_points and has_mouth_points:
      hull_points = np.delete(points, 2, axis=0)

    hull = cv2.convexHull(hull_points.astype(np.float32).reshape(-1, 1, 2))
    if hull is None or hull.shape[0] < 3:
      return None

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(hull)
    major = max(float(rw), float(rh)) * 1.40
    minor = min(float(rw), float(rh)) * 2.80
    return Ellipse(
      center=(float(cx), float(cy)),
      axes=(max(1e-6, major * 0.5), max(1e-6, minor * 0.5)),
      angle=float(angle),
    )

  @staticmethod
  def _create_eye_ellipse(*, landmarks5: typing.List[typing.List[float]]) -> typing.Optional[Ellipse]:
    if not landmarks5 or len(landmarks5) < 3:
      return None

    points = np.asarray(landmarks5, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
      return None

    left_eye, right_eye, nose = points[:3]
    lx, ly = float(left_eye[0]), float(left_eye[1])
    rx, ry = float(right_eye[0]), float(right_eye[1])
    nx, ny = float(nose[0]), float(nose[1])

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

    angle_deg = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
    return Ellipse(
      center=(eye_center_x, eye_center_y),
      axes=(max(1e-6, eye_dist * 0.90), max(1e-6, max(eye_dist * 0.38, nose_dist * 0.30))),
      angle=angle_deg,
    )

  @staticmethod
  def _create_mouth_ellipse(*, landmarks5: typing.List[typing.List[float]]) -> typing.Optional[Ellipse]:
    if not landmarks5 or len(landmarks5) < 5:
      return None

    points = np.asarray(landmarks5, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 5:
      return None

    left_eye, right_eye, _, mouth_left, mouth_right = points[:5]
    lx, ly = float(left_eye[0]), float(left_eye[1])
    rx, ry = float(right_eye[0]), float(right_eye[1])
    mlx, mly = float(mouth_left[0]), float(mouth_left[1])
    mrx, mry = float(mouth_right[0]), float(mouth_right[1])

    mouth_center_x = (mlx + mrx) * 0.5
    mouth_center_y = (mly + mry) * 0.5
    mouth_dist = float(np.hypot(mrx - mlx, mry - mly))
    eye_dist = float(np.hypot(rx - lx, ry - ly))
    if mouth_dist <= 1e-6 and eye_dist <= 1e-6:
      return None

    angle_deg = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    return Ellipse(
      center=(mouth_center_x, mouth_center_y),
      axes=(max(1e-6, max(mouth_dist * 0.80, eye_dist * 0.26)), max(1e-6, max(mouth_dist * 0.42, eye_dist * 0.18))),
      angle=angle_deg,
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

    # Keep all metrics that depend on per-pixel magnitude normalized by the
    # number of masked pixels they are computed over.
    lap_var = float(np.sum((lap_vals - np.mean(lap_vals)) ** 2) / float(masked_count)) if lap_vals.size else np.nan
    tenengrad = float(np.sum(tenengrad_vals) / float(masked_count)) if tenengrad_vals.size else np.nan
    pixel_var = float(np.sum((gray_vals - np.mean(gray_vals)) ** 2) / float(masked_count)) if gray_vals.size else np.nan

    metrics = RawBlurMetrics(
      lap_var=lap_var,
      tenengrad=tenengrad,
      edge_density=float(edges) / float(masked_count) if masked_count > 0 else np.nan,
      pixel_var=pixel_var,
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
