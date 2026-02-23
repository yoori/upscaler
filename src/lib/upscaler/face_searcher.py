import dataclasses
import typing

import cv2
import numpy as np

import facexlib.utils.face_restoration_helper


@dataclasses.dataclass(frozen=True)
class Ellipse:
  center: typing.Tuple[float, float]
  axes: typing.Tuple[float, float]
  angle: float


@dataclasses.dataclass(frozen=True)
class FaceDetection:
  bbox_px: typing.List[int]
  bbox_norm: typing.List[float]
  size_px: int
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
    bbox_px: typing.List[int],
    width: int,
    height: int,
    affine_matrix: typing.Optional[np.ndarray] = None,
    crop: typing.Optional[np.ndarray] = None,
    landmarks_5: typing.Optional[np.ndarray] = None,
    landmarks_all: typing.Optional[np.ndarray] = None,
    landmarks_all_face_crop: typing.Optional[typing.List[typing.List[float]]] = None,
    eye_ellipse: typing.Optional[Ellipse] = None,
    mouth_ellipse: typing.Optional[Ellipse] = None,
  ) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    object.__setattr__(self, "bbox_px", [x1, y1, x2, y2])
    object.__setattr__(
      self,
      "bbox_norm",
      [
        float(x1) / float(width) if width else 0.0,
        float(y1) / float(height) if height else 0.0,
        float(x2) / float(width) if width else 0.0,
        float(y2) / float(height) if height else 0.0,
      ],
    )
    object.__setattr__(self, "size_px", int(min(int(x2 - x1), int(y2 - y1))))
    object.__setattr__(self, "affine_matrix", affine_matrix)
    object.__setattr__(self, "crop", crop)
    object.__setattr__(self, "landmarks_5", landmarks_5)
    object.__setattr__(self, "landmarks_all", landmarks_all)
    object.__setattr__(self, "landmarks_all_face_crop", landmarks_all_face_crop)
    object.__setattr__(self, "eye_ellipse", eye_ellipse)
    object.__setattr__(self, "mouth_ellipse", mouth_ellipse)

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


class FaceSearcher:
  def __init__(self, *, device: str, face_size: int = 512) -> None:
    self._helper = facexlib.utils.face_restoration_helper.FaceRestoreHelper(
      upscale_factor=1,
      face_size=face_size,
      crop_ratio=(1, 1),
      det_model="retinaface_resnet50",
      save_ext="png",
      use_parse=True,
      device=device,
    )

  def get_faces(
    self,
    img: np.ndarray,
    is_bgr: bool = False,
    *,
    only_center_face: bool = False,
  ) -> typing.List[FaceDetection]:
    img_bgr = img if is_bgr else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    helper = self._helper
    helper.clean_all()
    helper.read_image(img_bgr)
    helper.get_face_landmarks_5(
      only_center_face=only_center_face,
      eye_dist_threshold=5,
    )
    helper.align_warp_face()

    det_faces = getattr(helper, "det_faces", [])
    affine_matrices = getattr(helper, "affine_matrices", [])
    cropped_faces = getattr(helper, "cropped_faces", [])
    all_landmarks_68 = getattr(helper, "all_landmarks_68", None)
    all_landmarks_5 = getattr(helper, "all_landmarks_5", [])

    h, w = img_bgr.shape[:2]
    faces: typing.List[FaceDetection] = []
    for i, bb in enumerate(det_faces):
      x1, y1, x2, y2 = bb[:4]
      landmarks_all = None
      if all_landmarks_68 is not None and i < len(all_landmarks_68):
        landmarks_all = np.asarray(all_landmarks_68[i], dtype=float)
      elif i < len(all_landmarks_5):
        landmarks_all = np.asarray(all_landmarks_5[i], dtype=float)

      landmarks_5 = None
      if i < len(all_landmarks_5):
        landmarks_5 = np.asarray(all_landmarks_5[i], dtype=float)

      faces.append(FaceDetection(
        bbox_px=[int(x1), int(y1), int(x2), int(y2)],
        width=w,
        height=h,
        affine_matrix=affine_matrices[i] if i < len(affine_matrices) else None,
        crop=cropped_faces[i] if i < len(cropped_faces) else None,
        landmarks_5=landmarks_5,
        landmarks_all=landmarks_all,
      ))
    return faces

  def paste_restored_faces(self, restored_faces: typing.List[np.ndarray]) -> np.ndarray:
    helper = self._helper
    for restored_face in restored_faces:
      helper.add_restored_face(restored_face)

    helper.get_inverse_affine(None)
    return helper.paste_faces_to_input_image(upsample_img=None)
