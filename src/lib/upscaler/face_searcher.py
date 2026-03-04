import typing

import cv2
import numpy as np

import facexlib.utils.face_restoration_helper

from .face_detection import FaceDetection


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

      affine_matrix = affine_matrices[i] if i < len(affine_matrices) else None
      crop = cropped_faces[i] if i < len(cropped_faces) else None
      landmarks_all_face_crop = self._project_landmarks_to_face_crop(
        landmarks_all=landmarks_all,
        bbox=[float(x1), float(y1), float(x2), float(y2)],
        image_shape=(h, w),
        crop_shape=(int(crop.shape[0]), int(crop.shape[1])) if crop is not None and crop.size else None,
        affine_matrix=affine_matrix,
      )
      faces.append(FaceDetection(
        bbox_px=[int(x1), int(y1), int(x2), int(y2)],
        width=w,
        height=h,
        affine_matrix=affine_matrix,
        crop=crop,
        landmarks_all_face_crop=landmarks_all_face_crop,
      ))
    return faces

  @staticmethod
  def _project_landmarks_to_face_crop(
    *,
    landmarks_all: typing.Optional[np.ndarray],
    bbox: typing.List[float],
    image_shape: typing.Tuple[int, int],
    crop_shape: typing.Optional[typing.Tuple[int, int]],
    affine_matrix: typing.Optional[np.ndarray],
  ) -> typing.Optional[typing.List[typing.List[float]]]:
    if landmarks_all is None:
      return None
    points = np.asarray(landmarks_all, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
      return None
    if crop_shape is None or crop_shape[0] <= 0 or crop_shape[1] <= 0:
      return None

    crop_h, crop_w = int(crop_shape[0]), int(crop_shape[1])
    image_h, image_w = int(image_shape[0]), int(image_shape[1])

    transformed = points.copy()
    if affine_matrix is not None:
      transformed = cv2.transform(transformed[None, :, :], np.asarray(affine_matrix, dtype=np.float32))[0]
      transformed[:, 0] /= float(crop_w) if crop_w else 1.0
      transformed[:, 1] /= float(crop_h) if crop_h else 1.0
      return transformed.tolist()

    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = max(1e-8, x2 - x1)
    box_h = max(1e-8, y2 - y1)
    transformed[:, 0] = (transformed[:, 0] - x1) / box_w
    transformed[:, 1] = (transformed[:, 1] - y1) / box_h
    transformed[:, 0] = np.clip(transformed[:, 0], 0.0, 1.0)
    transformed[:, 1] = np.clip(transformed[:, 1], 0.0, 1.0)
    if image_w <= 0 or image_h <= 0:
      return None
    return transformed.tolist()

  def create_original_face_detection(
    self,
    *,
    face_detection: FaceDetection,
    original_bgr: np.ndarray,
    upscaled_shape: typing.Tuple[int, int],
  ) -> typing.Optional[FaceDetection]:
    face_crop = face_detection.crop
    if face_crop is None or not face_crop.size:
      return None

    up_h, up_w = int(upscaled_shape[0]), int(upscaled_shape[1])
    orig_h, orig_w = original_bgr.shape[:2]
    bbox_norm = face_detection.bbox_norm

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
    else:
      ox1 = int(round(float(bbox_norm[0]) * orig_w))
      oy1 = int(round(float(bbox_norm[1]) * orig_h))
      ox2 = int(round(float(bbox_norm[2]) * orig_w))
      oy2 = int(round(float(bbox_norm[3]) * orig_h))
      if ox2 > ox1 and oy2 > oy1:
        original_crop = original_bgr[oy1:oy2, ox1:ox2].copy()
        if original_crop.size and original_crop.shape[:2] != face_crop.shape[:2]:
          original_crop = cv2.resize(
            original_crop,
            (face_crop.shape[1], face_crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
          )

    if original_crop is None or not original_crop.size:
      return None

    landmarks_all_face_crop = face_detection.landmarks_all_face_crop

    return FaceDetection(
      bbox_norm=face_detection.bbox_norm,
      affine_matrix=face_detection.affine_matrix,
      crop=original_crop,
      landmarks_all_face_crop=landmarks_all_face_crop,
    )

  def paste_restored_faces(self, restored_faces: typing.List[np.ndarray]) -> np.ndarray:
    helper = self._helper
    for restored_face in restored_faces:
      helper.add_restored_face(restored_face)

    helper.get_inverse_affine(None)
    return helper.paste_faces_to_input_image(upsample_img=None)
