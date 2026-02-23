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
