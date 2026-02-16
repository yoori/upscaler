import dataclasses
import pathlib
import random
import sys
import typing

import cv2
import numpy as np
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torch.utils.data import Dataset

from .face_blurrer import BlurMaskMode, BlurMode, FaceBlurrer


PARTS = ("eyes", "nose", "mouth")
STATES = ("visible", "occluded", "blurred", "uncertain")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclasses.dataclass(frozen=True)
class FaceMeta:
  image_path: pathlib.Path
  face_box: typing.Tuple[int, int, int, int]
  left_eye_center: typing.Tuple[float, float]
  right_eye_center: typing.Tuple[float, float]
  nose_center: typing.Tuple[float, float]
  mouth_center: typing.Tuple[float, float]
  landmarks_5: typing.Tuple[typing.Tuple[float, float], ...]


class FacePartStateDataset(Dataset):
  def __init__(
    self,
    images_dir: pathlib.Path,
    device: str = "cpu",
    face_size: int = 512,
    image_size: int = 224,
    repeat: int = 3,
    blur_probability: float = 0.80,
    occlusion_probability: float = 0.10,
    uncertain_probability: float = 0.10,
    seed: int = 42,
  ):
    self.images_dir = images_dir
    self.device = device
    self.face_size = face_size
    self.image_size = image_size
    self.repeat = max(1, repeat)
    self.blur_probability = blur_probability
    self.occlusion_probability = occlusion_probability
    self.uncertain_probability = uncertain_probability
    self.rng = random.Random(seed)
    self.face_blurrer = FaceBlurrer()

    self.face_samples = self._collect_faces()
    if not self.face_samples:
      raise RuntimeError(f"No faces found in directory: {images_dir}")

    self.samples: typing.List[typing.Tuple[FaceMeta, int]] = []
    for meta in self.face_samples:
      for part_id in range(len(PARTS)):
        for _ in range(self.repeat):
          self.samples.append((meta, part_id))
    self.rng.shuffle(self.samples)

  def _collect_faces(self) -> typing.List[FaceMeta]:
    image_paths = sorted([
      p
      for p in self.images_dir.rglob("*")
      if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ])
    if not image_paths:
      raise RuntimeError(f"No image files found in {self.images_dir}")

    helper = FaceRestoreHelper(
      upscale_factor=1,
      face_size=self.face_size,
      crop_ratio=(1, 1),
      det_model="retinaface_resnet50",
      use_parse=False,
      device=self.device,
    )

    all_faces: typing.List[FaceMeta] = []
    total_images = len(image_paths)
    for index, path in enumerate(image_paths, start=1):
      self._print_progress(index, total_images)
      image = cv2.imread(str(path))
      if image is None:
        continue

      helper.clean_all()
      helper.read_image(image)
      num_faces = helper.get_face_landmarks_5(
        only_center_face=False,
        resize=640,
        eye_dist_threshold=5,
      )
      if num_faces <= 0:
        continue

      landmarks_list = getattr(helper, "all_landmarks_5", None)
      if not landmarks_list:
        continue

      for landmarks in landmarks_list:
        lm = np.asarray(landmarks, dtype=np.float32)
        if lm.shape != (5, 2):
          continue
        face_box = self._face_box_from_landmarks(lm, image.shape[1], image.shape[0])
        if face_box is None:
          continue

        left_eye_center = tuple(lm[0].tolist())
        right_eye_center = tuple(lm[1].tolist())
        nose_center = tuple(lm[2].tolist())
        mouth_center = tuple(np.mean(lm[3:], axis=0).tolist())
        landmarks_5 = tuple((float(x), float(y)) for x, y in lm.tolist())

        all_faces.append(
          FaceMeta(
            image_path=path,
            face_box=face_box,
            left_eye_center=typing.cast(typing.Tuple[float, float], left_eye_center),
            right_eye_center=typing.cast(typing.Tuple[float, float], right_eye_center),
            nose_center=typing.cast(typing.Tuple[float, float], nose_center),
            mouth_center=typing.cast(typing.Tuple[float, float], mouth_center),
            landmarks_5=landmarks_5,
          )
        )

    if total_images > 0:
      print(file=sys.stderr)

    return all_faces

  def _print_progress(self, index: int, total: int) -> None:
    bar_width = 28
    progress = index / max(1, total)
    filled = int(round(bar_width * progress))
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
      f"\rCollecting faces: [{bar}] {index}/{total}",
      end="",
      file=sys.stderr,
      flush=True,
    )

  def _face_box_from_landmarks(
    self,
    landmarks: np.ndarray,
    width: int,
    height: int,
  ) -> typing.Optional[typing.Tuple[int, int, int, int]]:
    min_xy = landmarks.min(axis=0)
    max_xy = landmarks.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    side = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])) * 2.1
    side = max(side, 40.0)

    x1 = int(round(center[0] - side / 2))
    y1 = int(round(center[1] - side / 2))
    x2 = int(round(center[0] + side / 2))
    y2 = int(round(center[1] + side / 2))

    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 1, min(x2, width - 1))
    y2 = max(y1 + 1, min(y2, height - 1))

    if (x2 - x1) < 20 or (y2 - y1) < 20:
      return None
    return (x1, y1, x2, y2)

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    meta, part_id = self.samples[index]

    image = cv2.imread(str(meta.image_path))
    if image is None:
      raise RuntimeError(f"Failed to read image {meta.image_path}")

    x1, y1, x2, y2 = meta.face_box
    face_crop = image[y1:y2, x1:x2].copy()
    if face_crop.size == 0:
      raise RuntimeError(f"Invalid crop from {meta.image_path}")

    part_center = self._part_center(meta, part_id)
    landmarks_face_crop = self._landmarks_in_face_crop(meta)
    center_x = float((part_center[0] - x1) / max(1, (x2 - x1)))
    center_y = float((part_center[1] - y1) / max(1, (y2 - y1)))
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))

    state_id = self._sample_state()
    face_augmented = self._apply_state(face_crop, state_id, part_id, landmarks_face_crop)
    organ_crop = self._extract_part_crop(face_augmented, center_x, center_y, part_id, landmarks_face_crop)

    organ_crop = cv2.cvtColor(organ_crop, cv2.COLOR_BGR2RGB)
    organ_crop = cv2.resize(organ_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    img_tensor = torch.from_numpy(organ_crop).float().permute(2, 0, 1) / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5

    coords_tensor = torch.tensor([center_x, center_y], dtype=torch.float32)
    part_tensor = torch.tensor(part_id, dtype=torch.long)
    state_tensor = torch.tensor(state_id, dtype=torch.long)
    return img_tensor, coords_tensor, part_tensor, state_tensor

  def _part_center(self, meta: FaceMeta, part_id: int) -> typing.Tuple[float, float]:
    if part_id == 0:
      if self.rng.random() < 0.5:
        return meta.left_eye_center
      return meta.right_eye_center
    if part_id == 1:
      return meta.nose_center
    return meta.mouth_center

  def _landmarks_in_face_crop(self, meta: FaceMeta) -> np.ndarray:
    x1, y1, _, _ = meta.face_box
    points = np.asarray(meta.landmarks_5, dtype=np.float32).reshape(-1, 2)
    points[:, 0] -= float(x1)
    points[:, 1] -= float(y1)
    return points

  def _sample_state(self) -> int:
    p = self.rng.random()
    if p < self.uncertain_probability:
      return 3
    if p < self.uncertain_probability + self.occlusion_probability:
      return 1
    if p < self.uncertain_probability + self.occlusion_probability + self.blur_probability:
      return 2
    return 0

  def _apply_state(
    self,
    face_crop: np.ndarray,
    state_id: int,
    part_id: int,
    landmarks: np.ndarray,
  ) -> np.ndarray:
    out = face_crop.copy()

    if state_id == 0:
      return out

    if state_id == 2:
      return self.face_blurrer.apply(
        out,
        landmarks,
        blur_mode=self._sample_blur_mode(),
        mask_mode=self._sample_mask_mode(part_id),
        strong=True,
        rng=self.rng,
      )

    if state_id == 1:
      return self.face_blurrer.apply(
        out,
        landmarks,
        blur_mode=BlurMode.OCCLUDE,
        mask_mode=self._sample_mask_mode(part_id),
        strong=True,
        rng=self.rng,
      )

    if self.rng.random() < 0.5:
      out = self.face_blurrer.apply(
        out,
        landmarks,
        blur_mode=self._sample_blur_mode(),
        mask_mode=self._sample_mask_mode(part_id),
        strong=False,
        rng=self.rng,
      )
    if self.rng.random() < 0.7:
      out = self.face_blurrer.apply(
        out,
        landmarks,
        blur_mode=BlurMode.OCCLUDE,
        mask_mode=self._sample_mask_mode(part_id),
        strong=False,
        rng=self.rng,
      )
    return out


  def _sample_blur_mode(self) -> BlurMode:
    if self.rng.random() < 0.5:
      return BlurMode.GAUSSIAN
    return BlurMode.UNIFORM

  def _sample_mask_mode(self, part_id: int) -> BlurMaskMode:
    if part_id == 0 or self.rng.random() < 0.5:
      return BlurMaskMode.EYES
    return BlurMaskMode.FACE

  def _extract_part_crop(
    self,
    face_image: np.ndarray,
    cx: float,
    cy: float,
    part_id: int,
    landmarks: np.ndarray,
  ) -> np.ndarray:
    h, w = face_image.shape[:2]
    _, _, organ_w, organ_h = self._organ_extent(part_id, landmarks)
    min_side = max(16, int(np.ceil(max(organ_w, organ_h))))
    max_side = max(16, max(h, w))
    side = min_side if min_side >= max_side else self.rng.randint(min_side, max_side)

    center_px = int(round(cx * w))
    center_py = int(round(cy * h))
    pad_color = tuple(self.rng.randint(0, 255) for _ in range(3))
    crop = self._extract_square_with_padding(face_image, center_px, center_py, side, pad_color)
    if crop.size == 0:
      crop = cv2.resize(face_image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
    return crop

  def _organ_extent(
    self,
    part_id: int,
    landmarks: np.ndarray,
  ) -> typing.Tuple[float, float, float, float]:
    left_eye, right_eye, nose, mouth_left, mouth_right = landmarks[:5]
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    mouth_width = float(np.linalg.norm(mouth_left - mouth_right))

    if part_id == 0:
      width = max(12.0, eye_dist * 0.75)
      height = max(10.0, eye_dist * 0.55)
    elif part_id == 1:
      width = max(10.0, eye_dist * 0.55)
      height = max(10.0, eye_dist * 0.65)
    else:
      width = max(12.0, mouth_width * 1.05)
      height = max(10.0, mouth_width * 0.65)
    return (0.0, 0.0, width, height)

  def _extract_square_with_padding(
    self,
    image: np.ndarray,
    cx_px: int,
    cy_px: int,
    side: int,
    pad_color: typing.Tuple[int, int, int],
  ) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(1, int(side))
    half = side // 2
    src_x1 = cx_px - half
    src_y1 = cy_px - half
    src_x2 = src_x1 + side
    src_y2 = src_y1 + side

    canvas = np.full((side, side, 3), pad_color, dtype=image.dtype)
    clip_x1 = max(0, src_x1)
    clip_y1 = max(0, src_y1)
    clip_x2 = min(w, src_x2)
    clip_y2 = min(h, src_y2)
    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
      return canvas

    dst_x1 = clip_x1 - src_x1
    dst_y1 = clip_y1 - src_y1
    dst_x2 = dst_x1 + (clip_x2 - clip_x1)
    dst_y2 = dst_y1 + (clip_y2 - clip_y1)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[clip_y1:clip_y2, clip_x1:clip_x2]
    return canvas
