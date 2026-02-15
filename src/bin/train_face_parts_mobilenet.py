import argparse
import dataclasses
import pathlib
import random
import sys
import typing

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


PARTS = ("eyes", "nose", "mouth")
STATES = ("visible", "occluded", "blurred", "uncertain")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclasses.dataclass(frozen=True)
class FaceMeta:
  image_path: pathlib.Path
  face_box: typing.Tuple[int, int, int, int]
  eyes_center: typing.Tuple[float, float]
  nose_center: typing.Tuple[float, float]
  mouth_center: typing.Tuple[float, float]


class FacePartStateDataset(Dataset):
  def __init__(
    self,
    images_dir: pathlib.Path,
    device: str = "cpu",
    face_size: int = 512,
    image_size: int = 224,
    repeat: int = 3,
    blur_probability: float = 0.28,
    occlusion_probability: float = 0.24,
    uncertain_probability: float = 0.12,
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

        eyes_center = tuple(np.mean(lm[:2], axis=0).tolist())
        nose_center = tuple(lm[2].tolist())
        mouth_center = tuple(np.mean(lm[3:], axis=0).tolist())

        all_faces.append(
          FaceMeta(
            image_path=path,
            face_box=face_box,
            eyes_center=typing.cast(typing.Tuple[float, float], eyes_center),
            nose_center=typing.cast(typing.Tuple[float, float], nose_center),
            mouth_center=typing.cast(typing.Tuple[float, float], mouth_center),
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
    center_x = float((part_center[0] - x1) / max(1, (x2 - x1)))
    center_y = float((part_center[1] - y1) / max(1, (y2 - y1)))
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))

    state_id = self._sample_state()
    face_augmented = self._apply_state(face_crop, state_id, center_x, center_y)
    organ_crop = self._extract_part_crop(face_augmented, center_x, center_y, part_id)

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
      return meta.eyes_center
    if part_id == 1:
      return meta.nose_center
    return meta.mouth_center

  def _sample_state(self) -> int:
    p = self.rng.random()
    if p < self.uncertain_probability:
      return 3
    if p < self.uncertain_probability + self.occlusion_probability:
      return 1
    if p < self.uncertain_probability + self.occlusion_probability + self.blur_probability:
      return 2
    return 0

  def _apply_state(self, face_crop: np.ndarray, state_id: int, cx: float, cy: float) -> np.ndarray:
    out = face_crop.copy()
    h, w = out.shape[:2]

    if state_id == 0:
      return out

    target_whole_face = self.rng.random() < 0.5
    if state_id == 3 and self.rng.random() < 0.65:
      target_whole_face = False

    if target_whole_face:
      x1, y1, x2, y2 = 0, 0, w, h
    else:
      box_size = int(max(20, min(w, h) * self.rng.uniform(0.22, 0.4)))
      center_px = int(cx * w)
      center_py = int(cy * h)
      x1 = max(0, center_px - box_size // 2)
      y1 = max(0, center_py - box_size // 2)
      x2 = min(w, x1 + box_size)
      y2 = min(h, y1 + box_size)

    if state_id == 1:
      return self._apply_occlusion(out, x1, y1, x2, y2, strong=True)
    if state_id == 2:
      return self._apply_blur(out, x1, y1, x2, y2, strong=True)

    if self.rng.random() < 0.5:
      out = self._apply_blur(out, x1, y1, x2, y2, strong=False)
    if self.rng.random() < 0.7:
      out = self._apply_occlusion(out, x1, y1, x2, y2, strong=False)
    return out

  def _apply_blur(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, strong: bool) -> np.ndarray:
    out = image.copy()
    patch = out[y1:y2, x1:x2]
    if patch.size == 0:
      return out

    if self.rng.random() < 0.5:
      k = self.rng.choice([5, 7, 9, 11]) if strong else self.rng.choice([3, 5, 7])
      patch = cv2.GaussianBlur(patch, (k, k), 0)
    else:
      scale = self.rng.uniform(0.08, 0.22) if strong else self.rng.uniform(0.2, 0.35)
      down_w = max(1, int((x2 - x1) * scale))
      down_h = max(1, int((y2 - y1) * scale))
      patch = cv2.resize(patch, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
      patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    out[y1:y2, x1:x2] = patch
    return out

  def _apply_occlusion(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, strong: bool) -> np.ndarray:
    out = image.copy()
    base_color = int(self.rng.uniform(0, 255))
    color = (base_color, base_color, base_color)

    if strong:
      cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=-1)
      return out

    if self.rng.random() < 0.5:
      cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=-1)
    else:
      cx = (x1 + x2) // 2
      cy = (y1 + y2) // 2
      ax = max(1, (x2 - x1) // 2)
      ay = max(1, (y2 - y1) // 2)
      cv2.ellipse(out, (cx, cy), (ax, ay), 0, 0, 360, color, thickness=-1)
    return out

  def _extract_part_crop(self, face_image: np.ndarray, cx: float, cy: float, part_id: int) -> np.ndarray:
    h, w = face_image.shape[:2]
    base_scale = {0: (0.24, 0.34), 1: (0.18, 0.28), 2: (0.26, 0.36)}[part_id]
    side = int(max(16, min(h, w) * self.rng.uniform(base_scale[0], base_scale[1])))

    jitter_x = int(self.rng.uniform(-0.06, 0.06) * w)
    jitter_y = int(self.rng.uniform(-0.06, 0.06) * h)
    cx_px = int(cx * w) + jitter_x
    cy_px = int(cy * h) + jitter_y

    x1 = max(0, cx_px - side // 2)
    y1 = max(0, cy_px - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    crop = face_image[y1:y2, x1:x2]
    if crop.size == 0:
      crop = cv2.resize(face_image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
    return crop


class MobileNetPartState(nn.Module):
  def __init__(self, num_parts: int = len(PARTS), num_states: int = len(STATES)):
    super().__init__()
    backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    self.features = backbone.features
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.img_head = nn.Linear(960, 256)

    self.part_embedding = nn.Embedding(num_parts, 16)
    self.meta_head = nn.Sequential(
      nn.Linear(2 + 16, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 64),
      nn.ReLU(inplace=True),
    )
    self.classifier = nn.Sequential(
      nn.Linear(256 + 64, 128),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.2),
      nn.Linear(128, num_states),
    )

  def forward(self, image: torch.Tensor, coords: torch.Tensor, part_id: torch.Tensor) -> torch.Tensor:
    x = self.features(image)
    x = self.pool(x).flatten(1)
    x = F.relu(self.img_head(x), inplace=True)

    part_emb = self.part_embedding(part_id)
    meta = torch.cat([coords, part_emb], dim=1)
    meta = self.meta_head(meta)

    fused = torch.cat([x, meta], dim=1)
    return self.classifier(fused)


def run_epoch(
  model: nn.Module,
  loader: DataLoader,
  optimizer: typing.Optional[torch.optim.Optimizer],
  device: torch.device,
) -> typing.Tuple[float, float]:
  is_train = optimizer is not None
  model.train(is_train)

  loss_sum = 0.0
  total = 0
  correct = 0

  for images, coords, parts, targets in loader:
    images = images.to(device)
    coords = coords.to(device)
    parts = parts.to(device)
    targets = targets.to(device)

    with torch.set_grad_enabled(is_train):
      logits = model(images, coords, parts)
      loss = F.cross_entropy(logits, targets)
      if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_size = images.size(0)
    loss_sum += loss.item() * batch_size
    total += batch_size
    correct += (logits.argmax(dim=1) == targets).sum().item()

  return loss_sum / max(1, total), correct / max(1, total)


def train(args: argparse.Namespace):
  dataset = FacePartStateDataset(
    images_dir=pathlib.Path(args.images_dir),
    device=args.device,
    face_size=args.face_size,
    image_size=args.image_size,
    repeat=args.repeat,
    blur_probability=args.blur_probability,
    occlusion_probability=args.occlusion_probability,
    uncertain_probability=args.uncertain_probability,
    seed=args.seed,
  )

  train_len = int(len(dataset) * args.train_ratio)
  val_len = max(1, len(dataset) - train_len)
  if train_len < 1:
    raise RuntimeError("Dataset too small. Increase images or repeat.")

  train_ds, val_ds = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(args.seed),
  )

  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  device = torch.device(args.device)
  model = MobileNetPartState().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  best_acc = -1.0
  output_path = pathlib.Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = run_epoch(model, val_loader, None, device)

    print(
      f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

    if val_acc > best_acc:
      best_acc = val_acc
      torch.save(
        {
          "model_state": model.state_dict(),
          "parts": PARTS,
          "states": STATES,
          "image_size": args.image_size,
          "best_val_acc": best_acc,
        },
        output_path,
      )

  print(f"Best val_acc={best_acc:.4f}. Model saved to: {output_path}")


def build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Train MobileNetV3-Large for facial part state classification (visible/occluded/blurred/uncertain)."
  )
  parser.add_argument("--images-dir", required=True, help="Directory with face photos.")
  parser.add_argument("--output", default="artifacts/mobilenet_part_state.pth", help="Output checkpoint path.")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--epochs", type=int, default=12)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--num-workers", type=int, default=2)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--train-ratio", type=float, default=0.9)
  parser.add_argument("--image-size", type=int, default=224)
  parser.add_argument("--face-size", type=int, default=512)
  parser.add_argument("--repeat", type=int, default=3, help="Multiplier for each face-part sample.")
  parser.add_argument("--blur-probability", type=float, default=0.28)
  parser.add_argument("--occlusion-probability", type=float, default=0.24)
  parser.add_argument("--uncertain-probability", type=float, default=0.12)
  parser.add_argument("--seed", type=int, default=42)
  return parser


if __name__ == "__main__":
  args = build_argparser().parse_args()
  train(args)
