import typing

import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_large


FacePartName = typing.Literal["eyes", "nose", "mouth"]

DEFAULT_FACE_PARTS: typing.Tuple[str, ...] = ("eyes", "nose", "mouth")
DEFAULT_FACE_PART_STATES: typing.Tuple[str, ...] = ("visible", "blurred")


def _clamp_norm(value: float) -> float:
  return max(0.0, min(1.0, float(value)))


class _MobileNetPartState(torch.nn.Module):
  def __init__(self, num_parts: int, num_states: int):
    super().__init__()
    backbone = mobilenet_v3_large(weights=None)
    self.features = backbone.features
    self.pool = torch.nn.AdaptiveAvgPool2d(1)
    self.img_head = torch.nn.Linear(960, 256)

    self.part_embedding = torch.nn.Embedding(num_parts, 16)
    self.meta_head = torch.nn.Sequential(
      torch.nn.Linear(2 + 16, 64),
      torch.nn.ReLU(inplace=True),
      torch.nn.Linear(64, 64),
      torch.nn.ReLU(inplace=True),
    )
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(256 + 64, 128),
      torch.nn.ReLU(inplace=True),
      torch.nn.Dropout(p=0.2),
      torch.nn.Linear(128, num_states),
    )

  def forward(self, image: torch.Tensor, coords: torch.Tensor, part_id: torch.Tensor) -> torch.Tensor:
    x = self.features(image)
    x = self.pool(x).flatten(1)
    x = torch.nn.functional.relu(self.img_head(x), inplace=True)

    part_emb = self.part_embedding(part_id)
    meta = torch.cat([coords, part_emb], dim=1)
    meta = self.meta_head(meta)

    fused = torch.cat([x, meta], dim=1)
    return self.classifier(fused)


class FacePartStateEvaluator:
  def __init__(
    self,
    weights_path: str,
    *,
    device: typing.Optional[str] = None,
  ):
    self._device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(weights_path, map_location="cpu")
    self.parts: typing.Tuple[str, ...] = tuple(ckpt.get("parts", DEFAULT_FACE_PARTS))
    self.states: typing.Tuple[str, ...] = tuple(ckpt.get("states", DEFAULT_FACE_PART_STATES))
    self.image_size: int = int(ckpt.get("image_size", 224))

    self._part_to_id = {name: idx for idx, name in enumerate(self.parts)}

    self._model = _MobileNetPartState(num_parts=len(self.parts), num_states=len(self.states)).to(self._device)
    model_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    self._model.load_state_dict(model_state, strict=True)
    self._model.eval()

  def evaluate(
    self,
    part_image_bgr: np.ndarray,
    part: FacePartName,
    *,
    coords: typing.Tuple[float, float] = (0.5, 0.5),
  ) -> typing.Dict[str, typing.Any]:
    if part not in self._part_to_id:
      raise ValueError(f"Unsupported face part '{part}'. Expected one of: {self.parts}")

    if part_image_bgr is None or getattr(part_image_bgr, "size", 0) == 0:
      raise ValueError("Expected non-empty image for face part")

    image = part_image_bgr
    if image.ndim == 2:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
      raise ValueError(f"Expected image with shape HxW or HxWx3, got {image.shape}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    image_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_t = (image_t - 0.5) / 0.5
    image_t = image_t.to(self._device)

    cx = _clamp_norm(float(coords[0]))
    cy = _clamp_norm(float(coords[1]))
    coords_t = torch.tensor([[cx, cy]], dtype=torch.float32, device=self._device)
    part_t = torch.tensor([self._part_to_id[part]], dtype=torch.long, device=self._device)

    with torch.no_grad():
      logits = self._model(image_t, coords_t, part_t)
      if len(self.states) == 2:
        probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
      else:
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    state_idx = int(np.argmax(probabilities))
    state_scores = {self.states[i]: float(probabilities[i]) for i in range(len(self.states))}
    return {
      "part": str(part),
      "predicted_state": self.states[state_idx],
      "confidence": float(probabilities[state_idx]),
      "state_scores": state_scores,
    }

