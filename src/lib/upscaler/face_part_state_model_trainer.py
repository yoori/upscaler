import pathlib
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .face_part_state_dataset import PARTS, STATES


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


class FacePartStateModelTrainer:
  def __init__(
    self,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    train_ratio: float,
    seed: int,
    output_path: pathlib.Path,
  ):
    self.device = torch.device(device)
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.lr = lr
    self.weight_decay = weight_decay
    self.epochs = epochs
    self.train_ratio = train_ratio
    self.seed = seed
    self.output_path = output_path

  def train(self, dataset: Dataset, *, image_size: int) -> pathlib.Path:
    train_loader, val_loader = self._build_loaders(dataset)

    model = MobileNetPartState().to(self.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    best_acc = -1.0
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, self.epochs + 1):
      train_loss, train_acc = self._run_epoch(model, train_loader, optimizer)
      val_loss, val_acc = self._run_epoch(model, val_loader, None)

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
            "image_size": image_size,
            "best_val_acc": best_acc,
          },
          self.output_path,
        )

    print(f"Best val_acc={best_acc:.4f}. Model saved to: {self.output_path}")
    return self.output_path

  def _build_loaders(self, dataset: Dataset) -> typing.Tuple[DataLoader, DataLoader]:
    train_len = int(len(dataset) * self.train_ratio)
    val_len = max(1, len(dataset) - train_len)
    if train_len < 1:
      raise RuntimeError("Dataset too small. Increase images or repeat.")

    train_ds, val_ds = random_split(
      dataset,
      [train_len, val_len],
      generator=torch.Generator().manual_seed(self.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    return train_loader, val_loader

  def _run_epoch(
    self,
    model: nn.Module,
    loader: DataLoader,
    optimizer: typing.Optional[torch.optim.Optimizer],
  ) -> typing.Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    total = 0
    correct = 0

    for images, coords, parts, targets in loader:
      images = images.to(self.device)
      coords = coords.to(self.device)
      parts = parts.to(self.device)
      targets = targets.to(self.device)

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
