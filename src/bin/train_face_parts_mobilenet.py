import argparse
import pathlib
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from upscaler.face_part_state_dataset import FacePartStateDataset, PARTS, STATES


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
