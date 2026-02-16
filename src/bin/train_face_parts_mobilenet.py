import argparse
import pathlib

import torch

from upscaler.face_part_state_dataset import FacePartStateDataset
from upscaler.face_part_state_model_trainer import FacePartStateModelTrainer


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

  trainer = FacePartStateModelTrainer(
    device=args.device,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    lr=args.lr,
    weight_decay=args.weight_decay,
    epochs=args.epochs,
    train_ratio=args.train_ratio,
    seed=args.seed,
    output_path=pathlib.Path(args.output),
  )
  trainer.train(dataset, image_size=args.image_size)


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
