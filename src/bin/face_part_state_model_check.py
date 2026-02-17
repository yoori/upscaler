import argparse
import pathlib

import cv2
import numpy as np
import torch

from upscaler.face_part_state_dataset import FacePartStateDataset, PARTS
from upscaler.face_part_state_evaluator import FacePartStateEvaluator
from upscaler.face_part_state_model_trainer import FacePartStateModelTrainer


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Train FacePartState model on a dataset and run evaluator N times per dataset sample, "
      "counting fail/success by state mismatch."
    ),
  )
  parser.add_argument("--images-dir", type=pathlib.Path, required=True, help="Directory with source photos")
  parser.add_argument(
    "--output",
    type=pathlib.Path,
    default=pathlib.Path("artifacts/mobilenet_part_state_check.pth"),
    help="Output checkpoint path",
  )
  parser.add_argument("--n", type=int, default=1, help="How many evaluator runs for each dataset sample")
  parser.add_argument("--samples", type=int, default=0, help="Limit checked dataset samples (0 means all)")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--train-ratio", type=float, default=0.9)
  parser.add_argument("--image-size", type=int, default=224)
  parser.add_argument("--face-size", type=int, default=512)
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--blur-probability", type=float, default=0.28)
  parser.add_argument("--occlusion-probability", type=float, default=0.24)
  parser.add_argument("--seed", type=int, default=42)
  return parser


def _tensor_to_bgr_image(image_tensor: torch.Tensor) -> np.ndarray:
  image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
  image = ((image * 0.5) + 0.5) * 255.0
  image = np.clip(image, 0, 255).astype(np.uint8)
  return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def main() -> int:
  parser = _build_parser()
  args = parser.parse_args()

  if args.n <= 0:
    parser.error("--n must be positive")
  if args.samples < 0:
    parser.error("--samples must be >= 0")
  if not args.images_dir.is_dir():
    parser.error(f"Images directory not found: {args.images_dir}")

  dataset = FacePartStateDataset(
    images_dir=args.images_dir,
    device=args.device,
    face_size=args.face_size,
    image_size=args.image_size,
    repeat=args.repeat,
    blur_probability=args.blur_probability,
    occlusion_probability=args.occlusion_probability,
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
    output_path=args.output,
  )
  model_path = trainer.train(dataset, image_size=args.image_size)
  evaluator = FacePartStateEvaluator(str(model_path), device=args.device)

  success = 0
  fail = 0
  light_fail = 0
  sample_count = len(dataset) if args.samples == 0 else min(args.samples, len(dataset))

  for idx in range(sample_count):
    image_tensor, coords_tensor, part_tensor, state_tensor = dataset[idx]
    image_bgr = _tensor_to_bgr_image(image_tensor)
    part_name = PARTS[int(part_tensor.item())]
    expected_visible, expected_blurred = [float(v) for v in state_tensor.tolist()]
    coords = (float(coords_tensor[0].item()), float(coords_tensor[1].item()))

    for _ in range(args.n):
      prediction = evaluator.evaluate(image_bgr, part_name, coords=coords)
      predicted_visible = float(prediction["state_scores"].get("visible", 0.0))
      predicted_blurred = float(prediction["state_scores"].get("blurred", 0.0))
      pred_visible_bin = 1.0 if predicted_visible >= 0.5 else 0.0
      pred_blurred_bin = 1.0 if predicted_blurred >= 0.5 else 0.0
      if pred_visible_bin == (1.0 if expected_visible >= 0.5 else 0.0) and pred_blurred_bin == (1.0 if expected_blurred >= 0.5 else 0.0):
        success += 1
      elif pred_blurred_bin == (1.0 if expected_blurred >= 0.5 else 0.0):
        light_fail += 1
      else:
        fail += 1

  ratio = float("inf") if success == 0 else fail / success
  print(f"fail={fail} light_fail={light_fail} success={success} fail/success={ratio}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
