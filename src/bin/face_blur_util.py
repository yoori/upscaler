import argparse
import pathlib

import cv2
import numpy as np

from upscaler.face_blurrer import BlurMaskMode, BlurMode, FaceBlurrer


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Apply face blur using 5-point landmarks")
  parser.add_argument("--image", type=pathlib.Path, required=True, help="Path to input image")
  parser.add_argument("--landmarks", type=pathlib.Path, required=True, help="Path to .npy with shape (5,2)")
  parser.add_argument("--output", type=pathlib.Path, required=True, help="Path to output image")
  parser.add_argument("--mode", choices=[m.value for m in BlurMode], default=BlurMode.GAUSSIAN.value)
  parser.add_argument("--mask", choices=[m.value for m in BlurMaskMode], default=BlurMaskMode.FACE.value)
  parser.add_argument("--strong", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  image = cv2.imread(str(args.image))
  if image is None:
    raise RuntimeError(f"Failed to read image: {args.image}")

  landmarks = np.asarray(np.load(str(args.landmarks)), dtype=np.float32).reshape(-1, 2)
  if landmarks.shape[0] < 5:
    raise RuntimeError(f"Expected at least 5 landmarks, got shape {landmarks.shape}")

  blurrer = FaceBlurrer()
  out = blurrer.apply(
    image,
    landmarks=landmarks,
    blur_mode=BlurMode(args.mode),
    mask_mode=BlurMaskMode(args.mask),
    strong=bool(args.strong),
  )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  ok = cv2.imwrite(str(args.output), out)
  if not ok:
    raise RuntimeError(f"Failed to write output image: {args.output}")


if __name__ == "__main__":
  main()
