import argparse
import pathlib

import cv2
import numpy as np
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from upscaler.face_blurrer import BlurMaskMode, BlurMode, FaceBlurrer


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Detect a face and apply blur using 5-point landmarks"
  )
  parser.add_argument("--image", type=pathlib.Path, required=True, help="Path to input image")
  parser.add_argument("--output", type=pathlib.Path, required=True, help="Path to output image")
  parser.add_argument(
    "--mode",
    choices=[m.value for m in BlurMode],
    default=BlurMode.GAUSSIAN.value,
  )
  parser.add_argument(
    "--mask",
    choices=[m.value for m in BlurMaskMode],
    default=BlurMaskMode.FACE.value,
  )
  parser.add_argument(
    "--blur-level",
    type=float,
    default=0.5,
    help="Blur intensity in range [0, 1)",
  )
  parser.add_argument(
    "--det-model",
    type=str,
    default="retinaface_resnet50",
    help="Face detector model name for FaceRestoreHelper",
  )
  parser.add_argument(
    "--only-center-face",
    action="store_true",
    help="Detect only the center-most face",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Torch device for face detection (e.g. 'cpu' or 'cuda')",
  )
  return parser.parse_args()


def _detect_landmarks(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
  helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model=args.det_model,
    use_parse=False,
    device=args.device,
  )
  helper.clean_all()
  helper.read_image(image)
  num_faces = helper.get_face_landmarks_5(
    only_center_face=bool(args.only_center_face),
    resize=640,
    eye_dist_threshold=5,
  )

  landmarks_list = getattr(helper, "all_landmarks_5", None)
  if num_faces <= 0 or not landmarks_list:
    raise RuntimeError("No faces with 5-point landmarks were detected")

  landmarks = np.asarray(landmarks_list[0], dtype=np.float32).reshape(-1, 2)
  if landmarks.shape[0] < 5:
    raise RuntimeError(f"Expected at least 5 landmarks, got shape {landmarks.shape}")
  return landmarks[:5]


def main() -> None:
  args = parse_args()
  image = cv2.imread(str(args.image))
  if image is None:
    raise RuntimeError(f"Failed to read image: {args.image}")

  landmarks = _detect_landmarks(image, args)

  blurrer = FaceBlurrer()
  out = blurrer.apply(
    image,
    landmarks=landmarks,
    blur_mode=BlurMode(args.mode),
    mask_mode=BlurMaskMode(args.mask),
    blur_level=max(0.0, min(0.999999, float(args.blur_level))),
  )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  ok = cv2.imwrite(str(args.output), out)
  if not ok:
    raise RuntimeError(f"Failed to write output image: {args.output}")


if __name__ == "__main__":
  main()
