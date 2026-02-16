import argparse
import json
import pathlib
import typing

import cv2

from upscaler.face_part_state_evaluator import FacePartStateEvaluator


def _parse_coords(value: str) -> typing.Tuple[float, float]:
  raw = str(value).strip().replace(" ", "")
  if not raw:
    raise argparse.ArgumentTypeError("Coordinates cannot be empty")

  if "," in raw:
    chunks = raw.split(",")
  elif ":" in raw:
    chunks = raw.split(":")
  else:
    raise argparse.ArgumentTypeError(
      "Coordinates must be provided as 'x,y' or 'x:y' normalized values"
    )

  if len(chunks) != 2:
    raise argparse.ArgumentTypeError("Coordinates must contain exactly two values")

  try:
    x = float(chunks[0])
    y = float(chunks[1])
  except ValueError as err:
    raise argparse.ArgumentTypeError(f"Coordinates must be numeric: {value}") from err

  return (x, y)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Evaluate state of a face part and print result to console",
  )
  parser.add_argument("--weights", type=pathlib.Path, required=True, help="Path to evaluator weights .pth file")
  parser.add_argument("--image", type=pathlib.Path, required=True, help="Path to part image file")
  parser.add_argument(
    "--part",
    type=str,
    required=True,
    choices=["eyes", "nose", "mouth"],
    help="Face part name",
  )
  parser.add_argument(
    "--coords",
    type=_parse_coords,
    default=(0.5, 0.5),
    help="Normalized part center coordinates in format 'x,y' or 'x:y'",
  )
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Torch device (e.g. 'cpu' or 'cuda'). Defaults to auto",
  )
  parser.add_argument(
    "--indent",
    type=int,
    default=2,
    help="JSON indentation level for output",
  )
  return parser


def main() -> int:
  parser = _build_parser()
  args = parser.parse_args()

  if not args.weights.is_file():
    parser.error(f"Weights file not found: {args.weights}")
  if not args.image.is_file():
    parser.error(f"Image file not found: {args.image}")

  image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
  if image is None:
    parser.error(f"Failed to read image: {args.image}")

  evaluator = FacePartStateEvaluator(
    weights_path=str(args.weights),
    device=args.device,
  )

  result = evaluator.evaluate(
    part_image_bgr=image,
    part=args.part,
    coords=args.coords,
  )

  print(json.dumps(result, ensure_ascii=False, indent=args.indent))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
