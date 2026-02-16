import argparse
import json
import pathlib
import typing

import cv2
import numpy as np
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

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
    choices=["eyes", "nose", "mouth", "face"],
    help="Face part name, or 'face' to evaluate all detected face parts",
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
  parser.add_argument(
    "--only-center-face",
    action="store_true",
    help="When --part=face, evaluate only the center-most face",
  )
  parser.add_argument(
    "--face-size",
    type=int,
    default=512,
    help="Face alignment size for FaceRestoreHelper when --part=face",
  )
  parser.add_argument(
    "--det-model",
    type=str,
    default="retinaface_resnet50",
    help="Face detector model for FaceRestoreHelper when --part=face",
  )
  return parser


def _face_box_from_landmarks(
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


def _extract_part_crop(face_image: np.ndarray, cx: float, cy: float, part: str) -> np.ndarray:
  h, w = face_image.shape[:2]
  side_ratio = {
    "eyes": 0.29,
    "nose": 0.23,
    "mouth": 0.31,
  }[part]
  side = int(max(16, min(h, w) * side_ratio))

  cx_px = int(cx * w)
  cy_px = int(cy * h)

  x1 = max(0, cx_px - side // 2)
  y1 = max(0, cy_px - side // 2)
  x2 = min(w, x1 + side)
  y2 = min(h, y1 + side)

  crop = face_image[y1:y2, x1:x2]
  if crop.size == 0:
    return face_image
  return crop


def _evaluate_detected_faces(
  *,
  evaluator: FacePartStateEvaluator,
  image_bgr: np.ndarray,
  only_center_face: bool,
  face_size: int,
  det_model: str,
  device: typing.Optional[str],
) -> typing.Dict[str, typing.Any]:
  helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=face_size,
    crop_ratio=(1, 1),
    det_model=det_model,
    use_parse=False,
    device=device if device is not None else "cpu",
  )

  helper.clean_all()
  helper.read_image(image_bgr)
  num_faces = helper.get_face_landmarks_5(
    only_center_face=only_center_face,
    resize=640,
    eye_dist_threshold=5,
  )
  landmarks_list = getattr(helper, "all_landmarks_5", None)

  results: typing.List[typing.Dict[str, typing.Any]] = []
  if num_faces <= 0 or not landmarks_list:
    return {
      "mode": "face",
      "faces_detected": 0,
      "faces": results,
    }

  h, w = image_bgr.shape[:2]
  parts_to_eval = [p for p in ("eyes", "nose", "mouth") if p in evaluator.parts]
  for face_index, landmarks in enumerate(landmarks_list):
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape != (5, 2):
      continue

    face_box = _face_box_from_landmarks(lm, w, h)
    if face_box is None:
      continue

    x1, y1, x2, y2 = face_box
    face_crop = image_bgr[y1:y2, x1:x2].copy()
    if face_crop.size == 0:
      continue

    part_centers = {
      "eyes": tuple(np.mean(lm[:2], axis=0).tolist()),
      "nose": tuple(lm[2].tolist()),
      "mouth": tuple(np.mean(lm[3:], axis=0).tolist()),
    }

    part_results: typing.Dict[str, typing.Any] = {}
    for part in parts_to_eval:
      part_center = part_centers[part]
      center_x = float((part_center[0] - x1) / max(1, (x2 - x1)))
      center_y = float((part_center[1] - y1) / max(1, (y2 - y1)))
      center_x = max(0.0, min(1.0, center_x))
      center_y = max(0.0, min(1.0, center_y))

      part_crop = _extract_part_crop(face_crop, center_x, center_y, part)
      cv2.imwrite("./debug/part_" + str(part) + ".jpg", part_crop)
      part_results[part] = evaluator.evaluate(
        part_image_bgr=part_crop,
        part=typing.cast(typing.Literal["eyes", "nose", "mouth"], part),
        coords=(center_x, center_y),
      )

    results.append({
      "face_index": face_index,
      "face_box": [x1, y1, x2, y2],
      "parts": part_results,
    })

  return {
    "mode": "face",
    "faces_detected": len(results),
    "faces": results,
  }


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

  if args.part == "face":
    result = _evaluate_detected_faces(
      evaluator=evaluator,
      image_bgr=image,
      only_center_face=args.only_center_face,
      face_size=args.face_size,
      det_model=args.det_model,
      device=args.device,
    )
  else:
    result = evaluator.evaluate(
      part_image_bgr=image,
      part=typing.cast(typing.Literal["eyes", "nose", "mouth"], args.part),
      coords=args.coords,
    )

  print(json.dumps(result, ensure_ascii=False, indent=args.indent))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
