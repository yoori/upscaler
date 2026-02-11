import argparse
import asyncio
import dataclasses
import json
import pathlib
import typing

import cv2
import numpy as np
import upscaler


@dataclasses.dataclass(frozen=True)
class ImageExpectation:
  algorithm: str
  reversals: typing.List[str]


def collect_images(input_dir: pathlib.Path) -> typing.List[pathlib.Path]:
  image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
  return sorted(
    path
    for path in input_dir.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
  )


def load_expectations(json_path: pathlib.Path) -> typing.List[ImageExpectation]:
  with json_path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  faces = payload.get("faces", [])
  if not isinstance(faces, list):
    raise ValueError(f"Expected 'faces' list in {json_path}")
  expectations: typing.List[ImageExpectation] = []
  for idx, face in enumerate(faces):
    if not isinstance(face, dict):
      raise ValueError(f"Face entry #{idx} must be an object in {json_path}")
    algorithm = face.get("algorithm", "")
    reversals = face.get("reversals", [])
    if reversals is None:
      reversals = []
    if not isinstance(reversals, list):
      raise ValueError(f"reversals for face #{idx} must be a list in {json_path}")
    expectations.append(ImageExpectation(
      algorithm=algorithm,
      reversals=reversals,
    ))
  return expectations


def _normalize_reversals(values: typing.Iterable[str]) -> typing.List[str]:
  return sorted({str(value) for value in values if value})


def detect_reversals(
  face_info: upscaler.UpscaleFaceInfo,
  *,
  image_shape: typing.Tuple[int, int],
) -> typing.List[str]:
  if image_shape[0] <= 0 or image_shape[1] <= 0:
    return []
  mask = face_info.strong_change_eye_mask
  if mask is None:
    mask = face_info.get_strong_change_eye_mask()
  if mask.size == 0:
    return []

  if np.any(mask > 0):
    return ["revert-eyes"]
  return []


def extract_actual_faces(
  info: upscaler.UpscaleInfo,
  *,
  image_shape: typing.Tuple[int, int],
) -> typing.List[ImageExpectation]:
  faces_sorted = sorted(info.faces, key=lambda f: (f.bbox[0] if f.bbox else 0.0))
  actual_faces: typing.List[ImageExpectation] = []
  for face_info in faces_sorted:
    reversals = detect_reversals(face_info, image_shape=image_shape)
    actual_faces.append(ImageExpectation(
      algorithm=face_info.algorithm,
      reversals=reversals,
    ))
  return actual_faces


def compare_faces(
  expected: typing.List[ImageExpectation],
  actual: typing.List[ImageExpectation],
) -> typing.Tuple[int, int, typing.List[str]]:
  total_faces = max(len(expected), len(actual))
  mismatched = 0
  details: typing.List[str] = []
  for idx in range(total_faces):
    if idx >= len(expected):
      mismatched += 1
      details.append(f"face[{idx}]: extra actual {actual[idx]}")
      continue
    if idx >= len(actual):
      mismatched += 1
      details.append(f"face[{idx}]: missing actual, expected {expected[idx]}")
      continue
    exp = expected[idx]
    act = actual[idx]
    exp_algo = exp.algorithm
    exp_rev = _normalize_reversals(exp.reversals)
    act_algo = act.algorithm
    act_rev = _normalize_reversals(act.reversals)
    if exp_algo != act_algo or exp_rev != act_rev:
      mismatched += 1
      details.append(
        f"face[{idx}]: expected algorithm={exp_algo}, reversals={exp_rev}; "
        f"actual algorithm={act_algo}, reversals={act_rev}"
      )
  return mismatched, total_faces, details


async def check_folder(
  input_dir: pathlib.Path,
  *,
  codeformer_fidelity: float,
) -> None:
  upscalerr = upscaler.Upscaler(
    realesrgan_weights="RealESRGAN_x4plus.pth",
    gfpgan_weights="GFPGANv1.4.pth",
    codeformer_weights="codeformer.pth",
  )

  images = collect_images(input_dir)
  mismatched_images = 0
  total_images = len(images)
  mismatched_faces = 0
  total_faces = 0

  for image_path in images:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
      mismatched_images += 1
      print(f"Mismatch for {image_path.name}: missing {json_path.name}")
      continue

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
      mismatched_images += 1
      print(f"Mismatch for {image_path.name}: failed to read image")
      continue

    try:
      expected = load_expectations(json_path)
    except Exception as exc:
      mismatched_images += 1
      print(f"Mismatch for {image_path.name}: invalid json ({exc})")
      continue

    _, info = await upscalerr.upscale_with_info(
      img,
      upscaler.UpscaleParams(
        outscale=4,
        codeformer_fidelity=codeformer_fidelity,
      ),
    )

    actual = extract_actual_faces(info, image_shape=img.shape[:2])
    face_mismatch, face_total, details = compare_faces(expected, actual)
    total_faces += face_total
    mismatched_faces += face_mismatch
    if face_mismatch > 0:
      mismatched_images += 1
      print(f"Mismatch for {image_path.name}:")
      for line in details:
        print(f"  - {line}")

  print(f"Images mismatched: {mismatched_images}/{total_images}")
  print(f"Faces mismatched: {mismatched_faces}/{total_faces}")


def main() -> None:
  parser = argparse.ArgumentParser(description="upscale_check")
  parser.add_argument("-i", "--input", required=True, type=str)
  parser.add_argument("--codeformer-fidelity", type=float, default=0.3)
  args = parser.parse_args()

  input_dir = pathlib.Path(args.input)
  if not input_dir.is_dir():
    raise SystemExit(f"Input path is not a directory: {input_dir}")

  global_loop = asyncio.new_event_loop()
  global_loop.run_until_complete(
    check_folder(
      input_dir,
      codeformer_fidelity=args.codeformer_fidelity,
    )
  )


if __name__ == "__main__":
  main()
