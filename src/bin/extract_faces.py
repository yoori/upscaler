import argparse
import pathlib
import typing

import cv2

from upscaler.face_searcher import FaceSearcher


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Extract all detected faces from every image in a folder",
  )
  parser.add_argument(
    "--input",
    type=pathlib.Path,
    required=True,
    help="Path to a folder with source images",
  )
  parser.add_argument(
    "--output",
    type=pathlib.Path,
    required=True,
    help="Path to output folder for extracted faces",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Torch device for face detection (e.g. 'cpu' or 'cuda')",
  )
  parser.add_argument(
    "--face-size",
    type=int,
    default=512,
    help="Aligned face crop size used by FaceSearcher",
  )
  return parser.parse_args()


def list_images(input_dir: pathlib.Path) -> typing.List[pathlib.Path]:
  image_paths = [
    path for path in sorted(input_dir.iterdir())
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
  ]
  return image_paths


def main() -> None:
  args = parse_args()

  if not args.input.is_dir():
    raise RuntimeError(f"Input path is not a directory: {args.input}")

  args.output.mkdir(parents=True, exist_ok=True)

  searcher = FaceSearcher(device=args.device, face_size=int(args.face_size))

  image_paths = list_images(args.input)
  for image_path in image_paths:
    image = cv2.imread(str(image_path))
    if image is None:
      raise RuntimeError(f"Failed to read image: {image_path}")

    faces = searcher.get_faces(image, is_bgr=True)
    for face_index, face in enumerate(faces):
      if face.crop is None:
        continue

      output_name = f"{image_path.stem}_face{face_index}.jpg"
      output_path = args.output / output_name
      ok = cv2.imwrite(str(output_path), face.crop)
      if not ok:
        raise RuntimeError(f"Failed to write face image: {output_path}")


if __name__ == "__main__":
  main()
