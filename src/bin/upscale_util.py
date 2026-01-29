import argparse
import asyncio
import dataclasses
import os
import pathlib
import typing

import cv2
import torch
import upscaler


@dataclasses.dataclass(frozen=True)
class ProcessFile:
  input_file: pathlib.Path
  output_file: pathlib.Path


async def main(
  files: typing.List[ProcessFile],
  use_codeformer: bool = True,
  codeformer_fidelity=0.3,
  output_faces: str = None,
):
  upscalerr = upscaler.Upscaler(
    realesrgan_weights="RealESRGAN_x4plus.pth",
    gfpgan_weights="GFPGANv1.4.pth",
    codeformer_weights=("codeformer.pth" if use_codeformer else None),
  )

  for file_info in files:
    img = cv2.imread(str(file_info.input_file), cv2.IMREAD_COLOR)
    out, upscale_info = await upscalerr.upscale_with_info(
      img,
      upscaler.UpscaleParams(
        outscale=4,
        codeformer_fidelity=codeformer_fidelity,
      )
    )

    cv2.imwrite(str(file_info.output_file), out)
    print("Result:\n" + str(upscale_info))

    if output_faces:
      os.makedirs(output_faces, exist_ok=True)
      for idx, face_info in enumerate(upscale_info.faces):
        orig_face = face_info.visualize(img)
        up_face = face_info.visualize(out)
        eye_mask = face_info.get_eye_mask(width=img.shape[1], height=img.shape[0])
        if orig_face.size == 0 or up_face.size == 0 or eye_mask.size == 0:
          continue
        if eye_mask.shape[0] != img.shape[0] or eye_mask.shape[1] != img.shape[1]:
          continue
        eye_region = cv2.bitwise_and(img, img, mask=eye_mask)
        ys, xs = (eye_mask > 0).nonzero()
        if ys.size == 0 or xs.size == 0:
          continue
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        eye_face = eye_region[y1:y2, x1:x2].copy()
        if eye_face.size == 0:
          continue
        target_h = max(orig_face.shape[0], up_face.shape[0], eye_face.shape[0])
        if orig_face.shape[0] != target_h:
          new_w = max(1, int(round(orig_face.shape[1] * target_h / orig_face.shape[0])))
          orig_face = cv2.resize(orig_face, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if up_face.shape[0] != target_h:
          new_w = max(1, int(round(up_face.shape[1] * target_h / up_face.shape[0])))
          up_face = cv2.resize(up_face, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if eye_face.shape[0] != target_h:
          new_w = max(1, int(round(eye_face.shape[1] * target_h / eye_face.shape[0])))
          eye_face = cv2.resize(eye_face, (new_w, target_h), interpolation=cv2.INTER_AREA)
        face_img = cv2.hconcat([orig_face, up_face, eye_face])
        face_path = os.path.join(output_faces, f"face_{idx}.png")
        cv2.imwrite(face_path, face_img)


def resolve_output_path(
  input_path: pathlib.Path,
  output_path: typing.Optional[pathlib.Path],
) -> pathlib.Path:
  if output_path is None:
    output_dir = input_path.parent
    return output_dir / f"{input_path.stem}_result{input_path.suffix}"
  if output_path.is_dir() or (not output_path.exists() and output_path.suffix == ""):
    output_dir = output_path
    return output_dir / f"{input_path.stem}_result{input_path.suffix}"
  return output_path


def collect_images(input_dir: pathlib.Path) -> typing.List[pathlib.Path]:
  image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
  return sorted(
    path
    for path in input_dir.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'upscale_util.')
  parser.add_argument("-i", "--input", type=str)
  parser.add_argument("-o", "--output", type=str)
  parser.add_argument("--codeformer-fidelity", type=float, default=0.3)
  parser.add_argument('--no-codeformer', dest='use_codeformer', action='store_false')
  parser.add_argument("--output-faces", type=str, default=None)
  parser.set_defaults(use_codeformer=True)
  args = parser.parse_args()

  input_path = pathlib.Path(args.input)
  output_path = pathlib.Path(args.output) if args.output else None

  global_loop = asyncio.new_event_loop()

  if input_path.is_dir():
    output_dir = output_path or input_path
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
      ProcessFile(
        input_file=image_path,
        output_file=resolve_output_path(image_path, output_dir),
      )
      for image_path in collect_images(input_path)
    ]
  else:
    resolved_output = resolve_output_path(input_path, output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    files = [ProcessFile(input_file=input_path, output_file=resolved_output)]

  global_loop.run_until_complete(
    main(
      files,
      use_codeformer=args.use_codeformer,
      codeformer_fidelity=args.codeformer_fidelity,
      output_faces=args.output_faces,
    )
  )
