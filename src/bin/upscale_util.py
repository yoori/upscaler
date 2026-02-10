import argparse
import asyncio
import dataclasses
import os
import pathlib
import typing
import json
import numpy as np

import cv2
import torch
import upscaler


class NumpyAdaptEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return "np.ndarray" # Convert the array to a list
    elif isinstance(obj, np.integer):
      return int(obj) # Handle numpy specific integer types
    elif isinstance(obj, np.floating):
      return float(obj) # Handle numpy specific float types
    return json.JSONEncoder.default(self, obj)


@dataclasses.dataclass(frozen=True)
class ProcessFile:
  input_file: pathlib.Path
  output_file: pathlib.Path


def _resize_to_height(image: typing.Optional[typing.Any], target_h: int):
  if image is None:
    return None
  if getattr(image, "size", 0) == 0:
    return None
  if image.shape[0] == target_h:
    return image
  new_w = max(1, int(round(image.shape[1] * target_h / image.shape[0])))
  return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _with_label(image: typing.Optional[typing.Any], label: str) -> typing.Optional[typing.Any]:
  if image is None:
    return None
  if getattr(image, "size", 0) == 0:
    return None

  strip_h = 28
  labeled = cv2.copyMakeBorder(
    image,
    0,
    strip_h,
    0,
    0,
    borderType=cv2.BORDER_CONSTANT,
    value=(255, 255, 255),
  )

  baseline = 0
  font = cv2.FONT_HERSHEY_SIMPLEX
  text_scale = 0.5
  text_thickness = 1
  text_size, baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
  text_x = max(4, (labeled.shape[1] - text_size[0]) // 2)
  text_y = image.shape[0] + max(16, (strip_h + text_size[1]) // 2)
  cv2.putText(
    labeled,
    label,
    (text_x, text_y),
    font,
    text_scale,
    (20, 20, 20),
    text_thickness,
    cv2.LINE_AA,
  )
  return labeled


async def main(
  files: typing.List[ProcessFile],
  use_codeformer: bool = True,
  codeformer_fidelity=0.3,
  output_faces: str = None,
  outscale: float = 4.0,
  diff_thr: float = 18.0,
  diff_min_area: int = 80,
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
        outscale=outscale,
        codeformer_fidelity=codeformer_fidelity,
        fill_debug_images=bool(output_faces),
        diff_thr=diff_thr,
        diff_min_area=diff_min_area,
      )
    )

    cv2.imwrite(str(file_info.output_file), out)
    print(
      "Result:\n" + json.dumps(
        dataclasses.asdict(upscale_info),
        indent=2,
        cls=NumpyAdaptEncoder,
      )
    )

    if output_faces:
      os.makedirs(output_faces, exist_ok=True)
      for idx, face_info in enumerate(upscale_info.faces):
        orig_face = face_info.debug_original_crop
        helper_face = face_info.debug_helper_crop
        transformed_face = face_info.debug_transformed_face
        pasted_face = face_info.debug_pasted_face
        diff_mask = face_info.strong_change_mask
        diff_mask_color = face_info.strong_change_mask_color

        if diff_mask is not None and len(diff_mask.shape) == 2:
          diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
        if diff_mask_color is not None and len(diff_mask_color.shape) == 2:
          diff_mask_color = cv2.cvtColor(diff_mask_color, cv2.COLOR_GRAY2BGR)

        if orig_face is None:
          orig_face = face_info.visualize(img)
        if pasted_face is None:
          pasted_face = face_info.visualize(out)

        collage_parts = [
          (orig_face, "original crop"),
          (helper_face, "helper crop on upscaled"),
          (transformed_face, str(face_info.algorithm) + " transformed"),
          (diff_mask, "diff mask luma"),
          (diff_mask_color, "diff mask color sum"),
          (pasted_face, "pasted result"),
        ]

        valid_parts = [(face, label) for face, label in collage_parts if face is not None and face.size]
        if not valid_parts:
          continue

        target_h = max(face.shape[0] for face, _ in valid_parts)
        resized_faces = []
        for face, label in valid_parts:
          prepared = _resize_to_height(face, target_h)
          prepared = _with_label(prepared, label)
          if prepared is not None:
            resized_faces.append(prepared)
        if not resized_faces:
          continue

        face_img = cv2.hconcat(resized_faces)
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
  parser.add_argument('--outscale', type=float, default=4.0)
  parser.add_argument('--diff-thr', type=float, default=10.0)
  parser.add_argument('--diff-min-area', type=int, default=15)
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
      outscale=args.outscale,
      diff_thr=args.diff_thr,
      diff_min_area=args.diff_min_area,
    )
  )
