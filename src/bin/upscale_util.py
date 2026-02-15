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


def _with_header(image: typing.Optional[typing.Any], header: str) -> typing.Optional[typing.Any]:
  if image is None:
    return None
  if getattr(image, "size", 0) == 0:
    return None

  strip_h = 34
  out = cv2.copyMakeBorder(
    image,
    strip_h,
    0,
    0,
    0,
    borderType=cv2.BORDER_CONSTANT,
    value=(245, 245, 245),
  )

  font = cv2.FONT_HERSHEY_SIMPLEX
  text_scale = 0.55
  text_thickness = 1
  text_size, _ = cv2.getTextSize(header, font, text_scale, text_thickness)
  text_x = max(8, (out.shape[1] - text_size[0]) // 2)
  text_y = max(22, (strip_h + text_size[1]) // 2)
  cv2.putText(
    out,
    header,
    (text_x, text_y),
    font,
    text_scale,
    (20, 20, 20),
    text_thickness,
    cv2.LINE_AA,
  )
  return out


def _overlay_mask(
  image: typing.Optional[np.ndarray],
  mask: typing.Optional[np.ndarray],
  *,
  color: typing.Tuple[int, int, int] = (0, 255, 0),
  alpha: float = 0.45,
) -> typing.Optional[np.ndarray]:
  if image is None or getattr(image, "size", 0) == 0:
    return None
  if mask is None or getattr(mask, "size", 0) == 0:
    return image

  base = image.copy()
  if base.ndim == 2:
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
  elif base.ndim == 3 and base.shape[2] == 4:
    base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)

  mono_mask = mask
  if mono_mask.ndim == 3:
    mono_mask = cv2.cvtColor(mono_mask, cv2.COLOR_BGR2GRAY)

  if mono_mask.shape[:2] != base.shape[:2]:
    mono_mask = cv2.resize(mono_mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

  mask_bool = mono_mask > 0
  if not np.any(mask_bool):
    return base

  overlay = np.zeros_like(base)
  overlay[:, :] = color
  blended = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
  base[mask_bool] = blended[mask_bool]
  return base


def _parse_bool_flag(value: str) -> bool:
  v = str(value).strip().lower()
  return v in {"1", "true", "yes", "y", "on"}


def _parse_face_processor(raw: str) -> upscaler.FaceProcessor:
  parts = [x.strip() for x in str(raw).split(":")]
  if not parts or not parts[0]:
    raise ValueError(f"Empty face processor entry: {raw}")

  processor = parts[0]
  if processor not in {"codeformer", "gfpgan", "restoreformer", "rollback_diff"}:
    raise ValueError(f"Unknown processor: {processor}")
  max_apply_px = None
  stop_apply = True

  if len(parts) >= 2 and parts[1] != "":
    max_apply_px = int(parts[1])
  if len(parts) >= 3 and parts[2] != "":
    stop_apply = _parse_bool_flag(parts[2])

  return upscaler.FaceProcessor(
    processor=typing.cast(upscaler.FaceProcessorName, processor),
    max_apply_px=max_apply_px,
    stop_apply=stop_apply,
  )


def _default_face_processors() -> typing.List[upscaler.FaceProcessor]:
  return upscaler.default_face_processors()


def _draw_landmark_crosses(
  image: typing.Optional[np.ndarray],
  points_norm: typing.Optional[typing.List[typing.List[float]]],
  *,
  color: typing.Tuple[int, int, int] = (0, 0, 255),
  arm: int = 4,
  thickness: int = 1,
) -> typing.Optional[np.ndarray]:
  if image is None or getattr(image, "size", 0) == 0:
    return image
  if not points_norm:
    return image

  out = image.copy()
  h, w = out.shape[:2]
  for point in points_norm:
    if not point or len(point) < 2:
      continue
    x = int(round(float(point[0]) * w))
    y = int(round(float(point[1]) * h))
    if x < 0 or x >= w or y < 0 or y >= h:
      continue
    cv2.line(out, (x - arm, y), (x + arm, y), color, thickness, cv2.LINE_AA)
    cv2.line(out, (x, y - arm), (x, y + arm), color, thickness, cv2.LINE_AA)
  return out


async def main(
  files: typing.List[ProcessFile],
  codeformer_fidelity: float = 0.3,
  output_faces: str = None,
  outscale: float = 4.0,
  diff_thr: float = (20.0 / 255.0),
  diff_min_area: float = 0.0003,
  diff_opening_window: float = 0.007,
  face_processors: typing.Optional[typing.List[upscaler.FaceProcessor]] = None,
  restoreformer_weights: typing.Optional[str] = None,
  rollback_extend: float = 0.0,
):
  resolved_face_processors = face_processors
  if resolved_face_processors is None:
    resolved_face_processors = _default_face_processors()

  upscalerr = upscaler.Upscaler(
    realesrgan_weights="RealESRGAN_x4plus.pth",
    gfpgan_weights="GFPGANv1.4.pth",
    restoreformer_weights=restoreformer_weights,
    codeformer_weights="codeformer.pth",
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
        diff_opening_window=diff_opening_window,
        face_processors=resolved_face_processors,
        rollback_extend=rollback_extend,
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
        collage_parts = []
        for step in face_info.steps:
          if step is None:
            continue
          collage_parts.append((step.image, step.name))

        if not collage_parts:
          fallback = face_info.visualize(out)
          collage_parts = [(fallback, "result")]

        valid_parts = [(face, label) for face, label in collage_parts if face is not None and face.size]
        if not valid_parts:
          continue

        target_h = max(face.shape[0] for face, _ in valid_parts)
        resized_faces = []
        for face, label in valid_parts:
          prepared = _resize_to_height(face, target_h)
          source_h, source_w = face.shape[:2]
          prepared = _with_label(prepared, f"{label} ({source_w}x{source_h})")
          if prepared is not None:
            resized_faces.append(prepared)
        if not resized_faces:
          continue

        face_img = cv2.hconcat(resized_faces)
        collage_params = (
          f"outscale={float(outscale):g}; "
          f"diff_opening_window={float(diff_opening_window):g}; "
          f"rollback_extend={float(rollback_extend):g}; "
          f"diff_thr={float(diff_thr):g}; "
          f"diff_min_area={float(diff_min_area):g}"
        )
        face_img = _with_header(face_img, collage_params)
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
  parser.add_argument("--output-faces", type=str, default=None)
  parser.add_argument('--outscale', type=float, default=4.0)
  parser.add_argument('--diff-thr', type=float, default=0.02, help='Normalized threshold in [0, 1]')
  parser.add_argument('--diff-min-area', type=float, default=0.0003, help='Min area in [0, 1] as share of face crop')
  parser.add_argument('--diff-opening-window', type=float, default=0.007)
  parser.add_argument('--face-processor', action='append', default=None, help='format: processor[:max_apply_px[:stop_apply]]')
  parser.add_argument('--restoreformer-weights', type=str, default=None)
  parser.add_argument('--rollback-extend', type=float, default=0.0)
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

  face_processors = None
  if args.face_processor:
    face_processors = [_parse_face_processor(x) for x in args.face_processor]

  global_loop.run_until_complete(
    main(
      files,
      codeformer_fidelity=args.codeformer_fidelity,
      output_faces=args.output_faces,
      outscale=args.outscale,
      diff_thr=args.diff_thr,
      diff_min_area=args.diff_min_area,
      diff_opening_window=args.diff_opening_window,
      face_processors=face_processors,
      restoreformer_weights=args.restoreformer_weights,
      rollback_extend=args.rollback_extend,
    )
  )
