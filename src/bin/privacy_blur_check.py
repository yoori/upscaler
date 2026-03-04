import argparse
import dataclasses
import json
import math
import pathlib
import typing

import cv2
import numpy as np
import torch

from upscaler.face_detection import FacePrivacyBlurMetrics, ZoneBlurMetrics
from upscaler.face_searcher import FaceSearcher


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclasses.dataclass(frozen=True)
class LabeledSample:
  image_path: pathlib.Path
  has_blur: bool
  metrics: typing.Dict[str, float]


@dataclasses.dataclass(frozen=True)
class ThresholdResult:
  threshold: float
  less_or_equal_is_blur: bool
  weighted_error: float
  false_positive: int
  false_negative: int  # Number of cases when blur is present, but we didn't predict it.
  count_of_blurred: int
  count_of_non_blurred: int
  count_of_blur_predicted: int
  count_of_non_blur_predicted: int


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Find optimal thresholds for privacy blur metrics on a labeled dataset.",
  )
  parser.add_argument("--input-dir", type=pathlib.Path, required=True, help="Directory with images and sidecar JSON labels")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--face-size", type=int, default=512)
  parser.add_argument("--false-negative-weight", type=float, default=2)
  parser.add_argument(
    "--extensions",
    nargs="*",
    default=sorted(IMAGE_EXTENSIONS),
    help="Image extensions to scan (default: .jpg .jpeg .png .bmp .webp)",
  )
  return parser


def _flatten_zone_metrics(zone_metrics: ZoneBlurMetrics, prefix: str) -> typing.Dict[str, float]:
  out: typing.Dict[str, float] = {}

  for group_name in ["compare"]:
    group = getattr(zone_metrics, group_name)
    for field in dataclasses.fields(group):
      if field.name.endswith('_ratio'):
        value = float(getattr(group, field.name))
        out[f"{prefix}.{group_name}.{field.name}"] = value

  """
  for extra_name in [
    "zone_area_ratio",
    "reference_area_ratio",
    "zone_min_size_px",
    "reference_min_size_px",
    "valid_zone",
    "valid_reference",
  ]:
    out[f"{prefix}.{extra_name}"] = float(getattr(zone_metrics, extra_name))
  """

  return out


def _flatten_metrics(metrics: FacePrivacyBlurMetrics, prefix: str) -> typing.Dict[str, float]:
  if prefix == "eyes":
    return _flatten_zone_metrics(metrics.eyes_blur, prefix)
  if prefix == "face":
    return _flatten_zone_metrics(metrics.face_blur, prefix)
  raise ValueError(f"Unsupported metrics prefix: {prefix}")


def _collect_image_files(images_dir: pathlib.Path, extensions: typing.Set[str]) -> typing.List[pathlib.Path]:
  files = [
    path for path in sorted(images_dir.iterdir())
    if path.is_file() and path.suffix.lower() in extensions
  ]
  return files


def _read_label(json_path: pathlib.Path) -> typing.Tuple[bool, bool]:
  try:
    payload = json.loads(json_path.read_text())
  except Exception as error:
    raise ValueError(f"failed to read label json {json_path}: {error}") from error

  if not isinstance(payload, dict):
    raise ValueError(f"label json must be object: {json_path}")

  if "has_eyes_privacy_blur" not in payload or "has_face_privacy_blur" not in payload:
    raise ValueError(
      f"label json must contain has_eyes_privacy_blur and has_face_privacy_blur: {json_path}"
    )

  eyes_blur = bool(payload["has_eyes_privacy_blur"])
  face_blur = bool(payload["has_face_privacy_blur"])
  return eyes_blur, face_blur


def _load_labeled_samples(
  *,
  image_paths: typing.List[pathlib.Path],
  searcher: FaceSearcher,
) -> typing.Tuple[typing.List[LabeledSample], typing.List[LabeledSample]]:
  eyes_samples: typing.List[LabeledSample] = []
  face_samples: typing.List[LabeledSample] = []

  for image_path in image_paths:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
      raise ValueError(f"missing label json for image: {image_path}")

    eyes_blur, face_blur = _read_label(json_path)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
      raise ValueError(f"failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    faces = searcher.get_faces(image_rgb, is_bgr=False)
    if len(faces) > 0:
      if len(faces) > 1:
        raise ValueError(f"image must contain exactly 1 face ({len(faces)} found): {image_path}")

      metrics = faces[0].compute_privacy_blur_metrics()
      print(f">>>> {image_path}")
      print("eyes_blur: " + str(metrics.eyes_blur))
      print("face_blur: " + str(metrics.face_blur) + "\n")
      eyes_samples.append(LabeledSample(
        image_path=image_path,
        has_blur=eyes_blur,
        metrics=_flatten_metrics(metrics, "eyes"),
      ))
      face_samples.append(LabeledSample(
        image_path=image_path,
        has_blur=face_blur,
        metrics=_flatten_metrics(metrics, "face"),
      ))

  return eyes_samples, face_samples


def _compute_candidate_thresholds(values: typing.List[float]) -> typing.List[float]:
  unique_values = sorted(set(values))
  if not unique_values:
    return []

  thresholds = [unique_values[0] - 1.0]
  for i in range(len(unique_values) - 1):
    thresholds.append((unique_values[i] + unique_values[i + 1]) * 0.5)
  thresholds.append(unique_values[-1] + 1.0)
  return thresholds


def _evaluate_threshold(
  *,
  samples: typing.List[LabeledSample],
  metric_name: str,
  threshold: float,
  less_or_equal_is_blur: bool,
  false_negative_weight: float = 2,
) -> ThresholdResult:
  false_positive = 0
  false_negative = 0
  count_of_blurred = 0
  count_of_non_blurred = 0
  count_of_blur_predicted = 0

  for sample in samples:
    value = sample.metrics[metric_name]
    if less_or_equal_is_blur:
      predicted_blur = value <= threshold
    else:
      predicted_blur = value >= threshold

    if predicted_blur:
      count_of_blur_predicted += 1

    if sample.has_blur:
      count_of_blurred += 1
      if not predicted_blur:
        false_negative += 1
    else:
      count_of_non_blurred += 1
      if predicted_blur:
        false_positive += 1

  count_of_non_blur_predicted = len(samples) - count_of_blur_predicted
  weighted_error = float((false_negative_weight * false_negative) + false_positive)

  return ThresholdResult(
    threshold=threshold,
    less_or_equal_is_blur=less_or_equal_is_blur,
    weighted_error=weighted_error,
    false_positive=false_positive,
    false_negative=false_negative,
    count_of_blurred=count_of_blurred,
    count_of_non_blurred=count_of_non_blurred,
    count_of_blur_predicted=count_of_blur_predicted,
    count_of_non_blur_predicted=count_of_non_blur_predicted,
  )


def _find_best_threshold_for_metric(
  samples: typing.List[LabeledSample],
  metric_name: str,
  false_negative_weight: float = 2,
) -> typing.Optional[ThresholdResult]:
  values = [sample.metrics[metric_name] for sample in samples if math.isfinite(sample.metrics[metric_name])]
  if len(values) != len(samples):
    return None

  thresholds = _compute_candidate_thresholds(values)
  if not thresholds:
    return None

  best: typing.Optional[ThresholdResult] = None
  for threshold in thresholds:
    for less_or_equal_is_blur in [True, False]:
      result = _evaluate_threshold(
        samples=samples,
        metric_name=metric_name,
        threshold=threshold,
        less_or_equal_is_blur=less_or_equal_is_blur,
        false_negative_weight=false_negative_weight,
      )
      if best is None:
        best = result
        continue
      if result.weighted_error < best.weighted_error:
        best = result
        continue
      if result.weighted_error == best.weighted_error and result.false_negative < best.false_negative:
        best = result

  return best


def _fit_thresholds(
  samples: typing.List[LabeledSample],
  false_negative_weight: float = 2,
) -> typing.List[typing.Tuple[str, ThresholdResult]]:
  if not samples:
    return []

  metric_names = sorted(samples[0].metrics.keys())
  fitted: typing.List[typing.Tuple[str, ThresholdResult]] = []
  for metric_name in metric_names:
    best = _find_best_threshold_for_metric(
      samples,
      metric_name,
      false_negative_weight=false_negative_weight
    )
    if best is None:
      continue
    fitted.append((metric_name, best))

  fitted.sort(key=lambda item: (item[1].weighted_error, item[1].false_negative, item[0]))
  return fitted


def _print_results(zone_name: str, fitted: typing.List[typing.Tuple[str, ThresholdResult]]) -> None:
  print(f"\n=== {zone_name.upper()} ===")
  if not fitted:
    print("No valid metrics found")
    return

  for metric_name, result in sorted(fitted, key=lambda item: item[0]):
    direction = "<= threshold => blur" if result.less_or_equal_is_blur else ">= threshold => blur"
    print(
      " | ".join([
        metric_name,
        f"threshold={result.threshold:.6g}",
        direction,
        f"weighted_error={result.weighted_error:.0f}",
        f"false_negative={result.false_negative}",
        f"false_positive={result.false_positive}",
        f"count_of_blurred={result.count_of_blurred}",
        f"count_of_non_blurred={result.count_of_non_blurred}",
        f"count_of_blur_predicted={result.count_of_blur_predicted}",
        f"count_of_non_blur_predicted={result.count_of_non_blur_predicted}",
      ])
    )


def main() -> int:
  parser = _build_parser()
  args = parser.parse_args()

  if not args.input_dir.is_dir():
    parser.error(f"Input directory does not exist: {args.input_dir}")

  extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}
  image_paths = _collect_image_files(args.input_dir, extensions)
  if not image_paths:
    parser.error(f"No images found in: {args.input_dir}")

  searcher = FaceSearcher(device=args.device, face_size=args.face_size)
  eyes_samples, face_samples = _load_labeled_samples(image_paths=image_paths, searcher=searcher)

  print(f"Processed {len(image_paths)} images")

  eyes_fitted = _fit_thresholds(eyes_samples, false_negative_weight=args.false_negative_weight)
  face_fitted = _fit_thresholds(face_samples, false_negative_weight=args.false_negative_weight)

  _print_results("eyes", eyes_fitted)
  _print_results("face", face_fitted)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
