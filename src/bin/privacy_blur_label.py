import argparse
import json
import pathlib
import tkinter as tk
import typing
from tkinter import messagebox

from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PrivacyBlurLabelApp:
  def __init__(self, root: tk.Tk, image_paths: typing.List[pathlib.Path]) -> None:
    self.root = root
    self.image_paths = image_paths
    self.index = 0

    self.root.title("privacy_blur_label_util")

    self.has_eyes_privacy_blur = tk.BooleanVar(value=False)
    self.has_face_privacy_blur = tk.BooleanVar(value=False)
    self.has_face_privacy_blur.trace_add("write", self._on_face_blur_changed)

    self.image_title = tk.Label(root, text="", anchor="w", justify="left")
    self.image_title.pack(fill="x", padx=8, pady=(8, 4))

    self.image_label = tk.Label(root)
    self.image_label.pack(fill="both", expand=True, padx=8, pady=8)

    controls = tk.Frame(root)
    controls.pack(fill="x", padx=8, pady=(0, 8))

    tk.Checkbutton(
      controls,
      text="has_eyes_privacy_blur",
      variable=self.has_eyes_privacy_blur,
    ).pack(side="left", padx=(0, 12))

    tk.Checkbutton(
      controls,
      text="has_face_privacy_blur",
      variable=self.has_face_privacy_blur,
    ).pack(side="left", padx=(0, 12))

    tk.Button(
      controls,
      text="Drop",
      command=self.on_drop,
      width=14,
    ).pack(side="right", padx=(0, 8))

    tk.Button(
      controls,
      text="Next",
      command=self.on_next,
      width=14,
    ).pack(side="right")

    self.root.bind("<Return>", self.on_next)
    self.root.bind("<KP_Enter>", self.on_next)

    self._current_photo: typing.Optional[ImageTk.PhotoImage] = None
    self._show_current_image()

  def _show_current_image(self) -> None:
    if self.index >= len(self.image_paths):
      messagebox.showinfo("Done", "All images labeled.")
      self.root.destroy()
      return

    image_path = self.image_paths[self.index]
    payload = self._load_payload(image_path)
    self.has_eyes_privacy_blur.set(bool(payload.get("has_eyes_privacy_blur", False)))
    self.has_face_privacy_blur.set(bool(payload.get("has_face_privacy_blur", False)))

    self.image_title.config(
      text=f"[{self.index + 1}/{len(self.image_paths)}] {image_path.name}",
    )

    image = Image.open(image_path).convert("RGB")
    max_width = max(320, self.root.winfo_screenwidth() - 120)
    max_height = max(240, self.root.winfo_screenheight() - 260)
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

    self._current_photo = ImageTk.PhotoImage(image)
    self.image_label.config(image=self._current_photo)

  def _load_payload(self, image_path: pathlib.Path) -> typing.Dict[str, typing.Any]:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
      return {}

    try:
      return typing.cast(typing.Dict[str, typing.Any], json.loads(json_path.read_text()))
    except json.JSONDecodeError:
      return {}

  def _on_face_blur_changed(self, *_args: typing.Any) -> None:
    if self.has_face_privacy_blur.get():
      self.has_eyes_privacy_blur.set(True)

  def on_next(self, _event: typing.Optional[tk.Event] = None) -> None:
    image_path = self.image_paths[self.index]
    payload = {
      "has_eyes_privacy_blur": bool(self.has_eyes_privacy_blur.get()),
      "has_face_privacy_blur": bool(self.has_face_privacy_blur.get()),
    }

    json_path = image_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    self.index += 1
    self._show_current_image()

  def on_drop(self) -> None:
    image_path = self.image_paths[self.index]
    json_path = image_path.with_suffix(".json")

    if json_path.exists():
      json_path.unlink()
    image_path.unlink()

    self.index += 1
    self._show_current_image()


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Utility for label has_eyes_privacy_blur / has_face_privacy_blur",
  )
  parser.add_argument(
    "--input",
    type=pathlib.Path,
    required=True,
    help="Face images folder",
  )
  parser.add_argument(
    "--extensions",
    nargs="*",
    default=sorted(IMAGE_EXTENSIONS),
    help="Face image extensions",
  )
  parser.add_argument(
    "--revalidate",
    action="store_true",
    help="Open all images including those that already have a corresponding .json label",
  )
  return parser


def _collect_pending_images(
  images_dir: pathlib.Path,
  extensions: typing.Set[str],
  revalidate: bool,
) -> typing.List[pathlib.Path]:
  image_paths = [
    path for path in sorted(images_dir.iterdir())
    if path.is_file() and path.suffix.lower() in extensions
  ]
  if revalidate:
    return image_paths
  return [path for path in image_paths if not path.with_suffix(".json").exists()]


def main() -> int:
  parser = _build_parser()
  args = parser.parse_args()

  if not args.input.is_dir():
    parser.error(f"Input directory does not exist: {args.input}")

  extensions = {str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in args.extensions}
  pending_images = _collect_pending_images(args.input, extensions, args.revalidate)

  if not pending_images:
    print("No images found for labeling with the provided filters.")
    return 0

  root = tk.Tk()
  app = PrivacyBlurLabelApp(root, pending_images)
  root.mainloop()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
