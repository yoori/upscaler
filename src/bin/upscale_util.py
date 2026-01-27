import asyncio
import cv2
import torch
import upscaler
import argparse
import os


async def main(
  input_file,
  output_file,
  use_codeformer: bool = True,
  codeformer_fidelity=0.3,
  output_faces: str = None,
):
  upscalerr = upscaler.Upscaler(
    realesrgan_weights="RealESRGAN_x4plus.pth",
    gfpgan_weights="GFPGANv1.4.pth",
    codeformer_weights=("codeformer.pth" if use_codeformer else None),
  )

  img = cv2.imread(input_file, cv2.IMREAD_COLOR)
  out, upscale_info = await upscalerr.upscale_with_info(
    img,
    upscaler.UpscaleParams(
      outscale=4,
      codeformer_fidelity=codeformer_fidelity,
    )
  )

  cv2.imwrite(output_file, out)
  print("Result:\n" + str(upscale_info))

  if output_faces:
    os.makedirs(output_faces, exist_ok=True)
    for idx, face_info in enumerate(upscale_info.faces):
      orig_face = face_info.visualize(img)
      up_face = face_info.visualize(out)
      if orig_face.size == 0 or up_face.size == 0:
        continue
      target_h = max(orig_face.shape[0], up_face.shape[0])
      if orig_face.shape[0] != target_h:
        new_w = max(1, int(round(orig_face.shape[1] * target_h / orig_face.shape[0])))
        orig_face = cv2.resize(orig_face, (new_w, target_h), interpolation=cv2.INTER_AREA)
      if up_face.shape[0] != target_h:
        new_w = max(1, int(round(up_face.shape[1] * target_h / up_face.shape[0])))
        up_face = cv2.resize(up_face, (new_w, target_h), interpolation=cv2.INTER_AREA)
      face_img = cv2.hconcat([orig_face, up_face])
      face_path = os.path.join(output_faces, f"face_{idx}.png")
      cv2.imwrite(face_path, face_img)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'upscale_util.')
  parser.add_argument("-i", "--input", type=str)
  parser.add_argument("-o", "--output", type=str)
  parser.add_argument("--codeformer-fidelity", type=float, default=0.3)
  parser.add_argument('--no-codeformer', dest='use_codeformer', action='store_false')
  parser.add_argument("--output-faces", type=str, default=None)
  parser.set_defaults(use_codeformer=True)
  args = parser.parse_args()

  global_loop = asyncio.new_event_loop()
  global_loop.run_until_complete(
    main(
      args.input,
      args.output,
      use_codeformer=args.use_codeformer,
      codeformer_fidelity=args.codeformer_fidelity,
      output_faces=args.output_faces,
    )
  )
