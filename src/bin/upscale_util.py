import asyncio
import cv2
import torch
import upscaler
import argparse


async def main(
  input_file,
  output_file,
  use_codeformer: bool = True,
  codeformer_fidelity=0.3,
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'upscale_util.')
  parser.add_argument("-i", "--input", type=str)
  parser.add_argument("-o", "--output", type=str)
  parser.add_argument("--codeformer-fidelity", type=float, default=0.3)
  parser.add_argument('--no-codeformer', dest='use_codeformer', action='store_false')
  parser.set_defaults(use_codeformer=True)
  args = parser.parse_args()

  global_loop = asyncio.new_event_loop()
  global_loop.run_until_complete(
    main(
      args.input,
      args.output,
      use_codeformer=args.use_codeformer,
      codeformer_fidelity=args.codeformer_fidelity,
    )
  )
