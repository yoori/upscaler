import io
import os
import contextlib
import zipfile
import threading
import typing
import json

import cv2
import numpy as np
import torch
import fastapi
import upscaler


def read_image_bytes_as_bgr(data: bytes) -> np.ndarray:
  arr = np.frombuffer(data, dtype=np.uint8)
  img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
  if img is None:
    raise ValueError("Failed to decode image")
  return img


def encode_bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
  ok, buf = cv2.imencode(".png", img_bgr)
  if not ok:
    raise ValueError("Failed to encode image as PNG")
  return buf.tobytes()

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


def _resolve_face_processors(
  face_processor: typing.Optional[typing.List[str]],
) -> typing.List[upscaler.FaceProcessor]:
  if not face_processor:
    return []
  return [_parse_face_processor(x) for x in face_processor]


class Config(object):
  realesrgan_weights: str = None
  gfpgan_weights: str = None
  restoreformer_weights: str = None
  codeformer_weights: str = None

  def __init__(self):
    self.realesrgan_weights = 'RealESRGAN_x4plus.pth'
    self.gfpgan_weights = None
    self.restoreformer_weights = None
    self.codeformer_weights = None

  def init_json(self, config_json) :
    self.realesrgan_weights = config_json.get('realesrgan_weights', 'RealESRGAN_x4plus.pth')
    self.gfpgan_weights = config_json.get('gfpgan_weights', None)
    self.restoreformer_weights = config_json.get('restoreformer_weights', None)
    self.codeformer_weights = config_json.get('codeformer_weights', None)


upscalerr = None


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
  # read config
  config_file = os.environ.get("CONFIG_PATH", "./upscale_server.conf")
  config = Config()
  with open(config_file, 'r') as f:
    config_txt = f.read()
    config_json = json.loads(config_txt)
    config.init_json(config_json)

  global upscalerr
  upscalerr = upscaler.Upscaler(
    realesrgan_weights=config.realesrgan_weights,
    gfpgan_weights=config.gfpgan_weights,
    restoreformer_weights=config.restoreformer_weights,
    codeformer_weights=config.codeformer_weights,
  )

  yield

  # Shutdown
  await upscalerr.close()


app = fastapi.FastAPI(
  lifespan=lifespan,
  title="Upscale API (Real-ESRGAN + optional GFPGAN/CodeFormer)",
  docs_url='/rest/docs',
)


@app.get("/health")
async def health():
  return {
    "ok": True,
    "torch_cuda": torch.cuda.is_available(),
  }


@app.post("/upscale")
async def upscale(
  file: fastapi.UploadFile = fastapi.File(...),
  outscale: float = fastapi.Query(4.0, ge=1.0, le=8.0, description="Output scale multiplier (e.g. 2, 4)"),
  tile: int = fastapi.Query(0, ge=0, le=2048, description="Tile size to reduce VRAM, 0=disabled"),
  face_processor: typing.Optional[typing.List[str]] = fastapi.Query(
    None,
    description="Repeatable: processor[:max_apply_px[:stop_apply]]; omit to disable face processing",
  ),
):
  """
  Upload one image -> return upscaled PNG.
  """
  global upscalerr

  if upscalerr is None:
      raise fastapi.HTTPException(status_code=500, detail="Model not initialized")

  data = await file.read()
  try:
    img = read_image_bytes_as_bgr(data)
  except Exception as e:
    raise fastapi.HTTPException(status_code=400, detail=f"Bad image: {e}")

  try:
    processors = _resolve_face_processors(face_processor)
  except Exception as e:
    raise fastapi.HTTPException(status_code=400, detail=f"Bad face_processor: {e}")

  params = upscaler.UpscaleParams(
    outscale=outscale,
    tile=tile,
    face_processors=processors,
  )

  try:
    up_img = await upscalerr.upscale(img, params=params)
  except Exception as e:
    raise fastapi.HTTPException(status_code=500, detail=f"Enhancement failed: {e}")

  try:
    png = encode_bgr_to_png_bytes(up_img)
  except Exception as e:
    raise fastapi.HTTPException(status_code=500, detail=f"Encode failed: {e}")

  return fastapi.responses.Response(content=png, media_type="image/png")


@app.post("/upscale/batch")
async def upscale_batch(
  files: typing.List[fastapi.UploadFile] = fastapi.File(...),
  outscale: float = fastapi.Query(4.0, ge=1.0, le=8.0),
  tile: int = fastapi.Query(0, ge=0, le=2048),
  face_processor: typing.Optional[typing.List[str]] = fastapi.Query(None),
):
  """
  Upload multiple images -> returns ZIP with PNG results.
  """
  global upscalerr

  if upscalerr is None:
    raise fastapi.HTTPException(status_code=500, detail="Model not initialized")

  # Read all first (avoid holding lock during IO)
  payloads = []
  for f in files:
    data = await f.read()
    payloads.append((f.filename or "image", data))

  zip_buf = io.BytesIO()
  try:
    processors = _resolve_face_processors(face_processor)
  except Exception as e:
    raise fastapi.HTTPException(status_code=400, detail=f"Bad face_processor: {e}")

  params = upscaler.UpscaleParams(
    outscale=outscale,
    tile=tile,
    face_processors=processors,
  )

  with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name, data in payloads:
      try:
        img = read_image_bytes_as_bgr(data)
        up_img = await upscalerr.upscale(img, params=params)
        png = encode_bgr_to_png_bytes(up_img)
      except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"Failed on {name}: {e}")

      base = os.path.splitext(os.path.basename(name))[0] or "image"
      z.writestr(f"{base}.png", png)

  zip_buf.seek(0)
  return fastapi.responses.StreamingResponse(
    zip_buf,
    media_type="application/zip",
    headers={"Content-Disposition": "attachment; filename=upscaled.zip"},
  )
