import os
import asyncio
import typing
import dataclasses

import numpy as np
import torch
import cv2

import basicsr.archs.rrdbnet_arch
import realesrgan
import gfpgan
import facexlib.utils.face_restoration_helper
import codeformer.basicsr.archs.codeformer_arch


FaceMode = typing.Literal["off", "gfpgan", "auto_per_face", "auto_per_face_cf"]


@dataclasses.dataclass
class UpscaleFaceInfo:
  bbox: typing.List[int]
  face_px: int
  algorithm: str
  landmarks5: typing.Optional[typing.List[typing.List[float]]] = None


@dataclasses.dataclass
class UpscaleInfo:
  face_mode: FaceMode
  faces: typing.List[UpscaleFaceInfo] = dataclasses.field(default_factory=list)
  fallback: typing.Optional[str] = None


@dataclasses.dataclass
class UpscaleParams:
  outscale: float = 4.0
  tile: int = 0

  face_mode: FaceMode = "auto_per_face_cf"
  only_center_face: bool = False

  # per-face threshold on ORIGINAL image (min(w,h) in px)
  min_face_px: int = 96

  # GFPGAN strengths
  gfpgan_weight: float = 0.35
  gfpgan_weight_small: float = 0.20

  # CodeFormer knob (lower => stronger reconstruction; good for low-res)
  codeformer_fidelity: float = 0.30


class Upscaler(object):

  class Exception(Exception):
    pass

  def __init__(
    self,
    realesrgan_weights: str = "RealESRGAN_x4plus.pth",
    gfpgan_weights: typing.Optional[str] = None,
    codeformer_weights: typing.Optional[str] = None,
  ):
    self._realesrgan_weights = realesrgan_weights
    self._gfpgan_weights = gfpgan_weights
    self._codeformer_weights = codeformer_weights

    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    self._use_half = (self._device == "cuda")

    self._lock = asyncio.Lock()

    # Real-ESRGAN
    model = basicsr.archs.rrdbnet_arch.RRDBNet(
      num_in_ch=3,
      num_out_ch=3,
      num_feat=64,
      num_block=23,
      num_grow_ch=32,
      scale=4,
    )

    self._upsampler = realesrgan.RealESRGANer(
      scale=4,
      model_path=self._realesrgan_weights,
      model=model,
      tile=0,
      tile_pad=10,
      pre_pad=0,
      half=self._use_half,
      device=self._device,
    )

    # Face helper (used for align/paste-back for both GFPGAN and CodeFormer)
    self._face_helper = facexlib.utils.face_restoration_helper.FaceRestoreHelper(
      upscale_factor=1,
      face_size=512,
      crop_ratio=(1, 1),
      det_model="retinaface_resnet50",
      save_ext="png",
      use_parse=True,
      device=self._device,
    )

    # GFPGAN (optional)
    if self._gfpgan_weights is not None:
      self._face_enhancer = gfpgan.GFPGANer(
        model_path=self._gfpgan_weights,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=self._device,
      )
    else:
      self._face_enhancer = None

    # CodeFormer (optional)
    self._codeformer_net = None
    if self._codeformer_weights is not None:
      self._codeformer_net = self._load_codeformer(self._codeformer_weights)

  async def upscale(
    self,
    img_bgr: np.ndarray,
    params: typing.Optional[UpscaleParams] = None,
  ) -> np.ndarray:
    result, _ = await self.upscale_with_info(img_bgr, params=params)
    return result

  async def upscale_with_info(
    self,
    img_bgr: np.ndarray,
    params: typing.Optional[UpscaleParams] = None,
  ) -> typing.Tuple[np.ndarray, UpscaleInfo]:
    if params is None:
      params = UpscaleParams()

    result_info = UpscaleInfo(face_mode=params.face_mode)

    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
      raise Upscaler.Exception("Expected BGR image HxWx3")
    if img_bgr.dtype != np.uint8:
      raise Upscaler.Exception(f"Expected uint8 image, got {img_bgr.dtype}")

    tile = 0 if params.tile is None else int(params.tile)
    outscale = float(params.outscale) if params.outscale is not None else 4.0

    async with self._lock:
      # 1) Upscale
      self._upsampler.tile = tile
      up_bgr, _ = self._upsampler.enhance(img_bgr, outscale=outscale)

      # 2) Faces
      if params.face_mode == "off":
        return (up_bgr, result_info)

      elif params.face_mode == "gfpgan":
        if self._face_enhancer is None:
          raise Upscaler.Exception("GFPGAN requested but weights not configured")

        return (
          self._apply_gfpgan_whole(
            up_bgr,
            weight=float(params.gfpgan_weight),
            only_center_face=bool(params.only_center_face),
          ),
          result_info,
        )

      elif params.face_mode == "auto_per_face":
        if self._face_enhancer is None:
          raise Upscaler.Exception("auto_per_face requires GFPGAN weights")

        return self._apply_faces_routed(
          original_bgr=img_bgr,
          upscaled_bgr=up_bgr,
          min_face_px=int(params.min_face_px),
          only_center_face=bool(params.only_center_face),
          # routing: small->gfpgan(weight_small), big->gfpgan(weight_normal)
          gfpgan_weight_normal=float(params.gfpgan_weight),
          gfpgan_weight_small=float(params.gfpgan_weight_small),
          codeformer_fidelity=float(params.codeformer_fidelity),
          enable_codeformer=False,
          info=result_info,
        )

      elif params.face_mode == "auto_per_face_cf":
        # small -> CodeFormer (if available), else fallback to GFPGAN small weight
        if self._face_enhancer is None and self._codeformer_net is None:
          return (up_bgr, result_info)

        return self._apply_faces_routed(
          original_bgr=img_bgr,
          upscaled_bgr=up_bgr,
          min_face_px=int(params.min_face_px),
          only_center_face=bool(params.only_center_face),
          gfpgan_weight_normal=float(params.gfpgan_weight),
          gfpgan_weight_small=float(params.gfpgan_weight_small),
          codeformer_fidelity=float(params.codeformer_fidelity),
          enable_codeformer=True,
          info=result_info,
        )

      else:
        raise Upscaler.Exception(f"Unknown face_mode: {params.face_mode}")

  async def close(self):
    pass

  def _apply_gfpgan_whole(
    self,
    img_bgr: np.ndarray,
    *,
    weight: float,
    only_center_face: bool,
  ) -> np.ndarray:

    _, _, restored = self._face_enhancer.enhance(
      img_bgr,
      has_aligned=False,
      only_center_face=only_center_face,
      paste_back=True,
      weight=weight,
    )
    return restored

  def _detect_faces(
    self,
    img_bgr: np.ndarray,
    *,
    only_center_face: bool,
  ) -> typing.List[typing.Dict]:

    helper = self._face_helper
    helper.clean_all()
    helper.read_image(img_bgr)
    helper.get_face_landmarks_5(
      only_center_face=only_center_face,
      eye_dist_threshold=5,
    )

    det_faces = getattr(helper, "det_faces", None)
    if det_faces is None:
      return []

    out: typing.List[typing.Dict] = []
    for bb in det_faces:
      x1, y1, x2, y2 = bb[:4]
      out.append({
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "size_px": int(min(int(x2 - x1), int(y2 - y1))),
      })
    return out

  def _cv2_ready_bgr(self, img) -> np.ndarray:
    img = np.asarray(img)

    if img.dtype == np.dtype("O"):
      raise Upscaler.Exception("Image dtype=object (cv2 can't handle it)")

    if img.ndim == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
      raise Upscaler.Exception(f"Bad image shape: {img.shape}")

    if img.dtype == np.float16:
      img = img.astype(np.float32, copy=False)

    if img.dtype != np.uint8 and img.dtype != np.float32:
      if np.issubdtype(img.dtype, np.floating):
        m = float(np.max(img)) if img.size else 0.0
        if m <= 1.0:
          img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
      else:
        img = img.astype(np.uint8)

    if not img.flags["C_CONTIGUOUS"]:
      img = np.ascontiguousarray(img)

    return img

  def _create_face_oval_mask(
    self,
    face_crop: np.ndarray,
    *,
    landmarks_5: np.ndarray,
    affine_matrix: np.ndarray,
  ) -> np.ndarray:
    """
    Build an oval mask that covers the main facial area (eyes + mouth),
    aligned to the face orientation using FaceRestoreHelper landmarks.

    Returns a uint8 mask (0/255) matching the face crop size.
    """

    face = self._cv2_ready_bgr(face_crop)
    h, w = face.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    default_center = (int(w * 0.5), int(h * 0.55))
    default_axes = (max(1, int(w * 0.45)), max(1, int(h * 0.60)))

    ellipse = None
    if landmarks_5 is not None and affine_matrix is not None:
      points = np.asarray(landmarks_5, dtype=np.float32).reshape(-1, 2)
      if points.shape[0] >= 5:
        points = cv2.transform(points[None, :, :], affine_matrix)[0]
        eye_left, eye_right, _, mouth_left, mouth_right = points[:5]
        mouth_center = (mouth_left + mouth_right) * 0.5
        ellipse_points = np.stack(
          [eye_left, eye_right, mouth_left, mouth_right, mouth_center],
          axis=0,
        )
        try:
          ellipse = cv2.fitEllipse(ellipse_points.astype(np.float32))
        except cv2.error:
          ellipse = None

    if ellipse is not None:
      (cx, cy), (width, height), angle = ellipse
      scale = 1.02
      axes = (
        max(1, int(width * 0.5 * scale)),
        max(1, int(height * 0.5 * scale)),
      )
      center = (int(cx), int(cy))
      cv2.ellipse(mask, center, axes, float(angle), 0, 360, 255, -1)
    else:
      cv2.ellipse(mask, default_center, default_axes, 0, 0, 360, 255, -1)

    return mask

  def _apply_faces_routed(
    self,
    *,
    original_bgr: np.ndarray,
    upscaled_bgr: np.ndarray,
    min_face_px: int,
    only_center_face: bool,
    gfpgan_weight_normal: float,
    gfpgan_weight_small: float,
    codeformer_fidelity: float,
    enable_codeformer: bool,
    info: UpscaleInfo,
  ) -> typing.Tuple[np.ndarray, UpscaleInfo]:

    print("XXX P3\n")
    helper = self._face_helper

    faces = self._detect_faces(
      original_bgr,
      only_center_face=only_center_face,
    )

    helper.clean_all()
    helper.read_image(upscaled_bgr)
    helper.get_face_landmarks_5(
      only_center_face=only_center_face,
      eye_dist_threshold=5,
    )
    helper.align_warp_face()

    # if mismatch, safe fallback: whole-image GFPGAN if available, else no-op
    if len(helper.cropped_faces) > 0 and len(faces) != len(helper.cropped_faces):
      if self._face_enhancer is not None:
        return (
          self._apply_gfpgan_whole(
            upscaled_bgr,
            weight=gfpgan_weight_normal,
            only_center_face=only_center_face,
          ),
          dataclasses.replace(info, fallback="det_faces != cropped_faces"),
        )
      return (
        upscaled_bgr,
        dataclasses.replace(info, fallback="det_faces != cropped_faces"),
      )

    restored_faces: typing.List[np.ndarray] = []
    face_infos: typing.List[UpscaleFaceInfo] = []

    for i, face_crop in enumerate(helper.cropped_faces):
      face_px = faces[i]['size_px']
      landmarks5 = None
      if i < len(helper.all_landmarks_5):
        landmarks5 = helper.all_landmarks_5[i].astype(float).tolist()
      face_info = UpscaleFaceInfo(
        bbox=faces[i]["bbox"],
        face_px=face_px,
        algorithm="",
        landmarks5=landmarks5,
      )

      if (
        enable_codeformer and
        self._codeformer_net is not None and
        face_px < min_face_px
      ):
        face_info.algorithm = "codeformer"
        result_face, diff_mask, d = self._restore_face_codeformer(face_crop, fidelity=codeformer_fidelity)
        self._debug_save_codeformer(
          face_crop_bgr=face_crop,
          restored_bgr=result_face,
          diff_mask=diff_mask,
          d=d,
          out_dir='./debug/',
          tag="face" + str(i),
        )
        local_restored_faces = [ result_face ]
      elif self._face_enhancer is not None:
        face_info.algorithm = "gfpgan"
        w = gfpgan_weight_small if face_px < min_face_px else gfpgan_weight_normal
        _, local_restored_faces, _ = self._face_enhancer.enhance(
          face_crop,
          has_aligned=True,
          only_center_face=True,
          paste_back=False,
          weight=w,
        )
      else:
        # fallback
        face_info.algorithm = "fallback"
        local_restored_faces = [face_crop]

      if local_restored_faces:
        restored_faces.extend([self._cv2_ready_bgr(x) for x in local_restored_faces])

      face_infos.append(face_info)

    for rf in restored_faces:
      helper.add_restored_face(rf)

    helper.get_inverse_affine(None)

    #result_info = info.copy().update({'faces': face_infos})
    #print(f"XXX P4, result_info = {str(result_info)}")
    return (
      helper.paste_faces_to_input_image(upsample_img=None),
      dataclasses.replace(info, faces=face_infos),
    )

  def _load_codeformer(self, weights_path: str):
    device = torch.device(self._device)

    net = codeformer.basicsr.archs.codeformer_arch.CodeFormer(
      dim_embd=512,
      codebook_size=1024,
      n_head=8,
      n_layers=9,
      connect_list=["32", "64", "128", "256"],
    ).to(device)

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
      net.load_state_dict(ckpt["params_ema"], strict=True)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
      net.load_state_dict(ckpt["state_dict"], strict=False)
    else:
      net.load_state_dict(ckpt, strict=False)

    net.eval()
    #if self._use_half:
    #  net = net.half()

    return net

  def _restore_face_codeformer(self, face_crop_bgr: np.ndarray, *, fidelity: float) -> np.ndarray:
    print("> _restore_face_codeformer")
    # CodeFormer expects RGB, normalized to [-1,1], BCHW
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

    #if self._use_half:
    #  x = x.half()

    x = x.to(next(self._codeformer_net.parameters()).device)

    with torch.no_grad():
      out = self._codeformer_net(x, w=float(fidelity), adain=True)

    # out can be tensor or tuple/list depending on version
    if isinstance(out, (list, tuple)):
      out = out[0]

    out = out.detach().float().cpu().clamp_(-1, 1)
    out = (out * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).numpy()
    out = (out * 255.0).round().astype(np.uint8)

    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    d = self._diff_zones_mean_window(face_crop_bgr, bgr, win=3)

    return bgr, d['mask_u8'], d['d']

  def _diff_zones_mean_window(
    self,
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    *,
    win: int = 21,
    diff_thr: float = 18.0,
    min_area: int = 80,
  ) -> typing.Dict[str, typing.Any]:
    """
    Find zones where img1 deviates from img0 significantly, after averaging (box/gaussian) over a window.

    - img0_bgr, img1_bgr: uint8 BGR images of same size (face crops)
    - win: window size (odd recommended). Larger => more "semantic" difference, less sensitivity to noise.
    - diff_thr: threshold in L-channel units (0..255). Typical start: 12..25.
    - min_area: remove tiny blobs in the final mask.

    Returns dict with:
      diff_mean (float32), mask01 (float32 0..1), mask_u8 (uint8 0/255), boxes (list of [x1,y1,x2,y2])
    """

    if img0_bgr is None or img1_bgr is None:
      raise ValueError("img0_bgr/img1_bgr is None")
    if img0_bgr.shape != img1_bgr.shape:
      raise ValueError(f"Shape mismatch: {img0_bgr.shape} vs {img1_bgr.shape}")
    if img0_bgr.ndim != 3 or img0_bgr.shape[2] != 3:
      raise ValueError(f"Expected HxWx3, got {img0_bgr.shape}")
    if img0_bgr.dtype != np.uint8 or img1_bgr.dtype != np.uint8:
      raise ValueError(f"Expected uint8, got {img0_bgr.dtype} / {img1_bgr.dtype}")

    # 1) Use perceptual luminance: Lab L channel
    lab0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2LAB)
    lab1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2LAB)
    L0 = lab0[:, :, 0].astype(np.float32)
    L1 = lab1[:, :, 0].astype(np.float32)

    # 2) Per-pixel absolute difference
    d = np.abs(L1 - L0)  # 0..255 float32

    # 3) Local averaging over window
    k = int(win)
    if k < 1:
      k = 1
    if (k % 2) == 0:
      k += 1

    # box filter is fast and matches "усреднение по окну"
    diff_mean = cv2.blur(d, (k, k))

    # 4) Threshold -> mask
    mask01 = (diff_mean >= float(diff_thr)).astype(np.float32)

    # 5) Clean mask (optional but usually needed)
    # remove small noise with opening, then fill small gaps with closing
    if k >= 5:
      kk = 3
    else:
      kk = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    mask_u8 = (mask01 * 255.0).astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Remove tiny components
    boxes: typing.List[typing.List[int]] = []
    if int(min_area) > 0:
      num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
      cleaned = np.zeros_like(mask_u8)
      for i in range(1, num):
        x, y, w, h, area = stats[i]
        if int(area) >= int(min_area):
          cleaned[labels == i] = 255
          boxes.append([int(x), int(y), int(x + w), int(y + h)])
      mask_u8 = cleaned

    mask01 = (mask_u8.astype(np.float32) / 255.0)

    return {
      "diff_mean": diff_mean,
      "mask01": mask01,
      "mask_u8": mask_u8,
      "boxes": boxes,
      "d": d,
    }

  def _debug_save_codeformer(
    self,
    *,
    face_crop_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    out_dir: str,
    diff_mask: np.ndarray = None,
    d: np.ndarray = None,
    tag: str = "",
  ) -> None:

    try:
      os.makedirs(out_dir, exist_ok=True)

      face0 = self._cv2_ready_bgr(face_crop_bgr)
      face1 = self._cv2_ready_bgr(restored_bgr)
      face_mask = self._cv2_ready_bgr(diff_mask) if diff_mask is not None else None

      name = "codeformer"
      if tag:
        name += "_" + str(tag)

      p_in = os.path.join(out_dir, name + "_in.jpg")
      p_out = os.path.join(out_dir, name + "_out.jpg")
      p_pair = os.path.join(out_dir, name + "_pair.jpg")
      print(f"p_pair: {p_pair}")

      cv2.imwrite(p_in, face0)
      cv2.imwrite(p_out, face1)

      # side-by-side
      h0, w0 = face0.shape[:2]
      h1, w1 = face1.shape[:2]
      if h0 == h1:
        pair = np.concatenate(
          [face0, face1] +
          ([face_mask] if face_mask is not None else []) +
          ([self._cv2_ready_bgr(d)] if d is not None else []),
          axis=1
        )
        cv2.imwrite(p_pair, pair)

    except Exception as e:
      print("_debug_save_codeformer failed: " + str(e))
