"""Zapmatte green screen remover — the Standard pipeline, open source.

Input video shot in front of a green, blue or teal screen -> the subject on a
transparent background (ProRes 4444 MOV, VP9 WebM with alpha), a black/white
alpha matte, or an MP4 over a new colour or picture. Audio is kept.

How it works: Robust Video Matting (RVM, GPL-3.0, PeterL1n/RobustVideoMatting)
gives a soft human alpha; it is intersected with a chroma-key alpha that
matches the local screen colour near the subject, smoothed by cut density,
with a colour-only key away from people (props), hue rotation for blue/teal
screens, despill, and removal of tracking markers.

This is the same code that runs the Standard tier of https://zapmatte.com.
Runs on CUDA (a T4 does 1080p at ~19 fps) or, slowly, on CPU.

Outputs (each optional, keyed like `want`):
  webm    VP9 + alpha (yuva420p), original audio as Opus
  mov     ProRes 4444 + alpha (yuva444p10le, straight alpha), PCM audio
  matte   black/white alpha as H.264 MP4
  mp4     subject over the chosen colour/image, H.264 + AAC
  preview <=720p H.264 over the chosen background, or a checkerboard
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def model():
    global _model
    if _model is None:
        _model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True).eval().to(DEV)
    return _model


# ── v15 maths (verbatim from research/gpu-test/opt.py) ─────────────────────

def blur(x, k):  # separable box blur approx of gaussian, x [B,1,H,W]
    p = k // 2
    x = F.avg_pool2d(F.pad(x, (p, p, 0, 0), mode="replicate"), (1, k), 1)
    return F.avg_pool2d(F.pad(x, (0, 0, p, p), mode="replicate"), (k, 1), 1)


def dilate(x, k):
    return F.max_pool2d(x, (k, k), 1, k // 2)


def erode(x, k):  # separable (identical result, ~30x fewer ops)
    y = -F.max_pool2d(-x, (1, k), 1, (0, k // 2))
    return -F.max_pool2d(-y, (k, 1), 1, (k // 2, 0))


def rot(x, d):  # rotate hue by d degrees, x [B,3,H,W] RGB 0-1
    if abs(d) < 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    mx = x.max(1, keepdim=True)[0]
    mn = x.min(1, keepdim=True)[0]
    dl = mx - mn
    dz = dl.clamp_min(1e-6)
    h = torch.where(mx == r, ((g - b) / dz) % 6, torch.where(mx == g, (b - r) / dz + 2, (r - g) / dz + 4))
    h = torch.where(dl > 0, h, torch.zeros_like(h))
    h = (h + d / 60.0) % 6
    f = lambda n: mx - dl * (torch.minimum(torch.minimum((n + h) % 6, 4 - (n + h) % 6), torch.ones_like(h))).clamp_min(0)
    return torch.cat([f(5), f(3), f(1)], 1)


def post(rgb, a_ai):  # rgb [B,3,H,W] 0-1, a_ai [B,1,H,W] -> alpha, despilled foreground
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    green = g - torch.maximum(r, b)
    a_key = blur(1 - ((green - 0.06) / 0.14).clamp(0, 1), 3)
    core = blur(erode((a_ai > 0.9).float(), 15), 15)
    deep = blur(erode((a_ai > 0.9).float(), 61), 31)
    bgm = (a_ai < 0.05).float()
    H, W = rgb.shape[2:]
    sm = F.interpolate(bgm, (H // 8, W // 8), mode="area")
    sc = F.interpolate(rgb * bgm, (H // 8, W // 8), mode="area")
    ref = blur(sc, 41) / (blur(sm, 41) + 1e-4)
    ref = F.interpolate(ref, (H, W), mode="bilinear", align_corners=False)
    d = ((rgb - ref) ** 2).sum(1, keepdim=True).sqrt()
    a_scr = ((d - 0.10) / 0.08).clamp(0, 1)
    nr = rgb / (rgb.sum(1, keepdim=True) + 1e-3)
    nf = ref / (ref.sum(1, keepdim=True) + 1e-3)
    dh = ((nr - nf) ** 2).sum(1, keepdim=True).sqrt()
    gm = (green > 0.06).float()
    a_hue = torch.maximum(((dh - 0.04) / 0.05).clamp(0, 1), 1 - gm)
    a_in = deep * a_scr + (1 - deep) * torch.minimum(a_scr, a_hue)
    cs = blur(1 - a_in, 11)
    a_in = 1 - ((cs - 0.15) / 0.30).clamp(0, 1)
    a = core * torch.minimum(a_ai, torch.maximum(a_in, 1 - gm)) + (1 - core) * torch.minimum(a_ai, a_key)
    zone = blur(dilate((a_ai > 0.05).float(), 31), 31)
    g8 = F.interpolate(green, (H // 8, W // 8), mode="area")
    G0 = torch.full((rgb.shape[0], 1, 1, 1), 0.3, device=rgb.device)
    for i in range(rgb.shape[0]):
        v = g8[i][g8[i] > 0.04]
        if v.numel() > 0.01 * g8[i].numel():
            G0[i] = torch.quantile(v.float(), 0.6)
    scm = (green > 0.5 * G0).float()
    pg = blur(F.interpolate(green * scm, (H // 8, W // 8), mode="area"), 41) / (blur(F.interpolate(scm, (H // 8, W // 8), mode="area"), 41) + 1e-4)
    pg = torch.maximum(F.interpolate(pg, (H, W), mode="bilinear", align_corners=False), 0.5 * G0)
    a_obj = blur(1 - ((green / pg - 0.25) / 0.45).clamp(0, 1), 3)
    nong = 1 - ((green - 0.02) / 0.04).clamp(0, 1)
    sm2 = F.interpolate(scm, (H // 8, W // 8), mode="area")
    sr = F.interpolate(rgb * scm, (H // 8, W // 8), mode="area")
    ref2 = F.interpolate(blur(sr, 41) / (blur(sm2, 41) + 1e-4), (H, W), mode="bilinear", align_corners=False)
    nf2 = ref2 / (ref2.sum(1, keepdim=True) + 1e-3)
    dh2 = ((nr - nf2) ** 2).sum(1, keepdim=True).sqrt()
    a_far = a_obj * torch.maximum(((dh2 - 0.04) / 0.06).clamp(0, 1), 1 - ((green - 0.03) / 0.03).clamp(0, 1))
    a = zone * torch.maximum(a, a_obj * nong) + (1 - zone) * a_far
    a = drop_markers(a, zone)
    fg = torch.cat([r, torch.minimum(g, (r + b) / 2), b], 1)
    sp = torch.maximum(torch.maximum(1 - deep, (1 - ((dh - 0.04) / 0.05).clamp(0, 1)) * gm), 1 - zone)
    fg = fg * sp + rgb * (1 - sp)
    return a, fg


def drop_markers(a: torch.Tensor, zone: torch.Tensor) -> torch.Tensor:
    """Tracking markers: small kept blobs away from people that the screen
    encloses on every side (white crosses / tape on a pro green screen).
    v15 keeps clearly non-green pixels away from people so props survive;
    a prop is either big or touches the frame edge / something else, a
    marker is small and floats in screen. Runs on a 1/4-size mask on the CPU."""
    B, _, H, W = a.shape
    h, w = max(8, H // 4), max(8, W // 4)
    small = F.interpolate(a, (h, w), mode="area")
    far = F.interpolate(zone, (h, w), mode="area") < 0.02
    kept = ((small > 0.5) & far).squeeze(1).cpu().numpy().astype(np.uint8)
    screen = (small < 0.1).squeeze(1).cpu().numpy()
    max_area = 0.004 * h * w
    kill = np.zeros((B, h, w), np.uint8)
    for i in range(B):
        n, lab, st, _ = cv2.connectedComponentsWithStats(kept[i], connectivity=8)
        for j in range(1, n):
            x, y, bw, bh, area = st[j]
            if area > max_area or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
                continue
            x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(w, x + bw + 3), min(h, y + bh + 3)
            ring = np.ones((y1 - y0, x1 - x0), bool)
            ring[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = False
            if screen[i, y0:y1, x0:x1][ring].mean() > 0.9:
                kill[i][lab == j] = 1
    if not kill.any():
        return a
    k = torch.from_numpy(kill).to(a.device).float()[:, None]
    k = F.interpolate(dilate(k, 3), (H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    return a * (1 - k)


def screen_rotation(cap) -> float:
    """Screen hue = most common saturated hue over 5 frames; teal/blue (>170°) rotate to green."""
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hs = []
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(nf - 1, 0) * p))
        ok, fr = cap.read()
        if not ok:
            continue
        hv = cv2.cvtColor(fr.astype(np.float32) / 255, cv2.COLOR_BGR2HSV)
        m = (hv[..., 1] > 0.35) & (hv[..., 2] > 0.25) & (hv[..., 0] > 60) & (hv[..., 0] < 280)
        hs.append(hv[..., 0][m])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    hs = np.concatenate(hs) if hs else np.array([120.0])
    sh = float(np.argmax(np.histogram(hs, bins=22, range=(60, 280))[0]) * 10 + 65) if hs.size > 500 else 120.0
    return 0.0 if sh < 170 else 120 - sh


# ── I/O ───────────────────────────────────────────────────────────────────

def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1) if float(num) > 0 else 30.0
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": fps, "audio": has_audio,
            "duration": float(info["format"].get("duration") or 0)}


def even(n: int) -> int:
    return n - (n % 2)


def background_tensor(bg: dict, W: int, H: int) -> torch.Tensor | None:
    """[1,3,H,W] RGB 0-1 for colour/image backgrounds; None for transparent."""
    if bg.get("type") == "color":
        c = bg["color"].lstrip("#")
        rgb = [int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        return torch.tensor(rgb, device=DEV).view(1, 3, 1, 1).expand(1, 3, H, W).contiguous()
    if bg.get("type") == "image" and bg.get("path"):
        img = cv2.imread(bg["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("background image unreadable")
        ih, iw = img.shape[:2]
        s = max(W / iw, H / ih)  # cover
        img = cv2.resize(img, (max(W, round(iw * s)), max(H, round(ih * s))), interpolation=cv2.INTER_AREA)
        y0 = (img.shape[0] - H) // 2
        x0 = (img.shape[1] - W) // 2
        img = img[y0:y0 + H, x0:x0 + W, ::-1].copy()
        return torch.from_numpy(img).to(DEV).permute(2, 0, 1)[None].float() / 255
    return None


def checker(W: int, H: int, cell: int = 16) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    on = (((yy // cell) + (xx // cell)) % 2).float()[None, None]
    light = torch.tensor([0.97, 0.96, 0.93], device=DEV).view(1, 3, 1, 1)
    dark = torch.tensor([0.88, 0.86, 0.83], device=DEV).view(1, 3, 1, 1)
    return light * (1 - on) + dark * on


def encoder(cmd_video: list[str], src: str, has_audio: bool, seconds: float, audio_codec: list[str], out: str, W: int, H: int, fps: float, pix: str):
    """ffmpeg reading raw frames on stdin (+ the source's audio), writing `out`."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "pipe:0"]
    if has_audio and audio_codec:
        cmd += ["-t", f"{seconds:.3f}", "-i", src, "-map", "0:v", "-map", "1:a:0?", *audio_codec, "-shortest"]
    cmd += [*cmd_video, out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


H264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
AAC = ["-c:a", "aac", "-b:a", "160k"]


def open_outputs(want: dict, src: str, meta: dict, seconds: float, W: int, H: int, pw: int, ph: int):
    fps, au = meta["fps"], meta["audio"]
    enc = {}
    if "webm" in want:
        enc["webm"] = encoder(["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30",
                               "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-auto-alt-ref", "0"],
                              src, au, seconds, ["-c:a", "libopus", "-b:a", "128k"], want["webm"], W, H, fps, "rgba")
    if "mov" in want:
        enc["mov"] = encoder(["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
                              "-alpha_bits", "8", "-qscale:v", "9"],
                             src, au, seconds, ["-c:a", "pcm_s16le"], want["mov"], W, H, fps, "rgba")
    if "matte" in want:
        enc["matte"] = encoder(H264, src, False, seconds, [], want["matte"], W, H, fps, "gray")
    if "mp4" in want:
        enc["mp4"] = encoder(H264, src, au, seconds, AAC, want["mp4"], W, H, fps, "rgb24")
    if "preview" in want:
        enc["preview"] = encoder(["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                                 src, au, seconds, ["-c:a", "aac", "-b:a", "96k"], want["preview"], pw, ph, fps, "rgb24")
    return enc


def check_frame(w: int, h: int) -> None:
    """Frames up to 4K (4096 wide, 2160 on the short side)."""
    if max(w, h) > 4096 or min(w, h) > 2160:
        raise ValueError(f"frame {w}x{h} is over 4K")


def frame_size(src: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(src)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def run(src: str, want: dict, max_seconds: float = 1e9, background: dict | None = None,
        progress: Callable[[float], None] = lambda f: None) -> dict:
    """Process `src` into the files named in `want` ({kind: path}).

    background: {"type": "transparent"} (default), {"type": "color", "color": "#rrggbb"}
    or {"type": "image", "path": "bg.jpg"} — used by the mp4 and preview outputs.
    """
    background = background or {"type": "transparent"}
    meta = probe(src)
    cap = cv2.VideoCapture(src)
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    check_frame(W0, H0)
    W, H = even(W0), even(H0)
    fps = meta["fps"]
    total = max(1, min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9, int(round(max_seconds * fps))))
    seconds = total / fps
    s = min(1.0, 720 / min(W, H))
    pw, ph = even(round(W * s)), even(round(H * s))

    ds = 0.4 if max(W, H) <= 1400 else (0.25 if max(W, H) <= 2000 else 0.125)
    ROT = screen_rotation(cap)
    bg = background_tensor(background, W, H)
    bg_prev = F.interpolate(bg, (ph, pw), mode="area") if bg is not None else checker(pw, ph)

    enc = open_outputs(want, src, meta, seconds, W, H, pw, ph)
    BS = 8 if W * H <= 1920 * 1080 else 2
    qin: queue.Queue = queue.Queue(4)
    qout: queue.Queue = queue.Queue(4)
    err: list[BaseException] = []

    def reader():
        try:
            buf, n = [], 0
            while n < total:
                ok, f = cap.read()
                if not ok:
                    break
                if f.shape[1] != W or f.shape[0] != H:
                    f = f[:H, :W]
                buf.append(f)
                n += 1
                if len(buf) == BS:
                    qin.put(np.stack(buf))
                    buf = []
            if buf:
                qin.put(np.stack(buf))
        except BaseException as e:  # surfaced by the main loop
            err.append(e)
        finally:
            qin.put(None)

    def writer():
        try:
            while True:
                o = qout.get()
                if o is None:
                    break
                for k, arr in o.items():
                    enc[k].stdin.write(arr.tobytes())
        except BaseException as e:
            err.append(e)

    tr = threading.Thread(target=reader, daemon=True)
    tw = threading.Thread(target=writer, daemon=True)
    tr.start()
    tw.start()
    rec = [None] * 4
    done = 0
    last = 0.0
    try:
        while True:
            b = qin.get()
            if b is None:
                break
            if err:
                raise err[0]
            x = torch.from_numpy(b).to(DEV, non_blocking=True).permute(0, 3, 1, 2).flip(1).float().div_(255)
            with torch.no_grad(), torch.autocast("cuda", enabled=DEV == "cuda"):
                _, pha, *rec = model()(x[None], *rec, downsample_ratio=ds)
            a, fg = post(rot(x, ROT), pha[0].float())
            fg = rot(fg, -ROT)
            a = a.clamp(0, 1)
            fg = fg.clamp(0, 1)
            out = {}
            to8 = lambda t: t.mul(255).round().byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
            if "webm" in enc or "mov" in enc:
                rgba = to8(torch.cat([fg, a], 1))  # straight (not premultiplied) alpha
                for kk in ("webm", "mov"):
                    if kk in enc:
                        out[kk] = rgba
            if "matte" in enc:
                out["matte"] = to8(a)
            if "mp4" in enc and bg is not None:
                out["mp4"] = to8(fg * a + bg * (1 - a))
            if "preview" in enc:
                ap = F.interpolate(a, (ph, pw), mode="area") if (pw, ph) != (W, H) else a
                fp = F.interpolate(fg, (ph, pw), mode="area") if (pw, ph) != (W, H) else fg
                out["preview"] = to8(fp * ap + bg_prev * (1 - ap))
            qout.put(out)
            done += len(b)
            if time.time() - last > 1:
                last = time.time()
                progress(min(0.99, done / total))
    finally:
        qout.put(None)
        tw.join()
        for p in enc.values():
            try:
                p.stdin.close()
            except BrokenPipeError:
                pass
        for k, p in enc.items():
            p.wait()
            if p.returncode != 0 and not err:
                err.append(RuntimeError(f"ffmpeg {k} failed: {p.stderr.read().decode()[-400:]}"))
        cap.release()
    if err:
        raise err[0]
    progress(1.0)
    return {"frames": done, "seconds": round(done / fps, 3), "width": W, "height": H, "fps": fps}
