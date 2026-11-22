# Zapmatte Green Screen Remover

Open-source green screen remover for video. Give it a clip shot in front of a
green, blue or teal screen and get the subject back on a transparent
background — as ProRes 4444, a transparent WebM, a black/white alpha matte, or
an MP4 on a new colour or picture. Resolution (up to 4K) and audio are kept.

This is the Standard pipeline behind **[Zapmatte](https://zapmatte.com)**, the
online [green screen remover](https://zapmatte.com) — use the website if you
just want to upload a clip and download the result without installing anything.

## How it works

1. **Human matting** — [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)
   (RVM, MobileNetV3) gives a soft alpha for the people in each frame, with
   temporal memory so edges don't flicker.
2. **Chroma key** — the screen colour is matched locally around the subject,
   so uneven lighting and shadows on the screen don't leave halos.
3. **Combine** — RVM alpha ∩ key alpha near people (hair stays soft); a
   colour-only key away from people, so props and objects survive.
4. **Blue and teal screens** — the screen hue is detected from 5 frames and
   rotated to green before keying.
5. **Despill** — green spill is pulled off hair, skin and edges.
6. **Tracking markers** — small blobs enclosed by screen, away from people,
   are removed.

On a T4 GPU it runs about 41 fps at 720p, 19 fps at 1080p and 5.5 fps at 4K.
It also runs on CPU, slowly.

## Install

Python 3.10+, ffmpeg (with libvpx and prores_ks) on PATH.

```bash
git clone https://github.com/wsmhj/zapmatte-green-screen-remover
cd zapmatte-green-screen-remover
pip install -r requirements.txt
```

The RVM weights download from GitHub on first run (via `torch.hub`).

## Use

```bash
python remove_green_screen.py clip.mp4                    # -> clip.webm (transparent)
python remove_green_screen.py clip.mp4 --mov --matte      # + ProRes 4444 MOV and alpha matte
python remove_green_screen.py clip.mp4 --color "#ffffff"  # -> clip-bg.mp4 on white
python remove_green_screen.py clip.mp4 --image beach.jpg  # -> clip-bg.mp4 on a picture
```

From Python:

```python
from zapmatte import pipeline

pipeline.run("clip.mp4", {"webm": "out.webm", "matte": "matte.mp4"})
```

| Output | File | Use it in |
|---|---|---|
| `mov` | ProRes 4444 + alpha | Premiere Pro, After Effects, Final Cut Pro, DaVinci Resolve |
| `webm` | VP9 + alpha, Opus audio | OBS, web pages |
| `matte` | black/white H.264 MP4 | any editor as a track matte |
| `mp4` | subject over a colour/picture | CapCut, phones, social |

## Limits

- It is made for green/blue screen footage. Ordinary videos without a screen
  are not what it is for.
- Green clothing is kept but can look patchy; green props far from people are
  keyed out with the screen.
- Frames up to 4K (4096 wide, 2160 on the short side).

## Other ways to run it

- **Website:** [zapmatte.com](https://zapmatte.com) — no install, first 10 seconds free.
- **Replicate:** `predict.py` + `cog.yaml` (Cog). `deploy/push-replicate.yml` is a GitHub Actions workflow that pushes it (copy it into `.github/workflows/`).
- **Hugging Face Space:** `hf-space/` is a small Gradio demo (CPU, short clips).

## License

GPL-3.0, because it builds on Robust Video Matting (GPL-3.0). See `LICENSE`.
RVM © Shanchuan Lin et al., ByteDance — please cite their paper if you use
this in research.
