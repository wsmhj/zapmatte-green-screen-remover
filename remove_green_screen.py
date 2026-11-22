#!/usr/bin/env python3
"""Remove a green (or blue/teal) screen from a video.

  python remove_green_screen.py clip.mp4                    # -> clip.webm (transparent)
  python remove_green_screen.py clip.mp4 --mov --matte      # + ProRes 4444 and alpha matte
  python remove_green_screen.py clip.mp4 --color "#ffffff"  # -> clip-bg.mp4 on white
  python remove_green_screen.py clip.mp4 --image beach.jpg  # -> clip-bg.mp4 on a picture
"""
import argparse
import os
import sys

from zapmatte import pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove a green screen from a video.")
    ap.add_argument("input", help="video shot in front of a green, blue or teal screen")
    ap.add_argument("--webm", action="store_true", help="transparent VP9 WebM (default when nothing else is asked)")
    ap.add_argument("--mov", action="store_true", help="ProRes 4444 MOV with alpha (Premiere, After Effects, Final Cut, Resolve)")
    ap.add_argument("--matte", action="store_true", help="black/white alpha matte MP4")
    ap.add_argument("--color", help="MP4 over this colour, e.g. '#ffffff'")
    ap.add_argument("--image", help="MP4 over this picture")
    ap.add_argument("--max-seconds", type=float, default=1e9, help="only process the first N seconds")
    ap.add_argument("--out-dir", default=None, help="where to write (default: next to the input)")
    a = ap.parse_args()

    base = os.path.splitext(os.path.basename(a.input))[0]
    out = a.out_dir or os.path.dirname(os.path.abspath(a.input))
    os.makedirs(out, exist_ok=True)
    path = lambda suffix: os.path.join(out, base + suffix)

    want = {}
    if a.webm or not (a.mov or a.matte or a.color or a.image):
        want["webm"] = path(".webm")
    if a.mov:
        want["mov"] = path(".mov")
    if a.matte:
        want["matte"] = path("-matte.mp4")
    bg = {"type": "transparent"}
    if a.image:
        bg = {"type": "image", "path": a.image}
    elif a.color:
        bg = {"type": "color", "color": a.color}
    if bg["type"] != "transparent":
        want["mp4"] = path("-bg.mp4")

    print(f"device: {pipeline.DEV}", file=sys.stderr)
    meta = pipeline.run(a.input, want, a.max_seconds, bg,
                        progress=lambda f: print(f"\r{f * 100:5.1f}%", end="", file=sys.stderr))
    print(file=sys.stderr)
    print(f"{meta['frames']} frames, {meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps")
    for p in want.values():
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
