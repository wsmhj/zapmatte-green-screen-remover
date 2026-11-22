"""Replicate (Cog) predictor for the Zapmatte green screen remover.

This public demo takes the first 10 seconds at up to 720p. Full-length clips
up to 4K: run the code yourself (see README) or use https://zapmatte.com.
"""
import os
import subprocess
import tempfile

from cog import BasePredictor, Input, Path

from zapmatte import pipeline

MAX_SECONDS = 10
FORMATS = {
    "transparent_webm": ("webm", ".webm"),
    "prores_4444_mov": ("mov", ".mov"),
    "alpha_matte_mp4": ("matte", "-matte.mp4"),
    "mp4_on_color": ("mp4", "-bg.mp4"),
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        pipeline.model()  # downloads and loads RVM once

    def run(
        self,
        video: Path = Input(description="Video shot in front of a green, blue or teal screen. "
                                        "The first 10 seconds are processed at up to 720p."),
        output_format: str = Input(
            description="transparent_webm for web/OBS, prores_4444_mov for Premiere/After Effects/Final Cut/Resolve, "
                        "alpha_matte_mp4 for a black/white matte, mp4_on_color for a new solid background",
            choices=["transparent_webm", "prores_4444_mov", "alpha_matte_mp4", "mp4_on_color"],
            default="transparent_webm"),
        background_color: str = Input(description="Background colour for mp4_on_color", default="#ffffff"),
    ) -> Path:
        d = tempfile.mkdtemp()
        small = os.path.join(d, "in.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-t", str(MAX_SECONDS), "-i", str(video),
                        "-vf", "scale='if(gt(iw,ih),-2,min(720,iw))':'if(gt(iw,ih),min(720,ih),-2)'",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", small], check=True)
        kind, suffix = FORMATS[output_format]
        out = os.path.join(d, "output" + suffix)
        bg = {"type": "color", "color": background_color} if kind == "mp4" else {"type": "transparent"}
        pipeline.run(small, {kind: out}, MAX_SECONDS, bg)
        return Path(out)
