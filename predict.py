"""Replicate (Cog) predictor for the Zapmatte green screen remover."""
import os
import tempfile

from cog import BasePredictor, Input, Path

from zapmatte import pipeline

FORMATS = {
    "transparent_webm": ("webm", ".webm"),
    "prores_4444_mov": ("mov", ".mov"),
    "alpha_matte_mp4": ("matte", "-matte.mp4"),
    "mp4_on_color": ("mp4", "-bg.mp4"),
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        pipeline.model()  # downloads and loads RVM once

    def predict(
        self,
        video: Path = Input(description="Video shot in front of a green, blue or teal screen (up to 4K)"),
        output_format: str = Input(
            description="transparent_webm for web/OBS, prores_4444_mov for Premiere/After Effects/Final Cut/Resolve, "
                        "alpha_matte_mp4 for a black/white matte, mp4_on_color for a new solid background",
            choices=list(FORMATS), default="transparent_webm"),
        background_color: str = Input(description="Background colour for mp4_on_color", default="#ffffff"),
        max_seconds: float = Input(description="Only process the first N seconds", default=180, ge=1, le=600),
    ) -> Path:
        kind, suffix = FORMATS[output_format]
        out = os.path.join(tempfile.mkdtemp(), "output" + suffix)
        bg = {"type": "color", "color": background_color} if kind == "mp4" else {"type": "transparent"}
        pipeline.run(str(video), {kind: out}, max_seconds, bg)
        return Path(out)
