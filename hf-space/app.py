"""Hugging Face Space: green screen remover demo (CPU, short clips)."""
import os
import subprocess
import tempfile

import gradio as gr

from zapmatte import pipeline

MAX_SECONDS = 8


def remove(video, background, color):
    if not video:
        raise gr.Error("Upload a video first.")
    d = tempfile.mkdtemp()
    small = os.path.join(d, "in.mp4")
    # CPU Space: first 8 s, short side at most 720 px
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-t", str(MAX_SECONDS), "-i", video,
                    "-vf", "scale='if(gt(iw,ih),-2,min(720,iw))':'if(gt(iw,ih),min(720,ih),-2)'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", small], check=True)
    want = {"webm": os.path.join(d, "transparent.webm"), "matte": os.path.join(d, "alpha-matte.mp4"),
            "preview": os.path.join(d, "preview.mp4")}
    bg = {"type": "transparent"}
    if background == "Solid colour":
        bg = {"type": "color", "color": color}
        want["mp4"] = os.path.join(d, "new-background.mp4")
    pipeline.run(small, want, MAX_SECONDS, bg)
    files = [want[k] for k in ("webm", "matte", "mp4") if k in want]
    return want["preview"], files


with gr.Blocks(title="Green Screen Remover") as demo:
    gr.Markdown(
        "# Green Screen Remover\n"
        "Upload a clip shot in front of a green, blue or teal screen. You get a transparent WebM and an alpha matte. "
        f"This CPU demo takes the first {MAX_SECONDS} seconds at up to 720p — for full clips up to 4K and ProRes 4444, "
        "use [Zapmatte](https://zapmatte.com).")
    with gr.Row():
        with gr.Column():
            inp = gr.Video(label="Green screen video", sources=["upload"])
            bg = gr.Radio(["Transparent", "Solid colour"], value="Transparent", label="Background")
            color = gr.ColorPicker(value="#ffffff", label="Colour (for Solid colour)")
            go = gr.Button("Remove green screen", variant="primary")
        with gr.Column():
            prev = gr.Video(label="Preview (checkerboard = transparent)")
            out = gr.File(label="Downloads", file_count="multiple")
    go.click(remove, [inp, bg, color], [prev, out])

demo.queue(max_size=8).launch()
