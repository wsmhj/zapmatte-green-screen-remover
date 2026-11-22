---
title: Green Screen Remover
emoji: 🟩
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: gpl-3.0
short_description: Remove a green screen from a video, get transparent WebM
---

# Green Screen Remover

Upload a short clip shot in front of a green, blue or teal screen and download
the subject on a transparent background (WebM with alpha) plus an alpha matte.

This free Space runs on CPU, so it takes the first 8 seconds at up to 720p.
For full-length clips up to 4K, ProRes 4444 for Premiere / After Effects /
Final Cut / Resolve, and new backgrounds, use
**[Zapmatte](https://zapmatte.com)**, the online green screen remover.

Code (GPL-3.0): https://github.com/wsmhj/zapmatte-green-screen-remover
Built on [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting).
