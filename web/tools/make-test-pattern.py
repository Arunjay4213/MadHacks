"""Writes the fake server's moving test pattern to web/test-pattern/ as JPEG frames.

The fake server uses ffmpeg when it is installed; these frames are the fallback, so the page shows a
moving picture on any laptop. Regenerate with:  python web/tools/make-test-pattern.py  (needs Pillow).
"""
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 416, 240  # same 26:15 shape as the model's 832x480
FRAMES = 32      # 2 s loop at 16 fps
BARS = [(192, 192, 192), (192, 192, 0), (0, 192, 192), (0, 192, 0), (192, 0, 192), (192, 0, 0), (0, 0, 192)]

out = Path(__file__).resolve().parent.parent / "test-pattern"
out.mkdir(exist_ok=True)
for old in out.glob("*.jpg"):
    old.unlink()

for i in range(FRAMES):
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    bar_w = W / len(BARS)
    for b, colour in enumerate(BARS):
        d.rectangle([b * bar_w, 0, (b + 1) * bar_w, H * 0.7], fill=colour)
    d.rectangle([0, H * 0.7, W, H], fill=(24, 24, 24))
    x = (W - 40) * i / (FRAMES - 1)  # a box sweeping left to right, so motion is obvious
    d.rectangle([x, H * 0.72, x + 40, H * 0.98], fill=(255, 255, 255))
    d.text((8, 8), f"MOCK test pattern  frame {i:02d}", fill=(0, 0, 0))
    im.save(out / f"{i:02d}.jpg", quality=70)

print(f"wrote {FRAMES} frames to {out}")
