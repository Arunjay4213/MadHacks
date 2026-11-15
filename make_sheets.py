"""Sheets from compare_decoders.py output: TAEHV left, Wan right."""

import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

out = sys.argv[1]
sheets = os.path.join(out, "sheets")
os.makedirs(sheets, exist_ok=True)


def load(name, tag, c, f):
    return np.asarray(Image.open(os.path.join(out, f"{name}_{tag}", f"c{c:03d}_f{f}.png")))


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


for name in ("meadow", "city"):
    n = len(glob.glob(os.path.join(out, f"{name}_wan", "c*_f0.png")))
    points = [int(x) for x in np.linspace(4, n - 2, 6)]
    rows = []
    for c in points:
        t, w = load(name, "taehv", c, 1), load(name, "wan", c, 1)
        gap = np.full((t.shape[0], 8, 3), 255, np.uint8)
        rows.append(np.concatenate([label(t, f"TAEHV c{c}"), gap, label(w, f"WAN c{c}")], 1))
        rows.append(np.full((8, rows[-1].shape[1], 3), 255, np.uint8))
    sheet = np.concatenate(rows[:3 * 2], 0), np.concatenate(rows[3 * 2:], 0)
    for k, s in enumerate(sheet):
        Image.fromarray(s).save(os.path.join(sheets, f"{name}_side_by_side_{k}.jpg"), quality=92)

    # Fine-detail crops, 1:1 pixels then x2 nearest, from three regions of two time points.
    H, W = 384, 672
    regions = {"bottom_left": (H - 160, 0), "center": (H // 2 - 80, W // 2 - 120), "top_right": (0, W - 240)}
    crops = []
    for c in (points[1], points[4]):
        t, w = load(name, "taehv", c, 1), load(name, "wan", c, 1)
        for rname, (y, x) in regions.items():
            ct, cw = t[y:y + 160, x:x + 240], w[y:y + 160, x:x + 240]
            pair = np.concatenate([ct, np.full((160, 4, 3), 255, np.uint8), cw], 1).repeat(2, 0).repeat(2, 1)
            crops.append(label(np.ascontiguousarray(pair), f"c{c} {rname}: TAEHV | WAN (x2)"))
    for k in range(0, len(crops), 3):
        Image.fromarray(np.concatenate(crops[k:k + 3], 0)).save(os.path.join(sheets, f"{name}_crops_{k // 3}.jpg"), quality=95)

    # Chunk boundary: last frame of chunk c, first frames of chunk c+1, per decoder, plus x4 abs difference.
    c = points[3]
    strips = []
    for tag in ("taehv", "wan"):
        seq = [load(name, tag, c, 2), load(name, tag, c, 3), load(name, tag, c + 1, 0), load(name, tag, c + 1, 1)]
        diffs = [np.clip(np.abs(seq[i + 1].astype(int) - seq[i].astype(int)) * 4, 0, 255).astype(np.uint8) for i in range(3)]
        top = np.concatenate([label(s[::2, ::2].copy(), f"{tag} {lab}") for s, lab in
                              zip(seq, (f"c{c}f2", f"c{c}f3", f"c{c+1}f0", f"c{c+1}f1"))], 1)
        bot = np.concatenate([label(d[::2, ::2].copy(), f"diff x4 {lab}") for d, lab in
                              zip(diffs, ("inside", "BOUNDARY", "inside"))] + [np.zeros((192, 336, 3), np.uint8)], 1)
        strips += [top, bot]
    Image.fromarray(np.concatenate(strips, 0)).save(os.path.join(sheets, f"{name}_boundary.jpg"), quality=92)

    # Time slice: column x = W/2 of every frame side by side (time runs left to right, 2 px per frame).
    # Flicker or seams show as vertical stripes every 4 frames.
    sl = []
    for tag in ("taehv", "wan"):
        cols = [load(name, tag, ci, f)[:, W // 2:W // 2 + 1] for ci in range(1, n)
                for f in range(4) if os.path.exists(os.path.join(out, f"{name}_{tag}", f"c{ci:03d}_f{f}.png"))]
        img = np.concatenate(cols, 1).repeat(2, 1)
        sl.append(label(np.ascontiguousarray(img), f"{tag}: column x={W // 2} over time"))
        sl.append(np.full((8, img.shape[1], 3), 255, np.uint8))
    Image.fromarray(np.concatenate(sl, 0)).save(os.path.join(sheets, f"{name}_time_slice.png"))
print("sheets in", sheets)
