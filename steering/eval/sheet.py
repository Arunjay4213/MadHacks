"""Contact sheets: judge a clip from one image.

    python -m steering.eval.sheet CLIP [--every N] [--cols 8] [--mark 16,32] [--cuts] [--out sheet.png]
    python -m steering.eval.sheet --dir DIR [--every N] [--cuts]

CLIP is a `.frames.pt` (torch.save of float [T, 3, H, W] in 0..1) or `.mp4`.
`--dir` makes `<name>.sheet.png` for every clip in a run_engine output folder,
marking the frames where a message was applied (read from `<name>.report.txt`).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from .metrics import find_cuts

STEER_COLOUR = (255, 40, 40)
CUT_COLOUR = (40, 160, 255)
BACKGROUND = (24, 24, 24)
PAD = 2
APPLIED = {"provisional", "confirm", "final"}

# A report row: "<step> <frame>  <label padded to 44> <result> ...", see scenarios.Result.report.
_ROW = re.compile(r"^\s*(\d+)\s+(\d+)  .{44} (\S+)")


# ---------------------------------------------------------------- loading


def load_clip(path: str) -> Tensor:
    """Frames of a `.pt` or `.mp4` clip as float [T, 3, H, W] in 0..1 on CPU."""
    if path.endswith(".pt"):
        t = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(t, Tensor) or t.dim() != 4 or t.shape[1] != 3:
            raise ValueError(f"{path}: expected a [T, 3, H, W] tensor")
        return t.to(torch.float32)
    if path.endswith(".mp4"):
        try:
            import imageio.v2 as iio
        except ImportError as e:
            raise RuntimeError(
                f"reading {path} needs imageio and imageio-ffmpeg (pip install imageio "
                "imageio-ffmpeg), or use the .frames.pt from --save-frames"
            ) from e
        import numpy as np

        arr = np.stack(iio.mimread(path, memtest=False))[..., :3]  # [T, H, W, 3] uint8
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    raise ValueError(f"{path}: expected a .pt or .mp4 file")


def find_clips(folder: str) -> dict[str, str]:
    """{scenario name: clip path} in a run_engine folder; `.frames.pt` wins over `.mp4`."""
    clips: dict[str, str] = {}
    for fname in sorted(os.listdir(folder)):
        for ext in (".frames.pt", ".mp4"):
            if fname.endswith(ext):
                name = fname[: -len(ext)]
                if name not in clips or ext == ".frames.pt":
                    clips[name] = os.path.join(folder, fname)
    return clips


def read_report(path: str) -> list[tuple[int, int, str]]:
    """(step, first frame, result) per message row of a `<name>.report.txt`.

    `result` is the kind ("provisional", "confirm", "final") when the message
    was applied, else the reason ("stale", "unchanged"). Returns [] when the
    file is missing or has no rows in the expected format.
    """
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        m = _ROW.match(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return rows


def steer_frames(report_path: str) -> list[int]:
    """Frames where a message was applied, from a report; [] if none parse."""
    return sorted({frame for _, frame, res in read_report(report_path) if res in APPLIED})


# ---------------------------------------------------------------- drawing


def _tile(frame: Tensor, size: tuple[int, int]) -> Image.Image:
    u8 = (frame.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    img = Image.fromarray(u8.numpy(), "RGB")
    return img if img.size == size else img.resize(size, Image.BILINEAR)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    """White text on a black box: readable on any background."""
    font = ImageFont.load_default()
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=font)
    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def contact_sheet(
    frames: Tensor,
    every: int = 1,
    cols: int = 8,
    max_width: int = 1600,
    labels: Sequence[str] | None = None,
    mark: Iterable[int] = (),
    mark_cuts: Iterable[int] = (),
) -> Image.Image:
    """Grid of every `every`-th frame of float [T, 3, H, W] frames.

    Each tile is labelled with its frame index (or `labels[i]`, one per
    frame). Frames in `mark` (steers) get a red border, frames in
    `mark_cuts` a blue one inside the image edge; marked frames are always
    shown even when not on the `every` grid. Tiles are downscaled so the
    sheet is at most `max_width` pixels wide.
    """
    if frames.dim() != 4 or frames.shape[1] != 3 or frames.shape[0] < 1:
        raise ValueError(f"expected [T>=1, 3, H, W], got {tuple(frames.shape)}")
    if every < 1 or cols < 1:
        raise ValueError("every and cols must be >= 1")
    t, _, h, w = frames.shape
    if labels is not None and len(labels) != t:
        raise ValueError(f"labels has {len(labels)} entries for {t} frames")
    steers = {int(i) for i in mark if 0 <= int(i) < t}
    cuts = {int(i) for i in mark_cuts if 0 <= int(i) < t}
    idx = sorted(set(range(0, t, every)) | steers | cuts)

    cols = min(cols, len(idx))
    tw = max(8, min(w, (max_width - PAD * (cols + 1)) // cols))
    th = max(1, round(h * tw / w))
    rows = -(-len(idx) // cols)
    sheet = Image.new("RGB", (PAD + cols * (tw + PAD), PAD + rows * (th + PAD)), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    bw = max(2, tw // 24)
    for k, i in enumerate(idx):
        x = PAD + (k % cols) * (tw + PAD)
        y = PAD + (k // cols) * (th + PAD)
        sheet.paste(_tile(frames[i], (tw, th)), (x, y))
        _label(draw, (x + bw + 2, y + bw + 1), labels[i] if labels is not None else str(i))
        if i in steers:
            draw.rectangle((x, y, x + tw - 1, y + th - 1), outline=STEER_COLOUR, width=bw)
        if i in cuts:
            inset = bw if i in steers else 0
            draw.rectangle(
                (x + inset, y + inset, x + tw - 1 - inset, y + th - 1 - inset),
                outline=CUT_COLOUR, width=bw,
            )
    return sheet


# ---------------------------------------------------------------- CLI


def _sheet_for(path: str, out: str, every: int, cols: int, mark: list[int], cuts: bool) -> str:
    frames = load_clip(path)
    cut_idx = find_cuts(frames) if cuts and frames.shape[0] >= 2 else []
    contact_sheet(frames, every=every, cols=cols, mark=mark, mark_cuts=cut_idx).save(out)
    return f"{out}: {frames.shape[0]} frames, steer marks {mark}, cuts {cut_idx if cuts else '-'}"


def _ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("clip", nargs="?", help=".frames.pt or .mp4")
    p.add_argument("--dir", help="run_engine output folder: a sheet per clip")
    p.add_argument("--every", type=int, default=1, help="show every N-th frame (default 1)")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mark", type=_ints, default=[], help="comma list of steer frames")
    p.add_argument("--cuts", action="store_true", help="mark cuts found by metrics.find_cuts")
    p.add_argument("--out", help="output png (default <clip>.sheet.png)")
    args = p.parse_args(argv)
    if bool(args.clip) == bool(args.dir):
        p.error("give exactly one of CLIP or --dir")

    if args.clip:
        stem = re.sub(r"(\.frames\.pt|\.mp4|\.pt)$", "", args.clip)
        print(_sheet_for(args.clip, args.out or stem + ".sheet.png", args.every, args.cols,
                         args.mark, args.cuts))
        return 0

    clips = find_clips(args.dir)
    if not clips:
        print(f"no .frames.pt or .mp4 files in {args.dir}", file=sys.stderr)
        return 1
    failures = 0
    for name, path in clips.items():
        base = os.path.join(args.dir, name)
        try:
            mark = steer_frames(base + ".report.txt")
            print(_sheet_for(path, base + ".sheet.png", args.every, args.cols, mark, args.cuts))
        except Exception as e:  # fail soft per clip
            failures += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
