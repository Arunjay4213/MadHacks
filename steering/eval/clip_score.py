"""CLIP score: did the picture move toward the new prompt, and when.

    python -m steering.eval.clip_score CLIP --before "..." --after "..." [--steer-frame N]
        [--fps 16] [--every 1] [--model M] [--device cpu] [--out scores.txt]
    python -m steering.eval.clip_score --dir DIR [--ramp 4] [--fps 16] [--every 1]

Per frame: cosine similarity of the CLIP image embedding to each prompt, and
score = sim(after) - sim(before). The transition report times the score's
rise from the steer frame. `--dir` scores every applied message of each
known scenario in a run_engine folder and writes `<name>.clip.txt`.
Needs `transformers` and the model download only when run.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from . import scenarios as sc
from .metrics import transition_span
from .sheet import APPLIED, find_clips, load_clip, read_report

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------- pure part


def similarity_curve(image_feats: Tensor, text_feats: Tensor) -> Tensor:
    """Cosine similarity [T, P] of image features [T, D] to text features [P, D]."""
    return F.normalize(image_feats.double(), dim=-1) @ F.normalize(text_feats.double(), dim=-1).T


def relative_score(curve: Tensor, before_idx: int, after_idx: int) -> Tensor:
    """Per-frame sim(after) - sim(before), shape [T]."""
    return curve[:, after_idx] - curve[:, before_idx]


def _smooth(x: Tensor, k: int = 3) -> Tensor:
    """Moving average over `k` samples, edges repeated."""
    if len(x) < k:
        return x.clone()
    pad = F.pad(x.view(1, 1, -1), (k // 2, k // 2), mode="replicate")
    return F.avg_pool1d(pad, k, stride=1).flatten()


def transition_report(
    scores: Tensor,
    steer_frame: int,
    fps: float,
    frame_idx: Sequence[int] | None = None,
    end: int | None = None,
    pre: int | None = None,
) -> dict:
    """Time the rise of `scores` after `steer_frame`.

    `scores[k]` belongs to frame `frame_idx[k]` (default k). The window runs
    from `pre` frames before the steer (default: enough that the first 10%
    `transition_span` uses as "before" is all pre-steer) up to frame `end`
    (exclusive, default the end). Keys: window (first, last frame), rise
    (mean of the window's last 10% minus first 10%), t10_ms / t90_ms (time
    from the steer frame to where the normalised rise first reaches 10% / 90%,
    None if `transition_span` finds no rise), decreasing_frac (fraction of
    steps after the steer where the 3-frame smoothed score goes down by more
    than 2% of |rise|, so plateau jitter does not count) and monotone
    (decreasing_frac <= 0.2).
    """
    s = scores.to(torch.float64).flatten()
    fidx = list(range(len(s))) if frame_idx is None else [int(i) for i in frame_idx]
    if len(fidx) != len(s):
        raise ValueError(f"frame_idx has {len(fidx)} entries for {len(s)} scores")
    last = fidx[-1] + 1 if end is None else end
    if pre is None:
        pre = max(2, math.ceil((last - steer_frame) / 9))
    keep = [k for k, f in enumerate(fidx) if steer_frame - pre <= f < last]
    if len(keep) < 3 or not any(fidx[k] >= steer_frame for k in keep):
        raise ValueError("too few scored frames around the steer frame")
    w = s[keep]
    wf = [fidx[k] for k in keep]

    m = max(1, round(0.1 * len(w)))
    rise = float(w[-m:].mean() - w[:m].mean())
    span = transition_span(w)
    t10 = t90 = None
    if span is not None:
        t10 = (wf[span[0]] - steer_frame) * 1000.0 / fps
        t90 = (wf[span[1]] - steer_frame) * 1000.0 / fps

    sm = _smooth(w)
    post = [k for k, f in enumerate(wf) if f >= steer_frame]
    steps = [sm[k] - sm[k - 1] for k in post if k > 0]
    tol = 0.02 * abs(rise)
    dec = sum(1 for d in steps if d < -tol) / len(steps) if steps else 0.0
    return {
        "window": (wf[0], wf[-1]),
        "rise": rise,
        "t10_ms": t10,
        "t90_ms": t90,
        "decreasing_frac": dec,
        "monotone": dec <= 0.2,
    }


def format_table(frame_idx: Sequence[int], curve: Tensor, score: Tensor, steer: int | None) -> str:
    """Plain per-frame table: frame, sim(before), sim(after), score."""
    lines = [f"{'frame':>5} {'sim_before':>10} {'sim_after':>10} {'score':>8}"]
    for k, f in enumerate(frame_idx):
        tag = "  <- steer" if steer is not None and f == steer else ""
        lines.append(
            f"{f:>5} {float(curve[k, 0]):>10.4f} {float(curve[k, 1]):>10.4f} "
            f"{float(score[k]):>+8.4f}{tag}"
        )
    return "\n".join(lines)


def format_report(r: dict) -> str:
    """One-paragraph plain text of a transition_report dict."""
    def ms(x: float | None) -> str:
        return "-" if x is None else f"{x:.0f} ms"

    return (
        f"window frames {r['window'][0]}..{r['window'][1]}: rise {r['rise']:+.4f}, "
        f"10% at {ms(r['t10_ms'])}, 90% at {ms(r['t90_ms'])}, "
        f"decreasing {r['decreasing_frac']:.0%} of post-steer steps "
        f"({'monotone-ish' if r['monotone'] else 'NOT monotone'})"
    )


# ---------------------------------------------------------------- model part


class ClipScorer:
    """A transformers CLIP model; imports and downloads only when built."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model)

    @staticmethod
    def _feats(out: Any) -> Tensor:
        # transformers 4.x returns a tensor; newer versions may return a model output.
        if isinstance(out, Tensor):
            return out
        for name in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, name, None)
            if isinstance(v, Tensor):
                return v
        raise TypeError(f"unexpected CLIP feature output {type(out).__name__}")

    @torch.no_grad()
    def embed_images(self, frames: Tensor, batch: int = 32) -> Tensor:
        """[N, D] image features of float [N, 3, H, W] frames in 0..1."""
        from PIL import Image

        feats = []
        for i in range(0, frames.shape[0], batch):
            u8 = (frames[i : i + batch].clamp(0, 1) * 255).round().to(torch.uint8)
            imgs = [Image.fromarray(f.permute(1, 2, 0).numpy(), "RGB") for f in u8]
            inp = self.processor(images=imgs, return_tensors="pt").to(self.device)
            feats.append(self._feats(self.model.get_image_features(**inp)).cpu())
        return torch.cat(feats)

    @torch.no_grad()
    def embed_texts(self, prompts: Sequence[str]) -> Tensor:
        """[P, D] text features."""
        inp = self.processor(text=list(prompts), return_tensors="pt", padding=True,
                             truncation=True).to(self.device)
        return self._feats(self.model.get_text_features(**inp)).cpu()


def score_clip(
    scorer: ClipScorer,
    frames: Tensor,
    before: str,
    after: str,
    every: int = 1,
    image_feats: Tensor | None = None,
) -> tuple[list[int], Tensor, Tensor]:
    """(frame indices, [N, 2] curve vs (before, after), [N] score) for every `every`-th frame."""
    idx = list(range(0, frames.shape[0], every))
    if image_feats is None:
        image_feats = scorer.embed_images(frames[idx])
    curve = similarity_curve(image_feats, scorer.embed_texts([before, after]))
    return idx, curve, relative_score(curve, 0, 1)


def score_text(
    idx: list[int], curve: Tensor, score: Tensor, steer: int | None, fps: float,
    end: int | None = None,
) -> str:
    """Table plus transition report (or why there is none) as plain text."""
    text = format_table(idx, curve, score, steer)
    if steer is not None:
        try:
            text += "\n" + format_report(transition_report(score, steer, fps, idx, end=end))
        except ValueError as e:
            text += f"\nno transition report: {e}"
    return text


# ---------------------------------------------------------------- --dir


def scenario_pairs(
    scenario: sc.Scenario, report_rows: list[tuple[int, int, str]], frames_per_step: float
) -> list[tuple[int, str, str, int | None]]:
    """(steer frame, before, after, next steer frame) per applied event.

    Applied/frames come from the report rows when they match the scenario's
    events, else every event is assumed applied at step * frames_per_step.
    """
    events = list(scenario.events)
    if len(report_rows) == len(events) and all(
        r[0] == e.step for r, e in zip(report_rows, events)
    ):
        placed = [(r[1], e, r[2] in APPLIED) for r, e in zip(report_rows, events)]
    else:
        placed = [(round(e.step * frames_per_step), e, True) for e in events]
    applied = [(f, e) for f, e, ok in placed if ok]
    out = []
    prompt = scenario.initial_prompt
    for k, (frame, e) in enumerate(applied):
        nxt = next((f for f, _ in applied[k + 1 :] if f > frame), None)
        out.append((frame, prompt, e.msg["prompt"], nxt))
        prompt = e.msg["prompt"]
    return out


def run_dir(folder: str, scorer: ClipScorer, fps: float, every: int, ramp: int) -> int:
    """Write `<name>.clip.txt` for each known scenario clip in `folder`; return failures."""
    known = {s.name: s for s in sc.default_scenarios(ramp)}
    failures = 0
    for name, path in find_clips(folder).items():
        if name not in known:
            print(f"{name}: skipped (not in default_scenarios(ramp={ramp}))")
            continue
        try:
            scenario = known[name]
            frames = load_clip(path)
            rows = read_report(os.path.join(folder, name + ".report.txt"))
            pairs = scenario_pairs(scenario, rows, frames.shape[0] / scenario.steps)
            idx = list(range(0, frames.shape[0], every))
            feats = scorer.embed_images(frames[idx])
            parts = [f"== {name}: {path}, {frames.shape[0]} frames @ {fps:g} fps"]
            for steer, before, after, end in pairs:
                parts.append(f"\n-- steer at frame {steer}\nbefore: {before}\nafter:  {after}")
                if before == after:
                    parts.append("same prompt before and after (strength or confirm): not scored")
                    continue
                i, curve, score = score_clip(scorer, frames, before, after, every, feats)
                parts.append(score_text(i, curve, score, steer, fps, end))
            text = "\n".join(parts) + "\n"
            with open(os.path.join(folder, name + ".clip.txt"), "w") as f:
                f.write(text)
            print(f"{name}: wrote {name}.clip.txt ({len(pairs)} applied messages)")
        except Exception as e:  # fail soft per scenario
            failures += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
    return failures


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("clip", nargs="?", help=".frames.pt or .mp4")
    p.add_argument("--dir", help="run_engine output folder")
    p.add_argument("--before", help="prompt in force before the steer")
    p.add_argument("--after", help="prompt steered to")
    p.add_argument("--steer-frame", type=int, default=None)
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--every", type=int, default=1, help="score every N-th frame")
    p.add_argument("--ramp", type=int, default=4, help="with --dir: the run's --ramp")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="also write the output here")
    args = p.parse_args(argv)
    if bool(args.clip) == bool(args.dir):
        p.error("give exactly one of CLIP or --dir")
    if args.clip and not (args.before and args.after):
        p.error("CLIP needs --before and --after")
    if args.every < 1:
        p.error("--every must be >= 1")

    scorer = ClipScorer(args.model, args.device)
    if args.dir:
        return 1 if run_dir(args.dir, scorer, args.fps, args.every, args.ramp) else 0

    frames = load_clip(args.clip)
    idx, curve, score = score_clip(scorer, frames, args.before, args.after, args.every)
    text = score_text(idx, curve, score, args.steer_frame, args.fps)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
