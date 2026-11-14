"""Scripted steering scenarios: replay contract-2 messages against a model.

The model is plugged in through two callables, so the same scenarios run on
a fake model here and on the real one on a GPU machine:

    generate(cond) -> frames [F, C, H, W] in [0, 1]   (one call per step)
    encode(prompt) -> conditioning tensor
                      (or (tensor, real token ids) when run with an aligner)

`run` also calls `on_cond_changed()` right before every step whose
conditioning differs from the previous step's; the real engine must reset
the model's text cross-attention cache at exactly those moments.

    python -m steering.eval.scenarios --fake [--align]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from ..align import Aligner
from ..controller import Controller, Encoded
from ..steerer import Steerer
from .metrics import find_cuts, summarize

Generate = Callable[[Tensor], Tensor]
Encode = Callable[[str], Encoded]

BASE = "a wide shot of a meadow with a single oak tree, gentle wind, cinematic"
WINTER = ", in deep winter with snow on the ground"
NIGHT = ", at night under a full moon"
RAIN = ", in heavy rain"


@dataclass(frozen=True)
class Event:
    """A contract-2 message applied just before generate call number `step`."""

    step: int
    msg: Mapping


@dataclass(frozen=True)
class Scenario:
    """A named script: start prompt, timed messages, and total generate calls."""

    name: str
    initial_prompt: str
    events: tuple[Event, ...]
    steps: int

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError(f"{self.name}: steps must be an int >= 1, got {self.steps!r}")
        for e in self.events:
            if not 0 <= e.step < self.steps:
                raise ValueError(
                    f"{self.name}: event step {e.step} outside [0, {self.steps})"
                )


@dataclass
class StepRecord:
    """What happened in one generate call."""

    step: int
    first_frame: int  # index in Result.frames of this step's first frame
    num_frames: int
    progress: float  # steerer.progress after get_cond
    cond_changed: bool
    generate_s: float


@dataclass
class Result:
    """Frames, per-step records, handle() results and metrics of one run."""

    scenario: Scenario
    fps: float
    frames: Tensor  # [T, C, H, W]
    steps: list[StepRecord]
    handled: list[tuple[Event, dict]]  # in the order applied
    metrics: dict = field(default_factory=dict)

    def save_frames(self, path: str) -> None:
        """Save the frames tensor with torch.save."""
        torch.save(self.frames, path)

    def report(self) -> str:
        """A short plain-text table for a terminal."""
        m = self.metrics
        changed = [r.step for r in self.steps if r.cond_changed]
        lines = [
            f"== {self.scenario.name}: {len(self.steps)} steps, "
            f"{self.frames.shape[0]} frames @ {self.fps:g} fps, generate "
            f"mean {m['generate_mean_s'] * 1e3:.1f} ms / max {m['generate_max_s'] * 1e3:.1f} ms",
            f"cuts in clip (frame idx): {m['cuts']}",
            f"cond changed at steps: {changed}",
            f"{'step':>4} {'frame':>5}  {'message':<44} {'result':<12} "
            f"{'cuts':<8} {'peak_ratio':>10} {'react_ms':>8}",
        ]
        for (event, res), s in zip(self.handled, m["events"]):
            outcome = res["kind"] if res["applied"] else res["reason"]
            if s is None:
                cuts, ratio, react = "-", "-", "-"
            else:
                cuts = str(s["cuts_in_window"])
                ratio = "-" if s["peak_ratio"] is None else f"{s['peak_ratio']:.1f}"
                react = "-" if s["reaction_ms"] is None else f"{s['reaction_ms']:.0f}"
            lines.append(
                f"{event.step:>4} {self.steps[event.step].first_frame:>5}  "
                f"{_label(event.msg, self.scenario.initial_prompt):<44} {outcome:<12} "
                f"{cuts:<8} {ratio:>10} {react:>8}"
            )
        return "\n".join(lines)


def _label(msg: Mapping, base: str, width: int = 44) -> str:
    """Short description of a message: prompt minus the shared base, ramp, cap."""
    p = msg["prompt"]
    p = "+" + p[len(base):].lstrip(", ") if p.startswith(base) and p != base else p
    tag = f" r{msg['ramp_chunks']}"
    if msg.get("provisional"):
        tag += f" prov cap{msg.get('cap', 1.0):g}"
    p = p if len(p) + len(tag) <= width else p[: width - len(tag) - 1] + "~"
    return p + tag


def run(
    scenario: Scenario,
    generate: Generate,
    encode: Encode,
    fps: float,
    on_cond_changed: Callable[[], None] | None = None,
    aligner: Aligner | None = None,
) -> Result:
    """Play `scenario` step by step; see the module docstring.

    With `aligner`, `encode` returns (embedding, ids); the aligner is reset,
    places the initial prompt, and is passed to the Controller.
    """
    if not fps > 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    if aligner is None:
        steerer = Steerer(encode(scenario.initial_prompt))
    else:
        aligner.reset()
        steerer = Steerer(aligner.place(*encode(scenario.initial_prompt)))
    controller = Controller(steerer, encode, aligner=aligner)
    by_step: dict[int, list[Event]] = {}
    for e in scenario.events:
        by_step.setdefault(e.step, []).append(e)

    prev = steerer.get_cond()  # the initial tensor; no move yet, so no side effect
    chunks: list[Tensor] = []
    records: list[StepRecord] = []
    handled: list[tuple[Event, dict]] = []
    n_frames = 0
    for step in range(scenario.steps):
        for e in by_step.get(step, []):
            handled.append((e, controller.handle(e.msg)))
        cond = steerer.get_cond()
        changed = cond is not prev
        if changed and on_cond_changed is not None:
            on_cond_changed()
        t0 = time.perf_counter()
        frames = generate(cond)
        dt = time.perf_counter() - t0
        if frames.dim() != 4 or frames.shape[0] < 1:
            raise ValueError(f"generate must return [F>=1, C, H, W], got {tuple(frames.shape)}")
        records.append(StepRecord(step, n_frames, frames.shape[0], steerer.progress, changed, dt))
        chunks.append(frames.detach().cpu())
        n_frames += frames.shape[0]
        prev = cond

    video = torch.cat(chunks)
    return Result(scenario, fps, video, records, handled, _metrics(video, records, handled, fps))


def _metrics(
    video: Tensor, records: list[StepRecord], handled: list[tuple[Event, dict]], fps: float
) -> dict:
    """Per-event metrics, whole-clip cuts, and generate timing.

    Each event is judged on its own window: from the start of the clip up to
    (not including) the first frame of the next event at a later step, so a
    later message's transition is not credited to it. An event entry is
    `summarize(...)` of that window plus `steer_frame`, `window_end` and
    `cuts_in_window` (cuts landing in [steer_frame, window_end)), or None
    when the window is too short for `summarize` (steer frame < 2, or fewer
    than one frame after it).
    """
    starts = sorted({records[e.step].first_frame for e, _ in handled})
    events = []
    for event, _ in handled:
        steer = records[event.step].first_frame
        end = next((s for s in starts if s > steer), video.shape[0])
        if 2 <= steer <= end - 1:
            s = summarize(video[:end], steer, fps)
            own = [c for c in s["cuts"] if c >= steer]
            events.append({"steer_frame": steer, "window_end": end, "cuts_in_window": own, **s})
        else:
            events.append(None)
    times = [r.generate_s for r in records]
    return {
        "events": events,
        "cuts": find_cuts(video) if video.shape[0] >= 2 else [],
        "generate_mean_s": sum(times) / len(times),
        "generate_max_s": max(times),
    }


# ---------------------------------------------------------------- scenarios


def _msg(prompt: str, ramp: int, provisional: bool = False, cap: float = 1.0) -> dict:
    return {"prompt": prompt, "ramp_chunks": ramp, "provisional": provisional, "cap": cap}


def _sequence(name: str, prompts: list[str], ramp: int, lead: int) -> Scenario:
    """Final messages for prompts[1:], each given ramp + 2 steps to finish."""
    gap = ramp + 2
    events = tuple(Event(lead + i * gap, _msg(p, ramp)) for i, p in enumerate(prompts[1:]))
    return Scenario(name, prompts[0], events, lead + (len(prompts) - 1) * gap)


def pace_scenarios(ramps: tuple[int, ...] = (1, 2, 3, 4, 12), lead: int = 4) -> list[Scenario]:
    """One base -> winter scenario per ramp length, named pace_<ramp>."""
    return [_sequence(f"pace_{r}", [BASE, BASE + WINTER], r, lead) for r in ramps]


def default_scenarios(ramp: int = 4) -> list[Scenario]:
    """The PLAN.md steering acceptance checks as scripts.

    Spacing is in steps derived from `ramp`: `lead = max(4, ramp)` idle steps
    before the first message, ramp + 2 steps after each final message, and
    half a ramp (at least 1 step) before an interrupting message.
    """
    if isinstance(ramp, bool) or not isinstance(ramp, int) or ramp < 1:
        raise ValueError(f"ramp must be an int >= 1, got {ramp!r}")
    lead = max(4, ramp)
    half = max(1, ramp // 2)
    winter, night = BASE + WINTER, BASE + NIGHT
    tail = ramp + 2

    def early(name: str, final: str) -> Scenario:
        # The guess gets a full ramp to reach its cap before the final text lands.
        events = (
            Event(lead, _msg(winter, ramp, provisional=True, cap=0.4)),
            Event(lead + ramp, _msg(final, ramp)),
        )
        return Scenario(name, BASE, events, lead + ramp + tail)

    return [
        _sequence("single", [BASE, winter], ramp, lead),
        _sequence("hard_swap", [BASE, winter], 1, lead),
        Scenario(
            "interrupt",
            BASE,
            (Event(lead, _msg(winter, ramp)), Event(lead + half, _msg(night, ramp))),
            lead + half + tail,
        ),
        *pace_scenarios(lead=lead),
        _sequence(
            "stack_remove_middle",
            [BASE, BASE + WINTER, BASE + WINTER + NIGHT, BASE + WINTER + NIGHT + RAIN,
             BASE + WINTER + RAIN],
            ramp,
            lead,
        ),
        early("early_right", winter),
        early("early_wrong", night),
    ]


# ---------------------------------------------------------------- fake model


def fake_encode(prompt: str) -> Tensor:
    """Deterministic [1, 16, 32] tensor seeded from the prompt string."""
    seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "little")
    return torch.randn(1, 16, 32, generator=torch.Generator().manual_seed(seed))


FAKE_ALIGN_LEN = 64  # rows of fake_encode_ids; the longest default prompt has 37 tokens
_EOS = 1


def _fake_id(word: str) -> int:
    return 2 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "little")


def fake_encode_ids(prompt: str) -> tuple[Tensor, list[int]]:
    """([1, FAKE_ALIGN_LEN, 32] tensor, token ids) for runs with an aligner.

    Tokens are words and punctuation plus an end token; each id has one fixed
    row (seeded from the id) in every prompt, and rows after the tokens are 0.
    """
    ids = [_fake_id(w) for w in re.findall(r"\w+|[^\w\s]", prompt)] + [_EOS]
    if len(ids) > FAKE_ALIGN_LEN:
        raise ValueError(f"prompt has {len(ids)} tokens > {FAKE_ALIGN_LEN}")
    emb = torch.zeros(1, FAKE_ALIGN_LEN, 32)
    for k, i in enumerate(ids):
        emb[0, k] = torch.randn(32, generator=torch.Generator().manual_seed(i))
    return emb, ids


class FakeModel:
    """Renders 4 frames of 3x32x32 per call as a smooth function of cond.

    cond (any shape; the projection is sized on first use) is projected
    (fixed random matrix) to a 3x4x4 colour grid, upsampled
    to a smooth image, plus small per-frame seeded noise. All frames of a call
    share the image, so a conditioning jump between calls shows as a one-frame
    change: a full prompt swap is scaled to read as a cut, a 4-step eased ramp
    is not.
    """

    FRAMES, SIZE = 4, 32
    SCALE = 0.025  # sets the size of a full prompt swap relative to cuts' floor
    NOISE = 0.002

    def __init__(self, seed: int = 0) -> None:
        self._proj: Tensor | None = None
        self._seed = seed
        self._calls = 0

    def __call__(self, cond: Tensor) -> Tensor:
        n = cond.numel()
        if self._proj is None or self._proj.shape[0] != n:
            g = torch.Generator().manual_seed(self._seed)
            self._proj = torch.randn(n, 3 * 4 * 4, generator=g) / n**0.5
        coarse = (cond.reshape(-1).to(torch.float32) @ self._proj).view(1, 3, 4, 4)
        img = F.interpolate(coarse, size=self.SIZE, mode="bilinear", align_corners=True)
        img = 0.5 + self.SCALE * img
        g = torch.Generator().manual_seed(self._seed * 100_003 + self._calls)
        self._calls += 1
        noise = torch.randn(self.FRAMES, 3, self.SIZE, self.SIZE, generator=g) * self.NOISE
        return (img + noise).clamp(0, 1)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fake", action="store_true", help="run against the built-in fake model")
    p.add_argument("--ramp", type=int, default=4, help="default ramp in steps (default 4)")
    p.add_argument("--align", action="store_true",
                   help="with --fake: place token rows with steering.align.Aligner")
    p.add_argument("--fps", type=float, default=16.0, help="video fps for metrics (default 16)")
    args = p.parse_args(argv)
    if not args.fake:
        print(
            "No model bound. Running real scenarios needs a `generate(cond) -> frames` and "
            "`encode(prompt) -> cond` binding (plus the cross-attention cache reset hook); "
            "see 'scenarios.py: scripted steering runs' in steering/eval/README.md. "
            "Use --fake for a smoke test.",
            file=sys.stderr,
        )
        return 2
    encode = fake_encode_ids if args.align else fake_encode
    for sc in default_scenarios(args.ramp):
        aligner = Aligner(FAKE_ALIGN_LEN) if args.align else None
        res = run(sc, FakeModel(), encode, args.fps, aligner=aligner)
        print(res.report(), end="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
