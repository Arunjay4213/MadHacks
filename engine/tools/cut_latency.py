"""Command-to-screen delay of a steer cut, on the fake backend at the live model's pass timings.

    python -m engine.tools.cut_latency                 # the four variants, 24 commands each (about 5 minutes)
    python -m engine.tools.cut_latency --commands 8 --variants today,all

Each variant runs one engine in this process (no ports), with --reanchor on and the live refresh every 24 chunks.
The fake backend runs the real model's passes (pass_ms 69, decode 22 ms: a normal chunk is 3 passes, about 207 ms,
and a steer cut 5 passes today, as measured live). Commands are POST /target steers with ramp_chunks 1 (a normal
command), their prompts already encoded, sent 2.3 to 3.3 s apart (seeded) and never within the policy's minimum gap
after the last re-anchor (often the refresh), so each one is a steer cut on its first chunk.

For each variant it prints heard to shown (the cut's first frame handed to the /frames clients) as median and p90,
and the stream's stalls, the chunks stopped for a cut, the old frames dropped, and a check of every command's
stamps: its applies_from_chunk is the steer cut's chunk, and generated <= shown. The frames sent are checked too:
chunk numbers never go back, and with no flush every chunk is sent whole (4 frames), so a stopped chunk is never
shown and the cut takes its number with no gap.
"""

import argparse
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field

from engine.commands import CommandStage
from engine.engine import Engine, EngineConfig
from engine.framing import unpack_frame
from engine.lifecycle import Lifecycle

PROMPTS = ["a meadow in deep winter snow", "a meadow at night under a full moon", "a meadow in autumn, orange leaves",
           "a meadow in heavy rain", "a desert of red dunes at sunset", "a meadow in spring, full of flowers"]

LIVE_PASSES = dict(pass_ms=69, denoise_passes=2, decode_ms=22, encode_ms=5, frames_per_chunk=4, fps=16)


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    backend: dict
    flush: str = "off"
    interrupt: str = "off"


VARIANTS = {
    "today": Variant("today", "today (5-pass cut)", dict(cut_passes=4)),
    "fast": Variant("fast", "2-step cut + defer_context", dict(cut_passes=2, defer_context=True)),
    "flush": Variant("flush", "+ flush", dict(cut_passes=2, defer_context=True), flush="on"),
    "all": Variant("all", "+ flush + interrupt", dict(cut_passes=2, defer_context=True), flush="on", interrupt="on"),
}


@dataclass
class Result:
    variant: Variant
    delays_ms: list[int]
    stalls: int
    interrupted: int
    late_interrupts: int
    flushed_frames: int
    problems: list[str] = field(default_factory=list)

    def row(self) -> str:
        d = sorted(self.delays_ms)
        p90 = d[min(len(d) - 1, round(0.9 * (len(d) - 1)))] if d else None
        med = round(statistics.median(d)) if d else None
        return (f"| {self.variant.label} | {med} / {p90} | {self.stalls} | {self.interrupted} | "
                f"{self.flushed_frames} | {'ok' if not self.problems else '; '.join(self.problems[:3])} |")


def run(variant: Variant, commands: int, seed: int = 1, width: int = 336, height: int = 192,
        gap_s: tuple[float, float] = (2.3, 3.3)) -> Result:
    sent: list[tuple[int, int]] = []  # (chunk, index) of every frame handed to the clients
    lock = threading.Lock()

    def publish(message: bytes) -> None:
        chunk, index, _ = unpack_frame(message)
        with lock:
            sent.append((chunk, index))

    cfg = EngineConfig(backend="fake", width=width, height=height, backend_options={**LIVE_PASSES, **variant.backend},
                       reanchor="on", reanchor_options={"refresh_every_chunks": 24}, record_dir=None,
                       flush_on_cut=variant.flush, interrupt_on_steer=variant.interrupt)
    e = Engine(cfg, publish, lambda m: None)
    reasons: dict[int, str | None] = {}  # chunk number -> its re-anchor reason; a stopped chunk's rerun overwrites it
    take_cond = e.book.take_cond

    def noting_take_cond(get_cond, chunk):
        cond, progress, anchor = take_cond(get_cond, chunk)
        reasons[chunk] = None if anchor is None else anchor.reason
        return cond, progress, anchor

    e.book.take_cond = noting_take_cond
    e.start()
    problems: list[str] = []
    delays: list[int] = []
    try:
        if not _wait(lambda: e.lifecycle.state is Lifecycle.GENERATING, 30):
            raise RuntimeError("the engine did not start")
        for p in PROMPTS:
            e.encoder.encode(p)  # cached, as the demo's pinned prompts are
        _wait(lambda: e.chunk_shown >= 12, 30)  # past the start and the policy's minimum gap
        stalls0 = e.stats()["stalls"]
        rng = random.Random(seed)
        records = []
        for i in range(commands):
            time.sleep(rng.uniform(*gap_s))
            # A steer within min_gap_chunks of the last anchor (often the refresh) waits for the gap, and its
            # stamps then time an old-scene chunk; wait until the gap is met so every command is a cut.
            _wait(lambda: e.policy.chunks_since_anchor >= e.reanchor_config.min_gap_chunks, 5)
            target = {"prompt": PROMPTS[i % len(PROMPTS)], "ramp_chunks": 1, "provisional": False, "cap": 1.0}
            records.append(e.handle_target(target, time.monotonic_ns()))
        _wait(lambda: all(r.stage is CommandStage.SHOWN for r in records), 10)
        time.sleep(0.5)
        s = e.stats()
        for r in records:
            if r.stage is not CommandStage.SHOWN:
                problems.append(f"command {r.seq} ended {r.stage.value}")
                continue
            st = r.stamps_ms()
            delays.append(st["shown"])
            if not st["generated"] <= st["shown"]:
                problems.append(f"command {r.seq}: generated {st['generated']} after shown {st['shown']}")
        problems += _check_frames(sent, variant.flush == "on", e.backend.frames_per_chunk)
        problems += [f"command {r.seq}: applies_from_chunk {r.applies_from_chunk} is not a steer cut"
                     for r in records if r.stage is CommandStage.SHOWN and reasons.get(r.applies_from_chunk) != "steer"]
        return Result(variant, delays, s["stalls"] - stalls0, s["cut"]["interrupted_chunks"],
                      s["cut"]["late_interrupts"], s["cut"]["flushed_frames"], problems)
    finally:
        e.stop()


def _check_frames(sent: list[tuple[int, int]], flush: bool, frames_per_chunk: int) -> list[str]:
    """Chunk numbers never go back and each chunk's frames come in order; without a flush every chunk is whole
    and no chunk number is skipped (so a stopped chunk left no gap and was never shown)."""
    problems = []
    counts: dict[int, int] = {}
    prev = None
    for chunk, index in sent:
        if prev is not None and (chunk < prev[0] or (chunk == prev[0] and index != prev[1] + 1)):
            problems.append(f"frame order: ({prev[0]}, {prev[1]}) then ({chunk}, {index})")
        prev = (chunk, index)
        counts[chunk] = counts.get(chunk, 0) + 1
    if not flush and counts:
        chunks = sorted(counts)
        inner = chunks[1:-1]  # the first and last may be cut by the measuring window
        missing = sorted(set(range(chunks[0], chunks[-1] + 1)) - set(counts))
        if missing:
            problems.append(f"chunks never shown: {missing[:5]}")
        broken = [c for c in inner if counts[c] != frames_per_chunk]
        if broken:
            problems.append(f"chunks not shown whole: {broken[:5]}")
    return problems[:10]


def _wait(pred, timeout: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m engine.tools.cut_latency", description=__doc__.splitlines()[0])
    p.add_argument("--commands", type=int, default=24)
    p.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated, from {', '.join(VARIANTS)}")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    results = []
    print("| Variant | heard to shown, median / p90 ms | stalls | chunks stopped | old frames dropped | checks |")
    print("|---|---|---|---|---|---|")
    for name in args.variants.split(","):
        r = run(VARIANTS[name.strip()], args.commands, args.seed)
        results.append(r)
        print(r.row(), flush=True)
    return 1 if any(r.problems for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
