"""Repeatable demo run: play a timed script of commands against an engine and record what comes back.

    python -m engine.tools.demo_run --url http://127.0.0.1:8769 --script engine/tools/demo_scripts/draft.json \\
        --out runs/baseline --label baseline
    python -m engine.tools.demo_run --compare runs/baseline runs/new-reanchor --out runs/compare

A run connects to /frames and /control, sends each script step at its time (a `say` step is a contract 1 `final`
message with a fresh t_client, exactly as the demo page sends a typed command; a `reset` step is a contract 1 reset
tap, also with a t_client, which the contract allows on a tap, so contract 4 reports its first chunk like a
command's), and records every frame and every contract 4 message until the script's `end`.
It sends nothing else: no partials, no other taps, no POST requests. It only reads GET /stats.

The output folder gets:
    video.mp4     every frame received during the script, at the engine fps (libx264, CRF 20)
    sheet.jpg     one frame every 2 s, labelled with the time, the chunk and the command shown at that moment
    frames/       the sheet frames as JPEGs, exactly as the engine sent them
    timings.json  per command: t_client, contract 4 stamps, applies_from_chunk, unmatched, client-side delay;
                  /stats at start and end; the sheet marks
    summary.txt   the same in a short table

`--compare` puts several runs on one sheet, one row per run at the same time marks, to judge quality changes.

Against a URL that is not localhost the run changes the shared live video for every viewer, so it prints a warning
and waits a few seconds before it starts (`--yes` skips the wait).
"""

import argparse
import asyncio
import enum
import io
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from PIL import Image, ImageDraw, ImageFont

from engine.framing import unpack_frame

STAMPS = ("heard", "decided", "embedded", "generated", "shown")
STATS_KEYS = ("state", "backend", "reader", "gen_ms", "decode_ms", "fps", "fps_target", "stalls", "reanchor",
              "recorder")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SHEET_EVERY_S = 2.0
WARN_WAIT_S = 5.0


# --- the script ------------------------------------------------------------------------------------------------


class StepKind(enum.Enum):
    SAY = "say"      # a contract 1 final message
    RESET = "reset"  # a contract 1 reset tap


@dataclass(frozen=True)
class Step:
    at: float
    kind: StepKind
    text: str | None = None


@dataclass(frozen=True)
class Script:
    name: str
    end: float
    steps: tuple[Step, ...]


class ScriptError(ValueError):
    pass


def parse_script(data: dict, default_name: str = "script") -> Script:
    """Check and convert a script: {"name": ..., "end": seconds, "steps": [{"at": s, "say": text} | {"at": s, "reset": true}]}.

    Steps are sorted by time (a stable sort, so steps at the same time keep their file order).
    """
    if not isinstance(data, dict):
        raise ScriptError("a script is a JSON object")
    end = data.get("end")
    if not _is_number(end) or end <= 0:
        raise ScriptError(f"'end' must be a number of seconds above 0, got {end!r}")
    raw = data.get("steps")
    if not isinstance(raw, list) or not raw:
        raise ScriptError("'steps' must be a non-empty list")
    steps = []
    for i, s in enumerate(raw):
        where = f"step {i}"
        if not isinstance(s, dict):
            raise ScriptError(f"{where} must be an object")
        extra = set(s) - {"at", "say", "reset"}
        if extra:
            raise ScriptError(f"{where} has unknown keys {sorted(extra)}")
        at = s.get("at")
        if not _is_number(at) or at < 0 or at > end:
            raise ScriptError(f"{where}: 'at' must be a number from 0 to end ({end}), got {at!r}")
        has_say, has_reset = "say" in s, "reset" in s
        if has_say == has_reset:
            raise ScriptError(f"{where} needs exactly one of 'say' or 'reset'")
        if has_say:
            if not isinstance(s["say"], str) or not s["say"].strip():
                raise ScriptError(f"{where}: 'say' must be a non-empty string")
            steps.append(Step(float(at), StepKind.SAY, s["say"].strip()))
        else:
            if s["reset"] is not True:
                raise ScriptError(f"{where}: 'reset' must be true")
            steps.append(Step(float(at), StepKind.RESET))
    name = data.get("name", default_name)
    if not isinstance(name, str) or not name:
        raise ScriptError("'name' must be a non-empty string")
    return Script(name, float(end), tuple(sorted(steps, key=lambda st: st.at)))


def load_script(path: Path) -> Script:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ScriptError(f"{path} is not valid JSON: {exc}") from None
    return parse_script(data, default_name=path.stem)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# --- timing math (pure, works on script seconds) ----------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    t: float      # receive time in script seconds (0 is when the script started)
    chunk: int
    index: int


@dataclass
class Sent:
    """One step as sent, and what came back for it. A reset tap carries a t_client too (contract 1 allows it on a
    tap), so the engine tracks it like a command and contract 4 gives its applies_from_chunk: the first chunk of
    the new video."""
    at: float                 # planned script time
    kind: StepKind
    text: str | None          # the words of a SAY step, None for a RESET
    t_client: int             # epoch ms, as in contract 1
    sent_t: float             # actual send time in script seconds
    reply: dict | None = None  # the last contract 4 message that echoed this t_client


def sheet_marks(end: float, every: float = SHEET_EVERY_S) -> list[float]:
    """0, every, 2 every, ... up to and including end."""
    n = int(end / every + 1e-9)
    return [round(i * every, 3) for i in range(n + 1)]


def frame_at(frames: list[Frame], t: float) -> int | None:
    """Index of the last frame received at or before t, else the first frame after it, else None."""
    if not frames:
        return None
    best = None
    for i, f in enumerate(frames):
        if f.t <= t:
            best = i
        else:
            break
    return 0 if best is None else best


def first_frame_from(frames: list[Frame], after_t: float, chunk: int) -> Frame | None:
    """The first frame received at or after after_t whose chunk is at least `chunk`.

    This is how the page measures "shown": the first displayed frame of the command's first chunk or later.
    Frames before the send time are skipped, so a reset that restarts chunk numbers cannot match an old frame.
    """
    for f in frames:
        if f.t >= after_t and f.chunk >= chunk:
            return f
    return None


def command_delay(cmd: Sent, frames: list[Frame]) -> dict:
    """The client-side result of one command: first applied frame and its receive time minus the send time."""
    reply = cmd.reply or {}
    afc = reply.get("applies_from_chunk")
    out = {"at": cmd.at, "kind": cmd.kind.value, "say": cmd.text, "t_client": cmd.t_client, "sent_s": round(cmd.sent_t, 3),
           "replied": cmd.reply is not None, "stamps": reply.get("stamps", {}), "applies_from_chunk": afc,
           "unmatched": reply.get("unmatched"), "keyword": reply.get("keyword"),
           "changes": [c.get("label") for c in reply.get("changes", [])],
           "first_frame": None, "client_delay_ms": None}
    if isinstance(afc, int):
        f = first_frame_from(frames, cmd.sent_t, afc)
        if f is not None:
            out["first_frame"] = {"chunk": f.chunk, "index": f.index, "received_s": round(f.t, 3)}
            out["client_delay_ms"] = round((f.t - cmd.sent_t) * 1000)
    return out


BEFORE_RUN = "before the run"  # the video that was playing before the script's first applied step
START = "start"                  # the starting prompt of a new video, after a reset


def step_name(r: dict) -> str:
    return r["say"] if r["kind"] == StepKind.SAY.value else "reset"


def shown_at(results: list[dict], frame: Frame) -> tuple[str, str | None]:
    """(what the frame shows, a newer step sent but not shown yet) at that frame.

    `results` are command_delay() outputs in send order, resets included. A step is shown from its first applied
    frame on; an applied reset shows the starting prompt.
    """
    showing, pending = BEFORE_RUN, None
    for r in results:
        if r["sent_s"] > frame.t:
            break
        ff = r["first_frame"]
        if ff is not None and frame.t >= ff["received_s"] and frame.chunk >= ff["chunk"]:
            showing, pending = (START if r["kind"] == StepKind.RESET.value else r["say"]), None
        else:
            pending = step_name(r)
    return showing, pending


# --- the run -----------------------------------------------------------------------------------------------------


class Phase(enum.Enum):
    CONNECTING = "connecting"   # waiting for /stats to say GENERATING and for both sockets
    RUNNING = "running"         # playing the script and recording
    SAVING = "saving"           # writing the mp4, the sheet and the timings
    DONE = "done"


def is_local(url: str) -> bool:
    return (urllib.parse.urlsplit(url).hostname or "") in LOCAL_HOSTS


def ws_base(url: str) -> str:
    return url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)


def get_stats(base: str) -> dict | None:
    req = urllib.request.Request(base + "/stats", headers={"User-Agent": "madhacks-demo-run"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except (OSError, json.JSONDecodeError):
        return None


def stats_subset(s: dict | None) -> dict | None:
    return None if s is None else {k: s.get(k) for k in STATS_KEYS}


@dataclass
class Recording:
    t0: float | None = None                      # monotonic time of script second 0
    frames: list[tuple[float, int, int, bytes]] = field(default_factory=list)  # (monotonic, chunk, index, jpeg)
    states: list[tuple[float, dict]] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    frames_closed: str | None = None
    control_closed: str | None = None


async def read_frames(url: str, rec: Recording) -> None:
    try:
        async with websockets.connect(url + "/frames", max_size=None, open_timeout=20) as ws:
            async for msg in ws:
                chunk, index, jpeg = unpack_frame(msg)
                rec.frames.append((time.monotonic(), chunk, index, jpeg))
    except Exception as exc:  # noqa: BLE001 - reported in the summary
        rec.frames_closed = f"{type(exc).__name__}: {exc}"
    else:
        rec.frames_closed = "closed by the server"


async def read_control(ws, rec: Recording) -> None:
    try:
        async for msg in ws:
            data = json.loads(msg)
            if "error" in data:
                rec.errors.append(data)
            else:
                rec.states.append((time.monotonic(), data))
    except Exception as exc:  # noqa: BLE001
        rec.control_closed = f"{type(exc).__name__}: {exc}"


async def wait_for(predicate, timeout: float, every: float = 0.05):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(every)
    return None


def enter(phase: Phase) -> Phase:
    print(f"[{phase.value}]")
    return phase


async def run(args, script: Script) -> int:
    base = args.url.rstrip("/")
    out = Path(args.out)
    enter(Phase.CONNECTING)
    stats0 = None
    t_end = time.monotonic() + 60
    while time.monotonic() < t_end:
        stats0 = get_stats(base)
        if stats0 and stats0.get("state") == "GENERATING":
            break
        await asyncio.sleep(1.0)
    if not stats0 or stats0.get("state") != "GENERATING":
        print(f"error: {base}/stats is not GENERATING (got {stats0 and stats0.get('state')})", file=sys.stderr)
        return 1
    fps = stats0.get("fps_target") or 16
    print(f"  backend {stats0.get('backend')}, reader {stats0.get('reader')}, {stats0.get('width')}x"
          f"{stats0.get('height')}, {fps} fps")

    rec = Recording()
    wsb = ws_base(base)
    frames_task = asyncio.create_task(read_frames(wsb, rec))
    async with websockets.connect(wsb + "/control", open_timeout=20) as control:
        control_task = asyncio.create_task(read_control(control, rec))
        if not await wait_for(lambda: rec.frames or rec.frames_closed, 20.0) or not rec.frames:
            print(f"error: no frames on /frames ({rec.frames_closed})", file=sys.stderr)
            return 1

        enter(Phase.RUNNING)
        sent: list[Sent] = []
        rec.t0 = t0 = time.monotonic()
        last_t_client = 0
        for step in script.steps:
            delay = t0 + step.at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            # Strictly increasing, so two steps at the same second never share a t_client (replies are matched by it).
            t_client = last_t_client = max(int(time.time() * 1000), last_t_client + 1)
            if step.kind is StepKind.RESET:
                message = {"type": "tap", "action": "reset", "t_client": t_client}
            else:
                message = {"type": "final", "text": step.text, "t_client": t_client}
            await control.send(json.dumps(message))
            now = time.monotonic() - t0
            sent.append(Sent(step.at, step.kind, step.text, t_client, now))
            print(f"  {now:6.2f} s  {step.kind.value}" + (f" {step.text!r}" if step.text else ""))
        rest = t0 + script.end - time.monotonic()
        if rest > 0:
            await asyncio.sleep(rest)
        t_stop = time.monotonic()
        stats1 = get_stats(base)
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)
    frames_task.cancel()
    await asyncio.gather(frames_task, return_exceptions=True)

    enter(Phase.SAVING)
    for cmd in sent:
        replies = [m for t, m in rec.states if m.get("t_client") == cmd.t_client and t <= t_stop]
        cmd.reply = replies[-1] if replies else None
    window = [f for f in rec.frames if t0 <= f[0] <= t_stop]
    frames = [Frame(t - t0, c, i) for t, c, i, _ in window]
    results = [command_delay(cmd, frames) for cmd in sent]
    report = save(out, args.label or out.name, script, fps, window, frames, results, rec, stats0, stats1)
    print((out / "summary.txt").read_text())
    enter(Phase.DONE)
    print(f"  wrote {out}")
    if not report["frames_received"]:
        return 1
    last_frame = window[-1][0] if window else t0
    if t_stop - last_frame > FRAMES_STOPPED_S:
        print(f"  FAILED: no frame in the last {t_stop - last_frame:.1f} s of the run (the /frames connection dropped?)")
        return 1
    return 0


FRAMES_STOPPED_S = 2.0  # a run fails if the last frame arrived more than this before the end


# --- saving ------------------------------------------------------------------------------------------------------


def save(out: Path, label: str, script: Script, fps: float, window, frames: list[Frame], results: list[dict],
         rec: Recording, stats0: dict, stats1: dict | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()

    video = out / "video.mp4"
    encode_mp4([jpeg for *_, jpeg in window], fps, video)

    marks = []
    for m in sheet_marks(script.end):
        i = frame_at(frames, m)
        if i is None:
            continue
        f = frames[i]
        showing, pending = shown_at(results, f)
        name = f"t{m:05.1f}.jpg"
        (frames_dir / name).write_bytes(window[i][3])
        marks.append({"t": m, "file": f"frames/{name}", "chunk": f.chunk, "index": f.index,
                      "received_s": round(f.t, 3), "showing": showing, "pending": pending})
    make_sheet(out, label, script.name, marks)

    gaps = [b.t - a.t for a, b in zip(frames, frames[1:])]
    report = {
        "label": label, "script": script.name, "end_s": script.end, "fps": fps,
        "frames_received": len(frames), "max_frame_gap_ms": round(max(gaps) * 1000) if gaps else None,
        "commands": results, "error_replies": rec.errors,
        "frames_socket": rec.frames_closed, "control_socket": rec.control_closed,
        "stats_start": stats_subset(stats0), "stats_end": stats_subset(stats1),
        "sheet": marks,
    }
    (out / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.txt").write_text(summary_text(report))
    return report


def encode_mp4(jpegs: list[bytes], fps: float, path: Path) -> None:
    if not jpegs:
        return
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(fps),
           "-c:v", "mjpeg", "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(jpegs), check=True)


def summary_text(r: dict) -> str:
    lines = [f"demo run '{r['label']}', script '{r['script']}', {r['end_s']:.0f} s, {r['frames_received']} frames "
             f"at {r['fps']} fps, max frame gap {r['max_frame_gap_ms']} ms", ""]
    lines.append(f"{'sent s':>7}  {'step':<22} {'unmatched':<9} {'afc':>6} "
                 + " ".join(f"{k:>9}" for k in STAMPS) + f" {'client ms':>9}")
    for c in r["commands"]:
        s = c["stamps"]
        lines.append(f"{c['sent_s']:7.2f}  {step_name(c)[:22]:<22} {str(c['unmatched']):<9} {str(c['applies_from_chunk']):>6} "
                     + " ".join(f"{str(s.get(k, '-')):>9}" for k in STAMPS) + f" {str(c['client_delay_ms']):>9}")
    lines.append("")
    for name in ("stats_start", "stats_end"):
        s = r[name]
        if s is None:
            lines.append(f"{name}: unavailable")
            continue
        ra, rc = s.get("reanchor") or {}, s.get("recorder") or {}
        lines.append(f"{name}: gen_ms {(s.get('gen_ms') or {}).get('avg')}, decode_ms {(s.get('decode_ms') or {}).get('avg')}, "
                     f"fps {s.get('fps')}, stalls {s.get('stalls')}, reanchor {ra.get('state')} x{ra.get('count')}, "
                     f"recorder {rc.get('state')} ({rc.get('frames_written')} frames written, {rc.get('frames_dropped')} dropped)")
    if r["error_replies"]:
        lines.append(f"error replies: {r['error_replies']}")
    for k in ("frames_socket", "control_socket"):
        if r[k]:
            lines.append(f"{k}: {r[k]}")
    return "\n".join(lines) + "\n"


# --- sheets ------------------------------------------------------------------------------------------------------


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1 has no size argument
        return ImageFont.load_default()


def thumb(path: Path, width: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)


def cell_lines(mark: dict) -> list[str]:
    lines = [f"{mark['t']:.0f} s  chunk {mark['chunk']}", f"shows: {mark['showing']}"]
    if mark["pending"]:
        lines.append(f"sent: {mark['pending']}")
    return lines


def make_sheet(out: Path, label: str, script_name: str, marks: list[dict], cols: int = 6, width: int = 320) -> None:
    if not marks:
        return
    thumbs = [thumb(out / m["file"], width) for m in marks]
    th = thumbs[0].height
    f, fb = font(15), font(20)
    text_h, pad, head = 3 * 19 + 6, 8, 40
    rows = -(-len(thumbs) // cols)
    sheet = Image.new("RGB", (pad + cols * (width + pad), head + rows * (th + text_h + pad)), (20, 20, 24))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), f"{label}  |  script {script_name}  |  one frame every {SHEET_EVERY_S:.0f} s", font=fb,
           fill=(240, 240, 240))
    for n, (m, img) in enumerate(zip(marks, thumbs)):
        x, y = pad + (n % cols) * (width + pad), head + (n // cols) * (th + text_h + pad)
        sheet.paste(img, (x, y))
        for k, line in enumerate(cell_lines(m)):
            d.text((x, y + th + 3 + k * 19), fit(d, line, f, width), font=f,
                   fill=(255, 210, 120) if line.startswith("sent:") else (220, 220, 220))
    sheet.save(out / "sheet.jpg", quality=90)


def fit(d: ImageDraw.ImageDraw, text: str, f, width: int) -> str:
    """Cut text to fit the width, ending in "..." when it was cut."""
    if d.textlength(text, font=f) <= width:
        return text
    while text and d.textlength(text + "...", font=f) > width:
        text = text[:-1]
    return text + "..."


def compare(dirs: list[Path], out: Path, every: float, width: int = 200) -> int:
    runs = []
    for d in dirs:
        try:
            runs.append(json.loads((d / "timings.json").read_text()) | {"dir": d})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: {d} has no readable timings.json ({exc})", file=sys.stderr)
            return 1
    common = sorted(set.intersection(*({m["t"] for m in r["sheet"]} for r in runs)))
    step = max(1, round(every / SHEET_EVERY_S))
    times = common[::step]
    if not times:
        print("error: the runs have no time marks in common", file=sys.stderr)
        return 1
    first = next(m for m in runs[0]["sheet"] if m["t"] == times[0])
    th = thumb(runs[0]["dir"] / first["file"], width).height  # every row uses this height, so rows line up
    f, fb = font(13), font(17)
    left, pad, head, text_h = 190, 6, 30, 2 * 16 + 4
    row_h = th + text_h + pad
    sheet = Image.new("RGB", (left + len(times) * (width + pad), head + len(runs) * row_h), (20, 20, 24))
    d = ImageDraw.Draw(sheet)
    for j, t in enumerate(times):
        d.text((left + j * (width + pad), 8), f"{t:.0f} s", font=fb, fill=(240, 240, 240))
    for i, r in enumerate(runs):
        y = head + i * row_h
        delays = [c["client_delay_ms"] for c in r["commands"] if c["kind"] == StepKind.SAY.value and c["client_delay_ms"] is not None]
        info = [r["label"], f"script {r['script']}", f"{r['frames_received']} frames",
                f"delay ms {min(delays)}-{max(delays)}" if delays else "no delays"]
        for k, line in enumerate(info):
            d.text((pad, y + 4 + k * 20), fit(d, line, fb if k == 0 else f, left - 2 * pad), font=fb if k == 0 else f,
                   fill=(240, 240, 240) if k == 0 else (180, 180, 180))
        by_t = {m["t"]: m for m in r["sheet"]}
        for j, t in enumerate(times):
            m, x = by_t[t], left + j * (width + pad)
            sheet.paste(thumb(r["dir"] / m["file"], width).resize((width, th)), (x, y))
            lines = [f"chunk {m['chunk']}", m["showing"]]
            if m["pending"]:
                lines[1] = f"{lines[1]} (sent: {m['pending']})"
            for k, line in enumerate(lines):
                d.text((x, y + th + 2 + k * 16), fit(d, line, f, width), font=f, fill=(220, 220, 220))
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / "compare.jpg", quality=90)
    print(f"wrote {out / 'compare.jpg'}: {len(runs)} runs x {len(times)} marks ({', '.join(r['label'] for r in runs)})")
    return 0


# --- command line ------------------------------------------------------------------------------------------------


def warn_shared(url: str, wait: bool) -> None:
    bar = "!" * 78
    print(f"{bar}\nWARNING: {url} is not localhost.\n"
          "This run sends the script's commands and resets to that engine, which changes the shared live video\n"
          "for every viewer. Press Ctrl-C now to stop.\n" + bar, flush=True)
    if wait:
        time.sleep(WARN_WAIT_S)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m engine.tools.demo_run", description=__doc__.splitlines()[0])
    p.add_argument("--url", help="engine origin, http(s)://host:port")
    p.add_argument("--script", help="a JSON script, for example engine/tools/demo_scripts/draft.json")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--label", help="name of this run on the sheets (default: the output folder name)")
    p.add_argument("--yes", action="store_true", help="skip the wait after the non-localhost warning")
    p.add_argument("--compare", nargs="+", metavar="DIR", help="make a side-by-side sheet of these run folders")
    p.add_argument("--every", type=float, default=4.0, help="--compare: seconds between columns (a multiple of 2)")
    args = p.parse_args(argv)

    if args.compare:
        sys.exit(compare([Path(d) for d in args.compare], Path(args.out), args.every))
    if not args.url or not args.script:
        p.error("--url and --script are required unless --compare is given")
    if not args.url.startswith(("http://", "https://")):
        p.error("--url must start with http:// or https://")
    try:
        script = load_script(Path(args.script))
    except (OSError, ScriptError) as exc:
        p.error(str(exc))
    if shutil.which("ffmpeg") is None:
        p.error("ffmpeg is not on PATH")
    if not is_local(args.url):
        warn_shared(args.url, wait=not args.yes)
    sys.exit(asyncio.run(run(args, script)))


if __name__ == "__main__":
    main()
