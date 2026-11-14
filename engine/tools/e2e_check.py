"""End-to-end check against a running engine server.

    python -m engine.tools.e2e_check --url http://127.0.0.1:8000

Connects to /frames and /control, measures the frame rate, checks chunk numbers and JPEGs, sends `final`
commands, validates every contract 4 message against contracts/schemas, checks that all five stamps appear in
order, checks a bad message gets an error reply, and checks a reset tap. Prints the numbers; exits 1 on failure.
"""

import argparse
import asyncio
import io
import json
import sys
import time
import urllib.request

import websockets
from PIL import Image

from contracts.validate import errors
from engine.framing import unpack_frame

STAMPS = ("heard", "decided", "embedded", "generated", "shown")


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def that(self, ok: bool, what: str) -> bool:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")
        if not ok:
            self.failures.append(what)
        return ok


class Client:
    """Collects everything from both sockets in the background."""

    def __init__(self, base: str) -> None:
        self.base = base
        self.frames: list[tuple[float, int, int, bytes]] = []  # (arrival, chunk, index, jpeg)
        self.states: list[tuple[float, dict]] = []
        self.errors: list[dict] = []
        self.control = None

    async def run_frames(self) -> None:
        async with websockets.connect(self.base + "/frames", max_size=None) as ws:
            async for msg in ws:
                chunk, index, jpeg = unpack_frame(msg)
                self.frames.append((time.monotonic(), chunk, index, jpeg))

    async def run_control(self) -> None:
        async with websockets.connect(self.base + "/control") as ws:
            self.control = ws
            async for msg in ws:
                data = json.loads(msg)
                (self.errors if "error" in data else self.states).append((time.monotonic(), data) if "error" not in data else data)

    async def send(self, message: dict) -> None:
        await self.control.send(json.dumps(message))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


async def wait_for(predicate, timeout: float):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(0.02)
    return None


async def command(client: Client, check: Check, text: str, label: str) -> dict | None:
    """Send a final command and wait for a contract 4 message carrying all five stamps for it."""
    t_client = int(time.time() * 1000)
    await client.send({"type": "final", "text": text, "t_client": t_client})

    def done():
        for _, m in client.states:
            if m.get("t_client") == t_client and all(k in m["stamps"] for k in STAMPS):
                return m
        return None

    msg = await wait_for(done, 10.0)
    if not check.that(msg is not None, f"{label}: a contract 4 message with all five stamps arrived"):
        return None
    s = msg["stamps"]
    print(f"     stamps ms after heard: " + ", ".join(f"{k} {s[k]}" for k in STAMPS))
    check.that([s[k] for k in STAMPS] == sorted(s[k] for k in STAMPS), f"{label}: stamps are in order")
    check.that(s["heard"] == 0 and s["shown"] < 5000, f"{label}: heard is 0 and shown is under 5 s")
    check.that(msg["heard"] == text, f"{label}: heard text echoed")
    afc = msg["applies_from_chunk"]
    check.that(isinstance(afc, int), f"{label}: applies_from_chunk is set ({afc})")
    shown_frames = [f for f in client.frames if f[1] >= afc]
    check.that(bool(shown_frames), f"{label}: frames of chunk {afc} or later arrived")
    return msg


async def main_async(args) -> int:
    base_http = args.url.rstrip("/")
    base_ws = base_http.replace("http://", "ws://").replace("https://", "wss://")
    check = Check()

    stats = await wait_for(lambda: (s := _try_stats(base_http)) and s["state"] == "GENERATING" and s, 60.0)
    if not check.that(bool(stats), "server reachable and GENERATING"):
        return 1
    print(f"     backend {stats['backend']}, {stats['width']}x{stats['height']}, target {stats['fps_target']} fps, "
          f"{stats['frames_per_chunk']} frames per chunk")
    fps_target = stats["fps_target"]

    client = Client(base_ws)
    tasks = [asyncio.create_task(client.run_frames()), asyncio.create_task(client.run_control())]
    await wait_for(lambda: client.control is not None and len(client.frames) > 0, 10.0)

    print(f"\n1. frames for {args.seconds:.0f} s")
    await asyncio.sleep(args.seconds)
    window = [f for f in client.frames if f[0] >= client.frames[0][0] + 1.0]  # skip the connect burst
    span = window[-1][0] - window[0][0] if len(window) > 1 else 0
    fps = (len(window) - 1) / span if span else 0.0
    print(f"     {len(client.frames)} frames, measured {fps:.2f} fps over {span:.1f} s")
    check.that(abs(fps - fps_target) <= 0.1 * fps_target, f"frame rate within 10% of {fps_target} fps")
    order = [(c, i) for _, c, i, _ in client.frames]
    check.that(all(a < b for a, b in zip(order, order[1:])), "chunk numbers rise and frame indexes rise within a chunk")
    bad = 0
    for _, _, _, jpeg in client.frames[:: max(1, len(client.frames) // 20)]:
        try:
            img = Image.open(io.BytesIO(jpeg))
            img.load()
            bad += img.format != "JPEG" or img.size != (stats["width"], stats["height"])
        except Exception:
            bad += 1
    check.that(bad == 0, f"sampled frames are valid {stats['width']}x{stats['height']} JPEGs")
    gaps = [b[0] - a[0] for a, b in zip(window, window[1:])]
    if gaps:
        print(f"     frame gap ms: max {max(gaps) * 1e3:.0f}, mean {sum(gaps) / len(gaps) * 1e3:.1f}")

    print("\n2. first command (new prompt, needs encoding)")
    first = await command(client, check, f"a snowy winter forest at dusk {time.time():.0f}", "new prompt")

    await asyncio.sleep(1.0)
    print("\n3. second command (the base prompt again, a cache hit)")
    await command(client, check, stats_prompt(base_http), "cached prompt")

    print("\n4. contract 4 messages")
    bad = [e for _, m in client.states for e in errors(4, m)]
    check.that(not bad, f"all {len(client.states)} contract 4 messages validate against contracts/schemas"
               + (f" (first problem: {bad[0]})" if bad else ""))
    chunks = [m["chunk"] for _, m in client.states]
    check.that(all(a < b for a, b in zip(chunks, chunks[1:])), "one contract 4 message per chunk, in order")
    if len(client.states) > 2:
        dt = client.states[-1][0] - client.states[0][0]
        print(f"     {len(client.states)} messages, {(len(client.states) - 1) / dt:.2f} per second")

    print("\n5. bad message")
    n = len(client.errors)
    await client.send({"type": "final", "text": "missing t_client"})
    got = await wait_for(lambda: len(client.errors) > n, 3.0)
    check.that(bool(got), "a bad message gets an error reply" + (f": {client.errors[-1]['error']}" if got else ""))

    print("\n6. reset tap")
    before = client.frames[-1][1]
    await client.send({"type": "tap", "action": "reset"})
    after = await wait_for(lambda: client.frames[-1][1] > before + 4 and client.frames[-1], 10.0)
    s = get_json(base_http + "/stats")
    check.that(bool(after) and s["state"] == "GENERATING", f"frames continue after reset (state {s['state']})")

    for t in tasks:
        t.cancel()
    print()
    if check.failures:
        print(f"FAILED: {len(check.failures)} check(s): " + "; ".join(check.failures))
        return 1
    print("PASSED")
    return 0


def _try_stats(base: str) -> dict | None:
    try:
        return get_json(base + "/stats")
    except OSError:
        return None


def stats_prompt(_base: str) -> str:
    # The server's default base prompt; it was encoded at startup, so this command is a cache hit.
    from engine.engine import EngineConfig
    return EngineConfig.base_prompt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--seconds", type=float, default=5.0, help="how long to measure the frame rate")
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
