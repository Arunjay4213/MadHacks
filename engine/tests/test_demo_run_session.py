"""demo_run.run() against a small in-process engine: /frames and /control websockets, no GPU and no engine.server."""

import asyncio
import io
import json
import time
import types

from PIL import Image
from websockets.asyncio.server import serve

from engine.framing import pack_frame
from engine.tools import demo_run
from engine.tools.demo_run import parse_script, run


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (40, 90, 160)).save(buf, "JPEG")
    return buf.getvalue()


class TinyEngine:
    """Sends a frame every 20 ms (4 per chunk) and answers each control message with a contract 4 reply for the next
    chunk. `close_control_after` closes /control after that many messages; `close_frames_after_s` closes /frames."""

    def __init__(self, close_control_after: int | None = None, close_frames_after_s: float | None = None):
        self.close_control_after = close_control_after
        self.close_frames_after_s = close_frames_after_s
        self.received: list[dict] = []
        self.chunk = 0
        self.jpeg = jpeg()

    async def handler(self, ws):
        if ws.request.path == "/frames":
            t_end = None if self.close_frames_after_s is None else time.monotonic() + self.close_frames_after_s
            i = 0
            while t_end is None or time.monotonic() < t_end:
                self.chunk = i // 4
                await ws.send(pack_frame(self.chunk, i % 4, self.jpeg))
                i += 1
                await asyncio.sleep(0.02)
        elif ws.request.path == "/control":
            async for msg in ws:
                m = json.loads(msg)
                self.received.append(m)
                if self.close_control_after is not None and len(self.received) >= self.close_control_after:
                    return  # returning from the handler closes the socket
                await ws.send(json.dumps({"t_client": m["t_client"], "applies_from_chunk": self.chunk + 1,
                                          "stamps": {"heard": 0}, "unmatched": False, "changes": []}))


def play(tmp_path, monkeypatch, engine: TinyEngine, script: dict, wall_ms: float | None = None) -> tuple[int, dict]:
    """Run the harness against `engine`; returns (exit code, timings.json)."""
    monkeypatch.setattr(demo_run, "get_stats", lambda base: {"state": "GENERATING", "fps_target": 16})
    monkeypatch.setattr(demo_run, "encode_mp4", lambda jpegs, fps, path: None)
    if wall_ms is not None:  # a fixed wall clock: every t_client would be the same without the +1 rule
        monkeypatch.setattr(demo_run, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=time.sleep,
                                                                    time=lambda: wall_ms / 1000))

    async def go():
        async with serve(engine.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            args = types.SimpleNamespace(url=f"http://127.0.0.1:{port}", out=str(tmp_path / "run"), label="t")
            return await run(args, parse_script(script))

    code = asyncio.run(go())
    return code, json.loads((tmp_path / "run" / "timings.json").read_text())


def test_same_time_steps_get_their_own_t_client_and_reply(tmp_path, monkeypatch):
    engine = TinyEngine()
    script = {"end": 0.6, "steps": [{"at": 0.1, "say": "make it winter"}, {"at": 0.1, "say": "make it night"}]}
    code, report = play(tmp_path, monkeypatch, engine, script, wall_ms=1_000_000)
    assert code == 0 and report["complete"] and report["failures"] == []
    assert [m["t_client"] for m in engine.received] == [1_000_000, 1_000_001]
    winter, night = report["commands"]
    assert (winter["say"], winter["t_client"], winter["replied"]) == ("make it winter", 1_000_000, True)
    assert (night["say"], night["t_client"], night["replied"]) == ("make it night", 1_000_001, True)


def test_control_closing_mid_run_saves_the_partial_run_and_fails(tmp_path, monkeypatch, capsys):
    engine = TinyEngine(close_control_after=1)
    script = {"end": 8.0, "steps": [{"at": 0.1, "say": "make it winter"}, {"at": 5.0, "say": "make it night"},
                                    {"at": 6.0, "say": "add a lion"}]}
    t = time.monotonic()
    code, report = play(tmp_path, monkeypatch, engine, script)
    assert code == 1
    assert time.monotonic() - t < 2.0  # stopped at the close, not at the next step (5 s) or the end (8 s)
    assert report["complete"] is False
    assert any("/control connection closed" in r for r in report["failures"])
    assert [c["say"] for c in report["commands"]] == ["make it winter"]  # nothing sent after the close
    assert report["frames_received"] > 0 and report["control_socket"]
    assert [m["t"] for m in report["sheet"]] == [0.0]  # no sheet marks after the stop, which repeat the last frame
    assert "FAILED: the /control connection closed" in (tmp_path / "run" / "summary.txt").read_text()
    assert "FAILED: the /control connection closed" in capsys.readouterr().out


def test_frames_closing_mid_run_stops_sending_commands_at_once(tmp_path, monkeypatch, capsys):
    # Without video nothing can be measured, so the harness must stop changing the shared video right away.
    engine = TinyEngine(close_frames_after_s=0.4)
    script = {"end": 8.0, "steps": [{"at": 0.1, "say": "make it winter"}, {"at": 5.0, "say": "make it night"},
                                    {"at": 6.0, "say": "add a lion"}]}
    t = time.monotonic()
    code, report = play(tmp_path, monkeypatch, engine, script)
    assert code == 1 and report["complete"] is False
    assert time.monotonic() - t < 3.0  # stopped at the close, not at the next step (5 s) or the end (8 s)
    assert [m["text"] for m in engine.received] == ["make it winter"]  # nothing sent after the close
    assert any("/frames connection closed" in r for r in report["failures"])
    assert "/frames closed" in capsys.readouterr().out
