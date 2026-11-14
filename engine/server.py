"""HTTP and websocket server for the engine. Run from the repo root:

    python -m engine.server --backend fake

See engine/README.md for the endpoints.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .engine import Engine, EngineConfig
from .lifecycle import Lifecycle
from .messages import BadMessage, parse_json

log = logging.getLogger("engine.server")
STATIC = Path(__file__).resolve().parent / "static"
WEB = Path(__file__).resolve().parent.parent / "web"

# The demo page in web/ served under /app/. Only these four files; the rest of web/ (node_modules, the fake
# server, tests, tools, package files) is never served. Read from disk on each request so a sync of web/
# takes effect without a restart.
APP_FILES = {
    "index.html": "text/html",
    "style.css": "text/css",
    "app.js": "text/javascript",
    "lib.js": "text/javascript",
}

# The Lucid Studio build (a Vite app) served under /studio/. Its asset names are hashed, so any regular file
# inside the folder is served, but never a dotfile or anything that resolves outside the folder.
STUDIO = Path(__file__).resolve().parent.parent / "lucid-studio" / "dist"
STUDIO_TYPES = {
    ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
    ".json": "application/json", ".map": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".ico": "image/x-icon",
    ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".wasm": "application/wasm",
    ".txt": "text/plain", ".webmanifest": "application/manifest+json",
}

FRAME_CLIENT_QUEUE = 8     # frames buffered per /frames client before that client drops frames
CONTROL_CLIENT_QUEUE = 64  # messages buffered per /control client before that client drops messages


class Hub:
    """Fan-out from engine threads to websocket clients. Each client has its own bounded queue.

    A slow client loses messages from its own queue; it never blocks the engine or other clients.
    """

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.frames: dict[WebSocket, asyncio.Queue] = {}
        self.control: dict[WebSocket, asyncio.Queue] = {}
        self.dropped: dict[str, int] = {"frames": 0, "control": 0}

    def publish_frame(self, data: bytes) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fan, self.frames, data, "frames")

    def publish_control(self, message: dict) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fan, self.control, json.dumps(message), "control")

    def _fan(self, clients: dict, item, kind: str) -> None:
        for ws, q in clients.items():
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                self.dropped[kind] += 1
                if self.dropped[kind] % 50 == 1:
                    log.info("slow %s client %s: dropping (%d dropped so far)", kind, ws.client, self.dropped[kind])

    def counts(self) -> dict:
        return {"frames": len(self.frames), "control": len(self.control), "dropped": dict(self.dropped)}


async def _serve(ws: WebSocket, q: asyncio.Queue, send, on_receive) -> None:
    """Run the writer (queue -> socket) and the reader (socket -> on_receive) until either ends."""

    async def writer():
        while True:
            await send(await q.get())

    async def reader():
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            await on_receive(msg)

    tasks = [asyncio.create_task(writer()), asyncio.create_task(reader())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError, OSError):
                await t


def build_app(config: EngineConfig, web_dir: Path = WEB, studio_dir: Path = STUDIO) -> Starlette:
    hub = Hub()
    engine = Engine(config, hub.publish_frame, hub.publish_control)
    engine.client_counts = hub.counts

    async def frames_ws(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=FRAME_CLIENT_QUEUE)
        hub.frames[ws] = q
        log.info("frames client connected: %s (%d total)", ws.client, len(hub.frames))

        async def ignore(_msg):
            pass

        try:
            await _serve(ws, q, ws.send_bytes, ignore)
        finally:
            hub.frames.pop(ws, None)
            log.info("frames client gone: %s", ws.client)

    async def control_ws(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=CONTROL_CLIENT_QUEUE)
        hub.control[ws] = q
        log.info("control client connected: %s (%d total)", ws.client, len(hub.control))

        def put(message: dict) -> None:  # event loop only
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(json.dumps(message))

        loop = asyncio.get_running_loop()

        def reply(message: dict) -> None:  # any thread; the reader worker uses it for errors
            log.info("control: error reply to %s: %s", ws.client, message.get("error"))
            loop.call_soon_threadsafe(put, message)

        async def on_receive(msg):
            heard_ns = time.monotonic_ns()
            try:
                if msg.get("text") is None:
                    raise BadMessage("send JSON text messages, not binary")
                engine.handle_control(parse_json(msg["text"]), heard_ns, reply)
            except BadMessage as exc:
                log.info("control: rejected message from %s: %s", ws.client, exc)
                put({"error": str(exc)})

        try:
            await _serve(ws, q, ws.send_text, on_receive)
        finally:
            hub.control.pop(ws, None)
            log.info("control client gone: %s", ws.client)

    async def index(_request: Request):
        return FileResponse(STATIC / "index.html")

    async def app_redirect(request: Request):
        query = request.url.query
        return RedirectResponse("/app/" + (f"?{query}" if query else ""))

    def app_file(request: Request):  # sync: Starlette runs it in a worker thread, so the disk read never blocks the loop
        name = request.path_params["name"] or "index.html"
        if name not in APP_FILES:
            return PlainTextResponse("not found", status_code=404)
        if not web_dir.is_dir():
            return PlainTextResponse(f"the demo page is not on this machine: {web_dir} is missing", status_code=404)
        try:
            body = (web_dir / name).read_bytes()
        except FileNotFoundError:
            return PlainTextResponse(f"web/{name} is missing on this machine", status_code=404)
        return Response(body, media_type=APP_FILES[name], headers={"Cache-Control": "no-cache"})

    async def studio_redirect(request: Request):
        query = request.url.query
        return RedirectResponse("/studio/" + (f"?{query}" if query else ""))

    def studio_file(request: Request):  # sync for the same reason as app_file
        root = studio_dir.resolve()
        if not root.is_dir():
            return PlainTextResponse(f"Lucid Studio build not found at {studio_dir}", status_code=404)
        name = request.path_params["name"] or "index.html"
        parts = name.split("/")
        if "\\" in name or any(not part or part.startswith(".") for part in parts):
            return PlainTextResponse("not found", status_code=404)
        path = (root / name).resolve()  # follows symlinks, so one that points outside fails the check below
        if not path.is_relative_to(root) or not path.is_file():
            return PlainTextResponse("not found", status_code=404)
        hashed = parts[0] == "assets"
        return FileResponse(path, media_type=STUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                            headers={"Cache-Control": "public, max-age=31536000, immutable" if hashed else "no-cache"})

    async def stats(_request: Request):
        return JSONResponse(engine.stats())

    async def healthz(_request: Request):
        state = engine.lifecycle.state
        ok = state not in (Lifecycle.FAILED, Lifecycle.STOPPED)
        return JSONResponse({"ok": ok, "state": state.value}, status_code=200 if ok else 503)

    async def reset(_request: Request):
        heard_ns = time.monotonic_ns()
        try:
            done = await asyncio.wrap_future(engine.reset(heard_ns))
        except BadMessage as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        if done is None:
            return JSONResponse({"error": "reset refused; see the server log"}, status_code=409)
        finished = await asyncio.to_thread(done.wait, 15.0)
        return JSONResponse({"ok": finished, "state": engine.lifecycle.state.value},
                            status_code=200 if finished else 504)

    async def target(request: Request):
        heard_ns = time.monotonic_ns()
        try:
            rec = engine.handle_target(parse_json(await request.body()), heard_ns)
        except BadMessage as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "command": rec.seq, "stage": rec.stage.value})

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        hub.loop = asyncio.get_running_loop()
        engine.start()
        try:
            yield
        finally:
            await asyncio.to_thread(engine.stop)

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/app", app_redirect),
            Route("/app/{name:path}", app_file),
            Route("/studio", studio_redirect),
            Route("/studio/{name:path}", studio_file),
            Route("/stats", stats),
            Route("/healthz", healthz),
            Route("/reset", reset, methods=["POST"]),
            Route("/target", target, methods=["POST"]),
            WebSocketRoute("/frames", frames_ws),
            WebSocketRoute("/control", control_ws),
        ],
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.hub = hub
    return app


def _option_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m engine.server", description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="fake", help="fake or causal_forcing")
    p.add_argument("--reader", default="standin", choices=["standin", "commands"],
                   help="command reader: the engine's stand-in, or commands.service.CommandService")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--steps", type=int, default=0,
                   help="denoising steps per chunk; 0 uses the schedule the backend's checkpoint was distilled for")
    p.add_argument("--device", default="cuda:0", help="generation device (the fake backend ignores it)")
    p.add_argument("--decode-device", default="cuda:0", help="decode device (the fake backend ignores it)")
    p.add_argument("--base-prompt", default=EngineConfig.base_prompt,
                   help="starting prompt, used when the reader has no committed prompt")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--precompute-file", help="text file of prompts, one per line, encoded at startup")
    p.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE",
                   help="option passed to the backend's load(); repeatable")
    p.add_argument("--lead-margin", type=float, default=EngineConfig.lead_margin,
                   help="safety factor on how much video generation keeps ahead of the playhead")
    p.add_argument("--reanchor", default="off", choices=["off", "on"],
                   help="re-anchor the model at steers and at intervals (steering.reanchor policy); needs a backend "
                        "with reanchor(strength)")
    p.add_argument("--studio-dir", default=str(STUDIO), help="Lucid Studio build folder served at /studio/")
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    options = {}
    for item in args.backend_opt:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--backend-opt expects KEY=VALUE, got {item!r}")
        options[key] = _option_value(value)
    config = EngineConfig(backend=args.backend, reader=args.reader, width=args.width, height=args.height, steps=args.steps,
                          device=args.device, decode_device=args.decode_device, backend_options=options,
                          base_prompt=args.base_prompt, jpeg_quality=args.jpeg_quality,
                          precompute_file=args.precompute_file, lead_margin=args.lead_margin,
                          reanchor=args.reanchor)
    app = build_app(config, studio_dir=Path(args.studio_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
