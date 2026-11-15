"""The /app/ routes that serve the demo page from web/. No lifespan, so the engine is never started."""

import asyncio

import pytest
from starlette.testclient import TestClient

from engine.engine import EngineConfig
from engine.server import WEB, build_app

PAGE = {
    "index.html": "text/html",
    "style.css": "text/css",
    "app.js": "text/javascript",
    "lib.js": "text/javascript",
}


@pytest.fixture
def client():
    return TestClient(build_app(EngineConfig(backend="fake")))


def raw_get_status(app, raw: str) -> int:
    """Send a raw path straight to the ASGI app, as a proxy could, because httpx removes "/../" before sending."""
    sent = []

    async def run():
        scope = {"type": "http", "method": "GET", "path": raw, "raw_path": raw.encode(), "query_string": b"",
                 "headers": [], "http_version": "1.1", "scheme": "http", "server": ("t", 80), "root_path": ""}

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(msg):
            sent.append(msg)

        await app(scope, receive, send)

    asyncio.run(run())
    return sent[0]["status"]


def test_app_without_slash_redirects_and_keeps_the_query(client):
    r = client.get("/app?server=127.0.0.1:9000", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert r.headers["location"] == "/app/?server=127.0.0.1:9000"


def test_the_page_files_are_served_fresh_with_their_types(client):
    for path, name in [("/app/", "index.html")] + [(f"/app/{n}", n) for n in PAGE]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].split(";")[0] == PAGE[name], path
        assert r.headers["cache-control"] == "no-cache", path
        assert r.content == (WEB / name).read_bytes(), path


def test_an_edited_page_file_is_served_with_its_new_contents(tmp_path):
    client = TestClient(build_app(EngineConfig(backend="fake"), web_dir=tmp_path))
    (tmp_path / "app.js").write_text("one")
    assert client.get("/app/app.js").text == "one"
    (tmp_path / "app.js").write_text("two")
    assert client.get("/app/app.js").text == "two"


def test_other_files_and_traversal_are_not_served(client):
    for path in ["/app/package.json", "/app/README.md", "/app/node_modules/", "/app/INDEX.HTML",
                 "/app/%2e%2e/engine/server.py", "/app/..%5cengine%5cserver.py"]:
        r = client.get(path)
        assert r.status_code == 404, path
        assert b"import" not in r.content, path


def test_raw_traversal_is_rejected():
    assert raw_get_status(build_app(EngineConfig(backend="fake")), "/app/../engine/server.py") == 404


def test_missing_web_dir_gives_a_404_and_the_engine_still_answers(tmp_path):
    client = TestClient(build_app(EngineConfig(backend="fake"), web_dir=tmp_path / "web"))
    assert client.get("/app/").status_code == 404
    assert client.get("/stats").status_code == 200


def test_engine_viewer_at_root_is_unchanged(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_an_edited_page_file_is_served_with_its_new_contents(tmp_path):
    """A file edited while the server runs is sent as it is now: the server never keeps the first contents."""
    web = tmp_path / "web"
    web.mkdir()
    for name in PAGE:
        (web / name).write_text(f"/* {name} v1 */")
    client = TestClient(build_app(EngineConfig(backend="fake"), web_dir=web))
    assert client.get("/app/app.js").text == "/* app.js v1 */"
    (web / "app.js").write_text("/* app.js v2, longer than v1 */")
    r = client.get("/app/app.js")
    assert r.text == "/* app.js v2, longer than v1 */"
    assert r.headers["cache-control"] == "no-cache"
