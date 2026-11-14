"""The /app/ routes that serve the demo page from web/. No lifespan, so the engine is never started."""

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


def test_app_without_slash_redirects(client):
    r = client.get("/app", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert r.headers["location"] == "/app/"
    r = client.get("/app?server=127.0.0.1:9000", follow_redirects=False)
    assert r.headers["location"] == "/app/?server=127.0.0.1:9000"


def test_app_slash_serves_index(client):
    r = client.get("/app/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-cache"
    assert r.content == (WEB / "index.html").read_bytes()


@pytest.mark.parametrize("name", sorted(PAGE))
def test_each_page_file_with_its_content_type(client, name):
    r = client.get(f"/app/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] == PAGE[name]
    assert r.headers["cache-control"] == "no-cache"
    assert r.content == (WEB / name).read_bytes()


@pytest.mark.parametrize("path", [
    "/app/package.json", "/app/package-lock.json", "/app/fake-server.mjs", "/app/README.md", "/app/PLAN.md",
    "/app/node_modules/", "/app/test/", "/app/tools/make-test-pattern.py", "/app/test-pattern/",
    "/app/INDEX.HTML", "/app/index.html/", "/app/%69ndex.html.bak",
])
def test_other_files_in_web_are_not_served(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path", [
    "/app/../engine/server.py",
    "/app/%2e%2e/engine/server.py",
    "/app/..%2fengine%2fserver.py",
    "/app/%2e%2e%2fengine%2fserver.py",
    "/app/..%5cengine%5cserver.py",
])
def test_traversal_is_not_served(client, path):
    r = client.get(path)
    assert r.status_code == 404
    assert b"import" not in r.content


def test_traversal_reaches_the_handler_and_is_rejected():
    # httpx removes "/../" before sending, so send the raw path straight to the ASGI app as a proxy could.
    import asyncio

    app = build_app(EngineConfig(backend="fake"))
    sent = []

    async def run(raw: str):
        scope = {"type": "http", "method": "GET", "path": raw, "raw_path": raw.encode(), "query_string": b"",
                 "headers": [], "http_version": "1.1", "scheme": "http", "server": ("t", 80), "root_path": ""}

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(msg):
            sent.append(msg)

        await app(scope, receive, send)

    asyncio.run(run("/app/../engine/server.py"))
    assert sent[0]["status"] == 404


def test_missing_web_dir_gives_a_clear_404(tmp_path):
    client = TestClient(build_app(EngineConfig(backend="fake"), web_dir=tmp_path / "web"))
    r = client.get("/app/")
    assert r.status_code == 404
    assert "missing" in r.text
    assert client.get("/stats").status_code == 200


def test_files_are_read_on_each_request(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "app.js").write_text("one")
    client = TestClient(build_app(EngineConfig(backend="fake"), web_dir=web))
    assert client.get("/app/app.js").text == "one"
    (web / "app.js").write_text("two")
    assert client.get("/app/app.js").text == "two"


def test_engine_viewer_at_root_is_unchanged(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
