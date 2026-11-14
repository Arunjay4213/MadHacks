"""The /studio/ routes that serve the Lucid Studio build. A temp folder stands in for the build."""

import asyncio

import pytest
from starlette.testclient import TestClient

from engine.engine import EngineConfig
from engine.server import build_app


@pytest.fixture
def dist(tmp_path):
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>studio</title>")
    (d / "assets" / "x.js").write_text("export const x = 1;")
    (d / "assets" / "x.css").write_text("body{}")
    (d / "logo.svg").write_text("<svg/>")
    (d / ".env").write_text("SECRET=1")
    (d / "assets" / ".hidden.js").write_text("secret")
    (tmp_path / "outside.txt").write_text("outside")
    return d


@pytest.fixture
def client(dist):
    return TestClient(build_app(EngineConfig(backend="fake"), studio_dir=dist))


def test_the_build_is_served(client):
    r = client.get("/studio", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308) and r.headers["location"] == "/studio/"
    r = client.get("/studio/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "studio" in r.text
    # The page and unhashed files must be fresh after a rebuild; hashed assets may be cached.
    assert r.headers["cache-control"] == "no-cache"
    assert client.get("/studio/logo.svg").headers["cache-control"] == "no-cache"
    assert "max-age" in client.get("/studio/assets/x.js").headers["cache-control"]
    for path, ctype in [("/studio/assets/x.js", "text/javascript"), ("/studio/assets/x.css", "text/css"),
                        ("/studio/logo.svg", "image/svg+xml")]:
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].split(";")[0] == ctype, path


def test_dotfiles_missing_and_traversal_give_404(client):
    for path in ["/studio/.env", "/studio/assets/.hidden.js", "/studio/missing.js", "/studio/assets/",
                 "/studio/%2e%2e/outside.txt", "/studio/..%5coutside.txt"]:
        r = client.get(path)
        assert r.status_code == 404, path
        assert "outside" not in r.text and "SECRET" not in r.text, path


def test_raw_traversal_and_outside_symlinks_are_rejected(dist):
    (dist / "assets" / "leak.txt").symlink_to(dist.parent / "outside.txt")
    (dist / "inside.js").symlink_to(dist / "assets" / "x.js")
    app = build_app(EngineConfig(backend="fake"), studio_dir=dist)
    client = TestClient(app)
    assert client.get("/studio/assets/leak.txt").status_code == 404
    assert client.get("/studio/inside.js").status_code == 200

    # httpx removes "/../" before sending, so send the raw path straight to the ASGI app.
    sent = []

    async def run(raw: str):
        scope = {"type": "http", "method": "GET", "path": raw, "raw_path": raw.encode(), "query_string": b"",
                 "headers": [], "http_version": "1.1", "scheme": "http", "server": ("t", 80), "root_path": ""}

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(msg):
            sent.append(msg)

        await app(scope, receive, send)

    asyncio.run(run("/studio/../outside.txt"))
    assert sent[0]["status"] == 404


def test_missing_build_gives_a_404_and_the_engine_still_answers(tmp_path):
    client = TestClient(build_app(EngineConfig(backend="fake"), studio_dir=tmp_path / "nope" / "dist"))
    r = client.get("/studio/")
    assert r.status_code == 404 and "not found" in r.text
    assert client.get("/stats").status_code == 200
