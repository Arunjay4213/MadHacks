"""The /studio/ routes that serve the Lucid Studio build. A temp folder stands in for the build."""

import asyncio

import pytest
from starlette.testclient import TestClient

from engine.engine import EngineConfig
from engine.server import build_app, parse_args


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


def test_studio_without_slash_redirects(client):
    r = client.get("/studio", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert r.headers["location"] == "/studio/"


def test_studio_slash_serves_index_uncached(client):
    r = client.get("/studio/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-cache"
    assert "studio" in r.text
    assert client.get("/studio/index.html").headers["cache-control"] == "no-cache"


@pytest.mark.parametrize("path,ctype", [
    ("/studio/assets/x.js", "text/javascript"),
    ("/studio/assets/x.css", "text/css"),
    ("/studio/logo.svg", "image/svg+xml"),
])
def test_files_with_content_types(client, path, ctype):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] == ctype


def test_hashed_assets_may_be_cached(client):
    assert "max-age" in client.get("/studio/assets/x.js").headers["cache-control"]
    assert client.get("/studio/logo.svg").headers["cache-control"] == "no-cache"


@pytest.mark.parametrize("path", [
    "/studio/.env", "/studio/assets/.hidden.js", "/studio/missing.js", "/studio/assets/", "/studio/assets",
    "/studio/assets//x.js", "/studio/%2e%2e/outside.txt", "/studio/..%2foutside.txt", "/studio/..%5coutside.txt",
    "/studio/%2fetc%2fpasswd",
])
def test_dotfiles_missing_and_traversal_give_404(client, path):
    r = client.get(path)
    assert r.status_code == 404
    assert "outside" not in r.text and "SECRET" not in r.text


def test_raw_traversal_reaches_the_handler_and_is_rejected(dist):
    # httpx removes "/../" before sending, so send the raw path straight to the ASGI app.
    app = build_app(EngineConfig(backend="fake"), studio_dir=dist)
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


def test_symlink_pointing_outside_is_rejected(dist):
    (dist / "assets" / "leak.txt").symlink_to(dist.parent / "outside.txt")
    (dist / "inside.js").symlink_to(dist / "assets" / "x.js")
    client = TestClient(build_app(EngineConfig(backend="fake"), studio_dir=dist))
    assert client.get("/studio/assets/leak.txt").status_code == 404
    assert client.get("/studio/inside.js").status_code == 200


def test_missing_build_gives_a_clear_404(tmp_path):
    missing = tmp_path / "nope" / "dist"
    client = TestClient(build_app(EngineConfig(backend="fake"), studio_dir=missing))
    r = client.get("/studio/")
    assert r.status_code == 404
    assert r.text == f"Lucid Studio build not found at {missing}"
    assert client.get("/stats").status_code == 200


def test_studio_dir_flag():
    assert parse_args(["--studio-dir", "/x/dist"]).studio_dir == "/x/dist"
    assert parse_args([]).studio_dir.endswith("lucid-studio/dist")
