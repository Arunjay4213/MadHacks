import os

import pytest
import torch
from PIL import Image

from steering.eval import run_engine as re_
from steering.eval import scenarios as sc
from steering.eval import sheet
from steering.tests.test_eval_run_engine import FakeBackend

pytestmark = pytest.mark.tools  # a tool's tests: run with pytest -m tools (pytest.ini)


def _frames(t: int = 20, h: int = 24, w: int = 32) -> torch.Tensor:
    """Mid-grey frames whose brightness steps up a little per frame."""
    return (0.3 + 0.01 * torch.arange(t).view(t, 1, 1, 1)).expand(t, 3, h, w).clone()


def _tile_origin(k: int, cols: int, tw: int, th: int) -> tuple[int, int]:
    return sheet.PAD + (k % cols) * (tw + sheet.PAD), sheet.PAD + (k // cols) * (th + sheet.PAD)


def test_sheet_size_and_tiling():
    img = sheet.contact_sheet(_frames(20), every=2, cols=4)
    # 10 tiles in 4 columns -> 3 rows, tiles at native 32x24.
    assert img.size == (2 + 4 * 34, 2 + 3 * 26)
    # Tile 0 (frame 0) and tile 1 (frame 2) carry the frame's colour away from the label.
    x, y = _tile_origin(1, 4, 32, 24)
    assert img.getpixel((x + 28, y + 20)) == (82, 82, 82)  # 0.32 * 255


def test_sheet_downscales_to_max_width():
    img = sheet.contact_sheet(_frames(16, 90, 160), cols=8, max_width=400)
    assert img.width <= 400
    tw = (400 - 2 * 9) // 8
    assert img.width == 2 + 8 * (tw + 2)


def test_sheet_fewer_frames_than_cols():
    assert sheet.contact_sheet(_frames(3), cols=8).size == (2 + 3 * 34, 2 + 26)


def test_marks_draw_coloured_borders():
    img = sheet.contact_sheet(_frames(8), cols=4, mark=[2], mark_cuts=[5, 2])
    # Frame 2: red outer border, blue border inside it.
    x, y = _tile_origin(2, 4, 32, 24)
    assert img.getpixel((x, y + 12)) == sheet.STEER_COLOUR
    assert img.getpixel((x + 2, y + 12)) == sheet.CUT_COLOUR
    # Frame 5: blue border at the edge; frame 1 has none.
    x, y = _tile_origin(5, 4, 32, 24)
    assert img.getpixel((x, y + 12)) == sheet.CUT_COLOUR
    x, y = _tile_origin(1, 4, 32, 24)
    assert img.getpixel((x, y + 12)) == (79, 79, 79)


def test_marked_frames_shown_off_grid():
    img = sheet.contact_sheet(_frames(10), every=4, cols=8, mark=[5])
    # frames 0, 4, 5, 8 -> 4 tiles, frame 5 is the third.
    assert img.size == (2 + 4 * 34, 2 + 26)
    x, y = _tile_origin(2, 4, 32, 24)
    assert img.getpixel((x, y + 12)) == sheet.STEER_COLOUR


def test_labels_and_bad_input():
    sheet.contact_sheet(_frames(3), labels=["a", "b", "c"])
    with pytest.raises(ValueError):
        sheet.contact_sheet(_frames(3), labels=["a"])
    with pytest.raises(ValueError):
        sheet.contact_sheet(torch.zeros(3, 4, 4))
    with pytest.raises(ValueError):
        sheet.load_clip("x.gif")


def test_pt_round_trip(tmp_path):
    f = torch.rand(5, 3, 8, 8)
    path = str(tmp_path / "a.frames.pt")
    torch.save(f, path)
    assert torch.equal(sheet.load_clip(path), f)


def test_mp4_round_trip_if_imageio(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    f = _frames(6, 32, 32)
    path = str(tmp_path / "a.mp4")
    assert re_.write_video(path, f, 16.0) == "imageio"
    back = sheet.load_clip(path)
    assert back.shape == (6, 3, 32, 32)
    assert (back - f).abs().mean() < 0.03


def test_read_report_matches_real_report(tmp_path):
    res = sc.run(sc.default_scenarios(4)[2], sc.FakeModel(), sc.fake_encode, 16.0)  # interrupt
    path = tmp_path / "interrupt.report.txt"
    path.write_text(res.report() + "\n")
    rows = sheet.read_report(str(path))
    assert rows == [(4, 16, "final"), (6, 24, "final")]
    assert sheet.steer_frames(str(path)) == [16, 24]
    assert sheet.read_report(str(tmp_path / "missing.txt")) == []
    (tmp_path / "junk.report.txt").write_text("nothing here\n")
    assert sheet.steer_frames(str(tmp_path / "junk.report.txt")) == []


def test_read_report_skips_unapplied(tmp_path):
    line = f"   5    20  {'+x r4':<44} unchanged    -                 -        -"
    (tmp_path / "r.txt").write_text("header\n" + line + "\n")
    assert sheet.read_report(str(tmp_path / "r.txt")) == [(5, 20, "unchanged")]
    assert sheet.steer_frames(str(tmp_path / "r.txt")) == []


def test_cli_single_clip(tmp_path, capsys):
    path = str(tmp_path / "c.frames.pt")
    torch.save(_frames(12), path)
    assert sheet.main([path, "--mark", "4,8", "--cuts", "--every", "2"]) == 0
    img = Image.open(tmp_path / "c.sheet.png")
    assert img.size[0] > 0
    assert "steer marks [4, 8]" in capsys.readouterr().out


def test_cli_dir_on_run_engine_output(tmp_path, capsys):
    scen = [s for s in sc.default_scenarios(4) if s.name in ("single", "interrupt")]
    _, failures = re_.run_all(FakeBackend(), scen, str(tmp_path), 16.0,
                              save_frames=True, video=False)
    assert failures == 0
    (tmp_path / "broken.frames.pt").write_bytes(b"not a tensor")
    assert sheet.main(["--dir", str(tmp_path), "--cuts"]) == 1  # broken one fails soft
    out = capsys.readouterr()
    assert "broken: FAILED" in out.err
    for name in ("single", "interrupt"):
        assert os.path.exists(tmp_path / f"{name}.sheet.png")
    # FakeBackend makes 2 frames per step: interrupt's messages at steps 4 and 6.
    assert "interrupt.sheet.png: 24 frames, steer marks [8, 12]" in out.out


def test_cli_needs_clip_or_dir():
    with pytest.raises(SystemExit):
        sheet.main([])
