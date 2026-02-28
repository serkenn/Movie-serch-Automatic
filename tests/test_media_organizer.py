"""メディア整理モジュールのテスト"""

from src import media_organizer as mo


def test_sanitize_name_basic():
    name = mo.sanitize_name(' test<>:"/\\|?*name  ')
    assert "<" not in name
    assert ">" not in name
    assert name.startswith("test")


def test_parse_filename_for_site():
    parsed = mo.parse_filename_for_site("fantia-posts-3035118.mp4")
    assert parsed is not None
    assert parsed.site == "fantia"
    assert parsed.post_id == "3035118"


def test_organize_media_success_move(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    unknown_dir = tmp_path / "unknown"
    in_dir.mkdir()
    f = in_dir / "fantia-posts-1234.mp4"
    f.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        mo,
        "get_fantia_metadata",
        lambda post_id: ("テストタイトル", "作者A", "ok"),
    )

    moves = mo.organize_media(in_dir, out_dir, unknown_dir, dry_run=False)
    assert len(moves) == 1
    expected = out_dir / "fantia" / "作者A" / "テストタイトル.mp4"
    assert expected.exists()
    assert not f.exists()


def test_organize_media_unresolved_goes_unknown(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    unknown_dir = tmp_path / "unknown"
    in_dir.mkdir()
    f = in_dir / "fantia-posts-1234.mp4"
    f.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        mo,
        "get_fantia_metadata",
        lambda post_id: (None, None, "fetch_error"),
    )

    mo.organize_media(in_dir, out_dir, unknown_dir, dry_run=False)
    moved = unknown_dir / "in" / "fantia-posts-1234.mp4"
    assert moved.exists()


def test_organize_media_dry_run(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    unknown_dir = tmp_path / "unknown"
    in_dir.mkdir()
    f = in_dir / "fantia-posts-1234.mp4"
    f.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        mo,
        "get_fantia_metadata",
        lambda post_id: ("Title", "Creator", "ok"),
    )

    moves = mo.organize_media(in_dir, out_dir, unknown_dir, dry_run=True)
    assert len(moves) == 1
    assert f.exists()
