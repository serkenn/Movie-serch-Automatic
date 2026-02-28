"""取得モジュールのテスト"""

from src.ingest import VideoIngestor, collect_video_files


def test_collect_video_files_recursive(tmp_path):
    (tmp_path / "a.mp4").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.mkv").write_text("x", encoding="utf-8")
    (nested / "c.txt").write_text("x", encoding="utf-8")

    files = collect_video_files(tmp_path)
    names = {p.name for p in files}
    assert names == {"a.mp4", "b.mkv"}


def test_load_sources(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text(
        "\n".join([
            "# comment",
            "https://t.me/example/123",
            "telegram:https://t.me/example/456",
            "magnet:?xt=urn:btih:AAAA",
        ]),
        encoding="utf-8",
    )

    telegram, magnets = VideoIngestor._load_sources(str(src))
    assert len(telegram) == 2
    assert len(magnets) == 1
    assert magnets[0].startswith("magnet:")


def test_build_proxy_env():
    env = VideoIngestor._build_proxy_env("socks5h://127.0.0.1:1080")
    assert env["HTTP_PROXY"].startswith("socks5h://")
    assert env["HTTPS_PROXY"].startswith("socks5h://")
    assert env["ALL_PROXY"].startswith("socks5h://")


def test_ingest_detects_new_files(tmp_path):
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    (download_dir / "old.mp4").write_text("old", encoding="utf-8")

    ingestor = VideoIngestor(str(download_dir))

    def fake_tg(urls, env):
        (download_dir / "new_from_tg.mp4").write_text("new", encoding="utf-8")

    def fake_magnet(magnets, env):
        (download_dir / "new_from_magnet.mkv").write_text("new", encoding="utf-8")

    ingestor._download_telegram = fake_tg
    ingestor._download_magnet = fake_magnet

    new_files = ingestor.ingest(
        telegram_urls=["https://t.me/example/1"],
        magnets=["magnet:?xt=urn:btih:BBBB"],
    )
    names = {p.name for p in new_files}
    assert names == {"new_from_tg.mp4", "new_from_magnet.mkv"}
