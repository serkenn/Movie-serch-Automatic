"""外部ソースから動画を取得するモジュール。"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


def collect_video_files(video_dir: str | Path) -> set[Path]:
    """指定ディレクトリ配下の動画ファイルを再帰的に収集する。"""
    base = Path(video_dir)
    if not base.exists():
        return set()
    return {
        p.resolve()
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    }


class VideoIngestor:
    """Telegram/Magnet などから動画を取得する。"""

    def __init__(self, download_dir: str):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        telegram_urls: list[str] | None = None,
        magnets: list[str] | None = None,
        source_file: str | None = None,
        proxy: str | None = None,
    ) -> list[Path]:
        """指定ソースから動画を取得し、新規動画ファイル一覧を返す。"""
        telegram = list(telegram_urls or [])
        magnet_links = list(magnets or [])

        if source_file:
            t, m = self._load_sources(source_file)
            telegram.extend(t)
            magnet_links.extend(m)

        if not telegram and not magnet_links:
            return []

        before = collect_video_files(self.download_dir)
        env = self._build_proxy_env(proxy)

        if telegram:
            self._download_telegram(telegram, env)
        if magnet_links:
            self._download_magnet(magnet_links, env)

        after = collect_video_files(self.download_dir)
        new_files = sorted(after - before)
        logger.info("取得完了: 新規動画 %d 件", len(new_files))
        return new_files

    def _download_telegram(self, urls: list[str], env: dict[str, str]) -> None:
        if not shutil.which("yt-dlp"):
            raise RuntimeError(
                "Telegram 取得には yt-dlp が必要です。インストールしてください。"
            )

        for url in urls:
            cmd = [
                "yt-dlp",
                "--no-progress",
                "--no-warnings",
                "--restrict-filenames",
                "-P",
                str(self.download_dir),
                url,
            ]
            self._run_cmd(cmd, env, f"Telegram 取得失敗: {url}")

    def _download_magnet(self, magnets: list[str], env: dict[str, str]) -> None:
        if not shutil.which("aria2c"):
            raise RuntimeError(
                "Magnet 取得には aria2c が必要です。インストールしてください。"
            )

        for magnet in magnets:
            cmd = [
                "aria2c",
                "--dir",
                str(self.download_dir),
                "--seed-time=0",
                "--summary-interval=0",
                magnet,
            ]
            self._run_cmd(cmd, env, "Magnet 取得失敗")

    @staticmethod
    def _run_cmd(cmd: list[str], env: dict[str, str], err_msg: str) -> None:
        logger.info("実行: %s", shlex.join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            raise RuntimeError(f"{err_msg}: {stderr}")

    @staticmethod
    def _build_proxy_env(proxy: str | None) -> dict[str, str]:
        env = os.environ.copy()
        if not proxy:
            return env
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["ALL_PROXY"] = proxy
        return env

    @staticmethod
    def _load_sources(source_file: str) -> tuple[list[str], list[str]]:
        telegram_urls: list[str] = []
        magnets: list[str] = []

        with open(source_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("magnet:"):
                    magnets.append(line)
                    continue
                if "t.me/" in line or line.startswith("telegram:"):
                    clean = line.replace("telegram:", "", 1).strip()
                    telegram_urls.append(clean)

        return telegram_urls, magnets
