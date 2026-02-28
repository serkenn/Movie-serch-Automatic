"""メディア整理モジュール。

Fantia の投稿IDをファイル名から抽出し、投稿タイトル/作者名ベースで
ファイルを整理する。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SUPPORTED_EXTS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".zip",
    ".rar",
    ".7z",
    ".pdf",
}

INVALID_CHARS = r'<>:"/\|?*'
FANTIA_ID_PATTERNS = [
    re.compile(r"fantia[-_]?posts[-_]?(\d+)", re.IGNORECASE),
    re.compile(r"posts[-_]?(\d+)", re.IGNORECASE),
]


@dataclass
class ParsedInfo:
    site: str
    post_id: str


def sanitize_name(name: str, max_len: int = 140) -> str:
    """ファイルシステム安全な名前に変換する。"""
    name = re.sub(r"\s+", " ", name).strip()
    table = str.maketrans({c: "_" for c in INVALID_CHARS})
    name = name.translate(table)
    name = name.replace("\0", "_")
    name = name.rstrip(" .")
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip(" .")
    return name or "untitled"


def parse_filename_for_site(filename: str) -> Optional[ParsedInfo]:
    """ファイル名から投稿IDを抽出する。"""
    stem = Path(filename).stem
    for pattern in FANTIA_ID_PATTERNS:
        matched = pattern.search(stem)
        if matched:
            return ParsedInfo(site="fantia", post_id=matched.group(1))
    return None


def fetch_url(url: str, timeout: int = 20) -> str:
    """URLからHTMLを取得する。"""
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _extract_og_title(html: str) -> Optional[str]:
    patterns = [
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_title_tag(html: str) -> Optional[str]:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title or None


def _walk_author_name(node) -> Optional[str]:
    if isinstance(node, list):
        for item in node:
            found = _walk_author_name(item)
            if found:
                return found
        return None

    if isinstance(node, dict):
        author = node.get("author")
        if isinstance(author, dict):
            name = author.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(author, list):
            for author_item in author:
                if isinstance(author_item, dict):
                    name = author_item.get("name")
                    if isinstance(name, str) and name.strip():
                        return name.strip()
        for value in node.values():
            found = _walk_author_name(value)
            if found:
                return found
    return None


def _extract_author_from_jsonld(html: str) -> Optional[str]:
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for script in scripts:
        script = script.strip()
        if not script:
            continue
        try:
            data = json.loads(script)
        except json.JSONDecodeError:
            continue
        author = _walk_author_name(data)
        if author:
            return author
    return None


def _cleanup_fantia_title(raw_title: str) -> str:
    title = raw_title.strip()
    title = re.sub(r"\s*\|\s*Fantia.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*-\s*Fantia.*$", "", title, flags=re.IGNORECASE)
    return title.strip() or raw_title.strip()


def get_fantia_metadata(post_id: str) -> tuple[Optional[str], Optional[str], str]:
    """Fantia投稿からタイトル/作者を取得する。"""
    url = f"https://fantia.jp/posts/{post_id}"
    try:
        html = fetch_url(url)
    except (HTTPError, URLError, TimeoutError) as exc:
        return None, None, f"fetch_error: {exc}"

    title = _extract_og_title(html) or _extract_title_tag(html)
    author = _extract_author_from_jsonld(html)
    if title:
        title = _cleanup_fantia_title(title)
    return title, author, "ok"


def ensure_unique_path(dest: Path) -> Path:
    """衝突時に連番付きの新規パスを返す。"""
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    index = 2
    while True:
        candidate = dest.with_name(f"{stem} ({index}){suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def move_file(src: Path, dst: Path, dry_run: bool) -> tuple[Path, Path]:
    """ファイル移動（dry-run対応）を行う。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    unique_dst = ensure_unique_path(dst)
    if not dry_run:
        shutil.move(str(src), str(unique_dst))
    return src, unique_dst


def infer_creator_fallback(file_path: Path, input_root: Path) -> str:
    """作者名取得失敗時のフォルダ名を推定する。"""
    try:
        rel = file_path.relative_to(input_root)
        if len(rel.parts) >= 2:
            return sanitize_name(rel.parts[0], max_len=80)
    except ValueError:
        pass
    parent = file_path.parent.name
    return sanitize_name(parent or "unknown_creator", max_len=80)


def process_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    unknown_root: Path,
    dry_run: bool,
) -> tuple[Path, Path]:
    """単一ファイルを整理して移動先を返す。"""
    parsed = parse_filename_for_site(file_path.name)
    fallback_creator = infer_creator_fallback(file_path, input_root)

    if not parsed:
        dst = unknown_root / fallback_creator / file_path.name
        return move_file(file_path, dst, dry_run=dry_run)

    if parsed.site == "fantia":
        title, author, status = get_fantia_metadata(parsed.post_id)
        creator = sanitize_name(author, max_len=80) if author else fallback_creator
        if status != "ok" or not title:
            dst = unknown_root / creator / file_path.name
            return move_file(file_path, dst, dry_run=dry_run)

        safe_title = sanitize_name(title)
        new_name = f"{safe_title}{file_path.suffix.lower()}"
        dst = output_root / "fantia" / creator / new_name
        return move_file(file_path, dst, dry_run=dry_run)

    dst = unknown_root / fallback_creator / file_path.name
    return move_file(file_path, dst, dry_run=dry_run)


def iter_target_files(root: Path):
    """対象拡張子ファイルを再帰列挙する。"""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        yield path


def organize_media(
    input_root: str | Path,
    output_root: str | Path,
    unknown_root: str | Path,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """メディア整理を実行し、移動元/移動先の一覧を返す。"""
    input_dir = Path(input_root).expanduser().resolve()
    output_dir = Path(output_root).expanduser().resolve()
    unknown_dir = Path(unknown_root).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    files = list(iter_target_files(input_dir))
    moves: list[tuple[Path, Path]] = []
    for file_path in files:
        src, dst = process_file(
            file_path=file_path,
            input_root=input_dir,
            output_root=output_dir,
            unknown_root=unknown_dir,
            dry_run=dry_run,
        )
        moves.append((src, dst))
    return moves
