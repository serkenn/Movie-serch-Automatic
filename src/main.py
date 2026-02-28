"""CLI エントリーポイント - 動画出演者分析ツール"""

import logging
import sys
from pathlib import Path

import click

from src.network_status import get_network_status, get_traffic_status
from src.pipeline import AnalysisPipeline
from src.output.reporter import append_csv_log as append_csv_history
from src.output.reporter import save_results, print_summary


def _setup_logging(verbose: bool = False) -> None:
    """ロギングの初期化"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-V", is_flag=True, help="デバッグログを表示する")
def cli(verbose):
    """アリスホリック動画 出演者分析ツール"""
    _setup_logging(verbose=verbose)


@cli.command()
@click.option("--video", "-v", type=click.Path(exists=True),
              help="解析対象の動画ファイル")
@click.option("--dir", "-d", "video_dir", type=click.Path(exists=True),
              help="解析対象の動画フォルダ")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--output", "-o", default="output",
              help="出力ディレクトリ")
@click.option("--format", "-f", "fmt", default="json",
              type=click.Choice(["json", "csv", "both"]),
              help="出力形式")
@click.option("--visual/--no-visual", default=False,
              help="視覚分析を有効にする")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
def analyze(video, video_dir, config, output, fmt, visual, hf_token):
    """動画を解析して出演者を判定する。"""
    from src.preflight import run_preflight, PreflightError

    if not video and not video_dir:
        click.echo("エラー: --video または --dir を指定してください。", err=True)
        sys.exit(1)

    try:
        run_preflight(check_gpu_available=visual)
    except PreflightError as e:
        click.echo(f"エラー: {e}", err=True)
        sys.exit(1)

    click.echo("パイプラインを初期化中...")
    pipeline = AnalysisPipeline(config_path=config)
    pipeline.setup(enable_visual=visual, hf_token=hf_token)

    results = []

    if video:
        click.echo(f"解析中: {video}")
        result = pipeline.analyze_video(video)
        results.append(result)
    elif video_dir:
        click.echo(f"フォルダ解析中: {video_dir}")
        results = pipeline.analyze_batch(video_dir)

    if not results:
        click.echo("解析対象の動画が見つかりませんでした。")
        return

    # コンソール出力
    print_summary(results)

    # ファイル出力
    saved = save_results(results, output, fmt=fmt)
    for path in saved:
        click.echo(f"結果を保存しました: {path}")


@cli.command()
@click.option("--dir", "-d", "ref_dir", default="data/reference_voices",
              help="基準音声ディレクトリ")
def list_speakers(ref_dir):
    """登録済みの基準音声を一覧表示する。"""
    ref_path = Path(ref_dir)
    if not ref_path.exists():
        click.echo(f"ディレクトリが見つかりません: {ref_path}")
        return

    click.echo(f"\n基準音声ディレクトリ: {ref_path}\n")
    for speaker_dir in sorted(ref_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.mp3"))
        status = f"{len(audio_files)} ファイル" if audio_files else "未登録"
        click.echo(f"  {speaker_dir.name}: {status}")
        for f in audio_files:
            click.echo(f"    - {f.name}")

    click.echo()


@cli.command()
@click.option("--video", "-v", type=click.Path(exists=True), required=True,
              help="出演者がわかっている動画ファイル")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
def setup(video, config, hf_token):
    """初回セットアップ: 動画から話者を自動分離し、ラベル付けで基準データを作成。"""
    from src.setup_wizard import SetupWizard

    wizard = SetupWizard(config_path=config)
    wizard.run(video, hf_token=hf_token)


@cli.command(name="auto-analyze")
@click.option("--dir", "-d", "video_dir", type=click.Path(exists=True), required=True,
              help="動画フォルダのパス（中の動画を全て自動解析）")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--output", "-o", default="output",
              help="出力ディレクトリ")
@click.option("--format", "-f", "fmt", default="both",
              type=click.Choice(["json", "csv", "both"]),
              help="出力形式")
@click.option("--visual/--no-visual", default=False,
              help="視覚分析を有効にする")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
@click.option("--skip-analyzed/--no-skip", default=True,
              help="解析済みの動画をスキップする（デフォルト: スキップ）")
@click.option("--recursive/--no-recursive", default=False,
              help="サブフォルダも再帰的に検索する")
def auto_analyze(video_dir, config, output, fmt, visual, hf_token, skip_analyzed, recursive):
    """過去の動画を全て放り込んで自動解析する。

    指定フォルダ内の全動画を自動で解析し、結果を出力します。
    既に解析済みの動画はスキップされるため、新しい動画を追加して
    再実行するだけで差分解析が可能です。

    \b
    使い方:
      python src/main.py auto-analyze --dir /path/to/videos/
      python src/main.py auto-analyze --dir /path/to/videos/ --recursive
    """
    from src.preflight import run_preflight, PreflightError

    try:
        run_preflight(check_gpu_available=visual)
    except PreflightError as e:
        click.echo(f"エラー: {e}", err=True)
        sys.exit(1)

    # 動画ファイルを収集
    video_dir_path = Path(video_dir)
    extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

    if recursive:
        videos = sorted(
            f for f in video_dir_path.rglob("*")
            if f.suffix.lower() in extensions and f.is_file()
        )
    else:
        videos = sorted(
            f for f in video_dir_path.iterdir()
            if f.suffix.lower() in extensions and f.is_file()
        )

    if not videos:
        click.echo(f"動画ファイルが見つかりません: {video_dir}")
        return

    click.echo(f"\n動画 {len(videos)} 件を検出しました。")

    # パイプライン初期化
    click.echo("パイプラインを初期化中...")
    pipeline = AnalysisPipeline(config_path=config)
    pipeline.setup(enable_visual=visual, hf_token=hf_token)

    # 解析済みの動画名を取得
    analyzed_names: set[str] = set()
    if skip_analyzed:
        analyzed_names = pipeline._load_analyzed_names(output)
        if analyzed_names:
            click.echo(f"解析済み: {len(analyzed_names)} 件（スキップ）")

    # 解析実行
    all_results = []
    new_count = 0
    for i, video in enumerate(videos, 1):
        if video.name in analyzed_names:
            click.echo(f"  [{i}/{len(videos)}] スキップ: {video.name}")
            continue

        click.echo(f"  [{i}/{len(videos)}] 解析中: {video.name}")
        result = pipeline.analyze_video(str(video))
        all_results.append(result)
        new_count += 1

        # 途中結果を逐次保存（中断に備える）
        _save_incremental(all_results, output, fmt, analyzed_names)

    if not all_results:
        click.echo("\n新しく解析する動画はありません。全て解析済みです。")
        return

    # 最終サマリー
    print_summary(all_results)
    click.echo(f"\n新規解析: {new_count} 件 / 合計: {len(videos)} 件")

    saved = save_results(all_results, output, fmt=fmt)
    for path in saved:
        click.echo(f"結果を保存しました: {path}")


def _save_incremental(results, output_dir, fmt, existing_names):
    """中間結果を保存する（中断復帰用）。"""
    try:
        save_results(results, output_dir, fmt=fmt)
    except Exception:
        pass  # 途中保存の失敗は無視


def _print_network_status(proxy: str | None, expect_proxy: bool = False) -> None:
    """現在のIP/所在地をCLIへ表示する。"""
    status = get_network_status(proxy=proxy, expect_proxy=expect_proxy)
    traffic = get_traffic_status()

    if status.error:
        click.echo(f"Network: {status.error}", err=True)
    else:
        place = ", ".join([p for p in [status.city, status.region, status.country] if p]) or "-"
        click.echo(
            f"Network: IP={status.effective_ip} / Origin={status.origin_ip} / Loc={place}"
        )
        if status.warning:
            click.echo(f"警告: {status.warning}", err=True)

    if traffic.error:
        click.echo(f"Traffic: {traffic.error}", err=True)
        return
    click.echo(
        "Traffic: "
        f"up={traffic.upload_mbps:.3f} Mbps "
        f"down={traffic.download_mbps:.3f} Mbps "
        f"(total up={traffic.bytes_sent_total}B / down={traffic.bytes_recv_total}B)"
    )


@cli.command(name="network-status")
@click.option("--proxy", default=None, help="HTTP/HTTPS/SOCKS プロキシURL")
@click.option("--mullvad-socks5/--no-mullvad-socks5", default=False,
              help="Mullvad のローカルSOCKS5 (socks5h://127.0.0.1:1080) を使う")
@click.option("--watch/--no-watch", default=False,
              help="リアルタイム監視（Ctrl+Cで停止）")
@click.option("--interval", default=10, type=int,
              help="監視間隔（秒）")
def network_status_cmd(proxy, mullvad_socks5, watch, interval):
    """現在のIPアドレス/所在地を表示する。"""
    effective_proxy = "socks5h://127.0.0.1:1080" if mullvad_socks5 else proxy
    expect_proxy = bool(effective_proxy)

    if not watch:
        _print_network_status(proxy=effective_proxy, expect_proxy=expect_proxy)
        return

    import time
    click.echo("リアルタイム監視を開始します（停止: Ctrl+C）")
    try:
        while True:
            _print_network_status(proxy=effective_proxy, expect_proxy=expect_proxy)
            time.sleep(max(1, interval))
    except KeyboardInterrupt:
        click.echo("監視を停止しました。")


@cli.command()
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--output", "-o", default="output",
              help="解析結果ディレクトリ")
@click.option("--host", default="127.0.0.1",
              help="バインドするホスト")
@click.option("--port", "-p", default=5000, type=int,
              help="ポート番号")
def web(config, output, host, port):
    """Web ダッシュボードを起動する。

    ブラウザから解析結果の統計・分析・閾値最適化を行えます。

    \b
    使い方:
      python src/main.py web
      python src/main.py web --port 8080
    """
    from src.web.app import create_app

    app = create_app(config_path=config, output_dir=output)
    click.echo(f"\nWeb ダッシュボードを起動中: http://{host}:{port}")
    click.echo("停止するには Ctrl+C を押してください。\n")
    app.run(host=host, port=port, debug=False)


@cli.command(name="ingest-analyze")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--download-dir", default="data/videos",
              help="取得動画の保存先ディレクトリ")
@click.option("--telegram-url", "telegram_urls", multiple=True,
              help="取得対象の Telegram URL（複数指定可）")
@click.option("--magnet", "magnets", multiple=True,
              help="取得対象の Magnet Link（複数指定可）")
@click.option("--source-file", type=click.Path(exists=True),
              help="取得元リストファイル（1行1ソース）")
@click.option("--proxy", default=None,
              help="HTTP/HTTPS/SOCKS プロキシURL")
@click.option("--mullvad-socks5/--no-mullvad-socks5", default=False,
              help="Mullvad のローカルSOCKS5 (socks5h://127.0.0.1:1080) を使う")
@click.option("--output", "-o", default="output",
              help="解析結果の出力ディレクトリ")
@click.option("--format", "-f", "fmt", default="both",
              type=click.Choice(["json", "csv", "both"]),
              help="出力形式")
@click.option("--visual/--no-visual", default=False,
              help="視覚分析を有効にする")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
@click.option("--skip-analyzed/--no-skip", default=True,
              help="解析済み動画をスキップする")
@click.option("--append-csv-log/--no-append-csv-log", default=True,
              help="履歴CSV(output/results_log.csv)に追記する")
@click.option("--sheet-id", default=None,
              help="Google Sheets のスプレッドシートID")
@click.option("--sheet-name", default="Sheet1",
              help="Google Sheets のワークシート名")
@click.option("--sheet-credentials", default=None,
              help="GoogleサービスアカウントJSONのパス")
def ingest_analyze(
    config,
    download_dir,
    telegram_urls,
    magnets,
    source_file,
    proxy,
    mullvad_socks5,
    output,
    fmt,
    visual,
    hf_token,
    skip_analyzed,
    append_csv_log,
    sheet_id,
    sheet_name,
    sheet_credentials,
):
    """外部ソース取得→解析→CSV/Spreadsheet記録を一括実行する。"""
    from src.ingest import VideoIngestor, collect_video_files
    from src.preflight import PreflightError, run_preflight

    try:
        run_preflight(check_gpu_available=visual)
    except PreflightError as e:
        click.echo(f"エラー: {e}", err=True)
        sys.exit(1)

    effective_proxy = "socks5h://127.0.0.1:1080" if mullvad_socks5 else proxy
    _print_network_status(proxy=effective_proxy, expect_proxy=bool(effective_proxy))

    ingestor = VideoIngestor(download_dir=download_dir)
    try:
        ingestor.ingest(
            telegram_urls=list(telegram_urls),
            magnets=list(magnets),
            source_file=source_file,
            proxy=effective_proxy,
        )
    except Exception as e:
        click.echo(f"取得エラー: {e}", err=True)
        sys.exit(1)

    click.echo("パイプラインを初期化中...")
    pipeline = AnalysisPipeline(config_path=config)
    pipeline.setup(enable_visual=visual, hf_token=hf_token)

    all_videos = sorted(collect_video_files(download_dir))
    if not all_videos:
        click.echo("取得先に動画が見つかりませんでした。")
        return

    analyzed_names: set[str] = set()
    if skip_analyzed:
        analyzed_names = pipeline._load_analyzed_names(output)

    targets = [v for v in all_videos if v.name not in analyzed_names]
    if not targets:
        click.echo("新規解析対象の動画はありません。")
        return

    results = []
    total = len(targets)
    for i, video in enumerate(targets, 1):
        click.echo(f"  [{i}/{total}] 解析中: {video.name}")
        result = pipeline.analyze_video(str(video))
        results.append(result)

    print_summary(results)
    saved = save_results(results, output, fmt=fmt)
    for path in saved:
        click.echo(f"結果を保存しました: {path}")

    if append_csv_log:
        log_path = append_csv_log_fn(results, output)
        click.echo(f"履歴CSVへ追記しました: {log_path}")

    if sheet_id and sheet_credentials:
        try:
            from src.output.sheet_sync import append_results_to_sheet
            count = append_results_to_sheet(
                results=results,
                sheet_id=sheet_id,
                worksheet_name=sheet_name,
                credentials_json=sheet_credentials,
            )
            click.echo(f"Google Sheetsへ追記しました: {count} 行")
        except Exception as e:
            click.echo(f"Google Sheets 追記エラー: {e}", err=True)
    elif sheet_id or sheet_credentials:
        click.echo("Google Sheets 連携には --sheet-id と --sheet-credentials の両方が必要です。")


@cli.command(name="add-magnets-from-url")
@click.argument("url")
@click.option("--download-dir", default="data/videos",
              help="動画保存先ディレクトリ")
@click.option("--config", default="config.yaml",
              help="設定ファイルのパス")
@click.option("--output", default="output",
              help="解析結果の出力ディレクトリ")
@click.option("--format", "-f", "fmt", default="both",
              type=click.Choice(["json", "csv", "both"]),
              help="出力形式")
@click.option("--visual/--no-visual", default=False,
              help="視覚分析を有効にする")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace トークン（pyannote用）")
@click.option("--proxy", default=None, help="HTTP/HTTPS/SOCKS プロキシURL")
@click.option("--mullvad-socks5/--no-mullvad-socks5", default=False,
              help="Mullvad のローカルSOCKS5 (socks5h://127.0.0.1:1080) を使う")
def add_magnets_from_url(url, download_dir, config, output, fmt, visual, hf_token, proxy, mullvad_socks5):
    """指定URLからmagnetリンクを抽出し、取得と解析を実行する。"""
    from src.ingest import fetch_magnets_from_url

    click.echo(f"URLからmagnetリンクを抽出中: {url}")
    try:
        magnets = fetch_magnets_from_url(url)
    except Exception as e:
        click.echo(f"magnetリンク抽出失敗: {e}", err=True)
        sys.exit(1)

    if not magnets:
        click.echo("magnetリンクが見つかりませんでした。", err=True)
        sys.exit(1)

    click.echo(f"{len(magnets)}件のmagnetリンクを取得。ダウンロード・解析を開始します。")
    effective_proxy = "socks5h://127.0.0.1:1080" if mullvad_socks5 else proxy
    ingest_analyze(
        config=config,
        download_dir=download_dir,
        telegram_urls=(),
        magnets=tuple(magnets),
        source_file=None,
        proxy=effective_proxy,
        mullvad_socks5=False,
        output=output,
        fmt=fmt,
        visual=visual,
        hf_token=hf_token,
        skip_analyzed=True,
        append_csv_log=True,
        sheet_id=None,
        sheet_name="Sheet1",
        sheet_credentials=None,
    )


def append_csv_log_fn(results, output_dir):
    """履歴CSVに追記する。"""
    return append_csv_history(results, str(Path(output_dir) / "results_log.csv"))


@cli.command(name="organize-media")
@click.option("--input", "input_dir", type=click.Path(exists=True), required=True,
              help="整理対象フォルダ")
@click.option("--output", "output_dir", type=click.Path(), required=True,
              help="整理済み出力先（site/creator/title.ext）")
@click.option("--unknown", "unknown_dir", type=click.Path(), required=True,
              help="未解決ファイルの出力先（creator/file）")
@click.option("--dry-run", is_flag=True,
              help="移動は行わず予定のみ表示")
def organize_media_cmd(input_dir, output_dir, unknown_dir, dry_run):
    """Fantia投稿IDベースでメディアファイルを整理する。"""
    from src.media_organizer import organize_media

    try:
        moves = organize_media(
            input_root=input_dir,
            output_root=output_dir,
            unknown_root=unknown_dir,
            dry_run=dry_run,
        )
    except FileNotFoundError as e:
        click.echo(f"エラー: {e}", err=True)
        sys.exit(1)

    if not moves:
        click.echo("対象ファイルが見つかりませんでした。")
        return

    for src, dst in moves:
        if dry_run:
            click.echo(f"[DRY RUN] MOVE: {src} -> {dst}")
        else:
            click.echo(f"MOVE: {src} -> {dst}")

    click.echo(f"完了: {len(moves)} 件")


@cli.command()
@click.option("--video", "-v", type=click.Path(exists=True), required=True,
              help="テスト対象の動画ファイル")
@click.option("--config", "-c", default="config.yaml",
              help="設定ファイルのパス")
def test_voice(video, config):
    """声紋照合のテスト実行（ダイアライゼーションなし）。"""
    from src.audio.extractor import extract_audio
    from src.audio.voice_matcher import VoiceMatcher

    import yaml
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    click.echo("声紋モデルを読み込み中...")
    matcher = VoiceMatcher(threshold=cfg["thresholds"]["voice_similarity"])
    matcher.register_speakers_from_dir(cfg["paths"]["reference_voices"])

    if not matcher.reference_embeddings:
        click.echo("エラー: 基準音声が登録されていません。data/reference_voices/ に音声ファイルを配置してください。")
        return

    click.echo(f"登録話者: {list(matcher.reference_embeddings.keys())}")

    click.echo(f"\n音声を抽出中: {video}")
    audio_path = extract_audio(video, sample_rate=cfg["audio"]["sample_rate"])

    click.echo("声紋照合中...")
    scores = matcher.compare(str(audio_path))

    click.echo(f"\n声紋類似度スコア:")
    threshold = cfg["thresholds"]["voice_similarity"]
    for sid, score in sorted(scores.items(), key=lambda x: -x[1]):
        mark = "○" if score >= threshold else "×"
        click.echo(f"  {mark} {sid}: {score:.4f}")

    # クリーンアップ
    Path(audio_path).unlink(missing_ok=True)


if __name__ == "__main__":
    cli()
