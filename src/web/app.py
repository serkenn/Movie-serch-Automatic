"""Flask Web アプリケーション - 解析結果のダッシュボードと統計表示"""

import csv
import json
import logging
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template, request

from src.ingest import VideoIngestor, collect_video_files, fetch_magnets_from_url
from src.network_status import get_network_status, get_traffic_status
from src.optimizer import ThresholdOptimizer
from src.output.reporter import append_csv_log, save_results
from src.pipeline import AnalysisPipeline
from src.preflight import PreflightError, run_preflight
from src.stats import ResultsAnalyzer

logger = logging.getLogger(__name__)


def create_app(config_path: str = "config.yaml", output_dir: str = "output") -> Flask:
    """Flask アプリケーションファクトリ

    Args:
        config_path: config.yaml のパス
        output_dir: 解析結果の出力ディレクトリ
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["CONFIG_PATH"] = config_path
    app.config["OUTPUT_DIR"] = output_dir

    def _load_config() -> dict:
        p = Path(config_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_results() -> list[dict]:
        results_path = Path(output_dir) / "results.json"
        if not results_path.exists():
            return []
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", []) if isinstance(data, dict) else data

    # --- ページルート ---

    @app.route("/")
    def dashboard():
        """ダッシュボード（トップページ）"""
        return render_template("dashboard.html")

    @app.route("/results")
    def results_page():
        """解析結果一覧ページ"""
        return render_template("results.html")

    @app.route("/stats")
    def stats_page():
        """統計・分析ページ"""
        return render_template("stats.html")

    @app.route("/optimizer")
    def optimizer_page():
        """閾値最適化ページ"""
        return render_template("optimizer.html")

    @app.route("/ingest")
    def ingest_page():
        """取得・解析実行ページ"""
        return render_template("ingest.html")

    @app.route("/csv-preview")
    def csv_preview_page():
        """CSVプレビューページ"""
        return render_template("csv_preview.html")

    # --- API エンドポイント ---

    @app.route("/api/overview")
    def api_overview():
        """全体概要API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        config = _load_config()
        overview = analyzer.get_overview()
        overview["thresholds"] = config.get("thresholds", {})
        return jsonify(overview)

    @app.route("/api/results")
    def api_results():
        """解析結果一覧API"""
        results = _load_results()
        return jsonify({"results": results, "total": len(results)})

    @app.route("/api/results/<path:video_name>")
    def api_result_detail(video_name):
        """個別動画の結果API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        detail = analyzer.get_video_details(video_name)
        if not detail:
            return jsonify({"error": "動画が見つかりません"}), 404
        return jsonify(detail[0])

    @app.route("/api/performers")
    def api_performers():
        """出演者分析API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_performer_analysis())

    @app.route("/api/matrix")
    def api_matrix():
        """出演マトリクスAPI"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_detection_matrix())

    @app.route("/api/trends")
    def api_trends():
        """スコア推移API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_score_trends())

    @app.route("/api/confidence")
    def api_confidence():
        """信頼度分析API"""
        results = _load_results()
        analyzer = ResultsAnalyzer(results_data=results)
        return jsonify(analyzer.get_confidence_analysis())

    @app.route("/api/optimize")
    def api_optimize():
        """閾値最適化API"""
        results = _load_results()
        if not results:
            return jsonify({"error": "解析結果がありません。先に動画を解析してください。"}), 400
        optimizer = ThresholdOptimizer(results)
        return jsonify(optimizer.get_recommendation())

    @app.route("/api/config")
    def api_config():
        """現在の設定API"""
        return jsonify(_load_config())

    @app.route("/api/config/thresholds", methods=["POST"])
    def api_update_thresholds():
        """閾値設定の更新API"""
        data = request.get_json()
        if not data:
            return jsonify({"error": "リクエストデータがありません"}), 400

        config = _load_config()
        thresholds = config.get("thresholds", {})

        if "voice_similarity" in data:
            thresholds["voice_similarity"] = float(data["voice_similarity"])
        if "visual_similarity" in data:
            thresholds["visual_similarity"] = float(data["visual_similarity"])

        config["thresholds"] = thresholds

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        logger.info("閾値を更新: %s", thresholds)
        return jsonify({"status": "ok", "thresholds": thresholds})

    @app.route("/api/ingest/run", methods=["POST"])
    def api_ingest_run():
        """GUIから取得→解析を実行するAPI。"""
        data = request.get_json(silent=True) or {}

        def _split_lines(text: str | None) -> list[str]:
            if not text:
                return []
            return [line.strip() for line in text.splitlines() if line.strip()]

        download_dir = str(data.get("download_dir") or "data/videos")
        output = str(data.get("output_dir") or output_dir)
        config_for_pipeline = str(data.get("config_path") or config_path)
        visual = bool(data.get("visual", False))
        skip_analyzed = bool(data.get("skip_analyzed", True))
        append_log = bool(data.get("append_csv_log", True))

        proxy = (data.get("proxy") or "").strip() or None
        if bool(data.get("mullvad_socks5", False)):
            proxy = "socks5h://127.0.0.1:1080"

        telegram_urls = _split_lines(data.get("telegram_urls"))
        magnets = _split_lines(data.get("magnets"))
        source_url = (data.get("magnet_source_url") or "").strip()

        added_from_url = 0
        if source_url:
            try:
                fetched = fetch_magnets_from_url(source_url)
                magnets.extend(fetched)
                added_from_url = len(fetched)
            except Exception as e:
                return jsonify({"error": f"magnet URL解析エラー: {e}"}), 400

        if not telegram_urls and not magnets:
            return jsonify({"error": "Telegram URL または Magnet を1件以上入力してください。"}), 400

        try:
            run_preflight(check_gpu_available=visual)
        except PreflightError as e:
            return jsonify({"error": str(e)}), 400

        ingestor = VideoIngestor(download_dir=download_dir)
        try:
            ingestor.ingest(
                telegram_urls=telegram_urls,
                magnets=magnets,
                source_file=None,
                proxy=proxy,
            )
        except Exception as e:
            return jsonify({"error": f"取得エラー: {e}"}), 500

        pipeline = AnalysisPipeline(config_path=config_for_pipeline)
        pipeline.setup(enable_visual=visual, hf_token=None)

        all_videos = sorted(collect_video_files(download_dir))
        analyzed_names: set[str] = set()
        if skip_analyzed:
            analyzed_names = pipeline._load_analyzed_names(output)
        targets = [v for v in all_videos if v.name not in analyzed_names]

        results = []
        for video in targets:
            results.append(pipeline.analyze_video(str(video)))

        saved = save_results(results, output, fmt="both")
        csv_log_path = None
        if append_log and results:
            csv_log_path = append_csv_log(results, str(Path(output) / "results_log.csv"))

        return jsonify({
            "status": "ok",
            "added_magnets_from_url": added_from_url,
            "download_dir": download_dir,
            "output_dir": output,
            "total_video_candidates": len(all_videos),
            "analyzed_count": len(results),
            "skipped_count": len(all_videos) - len(results),
            "saved_files": [str(p) for p in saved],
            "csv_log": str(csv_log_path) if csv_log_path else None,
        })

    @app.route("/api/csv-preview")
    def api_csv_preview():
        """CSVファイルをプレビュー用JSONとして返す。"""
        filename = request.args.get("file", "results.csv")
        limit = request.args.get("limit", 200, type=int)
        limit = min(max(limit, 1), 2000)

        csv_path = Path(output_dir) / filename
        if not csv_path.exists():
            return jsonify({"error": f"CSVが見つかりません: {csv_path}"}), 404

        headers: list[str] = []
        rows: list[dict] = []
        total_rows = 0

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                total_rows += 1
                if len(rows) < limit:
                    rows.append(row)

        return jsonify({
            "file": str(csv_path),
            "headers": headers,
            "rows": rows,
            "row_count": total_rows,
            "returned": len(rows),
            "truncated": total_rows > len(rows),
        })

    @app.route("/api/network/status", methods=["GET", "POST"])
    def api_network_status():
        """現在のIP/所在地/Origin判定を返す。"""
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
        else:
            data = request.args

        proxy = (data.get("proxy") or "").strip() or None
        mullvad = bool(data.get("mullvad_socks5", False))
        expect_proxy = bool(data.get("expect_proxy", False))
        if mullvad:
            proxy = "socks5h://127.0.0.1:1080"
            expect_proxy = True

        status = get_network_status(proxy=proxy, expect_proxy=expect_proxy)
        return jsonify(status.to_dict())

    @app.route("/api/network/traffic")
    def api_network_traffic():
        """現在の上り/下り通信量を返す。"""
        return jsonify(get_traffic_status().to_dict())

    return app
