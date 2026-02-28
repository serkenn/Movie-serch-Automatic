"""Web ダッシュボードのテスト"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import src.web.app as web_app
from src.web.app import create_app


SAMPLE_CONFIG = {
    "performers": [
        {"id": "person_a", "name": "A"},
        {"id": "person_b", "name": "B"},
    ],
    "thresholds": {
        "voice_similarity": 0.75,
        "visual_similarity": 0.60,
        "combined_weight_voice": 0.7,
        "combined_weight_visual": 0.3,
    },
    "paths": {
        "reference_voices": "data/reference_voices",
        "reference_visuals": "data/reference_visuals",
    },
}

SAMPLE_RESULTS = {
    "total_videos": 1,
    "results": [
        {
            "video": "test.mp4",
            "duration": "2:00",
            "performers": {
                "person_a": {
                    "name": "A",
                    "detected": True,
                    "voice_score": 0.88,
                    "visual_score": 0.65,
                    "combined_score": 0.81,
                    "speaking_time": "1:00",
                    "matching_segments": 3,
                },
                "person_b": {
                    "name": "B",
                    "detected": False,
                    "voice_score": 0.30,
                    "visual_score": 0.20,
                    "combined_score": 0.27,
                    "speaking_time": "0:00",
                    "matching_segments": 0,
                },
            },
            "detected_count": 1,
            "summary": "A",
            "errors": [],
        }
    ],
}


@pytest.fixture
def app_with_data():
    """テストデータ付きのFlaskアプリを生成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(SAMPLE_CONFIG, f, allow_unicode=True)

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_RESULTS, f)
        csv_path = os.path.join(output_dir, "results.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("動画名,長さ,出演者数,サマリー\n")
            f.write("test.mp4,2:00,1,A\n")

        app = create_app(config_path=config_path, output_dir=output_dir)
        app.config["TESTING"] = True
        yield app


@pytest.fixture
def client(app_with_data):
    return app_with_data.test_client()


class TestPages:
    def test_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data

    def test_results_page(self, client):
        resp = client.get("/results")
        assert resp.status_code == 200
        assert b"Results" in resp.data

    def test_stats_page(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert b"Stats" in resp.data

    def test_optimizer_page(self, client):
        resp = client.get("/optimizer")
        assert resp.status_code == 200
        assert b"Optimizer" in resp.data

    def test_ingest_page(self, client):
        resp = client.get("/ingest")
        assert resp.status_code == 200
        assert b"Ingest" in resp.data

    def test_csv_preview_page(self, client):
        resp = client.get("/csv-preview")
        assert resp.status_code == 200
        assert b"CSV Preview" in resp.data


class TestAPI:
    def test_overview(self, client):
        resp = client.get("/api/overview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_videos"] == 1
        assert "thresholds" in data

    def test_results(self, client):
        resp = client.get("/api/results")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 1

    def test_result_detail(self, client):
        resp = client.get("/api/results/test.mp4")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["video"] == "test.mp4"

    def test_result_detail_not_found(self, client):
        resp = client.get("/api/results/nonexistent.mp4")
        assert resp.status_code == 404

    def test_performers(self, client):
        resp = client.get("/api/performers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "person_a" in data

    def test_matrix(self, client):
        resp = client.get("/api/matrix")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "performer_ids" in data
        assert "videos" in data

    def test_trends(self, client):
        resp = client.get("/api/trends")
        assert resp.status_code == 200

    def test_confidence(self, client):
        resp = client.get("/api/confidence")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "summary" in data

    def test_optimize(self, client):
        resp = client.get("/api/optimize")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recommended_voice" in data

    def test_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "thresholds" in data

    def test_update_thresholds(self, client):
        resp = client.post(
            "/api/config/thresholds",
            json={"voice_similarity": 0.80, "visual_similarity": 0.65},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["thresholds"]["voice_similarity"] == 0.80

    def test_update_thresholds_no_data(self, client):
        resp = client.post(
            "/api/config/thresholds",
            json=None,
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_ingest_run_requires_sources(self, client):
        resp = client.post("/api/ingest/run", json={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_ingest_run_success(self, client, monkeypatch):
        monkeypatch.setattr(web_app, "run_preflight", lambda check_gpu_available=False: None)

        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = []
        monkeypatch.setattr(web_app, "VideoIngestor", lambda download_dir: mock_ingestor)

        mock_pipeline = MagicMock()
        mock_pipeline._load_analyzed_names.return_value = set()
        mock_pipeline.analyze_video.return_value = {"video": "x.mp4"}
        mock_pipeline.setup.return_value = None
        monkeypatch.setattr(web_app, "AnalysisPipeline", lambda config_path: mock_pipeline)

        monkeypatch.setattr(
            web_app,
            "collect_video_files",
            lambda download_dir: {Path("/tmp/test1.mp4")},
        )
        monkeypatch.setattr(
            web_app,
            "save_results",
            lambda results, output, fmt="both": [Path(output) / "results.json", Path(output) / "results.csv"],
        )
        monkeypatch.setattr(
            web_app,
            "append_csv_log",
            lambda results, output_path: Path(output_path),
        )

        resp = client.post(
            "/api/ingest/run",
            json={
                "magnets": "magnet:?xt=urn:btih:AAAA",
                "download_dir": "/tmp",
                "output_dir": "/tmp",
                "config_path": "/tmp/config.yaml",
                "skip_analyzed": False,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["analyzed_count"] == 1

    def test_csv_preview_api(self, client):
        resp = client.get("/api/csv-preview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "headers" in data
        assert data["row_count"] >= 1
        assert data["returned"] >= 1

    def test_csv_preview_api_not_found(self, client):
        resp = client.get("/api/csv-preview?file=missing.csv")
        assert resp.status_code == 404

    def test_network_status_api(self, client, monkeypatch):
        class DummyStatus:
            def to_dict(self):
                return {
                    "effective_ip": "1.2.3.4",
                    "origin_ip": "1.2.3.4",
                    "city": "Tokyo",
                    "region": "Tokyo",
                    "country": "JP",
                    "warning": None,
                    "error": None,
                }

        monkeypatch.setattr(web_app, "get_network_status", lambda proxy=None, expect_proxy=False: DummyStatus())
        resp = client.get("/api/network/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["effective_ip"] == "1.2.3.4"

    def test_network_traffic_api(self, client, monkeypatch):
        class DummyTraffic:
            def to_dict(self):
                return {
                    "bytes_sent_total": 1000,
                    "bytes_recv_total": 2000,
                    "upload_bps": 10.0,
                    "download_bps": 20.0,
                    "upload_mbps": 0.00008,
                    "download_mbps": 0.00016,
                    "error": None,
                }

        monkeypatch.setattr(web_app, "get_traffic_status", lambda: DummyTraffic())
        resp = client.get("/api/network/traffic")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["bytes_sent_total"] == 1000
