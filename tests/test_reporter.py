"""結果出力モジュールのテスト"""

from src.output.reporter import append_csv_log
from src.pipeline import PerformerResult, VideoAnalysisResult


def _sample_result(video_name: str) -> VideoAnalysisResult:
    return VideoAnalysisResult(
        video_path=f"/tmp/{video_name}",
        video_name=video_name,
        duration=120.0,
        performers=[
            PerformerResult("person_a", "A", True, voice_score=0.9, combined_score=0.9),
        ],
        detected_count=1,
        errors=[],
    )


def test_append_csv_log_creates_header_and_rows(tmp_path):
    out = tmp_path / "results_log.csv"

    append_csv_log([_sample_result("v1.mp4")], str(out))
    append_csv_log([_sample_result("v2.mp4")], str(out))

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # ヘッダー + 2行
    assert "動画名" in lines[0]
    assert "v1.mp4" in lines[1]
    assert "v2.mp4" in lines[2]
