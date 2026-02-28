"""Google Sheets への追記出力。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.pipeline import VideoAnalysisResult


def append_results_to_sheet(
    results: list[VideoAnalysisResult],
    sheet_id: str,
    worksheet_name: str,
    credentials_json: str,
) -> int:
    """解析結果を Google Sheets に追記する。"""
    if not results:
        return 0

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:  # pragma: no cover - 環境依存
        raise RuntimeError(
            "Google Sheets 連携には gspread と google-auth が必要です。"
        ) from e

    cred_path = Path(credentials_json)
    if not cred_path.exists():
        raise FileNotFoundError(f"認証JSONが見つかりません: {cred_path}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(str(cred_path), scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for result in results:
        rows.append([
            now,
            result.video_name,
            result.duration,
            result.detected_count,
            result.to_dict()["summary"],
            "; ".join(result.errors),
        ])

    worksheet.append_rows(rows, value_input_option="USER_ENTERED")
    return len(rows)
