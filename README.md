# Movie-serch-Automatic

[![CI](https://github.com/serkenn/Movie-serch-Automatic/actions/workflows/ci.yml/badge.svg)](https://github.com/serkenn/Movie-serch-Automatic/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

動画内の出演者を声紋中心で判定し、結果を JSON/CSV/スプレッドシートへ出力する自動分析ツールです。

## 主な機能

- 声紋分析（Resemblyzer）による出演者判定
- 話者分離（pyannote / フォールバックあり）
- 視覚補助分析（YOLOv8 + OpenCLIP, 任意）
- 自動セットアップ（`setup`）
- バッチ差分解析（`auto-analyze`）
- 取得〜解析〜記録の一括実行（`ingest-analyze`）
- Fantia投稿IDベースのファイル整理（`organize-media`）
- Webダッシュボード（統計/最適化）

## クイックスタート

### 1) セットアップ

```bash
git clone https://github.com/serkenn/Movie-serch-Automatic.git
cd Movie-serch-Automatic
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

FFmpeg が必要です。

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 2) 基準音声登録（初回のみ）

```bash
python -m src.main setup --video /path/to/known_video.mp4
```

### 3) 解析実行

```bash
python -m src.main auto-analyze --dir /path/to/videos --format both
```

結果は `output/results.json` / `output/results.csv` に保存されます。

## 主要コマンド

### `analyze`
単体動画またはフォルダ解析。

```bash
python -m src.main analyze --video /path/to/video.mp4
python -m src.main analyze --dir /path/to/videos
```

### `auto-analyze`
フォルダ内動画を一括解析。既解析スキップ・再帰検索に対応。

```bash
python -m src.main auto-analyze --dir /path/to/videos --recursive --skip-analyzed
```

### `ingest-analyze`
Telegram URL / Magnet Link 取得後に解析し、CSV履歴やGoogle Sheetsへ追記。

```bash
python -m src.main ingest-analyze \
  --telegram-url https://t.me/example/123 \
  --magnet "magnet:?xt=urn:btih:..." \
  --mullvad-socks5 \
  --download-dir data/videos \
  --format both
```

Google Sheets 連携:

```bash
python -m src.main ingest-analyze \
  --source-file sources.txt \
  --sheet-id <spreadsheet_id> \
  --sheet-credentials /path/to/service_account.json \
  --sheet-name Sheet1
```

### `organize-media`
`fantia-posts-<id>` 形式のファイル名からメタデータを取得し、整理。

```bash
python -m src.main organize-media \
  --input /path/to/raw_media \
  --output /path/to/organized \
  --unknown /path/to/unresolved \
  --dry-run
```

### `web`
ダッシュボード起動。

```bash
python -m src.main web --port 5000
```

## CLI一覧

| コマンド | 説明 |
|---|---|
| `setup` | 基準音声の自動登録 |
| `analyze` | 単体/フォルダ解析 |
| `auto-analyze` | バッチ解析（差分実行対応） |
| `ingest-analyze` | 取得→解析→記録の一括実行 |
| `organize-media` | Fantia投稿IDベースの整理 |
| `list-speakers` | 登録済み話者一覧 |
| `test-voice` | 声紋照合のデバッグ実行 |
| `web` | 結果可視化ダッシュボード |

## 設定 (`config.yaml`)

代表的な項目:

- `performers`: 出演者ID/表示名
- `thresholds.voice_similarity`: 判定閾値
- `thresholds.combined_weight_voice`: 統合時の声紋重み
- `paths.reference_voices`: 基準音声パス
- `paths.videos`: 入力動画パス

必要に応じて `config.yaml` を編集してください。

## 出力

- `output/results.json`: 詳細結果（機械読取向け）
- `output/results.csv`: 一覧結果（表計算向け）
- `output/results_log.csv`: 解析履歴（`ingest-analyze` 利用時）

## テストとCI

ローカルテスト:

```bash
python -m pytest tests/ -v --tb=short
```

GitHub Actions (`.github/workflows/ci.yml`):

- `main` への push / PR で実行
- Python 3.10 / 3.11 / 3.12
- 3.12 は実験枠（失敗許容）、3.10/3.11 を主判定

## 注意事項

- 本ツールは、権利・利用規約・法令を遵守して利用してください。
- 取得元サービス（Telegram/Fantia 等）の規約に反する利用は行わないでください。

## ライセンス

Private repository のため、利用条件は管理者ポリシーに従ってください。
