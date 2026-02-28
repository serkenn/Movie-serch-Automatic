# Movie-serch-Automatic

[![CI](https://github.com/serkenn/Movie-serch-Automatic/actions/workflows/ci.yml/badge.svg)](https://github.com/serkenn/Movie-serch-Automatic/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

動画内の出演者を声紋中心で判定し、取得・解析・記録まで自動化するツールです。

## できること

- 声紋分析（Resemblyzer）による出演者判定
- 話者分離（pyannote / フォールバック）
- 視覚補助分析（YOLOv8 + OpenCLIP, 任意）
- セットアップ自動化（`setup`）
- 差分バッチ解析（`auto-analyze`）
- 取得→解析→記録（`ingest-analyze`）
- Fantia投稿IDベースのメディア整理（`organize-media`）
- Web GUI（結果確認、ingest実行、CSVプレビュー、閾値設定）
- IP/所在地と Origin IP 警告（Web/CLI）
- 上り/下り通信量の可視化（Web/CLI）

## クイックスタート

### 1. セットアップ

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

### 2. 基準音声登録（初回のみ）

```bash
python -m src.main setup --video /path/to/known_video.mp4
```

### 3. 解析

```bash
python -m src.main auto-analyze --dir /path/to/videos --format both
```

出力先: `output/results.json`, `output/results.csv`

## よく使うコマンド

### 単体/フォルダ解析

```bash
python -m src.main analyze --video /path/to/video.mp4
python -m src.main analyze --dir /path/to/videos
```

### バッチ差分解析

```bash
python -m src.main auto-analyze --dir /path/to/videos --recursive --skip-analyzed
```

### 取得→解析→記録

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

### URLからmagnet抽出して実行

```bash
python -m src.main add-magnets-from-url "https://example.com/page-with-magnets"
```

### メディア整理

```bash
python -m src.main organize-media \
  --input /path/to/raw_media \
  --output /path/to/organized \
  --unknown /path/to/unresolved \
  --dry-run
```

### ネットワーク状態確認（IP/所在地/通信量）

```bash
python -m src.main network-status
python -m src.main network-status --mullvad-socks5 --watch --interval 10
```

## Web GUI

```bash
python -m src.main web --port 5000
```

主な画面:

- `/` ダッシュボード
- `/results` 解析結果
- `/stats` 統計
- `/optimizer` 閾値調整
- `/ingest` Telegram/Magnet/Proxy 指定で実行 + 通信量グラフ
- `/csv-preview` CSVプレビュー

補足:

- ヘッダーに現在IP/所在地を表示
- プロキシ指定時に Origin IP のままなら警告
- `/ingest` で上り/下り通信（Mbps）をリアルタイム表示

## CLI一覧

| コマンド | 説明 |
|---|---|
| `setup` | 基準音声の自動登録 |
| `analyze` | 単体/フォルダ解析 |
| `auto-analyze` | バッチ解析（差分実行対応） |
| `ingest-analyze` | 取得→解析→記録 |
| `add-magnets-from-url` | URLからmagnet抽出して取得→解析 |
| `organize-media` | Fantia投稿IDベースの整理 |
| `network-status` | IP/所在地/通信量の表示・監視 |
| `list-speakers` | 登録済み話者一覧 |
| `test-voice` | 声紋照合デバッグ |
| `web` | Web GUI |

## 設定

設定ファイル: `config.yaml`

主なキー:

- `performers`: 出演者ID/表示名
- `paths.*`: 参照音声・動画・出力のパス
- `thresholds.*`: 声紋/視覚の閾値、重み
- `diarization.*`: 話者分離の設定
- `visual.*`: 視覚分析の設定

## 出力ファイル

- `output/results.json`: 詳細結果
- `output/results.csv`: 一覧結果
- `output/results_log.csv`: 履歴（ingest系で追記）

## テストとCI

ローカル:

```bash
python -m pytest tests/ -v --tb=short
```

CI (`.github/workflows/ci.yml`):

- `main` への push / PR で実行
- Python 3.10 / 3.11 / 3.12
- 3.12 は実験枠（失敗許容）

## 重要な注意

- 法令・利用規約・権利に従って利用してください。
- Magnet/Torrent は仕様上、ピア通信・上り通信が発生する可能性があります。
- `aria2c --seed-time=0` は完了後シード抑制であり、ダウンロード中の上り通信ゼロを保証しません。
- 「ピア登録なし / 上り通信なし」を厳密に求める場合は Magnet/Torrent を使用しないでください。

## ライセンス

Private repository。利用条件は管理者ポリシーに従ってください。
