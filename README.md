# Movie-serch

[![CI](https://github.com/hirorogo/Movie-serch/actions/workflows/ci.yml/badge.svg)](https://github.com/hirorogo/Movie-serch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Private-gray.svg)](#ライセンス)

動画に出演している人物を AI で自動判定するツール。

## 概要

- 動画には最大 **3人** の出演者が登場するが、動画によっては **2人だけ** の場合もある
- 顔にはモザイクがかかっているため、**声紋分析を主軸**とし、視覚的特徴を補助に使う
- 複数の AI / モデルを組み合わせて判定精度を高める

### 主な特徴

| 特徴 | 説明 |
|------|------|
| 声紋ベースの話者識別 | Resemblyzer による d-vector 埋め込みで高精度な話者照合 |
| 自動話者分離 | pyannote-audio による発話区間の自動検出 |
| 視覚的補助分析 | YOLOv8 + OpenCLIP で体型・髪型・服装を補助判定 |
| インタラクティブセットアップ | 動画1本から基準音声を自動登録するウィザード |
| バッチ一括解析 | フォルダ内の全動画を自動解析、差分解析にも対応 |
| 中断復帰 | 解析途中で中断しても、既存の結果を引き継いで再開可能 |
| 埋め込みキャッシュ | 一度計算した声紋ベクトルをキャッシュして高速化 |

## 出演者定義

| ID | 呼称（仮） | 備考 |
|----|-----------|------|
| A  | 未設定     | 基準音声を登録して使用 |
| B  | 未設定     | 基準音声を登録して使用 |
| C  | 未設定     | 基準音声を登録して使用 |

> 呼称は利用者が自由に設定できる。
> 各出演者の基準音声サンプル（10〜30秒程度）を `data/reference_voices/` に配置する。

## 分析パイプライン

```
動画ファイル
  │
  ├─ [Step 1] 音声抽出（FFmpeg）
  │     └─ WAV ファイル生成（16kHz モノラル）
  │
  ├─ [Step 2] 話者ダイアライゼーション（pyannote-audio）
  │     └─ 「誰がいつ話しているか」を時間区間で分離
  │
  ├─ [Step 3] 声紋照合（Resemblyzer）
  │     └─ 各区間の音声 ↔ 基準音声の類似度スコア算出
  │
  ├─ [Step 4] 視覚的特徴分析（補助・オプション）
  │     ├─ 体型・シルエット検出（YOLOv8）
  │     ├─ 髪型・髪色の判定（OpenCLIP）
  │     └─ 服装・アクセサリーの検出
  │
  └─ [Step 5] 総合判定
        ├─ 声紋スコア（主: 70%）+ 視覚スコア（副: 30%）を統合
        ├─ 閾値に基づき出演有無を判定
        └─ 結果出力（JSON / CSV）
```

### データフロー詳細

```
入力: video.mp4（顔モザイク・音声あり）
  ↓
[FFmpeg] 音声抽出 → audio.wav (16kHz mono)
  ↓
[pyannote-audio] 話者ダイアライゼーション
  → segments: [(0:00-0:15, speaker_0), (0:15-0:30, speaker_1), ...]
  ↓
[Resemblyzer] 各セグメントの声紋埋め込みベクトル生成
  ↓
[VoiceMatcher] 基準音声との コサイン類似度 算出
  → person_a=0.92, person_b=0.87, person_c=0.12
  ↓
[Optional: YOLOv8 + OpenCLIP] フレーム抽出 → 人物検出 → 視覚特徴比較
  ↓
[Pipeline] スコア統合: voice_score × 0.7 + visual_score × 0.3
  ↓
出力: JSON / CSV レポート
```

## 使用する AI / モデル

| 役割 | モデル / ライブラリ | 説明 |
|------|-------------------|------|
| 声紋埋め込み | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | 音声から話者の特徴ベクトル（d-vector）を生成 |
| 話者分離 | [pyannote-audio](https://github.com/pyannote/pyannote-audio) | 音声内の話者交代を時間区間で検出 |
| 視覚分析 | [YOLOv8](https://github.com/ultralytics/ultralytics) | 人物検出・体型推定 |
| 特徴抽出 | [OpenCLIP](https://github.com/mlfoundations/open_clip) | 画像から視覚的特徴ベクトルを抽出 |
| 音声処理 | [librosa](https://github.com/librosa/librosa) | ピッチ分析・音声前処理 |
| 総合判定 | カスタムロジック | 各 AI の出力スコアを統合して最終判定 |

> モデルは段階的に追加する。まず声紋系（Resemblyzer + pyannote）から始める。

## ディレクトリ構成

```
Movie-serch/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml                  # 閾値・モデル設定
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # CLI エントリーポイント
│   ├── pipeline.py              # 分析パイプライン統合
│   ├── preflight.py             # システム要件チェック
│   ├── setup_wizard.py          # 基準音声セットアップウィザード
│   ├── cache.py                 # 埋め込みベクトルのキャッシュ
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── extractor.py         # 動画 → 音声抽出（FFmpeg）
│   │   ├── diarizer.py          # 話者ダイアライゼーション
│   │   └── voice_matcher.py     # 声紋照合
│   │
│   ├── visual/
│   │   ├── __init__.py
│   │   ├── frame_extractor.py   # 動画 → フレーム画像抽出
│   │   ├── body_analyzer.py     # 体型・シルエット分析
│   │   └── appearance.py        # 髪型・服装分析
│   │
│   └── output/
│       ├── __init__.py
│       └── reporter.py          # 結果出力（JSON / CSV）
│
├── data/
│   ├── reference_voices/        # 基準音声サンプル
│   │   ├── person_a/
│   │   ├── person_b/
│   │   └── person_c/
│   ├── reference_visuals/       # 基準画像サンプル（任意）
│   │   ├── person_a/
│   │   ├── person_b/
│   │   └── person_c/
│   └── videos/                  # 解析対象の動画ファイル
│
├── output/                      # 解析結果の出力先
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI
│
└── tests/
    ├── __init__.py
    ├── test_extractor.py
    ├── test_voice_matcher.py
    ├── test_pipeline.py
    ├── test_cache.py
    ├── test_preflight.py
    └── test_visual.py
```

## セットアップ

### 前提条件

- Python 3.10 以上
- FFmpeg がインストールされていること
- GPU 推奨（pyannote-audio / YOLO の高速化）
- [HuggingFace アカウント](https://huggingface.co/)（pyannote-audio のモデルダウンロードに必要）

### FFmpeg のインストール

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# インストール確認
ffmpeg -version
```

### プロジェクトのインストール

```bash
git clone https://github.com/hirorogo/Movie-serch.git
cd Movie-serch
pip install -r requirements.txt
```

### HuggingFace トークンの設定

pyannote-audio による話者ダイアライゼーションを使用するには、HuggingFace トークンが必要です。

1. [HuggingFace](https://huggingface.co/settings/tokens) でアクセストークンを取得
2. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約に同意
3. 環境変数またはコマンドライン引数で設定

```bash
# 環境変数で設定（推奨）
export HF_TOKEN="hf_your_token_here"

# または、コマンド実行時に指定
python src/main.py analyze --video video.mp4 --hf-token "hf_your_token_here"
```

### 基準音声の準備（自動セットアップ）

出演者がわかっている動画を1本用意して、セットアップコマンドを実行するだけ。
音声の切り出しは自動で行われ、「この声は誰？」と聞かれるので番号で答えるだけ。

```bash
python src/main.py setup --video 出演者がわかる動画.mp4
```

```
[Step 1/5] 音声を抽出中...
[Step 2/5] 話者を自動分離中...
  3 人の話者を検出しました。
[Step 3/5] 各話者の音声サンプルを作成中...
[Step 4/5] 話者のラベル付け

  --- 話者クラスタ: speaker_0 ---
  発話区間: 12区間, 合計 45.3秒
  この話者は誰ですか？
    1. person_a (出演者A)
    2. person_b (出演者B)
    3. person_c (出演者C)
  選択 [1-3/s/q]: 1
  → speaker_0 = person_a (出演者A)
  ...

[Step 5/5] 基準音声を保存中...
セットアップ完了!
```

> 手動で配置する場合は `data/reference_voices/person_a/sample.wav` のように直接配置も可能。

## 使い方

### CLI コマンド一覧

| コマンド | 説明 |
|----------|------|
| `analyze` | 単一動画またはフォルダを解析して出演者を判定 |
| `setup` | 基準音声のインタラクティブセットアップ |
| `auto-analyze` | フォルダ内の全動画を自動バッチ解析 |
| `list-speakers` | 登録済みの基準音声を一覧表示 |
| `test-voice` | 声紋照合のテスト実行（デバッグ用） |

### グローバルオプション

```bash
python src/main.py --verbose [コマンド]   # デバッグログを表示
python src/main.py --help                 # ヘルプを表示
```

### 単一動画の解析

```bash
python src/main.py analyze --video path/to/video.mp4
```

### フォルダ一括解析

```bash
python src/main.py analyze --dir path/to/videos/
```

### 自動バッチ解析（推奨）

大量の動画をまとめて解析する場合は `auto-analyze` を使用します。
解析済みの動画は自動でスキップされるため、新しい動画を追加して再実行するだけで差分解析が可能です。

```bash
# 基本的な使い方
python src/main.py auto-analyze --dir /path/to/videos/

# サブフォルダも再帰的に検索
python src/main.py auto-analyze --dir /path/to/videos/ --recursive

# 視覚分析も有効にして JSON + CSV 両方を出力
python src/main.py auto-analyze --dir /path/to/videos/ --visual --format both
```

### 視覚分析を有効にする

デフォルトでは声紋分析のみが有効です。視覚分析も併用するには `--visual` フラグを追加します。

```bash
python src/main.py analyze --video video.mp4 --visual
```

### 声紋照合のテスト

基準音声の登録状態や照合精度を確認するためのデバッグコマンドです。

```bash
python src/main.py test-voice --video test.mp4
```

```
声紋モデルを読み込み中...
登録話者: ['person_a', 'person_b', 'person_c']

音声を抽出中: test.mp4
声紋照合中...

声紋類似度スコア:
  ○ person_a: 0.9234
  ○ person_b: 0.8712
  × person_c: 0.1245
```

### 登録済み話者の確認

```bash
python src/main.py list-speakers
```

### 出力例

```json
{
  "video": "example.mp4",
  "duration": "12:34",
  "performers": {
    "person_a": {
      "detected": true,
      "confidence": 0.92,
      "speaking_time": "3:21"
    },
    "person_b": {
      "detected": true,
      "confidence": 0.87,
      "speaking_time": "4:05"
    },
    "person_c": {
      "detected": false,
      "confidence": 0.12,
      "speaking_time": "0:00"
    }
  },
  "summary": "出演者: person_a, person_b（2名）"
}
```

## 設定ファイル（config.yaml）

```yaml
performers:
  - id: person_a
    name: "出演者A"
  - id: person_b
    name: "出演者B"
  - id: person_c
    name: "出演者C"

paths:
  reference_voices: "data/reference_voices"
  reference_visuals: "data/reference_visuals"
  videos: "data/videos"
  output: "output"

thresholds:
  voice_similarity: 0.75       # 声紋一致と判定する最低スコア
  visual_similarity: 0.60      # 視覚一致と判定する最低スコア
  combined_weight_voice: 0.7   # 総合判定での声紋の重み
  combined_weight_visual: 0.3  # 総合判定での視覚の重み

diarization:
  min_segment_duration: 1.0    # 最短発話区間（秒）
  max_speakers: 3              # 最大話者数

audio:
  sample_rate: 16000           # サンプリングレート（Hz）
  extract_format: "wav"        # 抽出形式

visual:
  frame_interval: 2.0          # フレーム抽出間隔（秒）
  confidence_threshold: 0.5    # YOLO 人物検出の信頼度閾値

output:
  format: "json"               # json / csv / both
```

### 設定のカスタマイズ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `thresholds.voice_similarity` | 0.75 | 声紋が一致と判定する最低コサイン類似度。高くすると厳密、低くすると寛容になる |
| `thresholds.visual_similarity` | 0.60 | 視覚的特徴が一致と判定する最低スコア |
| `thresholds.combined_weight_voice` | 0.7 | 総合判定における声紋スコアの重み（0.0〜1.0） |
| `thresholds.combined_weight_visual` | 0.3 | 総合判定における視覚スコアの重み（0.0〜1.0） |
| `diarization.min_segment_duration` | 1.0 | 最短発話区間（秒）。短い区間はノイズとして除外 |
| `diarization.max_speakers` | 3 | 最大話者数 |
| `audio.sample_rate` | 16000 | 音声抽出時のサンプリングレート（Hz） |
| `visual.frame_interval` | 2.0 | フレーム抽出間隔（秒）。短くすると精度向上するが処理時間が増加 |

## 対応動画形式

| 拡張子 | 形式 |
|--------|------|
| `.mp4` | MPEG-4 |
| `.avi` | AVI |
| `.mkv` | Matroska |
| `.mov` | QuickTime |
| `.wmv` | Windows Media Video |
| `.flv` | Flash Video |
| `.webm` | WebM |

## 開発

### テストの実行

```bash
# 全テスト実行
python -m pytest tests/ -v

# 特定のテストファイル
python -m pytest tests/test_voice_matcher.py -v

# カバレッジ付き
python -m pytest tests/ -v --tb=short
```

### CI/CD

GitHub Actions による自動テストが設定されています。

- **トリガー**: `main` ブランチへの push / pull request
- **環境**: Ubuntu (Python 3.10, 3.11, 3.12)
- **内容**: FFmpeg インストール → 依存関係インストール → pytest 実行

### コード構成

| モジュール | 責務 |
|-----------|------|
| `src/main.py` | CLI エントリーポイント。Click によるコマンド定義 |
| `src/pipeline.py` | 分析パイプライン全体の統合・オーケストレーション |
| `src/preflight.py` | FFmpeg・GPU 等のシステム要件チェック |
| `src/setup_wizard.py` | 基準音声のインタラクティブ登録ウィザード |
| `src/cache.py` | 声紋埋め込みベクトルの .npy キャッシュ管理 |
| `src/audio/extractor.py` | FFmpeg を使った動画→音声抽出 |
| `src/audio/diarizer.py` | pyannote-audio / フォールバック話者分離 |
| `src/audio/voice_matcher.py` | Resemblyzer による声紋照合・スコア算出 |
| `src/visual/frame_extractor.py` | FFmpeg を使ったフレーム画像抽出 |
| `src/visual/body_analyzer.py` | YOLOv8 による人物検出・体型分析 |
| `src/visual/appearance.py` | OpenCLIP による髪型・服装の特徴抽出 |
| `src/output/reporter.py` | JSON / CSV レポートの生成・保存 |

## 開発ロードマップ

### Phase 1: 基盤構築
- [x] README 作成
- [x] プロジェクト構造・依存関係の定義
- [x] 動画 → 音声抽出モジュール
- [x] システム要件チェック（preflight）
- [x] CI/CD パイプライン構築

### Phase 2: 声紋分析（主軸）
- [x] 基準音声の登録・声紋ベクトル生成
- [x] 話者ダイアライゼーション（区間分離）
- [x] 声紋照合による出演者判定
- [x] 2人のみ出演のケースへの対応
- [x] セットアップウィザード
- [x] 埋め込みキャッシュ

### Phase 3: 視覚分析（補助）
- [x] フレーム画像抽出
- [x] 人物検出（YOLO）
- [x] 体型・髪型・服装の特徴抽出
- [x] 視覚的特徴による補助判定

### Phase 4: 統合・出力
- [x] 声紋 + 視覚スコアの統合判定ロジック
- [x] JSON / CSV レポート出力
- [x] CLI インターフェース完成
- [x] 自動バッチ解析（auto-analyze）

### Phase 5: 精度向上
- [ ] 閾値チューニング
- [ ] テストデータでの検証
- [ ] エッジケース対応（BGM が大きい、複数人同時発話 等）

## トラブルシューティング

### FFmpeg が見つからない

```
エラー: FFmpeg がインストールされていません
```

FFmpeg がシステムの PATH に存在するか確認してください。

```bash
ffmpeg -version   # バージョンが表示されれば OK
```

### HuggingFace トークンエラー

```
エラー: HuggingFace トークンが無効です
```

- トークンが正しく設定されているか確認: `echo $HF_TOKEN`
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) のモデルページで利用規約に同意しているか確認

### 基準音声が登録されていない

```
エラー: 基準音声が登録されていません
```

`setup` コマンドで基準音声を登録するか、`data/reference_voices/person_a/` 等に WAV ファイルを手動配置してください。

### GPU が認識されない

pyannote-audio や YOLO は CPU でも動作しますが、処理速度が大幅に低下します。
GPU を使用する場合は、CUDA 対応の PyTorch がインストールされているか確認してください。

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 声紋スコアが低い

- 基準音声のサンプルが短すぎる可能性があります（10〜30秒推奨）
- BGM や環境音が大きい場合、精度が低下します
- `config.yaml` の `voice_similarity` 閾値を調整してみてください

## ライセンス

このプロジェクトは個人利用を目的としています。
