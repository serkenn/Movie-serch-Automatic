"""セットアップウィザード - 動画から自動で話者を分離し、ユーザーがラベル付けするだけで基準データを作成"""

import shutil
from pathlib import Path

import yaml
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from src.audio.extractor import extract_audio, extract_audio_segment, get_video_duration
from src.audio.diarizer import Diarizer, SpeakerSegment


class SetupWizard:
    """初回セットアップを半自動で行うウィザード"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path

    def run(self, video_path: str, hf_token: str | None = None) -> None:
        """セットアップウィザードを実行する。

        Args:
            video_path: 出演者がわかっている動画のパス
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画が見つかりません: {video_path}")

        print(f"\n{'='*60}")
        print(" セットアップウィザード")
        print(f"{'='*60}")
        print(f"\n対象動画: {video_path.name}")

        # Step 1: 音声抽出
        print("\n[Step 1/5] 音声を抽出中...")
        audio_path = extract_audio(
            str(video_path),
            sample_rate=self.config["audio"]["sample_rate"],
        )
        duration = get_video_duration(str(video_path))
        print(f"  動画の長さ: {int(duration//60)}分{int(duration%60)}秒")

        # Step 2: 話者ダイアライゼーション
        print("\n[Step 2/5] 話者を自動分離中...")
        diarizer = Diarizer(
            max_speakers=self.config["diarization"]["max_speakers"],
            min_segment_duration=self.config["diarization"]["min_segment_duration"],
            hf_token=hf_token,
        )
        segments = diarizer.diarize(str(audio_path))

        if not segments:
            print("  エラー: 話者を検出できませんでした。")
            Path(audio_path).unlink(missing_ok=True)
            return

        # クラスタごとに整理
        clusters = self._group_segments(segments)
        print(f"  {len(clusters)} 人の話者を検出しました。")

        # Step 3: 各クラスタの代表音声を保存
        print("\n[Step 3/5] 各話者の音声サンプルを作成中...")
        sample_dir = Path("data/setup_samples")
        sample_dir.mkdir(parents=True, exist_ok=True)
        cluster_samples = self._save_cluster_samples(
            str(video_path), clusters, str(sample_dir)
        )

        # Step 4: ユーザーにラベル付けしてもらう
        print(f"\n[Step 4/5] 話者のラベル付け")
        performer_ids = [p["id"] for p in self.config["performers"]]
        performer_names = {p["id"]: p["name"] for p in self.config["performers"]}

        print(f"\n  登録する出演者:")
        for pid in performer_ids:
            print(f"    - {pid} ({performer_names[pid]})")

        assignments = self._interactive_labeling(
            cluster_samples, performer_ids, performer_names
        )

        if not assignments:
            print("\n  ラベル付けがキャンセルされました。")
            self._cleanup(sample_dir, audio_path)
            return

        # Step 5: 基準音声として保存
        print(f"\n[Step 5/5] 基準音声を保存中...")
        self._save_references(str(video_path), clusters, assignments)

        # クリーンアップ
        self._cleanup(sample_dir, audio_path)

        print(f"\n{'='*60}")
        print(" セットアップ完了!")
        print(f"{'='*60}")
        print(f"\n  保存先: {self.config['paths']['reference_voices']}/")
        for cluster_label, person_id in assignments.items():
            name = performer_names.get(person_id, person_id)
            print(f"    {person_id} ({name}) ← 話者クラスタ {cluster_label}")
        print(f"\n  次のコマンドで解析を実行できます:")
        print(f"    python src/main.py analyze --video <動画ファイル>")
        print()

    def _group_segments(self, segments: list[SpeakerSegment]) -> dict[str, list[SpeakerSegment]]:
        """セグメントを話者ラベルごとにグループ化"""
        clusters: dict[str, list[SpeakerSegment]] = {}
        for seg in segments:
            if seg.speaker_label not in clusters:
                clusters[seg.speaker_label] = []
            clusters[seg.speaker_label].append(seg)
        return clusters

    def _save_cluster_samples(self, video_path: str,
                              clusters: dict[str, list[SpeakerSegment]],
                              output_dir: str) -> dict[str, dict]:
        """各クラスタの代表音声サンプルを保存する。

        Returns:
            {クラスタラベル: {"sample_paths": [...], "total_time": float, "segment_count": int}}
        """
        result = {}

        for label, segs in sorted(clusters.items()):
            # 長いセグメントを優先して最大3つ選ぶ
            sorted_segs = sorted(segs, key=lambda s: s.duration, reverse=True)
            selected = sorted_segs[:3]

            sample_paths = []
            for i, seg in enumerate(selected):
                sample_path = Path(output_dir) / f"{label}_sample_{i}.wav"
                try:
                    extract_audio_segment(
                        video_path, seg.start, seg.end,
                        output_path=str(sample_path),
                        sample_rate=self.config["audio"]["sample_rate"],
                    )
                    sample_paths.append(str(sample_path))
                except Exception as e:
                    print(f"  警告: セグメント抽出失敗 ({label}, {seg.start:.1f}-{seg.end:.1f}s): {e}")

            total_time = sum(s.duration for s in segs)
            result[label] = {
                "sample_paths": sample_paths,
                "total_time": total_time,
                "segment_count": len(segs),
            }

            print(f"  話者 {label}: {len(segs)}区間, 合計 {total_time:.1f}秒")

        return result

    def _interactive_labeling(self, cluster_samples: dict[str, dict],
                              performer_ids: list[str],
                              performer_names: dict[str, str]) -> dict[str, str]:
        """ユーザーに各クラスタのラベルを聞く。

        Returns:
            {クラスタラベル: 出演者ID} の辞書
        """
        assignments = {}
        used_ids = set()

        print(f"\n  各話者クラスタに出演者を割り当ててください。")
        print(f"  音声サンプルは data/setup_samples/ に保存されています。")
        print(f"  再生して確認してから割り当てを行ってください。")
        print(f"  (スキップする場合は 's'、中断する場合は 'q' を入力)")

        for label, info in sorted(cluster_samples.items()):
            print(f"\n  --- 話者クラスタ: {label} ---")
            print(f"  発話区間: {info['segment_count']}区間, 合計 {info['total_time']:.1f}秒")
            print(f"  サンプル音声:")
            for sp in info["sample_paths"]:
                print(f"    {sp}")

            # 選択肢を表示
            available = [pid for pid in performer_ids if pid not in used_ids]
            if not available:
                print(f"  全出演者が割り当て済みです。残りはスキップします。")
                break

            print(f"\n  この話者は誰ですか？")
            for i, pid in enumerate(available):
                name = performer_names.get(pid, pid)
                print(f"    {i+1}. {pid} ({name})")
            print(f"    s. スキップ（この動画に出ていない人）")
            print(f"    q. 中断")

            while True:
                try:
                    choice = input(f"\n  選択 [1-{len(available)}/s/q]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return {}

                if choice == "q":
                    return {}
                if choice == "s":
                    print(f"  → {label} をスキップしました。")
                    break

                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available):
                        pid = available[idx]
                        assignments[label] = pid
                        used_ids.add(pid)
                        name = performer_names.get(pid, pid)
                        print(f"  → {label} = {pid} ({name})")
                        break
                    else:
                        print(f"  1〜{len(available)} の数字を入力してください。")
                except ValueError:
                    print(f"  1〜{len(available)} の数字、's'、または 'q' を入力してください。")

        return assignments

    def _save_references(self, video_path: str,
                         clusters: dict[str, list[SpeakerSegment]],
                         assignments: dict[str, str]) -> None:
        """ラベル付けに基づいて基準音声を保存する。"""
        ref_dir = Path(self.config["paths"]["reference_voices"])

        for cluster_label, person_id in assignments.items():
            person_dir = ref_dir / person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            segs = clusters[cluster_label]
            # 長い順に最大5セグメントを基準音声として保存
            sorted_segs = sorted(segs, key=lambda s: s.duration, reverse=True)
            selected = sorted_segs[:5]

            for i, seg in enumerate(selected):
                output_path = person_dir / f"reference_{i}.wav"
                try:
                    extract_audio_segment(
                        video_path, seg.start, seg.end,
                        output_path=str(output_path),
                        sample_rate=self.config["audio"]["sample_rate"],
                    )
                    print(f"  保存: {output_path} ({seg.duration:.1f}秒)")
                except Exception as e:
                    print(f"  警告: 保存失敗 {output_path}: {e}")

    def _cleanup(self, sample_dir: Path, audio_path) -> None:
        """一時ファイルを削除"""
        try:
            shutil.rmtree(sample_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass


def run_auto_setup(video_path: str, config_path: str = "config.yaml",
                   hf_token: str | None = None) -> None:
    """セットアップウィザードを実行するヘルパー関数"""
    wizard = SetupWizard(config_path=config_path)
    wizard.run(video_path, hf_token=hf_token)
