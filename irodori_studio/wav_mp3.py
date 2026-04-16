"""WAV → MP3 変換（ffmpeg 呼び出し）とファイル列挙。"""

from __future__ import annotations

import subprocess
from pathlib import Path


def convert_wav_to_mp3(
    wav_path: Path,
    *,
    ffmpeg: str,
    bitrate: str,
) -> tuple[Path | None, str]:
    """WAV → MP3。成功時は (Path, \"\")、失敗時は (None, 理由)。"""
    mp3 = wav_path.with_suffix(".mp3")
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(wav_path), "-b:a", bitrate, str(mp3)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if not err:
            err = f"ffmpeg 終了コード {proc.returncode}"
        return None, err[:6000]
    if not mp3.is_file():
        return None, "MP3 ファイルが生成されませんでした"
    return mp3, ""


def list_wav_files(folder: Path, *, recursive: bool) -> list[Path]:
    """folder 内の .wav（拡張子は大小無視）。recursive なら子フォルダも。"""
    root = folder.resolve()
    if not root.is_dir():
        return []

    def is_wav(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() == ".wav"

    if recursive:
        out = [p for p in root.rglob("*") if is_wav(p)]
    else:
        out = [p for p in root.iterdir() if is_wav(p)]
    return sorted(out)
