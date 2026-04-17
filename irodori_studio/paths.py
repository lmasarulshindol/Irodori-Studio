"""プロジェクト付近のパス解決。"""

from __future__ import annotations

from pathlib import Path


def studio_root() -> Path:
    """Irodori-Studio リポジトリのルート（このパッケージのひとつ上）。"""
    return Path(__file__).resolve().parent.parent


def default_irodori_tts_dir() -> Path:
    """本リポジトリに同梱した Irodori-TTS（`infer.py` があるディレクトリ）のデフォルトパス。"""
    return studio_root() / "Irodori-TTS"
