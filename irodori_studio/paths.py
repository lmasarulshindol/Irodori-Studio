"""プロジェクト付近のパス解決。"""

from __future__ import annotations

from pathlib import Path


def studio_root() -> Path:
    """Irodori-Studio リポジトリのルート（このパッケージのひとつ上）。"""
    return Path(__file__).resolve().parent.parent


def default_irodori_tts_dir() -> Path:
    """同じ親フォルダ上の Irodori-TTS（公式クローン）のデフォルトパス。"""
    return studio_root().parent / "Irodori-TTS"
