"""アプリ全体の設定（config/studio_settings.json）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from irodori_studio.paths import studio_root

CONFIG_DIR = studio_root() / "config"
SETTINGS_PATH = CONFIG_DIR / "studio_settings.json"

MP3_BITRATES = ("128k", "192k", "256k", "320k")

DEFAULTS: dict[str, Any] = {
    "ffmpeg_path": "",
    "mp3_bitrate": "192k",
    "log_max_lines": 3000,
}


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load() -> dict[str, Any]:
    raw = _load_json(SETTINGS_PATH, {})
    out = dict(DEFAULTS)
    if isinstance(raw, dict):
        for k in DEFAULTS:
            if k in raw:
                out[k] = raw[k]
    # 正規化
    try:
        out["log_max_lines"] = int(out["log_max_lines"])
        out["log_max_lines"] = max(500, min(50_000, out["log_max_lines"]))
    except (TypeError, ValueError):
        out["log_max_lines"] = DEFAULTS["log_max_lines"]
    br = str(out.get("mp3_bitrate", "192k"))
    if br not in MP3_BITRATES:
        out["mp3_bitrate"] = "192k"
    out["ffmpeg_path"] = str(out.get("ffmpeg_path", "") or "").strip()
    return out


def save(data: dict[str, Any]) -> None:
    to_store = dict(DEFAULTS)
    to_store.update({k: data.get(k, DEFAULTS[k]) for k in DEFAULTS})
    to_store["log_max_lines"] = max(500, min(50_000, int(to_store["log_max_lines"])))
    if str(to_store.get("mp3_bitrate", "")) not in MP3_BITRATES:
        to_store["mp3_bitrate"] = "192k"
    _save_json(SETTINGS_PATH, to_store)


def resolve_ffmpeg_path(user_override: str = "") -> str | None:
    """ffmpeg 実行ファイルのパス。user_override → 設定 → PATH の順。"""
    for candidate in (user_override.strip(), load().get("ffmpeg_path", "")):
        if candidate and Path(str(candidate)).is_file():
            return str(candidate)
    return shutil.which("ffmpeg")
