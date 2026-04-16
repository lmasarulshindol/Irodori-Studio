"""settings モジュールの単体テスト。"""

from __future__ import annotations

import json
from pathlib import Path

from irodori_studio import settings as studio_settings


def test_load_merges_defaults_and_clamps_log_lines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studio_settings, "SETTINGS_PATH", tmp_path / "studio_settings.json")
    monkeypatch.setattr(studio_settings, "CONFIG_DIR", tmp_path)
    (tmp_path / "studio_settings.json").write_text(
        json.dumps({"log_max_lines": 10, "mp3_bitrate": "999k", "ffmpeg_path": "  "}),
        encoding="utf-8",
    )
    s = studio_settings.load()
    assert s["log_max_lines"] == 500
    assert s["mp3_bitrate"] == "192k"
    assert s["ffmpeg_path"] == ""


def test_save_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(studio_settings, "SETTINGS_PATH", tmp_path / "studio_settings.json")
    monkeypatch.setattr(studio_settings, "CONFIG_DIR", tmp_path)
    data = {
        "ffmpeg_path": "",
        "mp3_bitrate": "256k",
        "log_max_lines": 4000,
    }
    studio_settings.save(data)
    raw = json.loads((tmp_path / "studio_settings.json").read_text(encoding="utf-8"))
    assert raw["mp3_bitrate"] == "256k"
    assert raw["log_max_lines"] == 4000
