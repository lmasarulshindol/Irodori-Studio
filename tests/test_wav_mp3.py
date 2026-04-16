"""irodori_studio.wav_mp3 のテスト。"""

from __future__ import annotations

from pathlib import Path

from irodori_studio.wav_mp3 import list_wav_files


def test_list_wav_files_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.WAV").write_bytes(b"")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.wav").write_bytes(b"")

    got = list_wav_files(tmp_path, recursive=False)
    assert len(got) == 2
    names = {p.name.lower() for p in got}
    assert names == {"a.wav", "b.wav"}


def test_list_wav_files_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "b.wav").write_bytes(b"")

    got = list_wav_files(tmp_path, recursive=True)
    assert len(got) == 2


def test_list_wav_files_missing_dir(tmp_path: Path) -> None:
    assert list_wav_files(tmp_path / "nope", recursive=False) == []
