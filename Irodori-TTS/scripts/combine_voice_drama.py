#!/usr/bin/env python3
"""複数 WAV を「間」を挿入しながら結合し、1つのボイスドラマ WAV を出力する。

使い方:
  py -3 scripts/combine_voice_drama.py \
    --manifest scripts/examples/centerman_drama_timeline.json \
    --output outputs/centerman_drama/centerman_drama_full.wav

マニフェスト（JSON）の形式:
  {
    "sample_rate": 24000,
    "clips": [
      {"wav": "outputs/.../01.wav", "pause_after_ms": 800},
      ...
    ]
  }
pause_after_ms: そのクリップの「あと」に挿入する無音の長さ（ms）。
"""
from __future__ import annotations

import argparse
import json
import struct
import wave
from pathlib import Path


def _read_wav_pcm16(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: sample_width={sample_width}, expected 2 (16-bit PCM)")
    return frames, sample_rate, n_channels


def _silence_bytes(ms: int, sample_rate: int, n_channels: int) -> bytes:
    n_samples = int(sample_rate * ms / 1000.0) * n_channels
    return b"\x00\x00" * n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="WAV を「間」付きで結合してボイスドラマにする")
    parser.add_argument("--manifest", type=Path, required=True, help="結合マニフェスト JSON")
    parser.add_argument("--output", type=Path, required=True, help="出力 WAV パス")
    args = parser.parse_args()

    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_sr = int(raw.get("sample_rate", 24000))
    clips = raw["clips"]

    chunks: list[bytes] = []
    first_channels: int | None = None

    for i, clip in enumerate(clips):
        wav_path = Path(str(clip["wav"]))
        if not wav_path.is_absolute():
            wav_path = args.manifest.parent / wav_path
        if not wav_path.is_file():
            raise FileNotFoundError(f"clip {i}: {wav_path}")

        frames, sr, n_ch = _read_wav_pcm16(wav_path)
        if sr != expected_sr:
            raise ValueError(f"clip {i}: sample_rate={sr}, expected {expected_sr}")
        if first_channels is None:
            first_channels = n_ch
        elif n_ch != first_channels:
            raise ValueError(f"clip {i}: channels={n_ch}, expected {first_channels}")

        chunks.append(frames)

        pause_ms = int(clip.get("pause_after_ms", 0))
        if pause_ms > 0:
            chunks.append(_silence_bytes(pause_ms, expected_sr, first_channels))

    if first_channels is None:
        raise SystemExit("clips が空です")

    combined = b"".join(chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as out:
        out.setnchannels(first_channels)
        out.setsampwidth(2)
        out.setframerate(expected_sr)
        out.writeframes(combined)

    total_sec = len(combined) / (expected_sr * first_channels * 2)
    print(f"[combine] saved {args.output} ({total_sec:.1f}s, {len(clips)} clips)", flush=True)


if __name__ == "__main__":
    main()
