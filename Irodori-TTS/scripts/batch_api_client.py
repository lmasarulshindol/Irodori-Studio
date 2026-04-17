#!/usr/bin/env python3
"""tts_batch_server が起動している前提で、マニフェストに従い連続で POST /synthesize する。

マニフェストは JSON Lines（1行1 JSON）。例:
  {"text": "セリフ", "caption": "スタイル（任意）", "output": "out/001.wav"}

output または output_wav キーで保存先（--output-dir からの相対可）を指定。
依存は標準ライブラリのみ（urllib）。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _post_synthesize(base_url: str, payload: dict[str, object], timeout: float | None) -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{base_url.rstrip('/')}/synthesize"
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return body, headers
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {err_body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Irodori-TTS batch API クライアント（JSONL マニフェスト）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="tts_batch_server のベース URL")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON Lines（text / caption? / output|output_wav）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch_api"),
        help="output が相対パスのときの基準ディレクトリ",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="1 リクエストあたりの秒数タイムアウト（未指定は無制限に近い既定）",
    )
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    jobs: list[tuple[int, str]] = []
    for line_no, raw in enumerate(lines, start=1):
        s = raw.strip()
        if s == "" or s.startswith("#"):
            continue
        jobs.append((line_no, s))
    n = len(jobs)
    done = 0
    for line_no, raw in jobs:
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"line {line_no}: invalid JSON: {exc}") from exc

        text = row.get("text")
        if not isinstance(text, str) or text.strip() == "":
            raise SystemExit(f"line {line_no}: 'text' (non-empty string) is required")

        out_key = "output" if "output" in row else "output_wav"
        if out_key not in row:
            raise SystemExit(f"line {line_no}: need 'output' or 'output_wav'")
        rel = Path(str(row[out_key]))
        out_path = rel if rel.is_absolute() else (args.output_dir / rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, object] = {"text": text}
        if "caption" in row and row["caption"] is not None:
            payload["caption"] = str(row["caption"])
        for opt in ("ref_wav", "ref_latent", "no_ref", "num_steps", "seed"):
            if opt in row:
                payload[opt] = row[opt]

        done += 1
        print(f"[client] ({done}/{n}) -> {out_path}", flush=True)
        wav_bytes, headers = _post_synthesize(str(args.base_url), payload, args.timeout)
        out_path.write_bytes(wav_bytes)
        seed_hdr = headers.get("x-irodori-used-seed", "")
        if seed_hdr:
            print(f"[client] used_seed: {seed_hdr}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[client] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
