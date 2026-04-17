#!/usr/bin/env python3
"""既存出力をスキップした batch config を生成する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create resumed batch config by skipping existing output_wav files.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    items = raw.get("items")
    if not isinstance(items, list):
        raise SystemExit("config.items must be a list")

    base_dir = args.config.parents[2]
    remaining: list[dict[str, object]] = []
    skipped = 0
    for item in items:
        out_wav = Path(str(item["output_wav"]))
        candidate = base_dir / out_wav
        if candidate.exists():
            skipped += 1
            continue
        remaining.append(item)

    raw["items"] = remaining
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"skipped={skipped} remaining={len(remaining)} output={args.output}")


if __name__ == "__main__":
    main()
