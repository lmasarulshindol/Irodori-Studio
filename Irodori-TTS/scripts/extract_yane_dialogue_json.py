#!/usr/bin/env python3
"""屋上ライブはトラブルだらけ！.pdf からセリフを抽出し JSON マニフェストを書き出す。"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise SystemExit("PyMuPDF が必要です: py -3 -m pip install pymupdf") from e

PDF_NAME = "屋上ライブはトラブルだらけ！.pdf"


def paragraphs_from_vertical(lines: list[str]) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if not ln:
            if buf:
                paras.append("".join(buf))
                buf = []
            continue
        if re.fullmatch(r"\d+", ln):
            if buf:
                paras.append("".join(buf))
                buf = []
            continue
        buf.append(ln)
    if buf:
        paras.append("".join(buf))
    return paras


def concat_until_closing_quote(paras: list[str], start_i: int) -> tuple[str, int]:
    if start_i >= len(paras):
        return "", start_i
    parts: list[str] = []
    i = start_i
    while i < len(paras):
        parts.append(paras[i])
        merged = "".join(parts)
        if "」" in merged:
            return merged, i + 1
        i += 1
    return "".join(parts), i


SPEAKERS = ("鬼怒川", "まゆ", "いちご", "ふたり")
ROLE_KEYS = {
    "鬼怒川": "kinugawa",
    "まゆ": "mayu",
    "いちご": "ichigo",
    "ふたり": "futari",
}


def extract(pdf_path: Path) -> list[dict[str, object]]:
    doc = fitz.open(pdf_path)
    raw = "\n".join(page.get_text() for page in doc[1:])
    lines = [ln.strip() for ln in raw.splitlines()]
    paras = paragraphs_from_vertical(lines)

    items: list[dict[str, object]] = []
    counts: defaultdict[str, int] = defaultdict(int)

    i = 0
    while i < len(paras):
        p = paras[i]
        if p in SPEAKERS:
            name = p
            i += 1
            if i >= len(paras):
                break
            q, i = concat_until_closing_quote(paras, i)
            if "「" in q and "」" in q:
                inner = q[q.index("「") + 1 : q.rindex("」")]
                role = ROLE_KEYS[name]
                counts[role] += 1
                items.append(
                    {
                        "role": role,
                        "role_label": name,
                        "line_index": int(counts[role]),
                        "text": inner,
                    }
                )
            continue
        i += 1

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help=f"PDF パス（未指定時は親を辿って {PDF_NAME} を探す）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "yane_dialogue_manifest.json",
        help="出力 JSON パス",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    if pdf_path is None:
        here = Path(__file__).resolve()
        for base in [here.parent.parent.parent, here.parent.parent]:
            cand = base / "VoiceActorLaboratory" / "台本" / "PDF" / PDF_NAME
            if cand.is_file():
                pdf_path = cand
                break
        if pdf_path is None:
            raise SystemExit(f"PDF が見つかりません: {PDF_NAME}")

    items = extract(pdf_path)
    payload = {
        "source_pdf": str(pdf_path).replace("\\", "/"),
        "title": "屋上ライブはトラブルだらけ！",
        "line_count": len(items),
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(items)} lines -> {args.out}")


if __name__ == "__main__":
    main()
