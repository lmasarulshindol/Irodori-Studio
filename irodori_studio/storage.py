"""履歴・プリセットの JSON 永続化（利便性用）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from irodori_studio.paths import studio_root

CONFIG_DIR = studio_root() / "config"
HISTORY_PATH = CONFIG_DIR / "history.json"
PRESETS_PATH = CONFIG_DIR / "presets.json"

MAX_HISTORY = 40


def _load(path: Path, default: Any) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return default


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_history() -> list[dict[str, Any]]:
    raw = _load(HISTORY_PATH, [])
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def save_history(entries: list[dict[str, Any]]) -> None:
    _save(HISTORY_PATH, entries[:MAX_HISTORY])


def push_history(
    *,
    text: str,
    output_path: str,
    checkpoint_label: str,
    caption: str = "",
    ref_wav: str = "",
    no_ref: bool = True,
    device: str = "cpu",
    num_steps: str = "",
    seed: str = "",
) -> None:
    entries = load_history()
    item = {
        "at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "output": output_path,
        "checkpoint": checkpoint_label,
        "caption": caption,
        "ref_wav": ref_wav,
        "no_ref": no_ref,
        "device": device,
        "num_steps": num_steps,
        "seed": seed,
    }
    entries.insert(0, item)
    save_history(entries)


def load_presets() -> list[dict[str, Any]]:
    raw = _load(PRESETS_PATH, {"presets": []})
    if isinstance(raw, dict) and isinstance(raw.get("presets"), list):
        return [x for x in raw["presets"] if isinstance(x, dict)]
    return []


def save_presets(presets: list[dict[str, Any]]) -> None:
    _save(PRESETS_PATH, {"presets": presets})
