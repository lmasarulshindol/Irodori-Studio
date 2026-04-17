#!/usr/bin/env python3
"""Irodori-Studio 連続生成用: モデルを1回だけロードし、複数セリフを順に WAV 出力する。

infer.py を行ごとに起動する方式と比べ、100 行超でもロード時間が線形に増えない。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import sys

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
    resolve_cfg_scales,
    save_wav,
)

FIXED_SECONDS = 30.0


def _resolve_checkpoint_path(*, checkpoint: str | None, hf_checkpoint: str | None) -> str:
    if checkpoint is not None and str(checkpoint).strip() != "":
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] using local file: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    repo_id = str(hf_checkpoint or "").strip()
    if repo_id == "":
        raise ValueError("Either checkpoint (local) or hf_checkpoint must be set.")

    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(
        f"[checkpoint] downloaded model.safetensors from hf://{repo_id} -> {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch infer for Irodori-Studio (single model load).")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON config written by Irodori-Studio",
    )
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))

    checkpoint_path = _resolve_checkpoint_path(
        checkpoint=raw.get("checkpoint"),
        hf_checkpoint=raw.get("hf_checkpoint"),
    )

    model_device = str(raw["model_device"])
    codec_device = str(raw["codec_device"])
    codec_repo = str(raw.get("codec_repo") or "Aratako/Semantic-DACVAE-Japanese-32dim")
    model_precision = str(raw.get("model_precision") or "fp32")
    codec_precision = str(raw.get("codec_precision") or "fp32")
    codec_deterministic_encode = bool(raw.get("codec_deterministic_encode", True))
    codec_deterministic_decode = bool(raw.get("codec_deterministic_decode", True))
    enable_watermark = bool(raw.get("enable_watermark", False))
    compile_model = bool(raw.get("compile_model", False))
    compile_dynamic = bool(raw.get("compile_dynamic", False))

    num_steps = int(raw["num_steps"])
    seed_raw = raw.get("seed")
    seed = None if seed_raw is None else int(seed_raw)

    global_caption = raw.get("caption")
    if global_caption is not None:
        global_caption = str(global_caption)
    no_ref = bool(raw["no_ref"])
    ref_wav = raw.get("ref_wav")
    if ref_wav is not None:
        ref_wav = str(ref_wav)

    items = raw["items"]
    if not isinstance(items, list) or not items:
        raise SystemExit("config.items must be a non-empty list")

    key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=model_device,
        codec_repo=codec_repo,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        codec_deterministic_encode=codec_deterministic_encode,
        codec_deterministic_decode=codec_deterministic_decode,
        enable_watermark=enable_watermark,
        compile_model=compile_model,
        compile_dynamic=compile_dynamic,
    )
    runtime, reloaded = get_cached_runtime(key)
    if reloaded:
        print("[batch] model loaded into memory", flush=True)
    else:
        print("[batch] reusing cached runtime", flush=True)

    if runtime.model_cfg.use_speaker_condition and not (no_ref or ref_wav):
        raise SystemExit(
            "This checkpoint requires speaker conditioning: set ref_wav or no_ref."
        )

    n = len(items)
    for i, it in enumerate(items, start=1):
        text = str(it["text"])
        out_wav = Path(str(it["output_wav"]))
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        line_caption = it.get("caption")
        if line_caption is not None and str(line_caption).strip() != "":
            caption = str(line_caption)
        else:
            caption = global_caption

        use_caption = bool(
            runtime.model_cfg.use_caption_condition
            and caption is not None
            and str(caption).strip() != ""
        )
        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, _ = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
            use_caption_condition=use_caption,
            use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
        )

        result = runtime.synthesize(
            SamplingRequest(
                text=text,
                caption=caption,
                ref_wav=ref_wav,
                ref_latent=None,
                no_ref=no_ref,
                ref_normalize_db=-16.0,
                ref_ensure_max=True,
                num_candidates=1,
                decode_mode="sequential",
                seconds=FIXED_SECONDS,
                max_ref_seconds=30.0,
                max_text_len=None,
                max_caption_len=None,
                num_steps=num_steps,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_guidance_mode="independent",
                cfg_scale=None,
                cfg_min_t=0.5,
                cfg_max_t=1.0,
                truncation_factor=None,
                rescale_k=None,
                rescale_sigma=None,
                context_kv_cache=True,
                speaker_kv_scale=None,
                speaker_kv_min_t=0.9,
                speaker_kv_max_layers=None,
                seed=seed,
                trim_tail=True,
                tail_window_size=20,
                tail_std_threshold=0.05,
                tail_mean_threshold=0.1,
            ),
            log_fn=None,
        )
        save_wav(str(out_wav), result.audio, result.sample_rate)
        print(f"[batch] [{i}/{n}] Saved {out_wav}", flush=True)
        print(f"[batch] [{i}/{n}] used_seed: {result.used_seed}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[batch] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
