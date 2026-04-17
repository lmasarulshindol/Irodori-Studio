#!/usr/bin/env python3
"""マニフェスト JSON の各セリフを VoiceDesign で WAV 出力（1行1ファイル、ランタイム1回初期化）。"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
import warnings
from pathlib import Path

# PowerShell で stderr に出るとネイティブコマンドのエラー扱いになりやすいため、
# torch 等の FutureWarning は抑制する（合成失敗の原因にはならない）。
warnings.filterwarnings("ignore", category=FutureWarning)

# プロジェクトルートを import パスに（PYTHONPATH 不要で infer.py と同様に実行可）
_ROOT = Path(__file__).resolve().parents[1]
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

FIXED_SECONDS_DEFAULT = 30.0

# VoiceDesign 用キャプション（キャラの雰囲気）
DEFAULT_CAPTIONS: dict[str, str] = {
    "mayu": (
        "明るく前向きな10代後半の女性アイドル。親しみやすく、はきはきと自然に話す。"
    ),
    "ichigo": (
        "ゆるふわで天然な10代女性。やわらかく甘い雰囲気で、あたたかい口調。"
    ),
    "kinugawa": (
        "30代男性マネージャー。落ち着いた声で、やや硬いがためらいもある。"
    ),
    "futari": (
        "二人の若い女性が揃って元気よく掛け声を言う。明るくハキハキ。"
    ),
}


def _resolve_checkpoint_path(*, checkpoint: str | None, hf_checkpoint: str | None) -> str:
    if checkpoint is not None:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] local: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    if hf_checkpoint is None or str(hf_checkpoint).strip() == "":
        raise ValueError("Either --checkpoint or --hf-checkpoint is required.")

    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download
    repo_id = str(hf_checkpoint).strip()
    path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"[checkpoint] hf://{repo_id} -> {path}", flush=True)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch VoiceDesign WAV export.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "yane_dialogue_manifest.json",
        help="extract_yane_dialogue_json.py が出力した JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "yane_voicedesign_wav",
        help="出力ディレクトリ",
    )
    ck = parser.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint", default=None, help="ローカルの model.safetensors 等")
    ck.add_argument(
        "--hf-checkpoint",
        default="Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        help="Hugging Face リポジトリ ID（デフォルト: VoiceDesign）",
    )
    parser.add_argument(
        "--model-device",
        default=None,
        help="未指定時は irodori_tts の default（通常は cuda 優先）",
    )
    parser.add_argument("--codec-device", default=None, help="未指定時は model-device と同じ")
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=FIXED_SECONDS_DEFAULT)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-guidance-mode", default="independent")
    parser.add_argument("--seed", type=int, default=None, help="全行で同じシード（未指定は毎回ランダム）")
    parser.add_argument("--start", type=int, default=1, help="1-based 行番号（マニフェスト順）")
    parser.add_argument("--end", type=int, default=None, help="含む。未指定は最後まで")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="1行失敗しても続行",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="読み込みのみ（合成しない）",
    )
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items: list[dict[str, object]] = data["items"]
    n = len(items)
    start = max(1, int(args.start))
    end = n if args.end is None else min(int(args.end), n)
    if start > end:
        raise SystemExit(f"Invalid range: start={start} end={end} (total {n})")

    subset = items[start - 1 : end]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for i, it in enumerate(subset, start=start):
            role = str(it["role"])
            text = str(it["text"])
            print(f"{i:04d} [{role}] {text[:60]}...")
        print(f"[dry-run] {len(subset)} lines")
        return

    ir = importlib.import_module("irodori_tts.inference_runtime")
    InferenceRuntime = ir.InferenceRuntime
    RuntimeKey = ir.RuntimeKey
    SamplingRequest = ir.SamplingRequest
    default_runtime_device = ir.default_runtime_device
    resolve_cfg_scales = ir.resolve_cfg_scales
    save_wav = ir.save_wav

    model_device = args.model_device or default_runtime_device()
    codec_device = args.codec_device or model_device

    checkpoint_path = _resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        hf_checkpoint=args.hf_checkpoint if args.checkpoint is None else None,
    )

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=str(model_device),
            codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            model_precision=str(args.model_precision),
            codec_device=str(codec_device),
            codec_precision=str(args.codec_precision),
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            enable_watermark=False,
            compile_model=False,
            compile_dynamic=False,
        )
    )

    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        cfg_scale_text=float(args.cfg_scale_text),
        cfg_scale_caption=float(args.cfg_scale_caption),
        cfg_scale_speaker=5.0,
        cfg_scale=None,
        use_caption_condition=True,
        use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
    )
    for msg in scale_messages:
        print(msg, flush=True)

    failed: list[tuple[int, str]] = []
    for i, it in enumerate(subset, start=start):
        role = str(it["role"])
        line_index = int(it["line_index"])
        text = str(it["text"]).strip()
        if not text:
            print(f"[skip] empty line at manifest index {i}", flush=True)
            continue

        caption = DEFAULT_CAPTIONS.get(role)
        if caption is None:
            raise KeyError(f"Unknown role={role!r}. Add caption in DEFAULT_CAPTIONS.")

        stem = f"{role}_{line_index:03d}"
        out_path = args.out_dir / f"{stem}.wav"

        try:
            result = runtime.synthesize(
                SamplingRequest(
                    text=text,
                    caption=caption,
                    ref_wav=None,
                    ref_latent=None,
                    no_ref=True,
                    ref_normalize_db=-16.0,
                    ref_ensure_max=True,
                    num_candidates=1,
                    decode_mode="sequential",
                    seconds=float(args.seconds),
                    max_ref_seconds=30.0,
                    max_text_len=None,
                    max_caption_len=None,
                    num_steps=int(args.num_steps),
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_caption=cfg_scale_caption,
                    cfg_scale_speaker=cfg_scale_speaker,
                    cfg_guidance_mode=str(args.cfg_guidance_mode),
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
                    seed=int(args.seed) if args.seed is not None else None,
                    trim_tail=True,
                    tail_window_size=20,
                    tail_std_threshold=0.05,
                    tail_mean_threshold=0.1,
                ),
                log_fn=None,
            )
            save_wav(out_path, result.audio, result.sample_rate)
            print(f"[{i}/{n}] Saved {out_path}", flush=True)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failed.append((i, msg))
            print(f"[ERROR] line {i} {stem}: {msg}", flush=True)
            traceback.print_exc()
            if not args.continue_on_error:
                raise

    if failed:
        print(f"\nFailed {len(failed)} line(s):", file=sys.stderr)
        for idx, msg in failed:
            print(f"  #{idx}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
