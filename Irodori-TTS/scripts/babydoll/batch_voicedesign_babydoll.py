#!/usr/bin/env python3
"""BabyDoll 台本マニフェスト（各話の JSON）を VoiceDesign で WAV 出力するバッチスクリプト。

使い方:
    uv run python scripts/babydoll/batch_voicedesign_babydoll.py \
        --manifest scripts/babydoll/manifests/ep1_riko.json \
        --out-dir scripts/babydoll/outputs

    # 複数マニフェストを一括処理:
    uv run python scripts/babydoll/batch_voicedesign_babydoll.py \
        --manifest scripts/babydoll/manifests/ep1_riko.json \
        --manifest scripts/babydoll/manifests/ep2_kaede.json

    # manifests/ ディレクトリ配下を全部処理:
    uv run python scripts/babydoll/batch_voicedesign_babydoll.py --all

既存の scripts/batch_voicedesign_wav.py との違い:
    * キャラ別 caption をスクリプト内にハードコードせず、マニフェスト JSON の `caption` フィールドを使用
    * 行単位で `skip_synthesis: true` フラグがあればスキップ（ハミングなど TTS に向かない行）
    * 出力先は `<out-dir>/ep{N}_{role}/{role}_{line:03d}.wav` に自動でネスト
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT = Path(__file__).resolve().parents[2]
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

FIXED_SECONDS_DEFAULT = 30.0
MANIFEST_DIR_DEFAULT = Path(__file__).resolve().parent / "manifests"
OUT_DIR_DEFAULT = Path(__file__).resolve().parent / "outputs"


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


def _load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("items", "caption")
    for key in required:
        if key not in data:
            raise KeyError(f"manifest {path.name} missing required key: {key!r}")
    return data


def _episode_stem(manifest: dict) -> str:
    ep = manifest.get("episode", 0)
    role = manifest.get("character", "unknown")
    return f"ep{int(ep)}_{role}"


def main() -> None:
    parser = argparse.ArgumentParser(description="BabyDoll VoiceDesign batch synthesizer.")
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=None,
        help="マニフェスト JSON（複数回指定可）。未指定なら --all が必要。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="manifests/ ディレクトリ配下の ep*.json を全て処理",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=MANIFEST_DIR_DEFAULT,
        help="--all 指定時の探索対象ディレクトリ",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR_DEFAULT,
        help="出力親ディレクトリ（この下に ep{N}_{role}/ が作られる）",
    )
    ck = parser.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint", default=None, help="ローカルの model.safetensors 等")
    ck.add_argument(
        "--hf-checkpoint",
        default="Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        help="Hugging Face リポジトリ ID（デフォルト: VoiceDesign）",
    )
    parser.add_argument("--model-device", default=None)
    parser.add_argument("--codec-device", default=None)
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=FIXED_SECONDS_DEFAULT)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-guidance-mode", default="independent")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start", type=int, default=1, help="1-based 行番号（各マニフェスト共通の開始位置）")
    parser.add_argument("--end", type=int, default=None, help="含む。未指定は最後まで")
    parser.add_argument("--continue-on-error", action="store_true", help="1行失敗しても続行")
    parser.add_argument("--dry-run", action="store_true", help="読み込み確認のみ（合成しない）")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存 WAV を上書き（デフォルトはスキップ）",
    )
    args = parser.parse_args()

    manifests: list[Path] = []
    if args.all:
        manifests.extend(sorted(Path(args.manifest_dir).glob("ep*.json")))
    if args.manifest:
        manifests.extend(args.manifest)
    if not manifests:
        raise SystemExit(
            "No manifest specified. Use --manifest <path> (repeatable) or --all."
        )

    datasets = [(p, _load_manifest(p)) for p in manifests]
    print(f"[manifests] loaded {len(datasets)} file(s):")
    for p, d in datasets:
        print(f"  - {p.name}  title={d.get('title')!r}  items={len(d['items'])}")

    if args.dry_run:
        for path, manifest in datasets:
            ep_stem = _episode_stem(manifest)
            print(f"\n=== {ep_stem} ({path.name}) ===")
            print(f"caption: {manifest['caption'][:80]}...")
            items = manifest["items"]
            start = max(1, int(args.start))
            end = len(items) if args.end is None else min(int(args.end), len(items))
            for i, it in enumerate(items[start - 1 : end], start=start):
                flag = " [SKIP]" if it.get("skip_synthesis") else ""
                text = str(it.get("text", ""))
                print(f"  {i:03d}{flag} [{it.get('phase')}] ({it.get('voice_tags','-')}) {text[:60]}")
        print(f"\n[dry-run] total {sum(len(d['items']) for _, d in datasets)} lines")
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

    failed: list[tuple[str, int, str]] = []
    for path, manifest in datasets:
        ep_stem = _episode_stem(manifest)
        ep_out_dir = args.out_dir / ep_stem
        ep_out_dir.mkdir(parents=True, exist_ok=True)
        caption = str(manifest["caption"])
        items: list[dict] = manifest["items"]
        n = len(items)
        start = max(1, int(args.start))
        end = n if args.end is None else min(int(args.end), n)
        subset = items[start - 1 : end]

        print(f"\n=== Synthesizing {ep_stem} ({len(subset)} line(s)) ===", flush=True)

        for i, it in enumerate(subset, start=start):
            role = str(it.get("role", manifest.get("character", "unknown")))
            line_index = int(it["line_index"])
            text = str(it.get("text", "")).strip()
            stem = f"{role}_{line_index:03d}"
            out_path = ep_out_dir / f"{stem}.wav"

            if it.get("skip_synthesis"):
                note = it.get("note", "")
                print(f"[skip] {ep_stem}#{line_index} (skip_synthesis=true) {note}", flush=True)
                continue
            if not text:
                print(f"[skip] {ep_stem}#{line_index} empty text", flush=True)
                continue
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {ep_stem}#{line_index} already exists: {out_path.name}", flush=True)
                continue

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
                print(f"[{i}/{n}] {ep_stem} -> {out_path.name}", flush=True)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                failed.append((ep_stem, i, msg))
                print(f"[ERROR] {ep_stem}#{i} {stem}: {msg}", flush=True)
                traceback.print_exc()
                if not args.continue_on_error:
                    raise

    if failed:
        print(f"\nFailed {len(failed)} line(s):", file=sys.stderr)
        for ep, idx, msg in failed:
            print(f"  {ep}#{idx}: {msg}", file=sys.stderr)
        sys.exit(1)

    print("\n[done] all manifests synthesized.", flush=True)


if __name__ == "__main__":
    main()
