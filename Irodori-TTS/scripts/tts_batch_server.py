#!/usr/bin/env python3
"""モデルを1プロセスに常駐させ、HTTP で連続合成する（WebUI と同様にロードは1回）。

依存: `uv sync --extra api` などで fastapi / uvicorn を入れてから実行。

例:
  uv run --extra api python scripts/tts_batch_server.py \\
    --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \\
    --no-ref --host 127.0.0.1 --port 8765

クライアントは scripts/batch_api_client.py を参照。
"""
from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/this.py` 実行時にパッケージ解決が失敗する環境向け（リポジトリ直下を追加）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
import uvicorn

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    SamplingResult,
    default_runtime_device,
    get_cached_runtime,
    resolve_cfg_scales,
    save_wav,
)

FIXED_SECONDS = 30.0


def _resolve_checkpoint_path(raw: str) -> str:
    checkpoint = str(raw).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")
    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        p = Path(checkpoint).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return str(p)
    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    print(f"[tts_batch_server] hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


class ServerState:
    runtime: InferenceRuntime | None = None
    default_no_ref: bool = False
    default_ref_wav: str | None = None
    default_num_steps: int = 40
    default_seed: int | None = None


STATE = ServerState()


class SynthesizeIn(BaseModel):
    text: str = Field(..., description="合成するセリフ本文")
    caption: str | None = Field(None, description="VoiceDesign 等のスタイル指示（行ごとに変えられる）")
    ref_wav: str | None = Field(None, description="サーバー上の参照 WAV パス（未指定時は起動時デフォルト）")
    ref_latent: str | None = Field(None, description="参照 latent .pt（任意）")
    no_ref: bool | None = Field(None, description="話者参照なし（未指定時は起動時 --no-ref）")
    num_steps: int | None = Field(None, description="拡散ステップ数（未指定時は --num-steps）")
    seed: int | None = Field(None, description="シード（未指定時はランダム、または起動時デフォルト）")


def _build_runtime_key(args: argparse.Namespace) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(
        str(args.hf_checkpoint or args.checkpoint),
    )
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(args.model_device),
        codec_repo=str(args.codec_repo),
        model_precision=str(args.model_precision),
        codec_device=str(args.codec_device),
        codec_precision=str(args.codec_precision),
        codec_deterministic_encode=bool(args.codec_deterministic_encode),
        codec_deterministic_decode=bool(args.codec_deterministic_decode),
        enable_watermark=bool(args.enable_watermark),
        compile_model=bool(args.compile_model),
        compile_dynamic=bool(args.compile_dynamic),
    )


def _synthesize_one(
    runtime: InferenceRuntime, body: SynthesizeIn, defaults: ServerState
) -> SamplingResult:
    no_ref = defaults.default_no_ref if body.no_ref is None else bool(body.no_ref)
    ref_wav = defaults.default_ref_wav if body.ref_wav is None else body.ref_wav
    ref_latent = body.ref_latent
    num_steps = int(defaults.default_num_steps if body.num_steps is None else body.num_steps)
    seed = defaults.default_seed if body.seed is None else body.seed

    if runtime.model_cfg.use_speaker_condition and not (no_ref or ref_wav or ref_latent):
        raise HTTPException(
            status_code=400,
            detail="このチェックポイントは話者条件が必須です。ref_wav / ref_latent を渡すか no_ref=true にしてください。",
        )

    caption = None if body.caption is None else str(body.caption)
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

    return runtime.synthesize(
        SamplingRequest(
            text=str(body.text),
            caption=caption,
            ref_wav=ref_wav,
            ref_latent=ref_latent,
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


def create_app() -> FastAPI:
    app = FastAPI(title="Irodori-TTS batch API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "model_loaded": STATE.runtime is not None,
        }

    @app.post("/synthesize")
    def synthesize(body: SynthesizeIn) -> Response:
        if STATE.runtime is None:
            raise HTTPException(status_code=503, detail="モデルが未ロードです。")
        try:
            result = _synthesize_one(STATE.runtime, body, STATE)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_wav(tmp_path, result.audio, result.sample_rate)
            wav_bytes = Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Irodori-Used-Seed": str(result.used_seed)},
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Irodori-TTS: 連続合成用 HTTP API（単一プロセス・モデル常駐）")
    ck = parser.add_mutually_exclusive_group(required=True)
    ck.add_argument("--checkpoint", default=None, help="ローカル .pt / .safetensors")
    ck.add_argument("--hf-checkpoint", default=None, help="Hugging Face リポジトリ ID")
    parser.add_argument("--model-device", default=default_runtime_device())
    parser.add_argument("--codec-device", default=default_runtime_device())
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--codec-deterministic-encode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--codec-deterministic-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-watermark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None, help="全リクエスト共通のデフォルトシード（未指定は毎回ランダム）")
    ref = parser.add_mutually_exclusive_group(required=False)
    ref.add_argument("--ref-wav", default=None, help="デフォルトの参照音声（サーバー上パス）")
    ref.add_argument("--no-ref", action="store_true", help="デフォルトで話者参照なし（VoiceDesign 向け）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    key = _build_runtime_key(args)
    runtime, reloaded = get_cached_runtime(key)
    STATE.runtime = runtime
    STATE.default_no_ref = bool(args.no_ref)
    STATE.default_ref_wav = args.ref_wav
    STATE.default_num_steps = int(args.num_steps)
    STATE.default_seed = args.seed

    if runtime.model_cfg.use_speaker_condition and not (STATE.default_no_ref or STATE.default_ref_wav):
        parser.error("話者条件付きチェックポイントには --ref-wav か --no-ref が必要です。")

    print(
        f"[tts_batch_server] {'loaded' if reloaded else 'reused'} runtime; "
        f"default_no_ref={STATE.default_no_ref} ref_wav={STATE.default_ref_wav}",
        flush=True,
    )

    app = create_app()
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
