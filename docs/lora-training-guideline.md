# LoRA 作成ガイドライン（Irodori-Studio / Irodori-TTS）

本書は、**Irodori-TTS 公式の学習パイプライン**を使って **PEFT LoRA を学習する**ときの手順と判断基準をまとめたものです。Irodori-Studio の **「LoRA 作成（学習）」タブ**は `train.py` を呼び出すための補助であり、**データ作成そのものは Irodori-TTS 側のツール（`prepare_manifest.py` など）で行います**。

参考:

- [Irodori-TTS README（Training / Fine-Tuning / LoRA）](https://github.com/Aratako/Irodori-TTS/blob/main/README.md#training)
- [Irodori-TTS リポジトリ](https://github.com/Aratako/Irodori-TTS)

---

## 1. この機能でできること・できないこと

| できること | できないこと（現状） |
| ---------- | -------------------- |
| 公開ベースモデル（`.safetensors`）から **LoRA アダプタを学習**する | Irodori-Studio の **推論画面だけ**で、録音を大量に食わせて自動で LoRA ができるわけではない |
| `manifest`（JSONL）と学習 YAML を指定して **`uv run python train.py ...` を実行**する | 学習用データセットの**収集・ラベル付け・前処理のすべてを GUI 完結**ではない |

**自分の声を「たくさん」学習データにする**場合は、**十分な量・品質の音声と、対応するテキスト（およびモードに応じた `speaker_id` / `caption`）**を用意し、公式手順どおり **DACVAE 潜在の事前計算 → `train.py`** まで進める必要があります。

---

## 2. 事前準備（環境）

1. **Irodori-TTS 本体**が利用できること。本リポジトリでは **`Irodori-TTS/`** に同梱されているので、その直下で `uv sync` 済みであればよい（[Installation](https://github.com/Aratako/Irodori-TTS#installation)）。別途公式をクローンしている場合は、そのルートで同様に `uv sync` してください。
2. **uv** が PATH に通っていること（Studio の推論・学習の両方で使用）。
3. **学習デバイス**
   - **GPU（CUDA）推奨**。CPU でも動かせる設定はありますが、時間・実用性の面で厳しいことが多いです。
   - Studio の「学習デバイス」で `cuda` を選ぶ前に、NVIDIA 環境と PyTorch の CUDA 対応を確認してください。
4. **ディスク容量**
   - 潜在ベクトル保存、チェックポイント、ベースモデルのキャッシュで **数 GB〜** 単位になることがあります。

---

## 3. 全体フロー（ざっくり）

```text
[データ] 音声 + テキスト（+ 必要なら speaker / caption）
    ↓
prepare_manifest.py  … JSONL（manifest）+ DACVAE 潜在（latent）を生成
    ↓
train.py（LoRA 用 YAML + --init-checkpoint など）… LoRA アダプタを学習
    ↓
（必要に応じて）convert_checkpoint_to_safetensors.py … 推論用に変換
    ↓
infer などで音声合成（公式ドキュメントに従う）
```

詳細は **常に Irodori-TTS の README の Training 節**を優先してください。

---

## 4. manifest（JSONL）とは

学習は **`--manifest` で渡す JSONL** を起点にします。各行は少なくとも **テキストと潜在パス**に紐づき、モードによって **`speaker_id` や `caption` を含める**ことがあります（公式例・列名は README の *Prepare Manifest* を参照）。

**自分の声だけを特定話者として学習したい**場合は、データセット側で **話者 ID を一貫して付与**し、`prepare_manifest.py` で `--speaker-column` 等を使う流れ（公式例あり）を検討します。

---

## 5. LoRA 用 YAML の選び方（プリセット）

Irodori-TTS の `configs/` に、用途別の YAML が用意されています。Studio の **「設定プリセット」**は、次のファイルを **Irodori-TTS フォルダ**基準で指します。

| プリセット名（アプリ表示） | ファイル | 用途の目安 |
| ------------------------- | -------- | ---------- |
| `train_500m_v2_lora.yaml（基本モデル LoRA）` | `configs/train_500m_v2_lora.yaml` | 参照・話者条件など**基本系**の LoRA ファインチューニング |
| `train_500m_v2_voice_design_lora.yaml（VoiceDesign LoRA）` | `configs/train_500m_v2_voice_design_lora.yaml` | **キャプション条件**が中心の VoiceDesign 系 |

**manifest の内容**（`caption` の有無、`speaker_id` の使い方）と **YAML の `model` / `train` 設定**は整合させる必要があります。迷ったら **公式 README の該当セクション**と、リポジトリ内の `configs/*.yaml` のコメントを確認してください。

---

## 6. 初期重み（`--init-checkpoint`）

LoRA の**新規学習**では、公開されている **ベースモデルの `.safetensors`** を `--init-checkpoint` に指定するのが一般的です（[Fine-Tuning from Released Weights](https://github.com/Aratako/Irodori-TTS/blob/main/README.md#fine-tuning-from-released-weights)）。

- ファイルパスは **ローカルにダウンロード済みの絶対パス**で指定するのが確実です。
- Hugging Face のモデル ID をそのまま渡せるかは **実行環境・バージョンに依存**するため、公式の推奨に従ってください。

---

## 7. 中断からの再開（`--resume`）

**学習を再開する**ときは、LoRA の場合 **アダプタのチェックポイント「ディレクトリ」**を `--resume` に渡します（拡張子 `.safetensors` は **再開用ではなく初期化用**として区別されるので注意。README に明記あり）。

別マシンへ移したあとベースモデルのパスが無効になった場合は、`--resume` とあわせて **`--init-checkpoint` でベースを上書き指定**できるケースがあります（README の *Resuming* / LoRA の説明を参照）。

---

## 8. Irodori-Studio の「LoRA 作成（学習）」タブの使い方

1. **Irodori-TTS フォルダ**に、`train.py` があるディレクトリを指定する。
2. **学習 YAML** … 上記プリセットを選んで「パスに反映」するか、ファイルを直接指定。
3. **manifest（JSONL）** … `prepare_manifest.py` で生成したファイル。
4. **出力フォルダ** … チェックポイント等が保存される場所（空のディレクトリや新規名を推奨）。
5. **初期重み** … 新規学習時はベースの `.safetensors`。**再開のみ**の場合は空にして **再開フォルダ**だけ指定してもよい（どちらか必須になるよう UI が案内します）。
6. **学習デバイス** … `cpu` / `cuda`。
7. **コマンドをコピー** … ターミナルで手動実行・調整するときに便利。
8. **学習を開始** … 下ペインのログに `train.py` の標準出力が流れます（長時間処理になり得ます）。

**推論（音声を生成）**は **「参照音声」「VoiceDesign」タブ**で行います。LoRA タブでは推論しません。

---

## 9. LoRA のターゲット層（上級者向け）

`train.py` の `--lora-target-modules` では、公式 README に **プリセット名**（例: `diffusion_attn`, `all_attn_mlp` など）が列挙されています。YAML または CLI で上書きします。通常は **付属の `train_500m_v2_lora.yaml` の既定**から始め、必要に応じて公式議論・issue を参照してください。

---

## 10. 倫理・権利・品質

- **本人の同意のない他人の声**や、**権利のない素材**で学習しないでください。
- **実在人物の声の再現**を目的とする利用は、プライバシー・肖像・商用利用の観点で問題になることがあります。創作・自分のデータ・適切にライセンスされたデータに限定してください。
- **録音品質**（ノイズ、クリップ、部屋鳴り）や、**テキストと音声の対応ミス**は学習劣化の主因になります。データクレンジングを軽視しないでください。

---

## 11. うまくいかないとき

| 症状 | 確認すること |
| ---- | ------------- |
| `train.py が見つかりません` | **Irodori-TTS フォルダ**がリポジトリのルートか（親に `train.py` があるか） |
| `manifest が見つかりません` | パス誤り、`prepare_manifest.py` 未実行 |
| `CUDA が使えません` | ドライバ、`uv run python -c "import torch; print(torch.cuda.is_available())"` など |
| 学習は動くが品質が悪い | データ量・対応付け・YAML と manifest の整合・ステップ数・学習率（公式の推奨レンジ） |

---

## 12. 参考リンク（再掲）

- [Irodori-TTS README — Training](https://github.com/Aratako/Irodori-TTS/blob/main/README.md#training)
- [Fine-Tuning from Released Weights / LoRA（同一 README 内）](https://github.com/Aratako/Irodori-TTS/blob/main/README.md#fine-tuning-from-released-weights)

---

*このドキュメントは Irodori-Studio 用の概要です。挙動の正確な仕様は Irodori-TTS のバージョンごとの README・ソースに従ってください。*
