# セリフ一括生成ガイドライン（Irodori-Studio / Irodori-TTS）

複数セリフを **モデル1回ロード**で連続合成する手順と、マニフェスト・スタイル指定の指針です。公式エンジンの仕様は [Irodori-TTS README](https://github.com/Aratako/Irodori-TTS) を優先してください。

## 目的

- **数十〜百行規模**のセリフを、WebUI で1行ずつ叩く代わりに **一括で WAV 出力**する。
- **行ごとに本文・キャプション（スタイル）を変えられる**（VoiceDesign 系チェックポイント向け）。

## 方式の選び方

| 方式 | 向いていること |
|------|----------------|
| **GUI「連続生成」**（ファイルメニュー） | Studio 上の設定（参照／VoiceDesign・チェックポイント）をそのまま使い、**本文だけ行分割**したいとき。 |
| **JSON バッチ**（`studio_batch_infer.py`） | スクリプトや CI から再現したい。**行ごとの `caption` もマニフェストで指定**できる。 |
| **HTTP API**（`tts_batch_server.py` + `batch_api_client.py`） | 別プロセスから `POST` したい・クライアントを自分で書きたいとき（要 `uv sync --extra api`）。 |

いずれも **Irodori-TTS はリポジトリ内 `Irodori-TTS/` に同梱**されています（`infer.py` のあるディレクトリ）。

## 事前準備

1. **依存関係**（Irodori-TTS ルートで一度だけ）:

   ```powershell
   cd Irodori-TTS
   uv sync
   ```

   HTTP API を使う場合はさらに `uv sync --extra api`。

2. **チェックポイント**: Hugging Face から初回実行時に取得されます。VoiceDesign で **行ごとにキャプション**を変える場合は、VoiceDesign 用のモデル（例: `Aratako/Irodori-TTS-500M-v2-VoiceDesign`）をマニフェストの `hf_checkpoint` に指定します。

3. **デバイス**: GPU がある環境では `model_device` / `codec_device` を `cuda` にすると速いです。CPU のみの場合は `cpu` とし、`num_steps` を下げると試しやすいです。

## JSON バッチ（推奨・再現性が高い）

### マニフェストの骨子

- ルートに **`hf_checkpoint`**（またはローカル **`checkpoint`**）、**`model_device`**、**`codec_device`**、**`num_steps`**、話者条件（**`no_ref`** または **`ref_wav`**）など。
- **`items`** は配列。各要素に最低限:
  - **`text`**: セリフ本文
  - **`output_wav`**: 出力 WAV パス（Irodori-TTS ルートからの相対パスでよい）

### 行ごとのスタイル（VoiceDesign）

`items[]` の各要素に任意で **`caption`** を書きます。

- **行に `caption` がある**: その行だけそのキャプションを使う。
- **省略**: ルートの **`caption`**（あれば）にフォールバック（従来どおり）。

例（抜粋）は `Irodori-TTS/scripts/examples/loli_trial_batch.json` を参照。

### 実行例

```powershell
cd Irodori-TTS
uv run python scripts/studio_batch_infer.py --config scripts/examples/loli_trial_batch.json
```

ログに `[batch] model loaded into memory` が一度出たあと、各行が順に保存されます。

## GUI「連続生成」との関係

Studio の **ファイル → 連続生成（本文を行ごと）** は、現在のタブ設定から **バッチ用 JSON を生成**し、内部で `studio_batch_infer.py` に相当する一括実行を行います。細かい **行ごとキャプション**をマニフェストで管理したい場合は、上記 JSON を手で編集してから `studio_batch_infer.py` を直接叩くとよいです。

## HTTP API（任意）

- **サーバー**（モデル常駐）: `Irodori-TTS/scripts/tts_batch_server.py`
- **クライアント**（JSON Lines マニフェスト）: `Irodori-TTS/scripts/batch_api_client.py`

起動例・エンドポイントは各スクリプト先頭の docstring を参照してください。

## マニフェスト作成の指針

1. **1行1セリフ**: 長すぎる文は分割する（学習・推論ともトークン上限に注意）。
2. **キャプション**（VoiceDesign）: 話者の属性・距離感・口調・感情を **日本語で具体的に**。テキスト作成の一般論は [text-creation-guideline.md](text-creation-guideline.md) とあわせる。
3. **ファイル名**: `001.wav` のように連番にすると管理しやすい。役名・シーンをプレフィックスにしてもよい。
4. **再現性**: 同じノイズに揃えたいときはマニフェストに **`seed`** を固定（全行同一シードになる点に注意）。
5. **Git**: 生成した **`outputs/` 以下の WAV はコミットしない**（`.gitignore` 済み）。マニフェスト JSON だけバージョン管理する運用を推奨。

## トラブルシュート

| 症状 | 確認すること |
|------|----------------|
| `infer.py が見つかりません` | Studio の **Irodori-TTS フォルダ**が `…/Irodori-Studio/Irodori-TTS` を指しているか。 |
| 話者条件エラー | 参照音声モデルなら **`ref_wav`**、VoiceDesign なら **`no_ref`: true** など、チェックポイントに合った設定か。 |
| 遅い | CPU か、ステップ数・秒数が大きい。GPU 利用または `num_steps` 削減。 |

---

* upstream の挙動は [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のバージョンによります。*
