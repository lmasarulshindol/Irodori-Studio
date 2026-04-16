# Irodori-Studio

ローカルで動かしている [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) に、テキスト（絵文字による感情表現を含む）を読ませるための **Windows 向け GUI** です。

## 前提

- **Irodori-TTS 本体**を同じ親フォルダにクローンし、`uv sync` まで完了していること（例: `001_声優・音声/Irodori-TTS`）。
- **uv** が入っていること（未導入の場合は PowerShell で次を実行）。

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後はターミナルを開き直すか、`%USERPROFILE%\.local\bin` が PATH に通っていることを確認してください。

## 起動

エクスプローラで `Irodori-Studio` を開き、`run_gui.bat` をダブルクリックするか、次を実行します。

```powershell
cd Irodori-Studio
py -3 main.py
```

初回の音声生成時は、Hugging Face からチェックポイントがダウンロードされます（時間とディスク容量がかかります）。

## テキスト作成のガイドライン

読み上げ本文・キャプション・絵文字の書き方の指針は次を参照してください。

- [docs/text-creation-guideline.md](docs/text-creation-guideline.md)

## GUI の使い方

1. **Irodori-TTS フォルダ**: `infer.py` があるディレクトリ（既定は隣の `Irodori-TTS`）。
2. **モードのタブ**:
   - `参照音声（基本モデル）`: 参照 WAV で声を寄せる
   - `VoiceDesign（キャプション）`: 参照なしで、キャプションで話者を指定する
3. **保存プロファイル**: 自分で保存した設定・本文・キャプション等をまとめて呼び出します。
4. **台本サンプル**: 最初から入っている本文例を差し替えます。
5. **読み上げテキスト**: 本文。絵文字ボタンから感情用の絵文字を挿入できます。
6. **キャプション**: VoiceDesign 利用時に必須です。話者・感情・話し方を日本語で指定します。
7. **参照音声**: 参照音声タブで使用。オフにすると `--no-ref` で生成します。
8. **出力 WAV**: 既定では自動ファイル名です。手動指定時は保存先を選べます。
9. **音声を生成**: `uv run python infer.py ...` をバックグラウンドで実行します。

## ヘルプ

- アプリの **ヘルプ** メニューから、テキスト作成ガイドと VoiceDesign の参考リンクを開けます。

## ディレクトリ構成の例

```text
001_声優・音声/
  Irodori-TTS/      # git clone https://github.com/Aratako/Irodori-TTS.git && uv sync
  Irodori-Studio/   # このリポジトリ（GUI）
```

## ライセンス

GUI 部分のライセンスは本リポジトリに従います。音声合成エンジンは [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のライセンスに従ってください。
