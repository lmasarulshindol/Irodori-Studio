# BabyDoll 台本 TTS マニフェスト

新宿・歌舞伎町の完全会員制ソープランド「BabyDoll」初接客オムニバス（全5話）の女性キャスト台本を、
Irodori-TTS VoiceDesign モデルで音声合成するためのマニフェストとバッチスクリプト一式です。

元の人間向け台本（収録ディレクション付き）は `001_声優・音声/Irodori-Studio/scripts/BabyDoll/` 配下、
本文原典は `999_その他/N000001/001_BabyDoll/本文/` 配下にあります。  
プロット原典は `999_その他/N000001/001_BabyDoll/プロット_初接客オムニバス.md`（**第11章「新フロー・絶頂回数設計」準拠**）。

## ディレクトリ構成

```text
scripts/babydoll/
├── README.md                           # このファイル
├── batch_voicedesign_babydoll.py       # バッチ合成スクリプト
├── manifests/                          # 各話のマニフェスト JSON
│   ├── ep1_riko.json                   # 第1話 「ピンクの契り」  佐藤 莉子（10歳／リコ）    44行
│   ├── ep2_kaede.json                  # 第2話 「紅の支配者」    桐原 楓（14歳／カエデ）   18行
│   ├── ep3_hinata.json                 # 第3話 「ひまわりの涙」  月島 ひなた（11歳／ヒナ） 43行
│   ├── ep4_kanon.json                  # 第4話 「ラベンダーの温度」 白石 花音（12歳／カノン）55行（ハミング1行 skip）
│   └── ep5_mio.json                    # 第5話 「蝶の標本」      御園 澪（13歳／ミオ）     49行
└── outputs/                            # WAV 出力先（自動生成・gitignore 推奨）
    └── ep{N}_{role}/                   # ep1_riko, ep2_kaede, ...
        └── {role}_{line:03d}.wav       # riko_001.wav, riko_002.wav, ...
```

**合計: 209行（合成対象 208行、ハミング 1行は skip_synthesis）**

## 行為本体の共通フロー（プロット第11章準拠）

```text
phase 2  入浴 → 口奉仕（muffled + breath_heavy） → 口内射精（1回目の射精）
phase 3  指愛撫 → 指絶頂（★花音は連続2回）
phase 4  生挿入 → 挿入中絶頂（★花音・澪のみ） → 射精時同時絶頂 → 中出し（2回目の射精）
phase 5  余韻
```

## キャスト別絶頂回数

| キャスト | 口内射精 | 指絶頂 | 挿入中絶頂 | 射精時絶頂 | 合計 | 声の特徴 |
|----------|:-:|:-:|:-:|:-:|:-:|-----|
| 莉子 10歳 | ○ | 1 | ― | 1 | **2** | 小悪魔→素の涙声 |
| 楓 14歳 | ○ | 1（無声） | ― | 1（無言） | **2** | 完全抑制・沈黙 |
| ひなた 11歳 | ○ | 1 | ― | 1 | **2** | ツンデレ→素の崩壊 |
| **花音 12歳 ★** | ○ | **2（連続）** | **1** | **1** | **4** | 溶ける癒し・ハミング |
| 澪 13歳 | ○ | 1 | 1 | 1 | **3** | 分析→被観察への反転 |
| **総計** | 5 | 6 | 2 | 5 | **13** | |

★ = 感じやすい子（全キャスト中最多の4回絶頂）

## マニフェスト JSON フォーマット

```jsonc
{
  "title": "BabyDoll 第1話「ピンクの契り」",
  "episode": 1,                            // エピソード番号
  "character": "riko",                     // role キー（英字・出力ファイル名に使用）
  "character_label": "佐藤 莉子",          // 表示名
  "genji_name": "リコ",                    // 源氏名
  "source_md": "001_声優・音声/.../第1話_莉子.md",
  "caption": "10歳の少女の声。ピッチは明らかに高め……",   // VoiceDesign 用長文キャプション
  "caption_short": "ピンクのシフォン越しに、小悪魔と10歳が入れ替わる。",
  "line_count": 44,
  "items": [
    {
      "role": "riko",                     // このアイテムの話者
      "role_label": "莉子",
      "line_index": 1,                    // 1-based の行番号
      "phase": 1,                         // 1=導入, 2=近づき/口奉仕, 3=高まり/指絶頂, 4=頂点/挿入・射精, 5=余韻
      "voice_tags": "pitch_high, pace_slow",
      "text": "失礼しまぁす。"
    }
    // ...
  ]
}
```

### 行レベルのオプションキー

| キー              | 型     | 用途                                                                 |
|-------------------|--------|----------------------------------------------------------------------|
| `skip_synthesis`  | bool   | `true` なら TTS でスキップ（例: ep4_kanon のパッヘルベルのカノン・ハミング） |
| `note`            | string | スキップ理由や別テイク指示などの補足メモ                             |

## 使い方

### 1. 環境準備

リポジトリのルート（`Irodori-TTS/`）で依存関係を入れておきます。

```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
cd "001_声優・音声/Irodori-Studio/Irodori-TTS"
uv sync
```

### 2. dry-run で行リストを確認

合成前に、マニフェストが正しく読めて行が揃っているか確認します。

```powershell
uv run python scripts/babydoll/batch_voicedesign_babydoll.py --all --dry-run
```

### 3. 1話ずつ合成

```powershell
# 第1話（莉子）だけ
uv run python scripts/babydoll/batch_voicedesign_babydoll.py `
    --manifest scripts/babydoll/manifests/ep1_riko.json

# 複数話をまとめて
uv run python scripts/babydoll/batch_voicedesign_babydoll.py `
    --manifest scripts/babydoll/manifests/ep1_riko.json `
    --manifest scripts/babydoll/manifests/ep3_hinata.json
```

### 4. 全話一括

```powershell
uv run python scripts/babydoll/batch_voicedesign_babydoll.py --all
```

### 5. 特定の行だけ（行番号範囲指定）

```powershell
# 花音の連続絶頂部分（仮に #26〜#35 に配置されている場合）だけ
uv run python scripts/babydoll/batch_voicedesign_babydoll.py `
    --manifest scripts/babydoll/manifests/ep4_kanon.json `
    --start 26 --end 35
```

### 6. 失敗してもスキップして続行

```powershell
uv run python scripts/babydoll/batch_voicedesign_babydoll.py --all --continue-on-error
```

### 7. 既存 WAV を上書き

デフォルトは既存 WAV があればスキップ。上書きしたいときだけ `--overwrite`。

## voice_tags グラデーション（プロット第11章準拠）

| phase | 場面 | 基本タグ | 追加タグ |
|---|---|---|---|
| 1 | 導入 | (キャラ基音) | `whisper` |
| 2 | 入浴 | (キャラ基音) | `whisper, pace_slow` |
| 2 | 口奉仕 | `breath_heavy` | `muffled` |
| 2 | 口内射精直後 | `breath_heavy` | `broken` |
| 3 | 指愛撫前半 | `breath_heavy` | (キャラ基音) |
| 3 | 指絶頂 | `broken, breath_heavy` | `cry` |
| 4 | 挿入瞬間 | `broken, breath_heavy` | `whisper` |
| 4 | 慣れ〜感じ | `breath_heavy` | (キャラ基音) |
| 4 | 挿入中絶頂 | `broken, breath_heavy` | `cry, pitch_high` |
| 4 | 射精時絶頂 | `broken, breath_heavy` | `cry, pitch_high` |
| 4 | 中出し直後 | `broken, breath_heavy` | `whisper, pace_slow` |
| 5 | 余韻 | (キャラ基音) | `whisper / pace_slow` |

## voice_tags 参考表

| タグ            | 意味                                               |
|-----------------|----------------------------------------------------|
| `pitch_high`    | 基音を高めに取る（小悪魔・年少寄り）              |
| `pitch_low`     | 基音を低めに取る（大人びた・支配的）              |
| `whisper`       | マイク近接の囁き、声量を極端に落とす              |
| `pace_slow`     | テンポを落とす、語尾を伸ばす                      |
| `breath_heavy`  | 息混じり・吐息                                    |
| `broken`        | 語尾の途切れ、掠れ、涙声                          |
| `cry`           | 涙声・泣きが混じる（絶頂時用）                    |
| `muffled`       | 口を塞がれた状態（口奉仕中）                      |

組み合わせ運用例:

- 莉子の小悪魔挑発: `pitch_high, pace_slow`
- 莉子の指絶頂: `broken, breath_heavy, cry, pitch_high`
- 楓の沈黙の支配: `whisper, pitch_low, pace_slow`
- 楓の口奉仕: `breath_heavy, muffled, pitch_low`
- ひなたの強がり: `pitch_high, broken`
- 花音の連続絶頂: `broken, breath_heavy, cry, pitch_high`
- 澪の分析崩壊: `broken, breath_heavy, cry, pitch_high`

## phase 段階

| phase | 意味   | 狙い                                     |
|-------|--------|------------------------------------------|
| 1     | 導入   | 入室・自己紹介。キャラの仮面が最も堅い   |
| 2     | 近づき | 入浴・口奉仕・口内射精。距離が縮む       |
| 3     | 高まり | 指愛撫〜指絶頂。素が滲み始める           |
| 4     | 頂点   | 挿入〜挿入中/射精時絶頂〜中出し          |
| 5     | 余韻   | 見送り。仮面を被り直す過程              |

## 備考

- `ep4_kanon.json` のハミング行は `skip_synthesis: true` を付けてあります。
  TTS では生成せず、別テイクの鼻歌 / ピアノ素材を推奨（パッヘルベルのカノン・ニ長調第一主題）。
- 口奉仕中の非言語行（「んっ」「ぅっ」「ごくん」等）は短いが独立 WAV として合成。
  長さが極端に短くなるため、必要に応じて `--seconds` を `5.0` 程度に下げて時間短縮も可。
- 出力 WAV のサンプルレートは VoiceDesign モデルの codec（Semantic-DACVAE-Japanese-32dim, 48kHz）に準じます。
- `outputs/` 配下は gitignore 対象にすることを推奨。
