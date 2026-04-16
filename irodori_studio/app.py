"""Irodori-TTS 連携のメインウィンドウ（tkinter）。"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import (
    END,
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    filedialog,
    Frame,
    Label,
    Menu,
    PanedWindow,
    messagebox,
    scrolledtext,
    StringVar,
    TclError,
    Tk,
    ttk,
)
from tkinter.simpledialog import askstring

from irodori_studio.paths import default_irodori_tts_dir, studio_root
from irodori_studio import storage
from irodori_studio import voice_design_presets as vd_presets
from irodori_studio.text_presets import (
    DEFAULT_SAMPLE_TEXT,
    PRESET_LABELS,
    body_text_for_label,
)

CHECKPOINTS = [
    ("Irodori-TTS-500M-v2（基本）", "Aratako/Irodori-TTS-500M-v2"),
    ("Irodori-TTS-500M-v2-VoiceDesign", "Aratako/Irodori-TTS-500M-v2-VoiceDesign"),
]
CHECKPOINT_LABEL_TO_ID = dict(CHECKPOINTS)
CHECKPOINT_LABELS = [c[0] for c in CHECKPOINTS]
BASE_CHECKPOINT_LABEL = CHECKPOINTS[0][0]
TEXT_GUIDE_PATH = studio_root() / "docs" / "text-creation-guideline.md"
PROFILE_PLACEHOLDER = "保存プロファイルを選ぶ…"
HISTORY_PLACEHOLDER = "履歴を選ぶ…"

# Windows 等で使えない文字を除いたファイル名用セグメント
_INVALID_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _sanitize_filename_segment(s: str, max_len: int) -> str:
    t = _INVALID_FILENAME.sub("", s.replace("\r", "").replace("\n", ""))
    t = t.strip()
    if len(t) > max_len:
        t = t[:max_len]
    return t if t else "x"


def _next_numbered_wav_path(out_dir: Path, base: str) -> Path:
    """base_NNN.wav で既存と重複しないパス（NNN は 001 から）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    b = base.strip("_") or "out"
    n = 1
    while n <= 9999:
        p = out_dir / f"{b}_{n:03d}.wav"
        if not p.exists():
            return p
        n += 1
    from time import time

    return out_dir / f"{b}_{int(time() * 1000)}.wav"


EMOJI_PRESETS = [
    "😊",
    "😢",
    "😤",
    "😴",
    "🎉",
    "😮",
    "🥺",
    "✨",
    "😭",
    "🤔",
    "💢",
    "🤫",
    "🔥",
    "💤",
]

CAPTION_TEMPLATES: list[tuple[str, str]] = [
    ("（テンプレなし）", ""),
    ("落ち着いた女性・近距離", "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"),
    ("ナレーション・落ち着き", "落ち着いたナレーション調で、はっきり聞き取りやすく読み上げてください。"),
    ("元気・明るい", "明るく元気な雰囲気で、テンポよく読み上げてください。"),
    ("ささやき・小声", "小声でささやくように、息づかいを感じる近い距離感で読み上げてください。"),
    ("落ち込み・しっとり", "少し落ち込んだ雰囲気で、しっとりと読み上げてください。"),
]

def _which_uv() -> str | None:
    local = Path.home() / ".local" / "bin"
    for name in ("uv.exe", "uv"):
        p = local / name
        if p.is_file():
            return str(p)
    import shutil

    return shutil.which("uv")


def _env_with_uv_on_path() -> dict[str, str]:
    env = os.environ.copy()
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.is_dir():
        env["Path"] = f"{local_bin};{env.get('Path', '')}"
    return env


def torch_cuda_available(tts_dir: Path, uv_bin: str, *, timeout: float = 120.0) -> bool:
    """Irodori-TTS の venv 上で torch.cuda.is_available() が真かどうか。"""
    try:
        proc = subprocess.run(
            [
                uv_bin,
                "run",
                "python",
                "-c",
                "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)",
            ],
            cwd=str(tts_dir),
            env=_env_with_uv_on_path(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _open_path_windows(path: Path) -> None:
    """エクスプローラでフォルダを開く、またはファイルの既定アプリで開く。"""
    try:
        os.startfile(path)  # type: ignore[attr-defined]
    except OSError:
        subprocess.run(["explorer", "/select,", str(path)], check=False)


def _open_folder_windows(folder: Path) -> None:
    try:
        os.startfile(folder)  # type: ignore[attr-defined]
    except OSError:
        subprocess.run(["explorer", str(folder)], check=False)


def _device_choices() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("cpu", "mps")
    return ("cpu", "cuda")


class IrodoriStudioApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("Irodori-Studio")
        root.minsize(720, 640)
        root.geometry("900x720")

        self.var_tts_dir = StringVar(value=str(default_irodori_tts_dir()))
        self.var_text = StringVar()
        self.var_caption = StringVar()
        self.var_ref_wav = StringVar()
        self.var_no_ref = BooleanVar(value=True)
        self.var_output_wav = StringVar(
            value=str(studio_root() / "outputs" / "last.wav")
        )
        self.var_device = StringVar(value="cpu")
        self.var_num_steps = StringVar(value="40")
        self.var_seed = StringVar(value="")
        self.var_caption_template = StringVar(value=CAPTION_TEMPLATES[0][0])
        self.var_preset_pick = StringVar(value=PROFILE_PLACEHOLDER)
        self.var_notify_done = BooleanVar(value=False)
        self.var_history_pick = StringVar(value=HISTORY_PLACEHOLDER)
        self.var_voice_design_char = StringVar(value=vd_presets.NONE_LABEL)
        self.var_auto_filename = BooleanVar(value=True)
        self.var_body_preset_label = StringVar(value=PRESET_LABELS[0])
        self.status = StringVar(value="準備OK")

        self._build_menu(root)
        self._build_main_paned(root)

    def _build_menu(self, root: Tk) -> None:
        menubar = Menu(root)
        root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="台本テキストを開く…", command=self._load_text_file)
        file_menu.add_command(label="クリップボードを貼り付け", command=self._paste_clipboard)
        file_menu.add_command(
            label="台本サンプルを挿入",
            command=self._insert_sample_text,
        )
        file_menu.add_separator()
        file_menu.add_command(label="履歴をクリア", command=self._clear_history_storage)
        file_menu.add_separator()
        file_menu.add_command(label="アプリを再起動", command=self._restart_application)

        preset_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="保存プロファイル", menu=preset_menu)
        preset_menu.add_command(
            label="現在の設定を保存…", command=self._save_preset_dialog
        )

        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="Irodori-TTS（公式）", command=self._open_upstream)
        help_menu.add_command(
            label="テキスト作成ガイドを開く",
            command=self._open_text_creation_guide,
        )
        help_menu.add_command(
            label="VoiceDesign モデルカード（HF）",
            command=self._open_voicedesign_card,
        )
        help_menu.add_command(
            label="絵文字アノテーション一覧（HF）",
            command=self._open_emoji_annotations,
        )
        help_menu.add_command(label="このアプリについて", command=self._about)

    def _build_main_paned(self, root: Tk) -> None:
        paned = PanedWindow(root, orient="vertical", sashrelief="raised")
        paned.pack(fill="both", expand=True)

        upper = Frame(paned)
        lower = Frame(paned)
        paned.add(upper)
        paned.add(lower)
        paned.paneconfig(upper, minsize=380)
        paned.paneconfig(lower, minsize=120)

        self._build_form(upper)
        self._build_log(lower)

    def _build_log(self, parent: Frame) -> None:
        f = Frame(parent)
        f.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        action = Frame(f)
        action.pack(fill="x", pady=(0, 4))
        self.btn_gen = Button(
            action, text="音声を生成", command=self._on_generate, height=2
        )
        self.btn_gen.pack(fill="x", pady=(0, 4))
        Label(action, textvariable=self.status, anchor="w").pack(fill="x")
        top = Frame(f)
        top.pack(fill="x")
        Label(top, text="実行ログ").pack(side="left")
        Button(top, text="ログをコピー", command=self._copy_log).pack(side="right", padx=2)
        Button(top, text="ログをクリア", command=self._clear_log).pack(side="right", padx=2)
        self.txt_log = scrolledtext.ScrolledText(
            f, height=8, wrap="word", state="disabled", font=("Consolas", 9)
        )
        self.txt_log.pack(fill="both", expand=True)

    def _log_append(self, text: str) -> None:
        self.txt_log.configure(state="normal")
        self.txt_log.insert(END, text)
        self.txt_log.see(END)
        self.txt_log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", END)
        self.txt_log.configure(state="disabled")

    def _copy_log(self) -> None:
        self.txt_log.configure(state="normal")
        content = self.txt_log.get("1.0", END)
        self.txt_log.configure(state="disabled")
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status.set("ログをクリップボードにコピーしました")

    def _build_form(self, root: Frame) -> None:
        pad = {"padx": 8, "pady": 3}

        f0 = Frame(root)
        f0.pack(fill="x", **pad)
        Label(f0, text="Irodori-TTS フォルダ").pack(side="left")
        Entry(f0, textvariable=self.var_tts_dir, width=72).pack(
            side="left", fill="x", expand=True, padx=4
        )
        Button(f0, text="フォルダを選択…", command=self._browse_tts_dir).pack(side="left")

        self._notebook = ttk.Notebook(root)
        self._notebook.pack(fill="x", **pad)

        tab_ref = Frame(self._notebook, padx=6, pady=6)
        self._notebook.add(tab_ref, text="参照音声（基本モデル）")
        Label(
            tab_ref,
            text=f"チェックポイント: {BASE_CHECKPOINT_LABEL}",
            anchor="w",
        ).pack(fill="x")
        Label(
            tab_ref,
            text="特定の声に寄せたいときのモードです。参照 WAV を使うと声を複製し、オフにすると汎用読み上げになります。",
            fg="#555555",
            anchor="w",
            wraplength=820,
            justify="left",
        ).pack(fill="x", pady=(0, 6))
        f4 = Frame(tab_ref)
        f4.pack(fill="x")
        Checkbutton(
            f4,
            text="参照音声を使わない（--no-ref）",
            variable=self.var_no_ref,
            command=self._toggle_ref,
        ).pack(side="left")
        Label(f4, text="参照 WAV").pack(side="left", padx=(16, 0))
        self.entry_ref = Entry(f4, textvariable=self.var_ref_wav, width=52)
        self.entry_ref.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_ref = Button(f4, text="WAVを選択…", command=self._browse_ref_wav)
        self.btn_ref.pack(side="left")
        self._toggle_ref()

        tab_vd = Frame(self._notebook, padx=6, pady=6)
        self._notebook.add(tab_vd, text="VoiceDesign（キャプション）")
        Label(
            tab_vd,
            text=f"モデル: {vd_presets.VOICEDESIGN_CHECKPOINT_LABEL}",
            anchor="w",
        ).pack(fill="x")
        Label(
            tab_vd,
            text="参照音声なしで、キャプションだけで話者・感情・話し方を決めます。本文の絵文字と併用できます。",
            fg="#555555",
            anchor="w",
            wraplength=820,
            justify="left",
        ).pack(fill="x", pady=(0, 4))
        fvd = Frame(tab_vd)
        fvd.pack(fill="x", pady=(0, 4))
        Label(fvd, text="キャラプリセット").pack(side="left")
        self._combo_vd_char = ttk.Combobox(
            fvd,
            textvariable=self.var_voice_design_char,
            values=vd_presets.combo_labels(),
            width=34,
            state="readonly",
        )
        self._combo_vd_char.pack(side="left", padx=4)
        Button(fvd, text="適用", command=self._apply_voice_design_character).pack(
            side="left"
        )
        Label(
            tab_vd,
            text="キャプション（話者・演技の指示）",
        ).pack(anchor="w")
        row_c = Frame(tab_vd)
        row_c.pack(fill="x")
        Label(row_c, text="テンプレ").pack(side="left")
        tmpl_labels = [t[0] for t in CAPTION_TEMPLATES]
        ttk.Combobox(
            row_c,
            textvariable=self.var_caption_template,
            values=tmpl_labels,
            width=20,
            state="readonly",
        ).pack(side="left", padx=4)
        Button(row_c, text="反映", command=self._apply_caption_template).pack(
            side="left"
        )
        Entry(row_c, textvariable=self.var_caption, width=58).pack(
            side="left", fill="x", expand=True, padx=4
        )

        self._notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)
        self._on_notebook_tab_changed()
        self._combo_vd_char.bind("<<ComboboxSelected>>", self._on_voice_character_selected)

        fp = Frame(root)
        fp.pack(fill="x", **pad)
        Label(fp, text="保存プロファイル").pack(side="left")
        self._combo_preset = ttk.Combobox(
            fp,
            textvariable=self.var_preset_pick,
            values=self._preset_names(),
            width=28,
            state="readonly",
        )
        self._combo_preset.pack(side="left", padx=4)
        self.btn_apply_profile = Button(fp, text="読み込む", command=self._apply_preset)
        self.btn_apply_profile.pack(side="left")
        Button(fp, text="一覧更新", command=self._refresh_preset_combo).pack(side="left", padx=4)
        Label(
            fp,
            text="自分で保存した設定・本文・キャプションをまとめて呼び出します。",
            fg="#555555",
        ).pack(side="left", padx=6)

        fh = Frame(root)
        fh.pack(fill="x", **pad)
        Label(fh, text="履歴").pack(side="left")
        self._combo_history = ttk.Combobox(
            fh,
            textvariable=self.var_history_pick,
            values=self._history_labels(),
            width=52,
            state="readonly",
        )
        self._combo_history.pack(side="left", padx=4, fill="x", expand=True)
        self.btn_apply_history = Button(fh, text="読み込む", command=self._apply_history_selection)
        self.btn_apply_history.pack(side="left")
        self._combo_preset.bind("<<ComboboxSelected>>", self._refresh_action_buttons)
        self._combo_history.bind("<<ComboboxSelected>>", self._refresh_action_buttons)
        self._refresh_action_buttons()

        f2 = Frame(root)
        f2.pack(fill="both", expand=True, **pad)
        row_t = Frame(f2)
        row_t.pack(fill="x")
        Label(row_t, text="読み上げテキスト（絵文字で感情表現可）").pack(side="left")
        Button(row_t, text="クリップボード貼付", command=self._paste_clipboard).pack(
            side="right"
        )
        Button(row_t, text="サンプル", command=self._insert_sample_text).pack(
            side="right", padx=4
        )
        row_tp = Frame(f2)
        row_tp.pack(fill="x", pady=(0, 2))
        Label(row_tp, text="台本サンプル:").pack(side="left")
        self._combo_body_preset = ttk.Combobox(
            row_tp,
            textvariable=self.var_body_preset_label,
            values=PRESET_LABELS,
            state="readonly",
            width=26,
        )
        self._combo_body_preset.pack(side="left", padx=4)
        Button(
            row_tp,
            text="このサンプルに差し替え",
            command=self._apply_selected_body_preset,
        ).pack(side="left", padx=2)
        Label(
            row_tp,
            text="最初から入っている台本サンプルです。保存プロファイルとは別物です。",
            fg="#555555",
        ).pack(side="left", padx=6)
        self.txt_body = scrolledtext.ScrolledText(f2, height=9, wrap="word")
        self.txt_body.pack(fill="both", expand=True, pady=(2, 0))
        self.txt_body.insert("1.0", DEFAULT_SAMPLE_TEXT)

        f_emoji = Frame(root)
        f_emoji.pack(fill="x", **pad)
        Label(f_emoji, text="絵文字:").pack(side="left")
        for em in EMOJI_PRESETS[:8]:
            Button(f_emoji, text=em, width=3, command=lambda e=em: self._insert_emoji(e)).pack(
                side="left", padx=1
            )
        f_emoji2 = Frame(root)
        f_emoji2.pack(fill="x", **pad)
        Label(f_emoji2, text="").pack(side="left", padx=32)
        for em in EMOJI_PRESETS[8:]:
            Button(f_emoji2, text=em, width=3, command=lambda e=em: self._insert_emoji(e)).pack(
                side="left", padx=1
            )

        f5 = Frame(root)
        f5.pack(fill="x", **pad)
        Checkbutton(
            f5,
            text="ファイル名自動",
            variable=self.var_auto_filename,
        ).pack(side="left")
        Label(f5, text="出力 WAV").pack(side="left", padx=(4, 0))
        Entry(f5, textvariable=self.var_output_wav, width=48).pack(
            side="left", fill="x", expand=True, padx=4
        )
        Button(f5, text="保存先を選択…", command=self._browse_output_wav).pack(side="left")
        Button(f5, text="フォルダ", command=self._open_output_folder).pack(side="left", padx=2)
        Button(f5, text="パスコピー", command=self._copy_output_path).pack(side="left", padx=2)
        Button(f5, text="再生", command=self._play_output_wav).pack(side="left", padx=2)
        f5h = Frame(root)
        f5h.pack(fill="x", padx=8, pady=(0, 2))
        Label(
            f5h,
            text="自動ON: outputs/generated に「声の種類_セリフ先頭_連番.wav」で保存（探しやすい）",
            fg="#555555",
            font=("", 8),
        ).pack(anchor="w")

        f6 = Frame(root)
        f6.pack(fill="x", **pad)
        Label(f6, text="デバイス").pack(side="left")
        self._device_combo = ttk.Combobox(
            f6,
            textvariable=self.var_device,
            values=_device_choices(),
            width=8,
            state="readonly",
        )
        self._device_combo.pack(side="left", padx=4)
        Label(f6, text="品質（ステップ数）").pack(side="left", padx=(12, 0))
        Entry(f6, textvariable=self.var_num_steps, width=5).pack(side="left")
        Label(f6, text="シード（空＝毎回ランダム）").pack(side="left", padx=(12, 0))
        Entry(f6, textvariable=self.var_seed, width=10).pack(side="left", padx=4)
        Checkbutton(
            f6,
            text="完了ダイアログ",
            variable=self.var_notify_done,
        ).pack(side="left", padx=(16, 0))

    def _current_checkpoint_label(self) -> str:
        """履歴・プリセット用のチェックポイント表示名。"""
        try:
            idx = self._notebook.index(self._notebook.select())
        except TclError:
            return BASE_CHECKPOINT_LABEL
        if idx == 1:
            return vd_presets.VOICEDESIGN_CHECKPOINT_LABEL
        return BASE_CHECKPOINT_LABEL

    def _on_notebook_tab_changed(self, _event: object = None) -> None:
        try:
            title = self._notebook.tab(self._notebook.select(), "text")
        except TclError:
            return
        if "VoiceDesign" in title:
            self.status.set(f"モード: {title}（キャプション必須）")
        else:
            self.status.set(f"モード: {title}")

    def _refresh_action_buttons(self, _event: object = None) -> None:
        profile_ready = self.var_preset_pick.get().strip() not in {"", PROFILE_PLACEHOLDER}
        history_ready = self.var_history_pick.get().strip() not in {"", HISTORY_PLACEHOLDER, "（履歴なし）"}
        self.btn_apply_profile.configure(state="normal" if profile_ready else "disabled")
        self.btn_apply_history.configure(state="normal" if history_ready else "disabled")

    def _on_voice_character_selected(self, _event: object = None) -> None:
        name = self.var_voice_design_char.get().strip()
        if name and name != vd_presets.NONE_LABEL:
            self.status.set(f"VoiceDesignキャラを選択: {name}（適用で反映）")

    def _voice_slug_for_filename(self) -> str:
        """自動ファイル名の先頭（声のタイプ／キャラ）。"""
        try:
            mode_idx = self._notebook.index(self._notebook.select())
        except TclError:
            mode_idx = 0
        if mode_idx == 0:
            return "NOREF" if self.var_no_ref.get() else "REF"
        char = self.var_voice_design_char.get().strip()
        if char and char != vd_presets.NONE_LABEL:
            short = char.replace("【公式】", "").strip()
            return _sanitize_filename_segment(short, 28)
        return "VD"

    def _compute_auto_output_wav_path(self) -> Path:
        """outputs/generated / 声種_セリフ抜粋_連番.wav"""
        out_dir = studio_root() / "outputs" / "generated"
        voice = self._voice_slug_for_filename()
        body = self.txt_body.get("1.0", END).strip()
        first_line = body.split("\n")[0] if body else ""
        text_part = _sanitize_filename_segment(first_line, 42)
        base = f"{voice}_{text_part}"
        if len(base) > 115:
            base = base[:115]
        return _next_numbered_wav_path(out_dir, base)

    def _apply_voice_design_character(self) -> None:
        """VoiceDesign タブへ切替え、キャプションを設定。"""
        name = self.var_voice_design_char.get().strip()
        if not name or name == vd_presets.NONE_LABEL:
            messagebox.showinfo(
                "VoiceDesign",
                "キャラを選んでから「適用」を押してください。\n\n"
                "VoiceDesign は --caption で話者や演技を日本語で指示し、"
                "参照音声は使いません（--no-ref）。\n"
                "本文に絵文字を入れると、さらに細かく演出できます。",
            )
            return
        cap = vd_presets.caption_for_display_name(name)
        if cap is None:
            messagebox.showerror("VoiceDesign", "不明なキャラ名です。")
            return
        self._notebook.select(1)
        self.var_no_ref.set(True)
        self._toggle_ref()
        self.var_caption.set(cap)
        self.status.set(f"VoiceDesign キャラ適用: {name}")

    def _open_voicedesign_card(self) -> None:
        webbrowser.open(
            "https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign"
        )

    def _open_text_creation_guide(self) -> None:
        if not TEXT_GUIDE_PATH.is_file():
            messagebox.showerror("ガイド", f"ガイドが見つかりません:\n{TEXT_GUIDE_PATH}")
            return
        _open_path_windows(TEXT_GUIDE_PATH)

    def _open_emoji_annotations(self) -> None:
        webbrowser.open(
            "https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign/blob/main/EMOJI_ANNOTATIONS.md"
        )

    def _preset_names(self) -> list[str]:
        presets = storage.load_presets()
        return [str(p.get("name", "")) for p in presets if p.get("name")]

    def _refresh_preset_combo(self) -> None:
        names = self._preset_names()
        self._combo_preset["values"] = names if names else [PROFILE_PLACEHOLDER]
        if self.var_preset_pick.get().strip() not in names:
            self.var_preset_pick.set(PROFILE_PLACEHOLDER)
        self._refresh_action_buttons()
        self.status.set(f"保存プロファイル {len(names)} 件")

    def _apply_preset(self) -> None:
        name = self.var_preset_pick.get().strip()
        if not name or name == PROFILE_PLACEHOLDER:
            self.status.set("保存プロファイルを選んでください")
            return
        for p in storage.load_presets():
            if str(p.get("name")) == name:
                cl = str(p.get("checkpoint_label", ""))
                if cl == vd_presets.VOICEDESIGN_CHECKPOINT_LABEL:
                    self._notebook.select(1)
                elif cl in CHECKPOINT_LABELS:
                    self._notebook.select(0)
                if p.get("device") in _device_choices():
                    self.var_device.set(str(p["device"]))
                else:
                    self.var_device.set("cpu")
                if p.get("num_steps") is not None:
                    self.var_num_steps.set(str(p["num_steps"]))
                else:
                    self.var_num_steps.set("40")
                if "caption" in p and p["caption"] is not None:
                    self.var_caption.set(str(p["caption"]))
                else:
                    self.var_caption.set("")
                if "no_ref" in p:
                    self.var_no_ref.set(bool(p["no_ref"]))
                else:
                    self.var_no_ref.set(True)
                if p.get("ref_wav"):
                    self.var_ref_wav.set(str(p["ref_wav"]))
                else:
                    self.var_ref_wav.set("")
                self.var_seed.set(str(p.get("seed", "")))
                self._toggle_ref()
                if "reading_text" in p:
                    self.txt_body.delete("1.0", END)
                    self.txt_body.insert("1.0", str(p["reading_text"]))
                self.status.set(f"保存プロファイル読込: {name}")
                return

    def _save_preset_dialog(self) -> None:
        name = askstring(
            "保存プロファイル",
            "保存名（タブ、本文、キャプション、参照設定などを丸ごと保存）:",
        )
        if not name or not name.strip():
            return
        name = name.strip()
        reading_text = self.txt_body.get("1.0", END).replace("\r\n", "\n").rstrip("\n")
        preset = {
            "name": name,
            "checkpoint_label": self._current_checkpoint_label(),
            "device": self.var_device.get().strip(),
            "num_steps": self.var_num_steps.get().strip(),
            "seed": self.var_seed.get().strip(),
            "caption": self.var_caption.get().strip(),
            "no_ref": self.var_no_ref.get(),
            "ref_wav": self.var_ref_wav.get().strip(),
            "reading_text": reading_text,
        }
        presets = storage.load_presets()
        presets = [p for p in presets if str(p.get("name")) != name]
        presets.append(preset)
        storage.save_presets(presets)
        self._refresh_preset_combo()
        self.var_preset_pick.set(name)
        self._refresh_action_buttons()
        self.status.set(f"保存プロファイル保存: {name}")

    def _history_labels(self) -> list[str]:
        entries = storage.load_history()
        out: list[str] = []
        for i, e in enumerate(entries[:25]):
            at = str(e.get("at", ""))[:19].replace("T", " ")
            preview = str(e.get("text", "")).replace("\n", " ")[:36]
            mode = "VoiceDesign" if str(e.get("checkpoint", "")) == vd_presets.VOICEDESIGN_CHECKPOINT_LABEL else "参照"
            out.append(f"{i + 1}. [{at}] ({mode}) {preview}")
        return out if out else ["（履歴なし）"]

    def _refresh_history_combo(self) -> None:
        labels = self._history_labels()
        self._combo_history["values"] = labels
        self.var_history_pick.set(HISTORY_PLACEHOLDER if labels != ["（履歴なし）"] else "（履歴なし）")
        self._refresh_action_buttons()

    def _apply_history_selection(self) -> None:
        label = self.var_history_pick.get()
        if not label or label in {HISTORY_PLACEHOLDER, "（履歴なし）"}:
            self.status.set("履歴を選んでください")
            return
        idx_s = label.split(".", 1)[0].strip()
        try:
            idx = int(idx_s) - 1
        except ValueError:
            return
        entries = storage.load_history()
        if 0 <= idx < len(entries):
            e = entries[idx]
            self.txt_body.delete("1.0", END)
            self.txt_body.insert("1.0", str(e.get("text", "")))
            self.var_output_wav.set(str(e.get("output", self.var_output_wav.get())))
            ck = str(e.get("checkpoint", ""))
            if ck == vd_presets.VOICEDESIGN_CHECKPOINT_LABEL:
                self._notebook.select(1)
            else:
                self._notebook.select(0)
            self.var_caption.set(str(e.get("caption", "")))
            self.var_ref_wav.set(str(e.get("ref_wav", "")))
            self.var_no_ref.set(bool(e.get("no_ref", True)))
            device = str(e.get("device", "cpu"))
            self.var_device.set(device if device in _device_choices() else "cpu")
            self.var_num_steps.set(str(e.get("num_steps", "40")))
            self.var_seed.set(str(e.get("seed", "")))
            self._toggle_ref()
            self.status.set("履歴をフォームに読み込みました")

    def _clear_history_storage(self) -> None:
        if not messagebox.askyesno("確認", "履歴をすべて削除しますか？"):
            return
        storage.save_history([])
        self._refresh_history_combo()
        self.status.set("履歴をクリアしました")

    def _restart_application(self) -> None:
        """新しいプロセスで main.py を起動し、現在のウィンドウを終了する。"""
        main_py = studio_root() / "main.py"
        if not main_py.is_file():
            messagebox.showerror(
                "再起動",
                f"main.py が見つかりません。\n{main_py}",
            )
            return
        try:
            subprocess.Popen(
                [sys.executable, str(main_py)],
                cwd=str(studio_root()),
                env=_env_with_uv_on_path(),
                close_fds=False,
            )
        except OSError as e:
            messagebox.showerror("再起動", str(e))
            return
        self.root.destroy()

    def _load_text_file(self) -> None:
        p = filedialog.askopenfilename(
            title="台本テキスト",
            filetypes=[("テキスト", "*.txt"), ("すべて", "*.*")],
        )
        if not p:
            return
        try:
            text = Path(p).read_text(encoding="utf-8")
        except OSError as e:
            messagebox.showerror("エラー", str(e))
            return
        self.txt_body.delete("1.0", END)
        self.txt_body.insert("1.0", text)
        self.status.set(f"読み込み: {p}")

    def _replace_body_with_preset(self, label: str) -> None:
        text = body_text_for_label(label)
        if text is None:
            messagebox.showerror("エラー", f"本文プリセットが見つかりません: {label}")
            return
        cur = self.txt_body.get("1.0", END).strip()
        if cur:
            if not messagebox.askyesno(
                "本文プリセット",
                f"いまの本文を「{label}」に差し替えますか？",
            ):
                return
        self.txt_body.delete("1.0", END)
        self.txt_body.insert("1.0", text)
        self.var_body_preset_label.set(label)
        self.status.set(f"本文プリセット: {label}")

    def _apply_selected_body_preset(self) -> None:
        self._replace_body_with_preset(self.var_body_preset_label.get().strip())

    def _insert_sample_text(self) -> None:
        """メニュー・サンプルボタン: 一覧の先頭サンプル。"""
        self._replace_body_with_preset(PRESET_LABELS[0])

    def _paste_clipboard(self) -> None:
        try:
            data = self.root.clipboard_get()
        except Exception:
            messagebox.showinfo("クリップボード", "テキストを取得できませんでした。")
            return
        self.txt_body.insert("insert", data)
        self.status.set("クリップボードの文字列を挿入しました")

    def _apply_caption_template(self) -> None:
        label = self.var_caption_template.get()
        for lab, cap in CAPTION_TEMPLATES:
            if lab == label:
                self.var_caption.set(cap)
                return

    def _open_output_folder(self) -> None:
        out = Path(self.var_output_wav.get().strip())
        folder = out.parent if out.suffix else out
        if folder.is_dir():
            _open_folder_windows(folder)
        elif folder.parent.is_dir():
            _open_folder_windows(folder.parent)
        else:
            messagebox.showinfo("フォルダを開く", "フォルダがまだありません。")

    def _copy_output_path(self) -> None:
        p = self.var_output_wav.get().strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(p)
        self.status.set("出力パスをコピーしました")

    def _play_output_wav(self) -> None:
        p = Path(self.var_output_wav.get().strip())
        if not p.is_file():
            messagebox.showinfo("再生", f"ファイルがありません:\n{p}")
            return
        _open_path_windows(p)

    def _toggle_ref(self) -> None:
        on = self.var_no_ref.get()
        state = "disabled" if on else "normal"
        self.entry_ref.configure(state=state)
        self.btn_ref.configure(state=state)

    def _insert_emoji(self, em: str) -> None:
        self.txt_body.insert("insert", em)

    def _browse_tts_dir(self) -> None:
        p = filedialog.askdirectory(title="Irodori-TTS フォルダを選択")
        if p:
            self.var_tts_dir.set(p)

    def _browse_ref_wav(self) -> None:
        p = filedialog.askopenfilename(
            title="参照 WAV を選択",
            filetypes=[("WAV", "*.wav"), ("すべて", "*.*")],
        )
        if p:
            self.var_ref_wav.set(p)

    def _browse_output_wav(self) -> None:
        p = filedialog.asksaveasfilename(
            title="出力 WAV の保存先を選択",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("すべて", "*.*")],
        )
        if p:
            self.var_auto_filename.set(False)
            self.var_output_wav.set(p)

    def _validate(self) -> tuple[list[str], Path] | None:
        tts = Path(self.var_tts_dir.get().strip())
        infer = tts / "infer.py"
        if not infer.is_file():
            messagebox.showerror(
                "エラー",
                f"infer.py が見つかりません。\n{infer}\n\n"
                "Irodori-TTS を git clone し、そのフォルダを指定してください。",
            )
            return None

        body = self.txt_body.get("1.0", END).strip()
        if not body:
            messagebox.showerror("エラー", "読み上げテキストを入力してください。")
            return None

        uv_bin = _which_uv()
        if not uv_bin:
            messagebox.showerror(
                "エラー",
                "uv が見つかりません。\n"
                "PowerShell: irm https://astral.sh/uv/install.ps1 | iex\n"
                "を実行してから再度お試しください。",
            )
            return None

        if self.var_auto_filename.get():
            out = self._compute_auto_output_wav_path()
            self.var_output_wav.set(str(out))
        else:
            raw_out = self.var_output_wav.get().strip()
            if not raw_out:
                messagebox.showerror(
                    "エラー",
                    "出力 WAV のパスを指定するか、「ファイル名自動」をオンにしてください。",
                )
                return None
            out = Path(raw_out)
            if out.exists() and not messagebox.askyesno(
                "上書き確認",
                f"既存ファイルを上書きしますか？\n{out}",
            ):
                return None
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("エラー", f"出力フォルダを作成できません: {e}")
            return None

        try:
            mode_idx = self._notebook.index(self._notebook.select())
        except TclError:
            mode_idx = 0

        if mode_idx == 0:
            no_ref = self.var_no_ref.get()
            ref = self.var_ref_wav.get().strip()
            if not no_ref:
                if not ref or not Path(ref).is_file():
                    messagebox.showerror(
                        "エラー",
                        "参照音声を使う場合は有効な WAV ファイルを指定してください。",
                    )
                    return None
            ckpt_id = CHECKPOINT_LABEL_TO_ID[BASE_CHECKPOINT_LABEL]
        else:
            no_ref = True
            ref = ""
            ckpt_id = CHECKPOINT_LABEL_TO_ID[vd_presets.VOICEDESIGN_CHECKPOINT_LABEL]
            cap = self.var_caption.get().strip()
            if not cap:
                messagebox.showerror(
                    "VoiceDesign",
                    "VoiceDesign モードではキャプションを入力してください。\n"
                    "キャラプリセットまたはテンプレを使うと簡単です。",
                )
                return None

        try:
            steps = int(self.var_num_steps.get().strip())
            if steps < 1 or steps > 500:
                raise ValueError
        except ValueError:
            messagebox.showerror("エラー", "品質（ステップ数）は 1〜500 の整数にしてください。")
            return None

        seed_str = self.var_seed.get().strip()
        if seed_str:
            try:
                int(seed_str)
            except ValueError:
                messagebox.showerror("エラー", "シードは整数で指定するか、空にしてください。")
                return None

        device = self.var_device.get().strip()
        if device == "cuda" and not torch_cuda_available(tts, uv_bin):
            messagebox.showerror(
                "CUDA が使えません",
                "デバイスに cuda を選んでいますが、この環境では "
                "torch.cuda.is_available() が False です。\n\n"
                "・NVIDIA GPU とドライバが入っているか\n"
                "・Irodori-TTS の venv に CUDA 対応 PyTorch が入っているか\n"
                "を確認するか、デバイスを cpu に変更してください。",
            )
            return None

        cmd: list[str] = [
            uv_bin,
            "run",
            "python",
            "infer.py",
            "--hf-checkpoint",
            ckpt_id,
            "--text",
            body,
            "--output-wav",
            str(out),
            "--model-device",
            device,
            "--codec-device",
            device,
            "--num-steps",
            str(steps),
        ]

        if mode_idx == 1:
            cmd.extend(["--caption", cap])

        if mode_idx == 0:
            if no_ref:
                cmd.append("--no-ref")
            else:
                cmd.extend(["--ref-wav", ref])
        else:
            cmd.append("--no-ref")

        if seed_str:
            cmd.extend(["--seed", seed_str])

        return cmd, tts

    def _on_generate(self) -> None:
        validated = self._validate()
        if not validated:
            return
        cmd, cwd = validated
        self.btn_gen.configure(state="disabled")
        self.status.set("生成中…（初回はモデル取得に時間がかかります）")
        self._log_append("\n----------\n")
        self._log_append("実行: " + " ".join(cmd[:6]) + " ... （テキスト省略）\n")

        def work() -> None:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    env=_env_with_uv_on_path(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                msg = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
                if proc.returncode == 0:
                    self.root.after(0, lambda: self._done_ok(msg))
                else:
                    self.root.after(
                        0,
                        lambda: self._done_err(
                            f"終了コード {proc.returncode}\n\n{msg[-12000:]}"
                        ),
                    )
            except OSError as e:
                self.root.after(0, lambda: self._done_err(str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _done_ok(self, log_tail: str) -> None:
        self.btn_gen.configure(state="normal")
        self.status.set("完了")
        self._log_append(log_tail)
        if not log_tail.endswith("\n"):
            self._log_append("\n")

        out = self.var_output_wav.get().strip()
        body = self.txt_body.get("1.0", END).strip()
        storage.push_history(
            text=body,
            output_path=out,
            checkpoint_label=self._current_checkpoint_label(),
            caption=self.var_caption.get().strip(),
            ref_wav=self.var_ref_wav.get().strip(),
            no_ref=self.var_no_ref.get(),
            device=self.var_device.get().strip(),
            num_steps=self.var_num_steps.get().strip(),
            seed=self.var_seed.get().strip(),
        )
        self._refresh_history_combo()

        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

        if self.var_notify_done.get():
            messagebox.showinfo("完了", f"保存しました:\n{out}")

    def _done_err(self, err: str) -> None:
        self.btn_gen.configure(state="normal")
        self.status.set("エラー")
        self._log_append(err)
        if not err.endswith("\n"):
            self._log_append("\n")
        messagebox.showerror("生成に失敗", err[-8000:])

    def _open_upstream(self) -> None:
        webbrowser.open("https://github.com/Aratako/Irodori-TTS")

    def _about(self) -> None:
        messagebox.showinfo(
            "Irodori-Studio",
            "ローカルの Irodori-TTS（infer.py）を呼び出して音声を生成します。\n\n"
            "【タブ】\n"
            "・参照音声 … 基本モデル＋参照 WAV（または参照なし）\n"
            "・VoiceDesign … キャプションで話者指定（キャラプリセット利用可）\n\n"
            "・保存プロファイルは、自分で保存した設定と本文の呼び出し用です。\n"
            "・台本サンプルは、最初から入っている例文の差し替え用です。\n"
            "・起動時は先頭プリセットが入っています。\n"
            "・ヘルプからテキスト作成ガイドを開けます。\n"
            "・ログ・履歴・保存プロファイルは config に保存されます。\n"
            "・初回は Hugging Face からモデルがダウンロードされます。",
        )


def run_app() -> None:
    root = Tk()
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = IrodoriStudioApp(root)
    app._refresh_preset_combo()
    app._refresh_history_combo()
    root.mainloop()
