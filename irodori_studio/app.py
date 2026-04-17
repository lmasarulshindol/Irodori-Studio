"""Irodori-TTS 連携のメインウィンドウ（tkinter）。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path
from tkinter import (
    END,
    BooleanVar,
    Button,
    Canvas,
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
    Toplevel,
    ttk,
)
from tkinter.simpledialog import askstring

from irodori_studio.paths import default_irodori_tts_dir, studio_root
from irodori_studio import settings as studio_settings
from irodori_studio import storage
from irodori_studio import voice_design_presets as vd_presets
from irodori_studio.text_presets import (
    DEFAULT_SAMPLE_TEXT,
    PRESET_LABELS,
    body_text_for_label,
)
from irodori_studio.wav_mp3 import convert_wav_to_mp3

CHECKPOINTS = [
    ("Irodori-TTS-500M-v2（基本）", "Aratako/Irodori-TTS-500M-v2"),
    ("Irodori-TTS-500M-v2-VoiceDesign", "Aratako/Irodori-TTS-500M-v2-VoiceDesign"),
]
CHECKPOINT_LABEL_TO_ID = dict(CHECKPOINTS)
CHECKPOINT_LABELS = [c[0] for c in CHECKPOINTS]
BASE_CHECKPOINT_LABEL = CHECKPOINTS[0][0]
TEXT_GUIDE_PATH = studio_root() / "docs" / "text-creation-guideline.md"
LORA_GUIDE_PATH = studio_root() / "docs" / "lora-training-guideline.md"
BATCH_GUIDE_PATH = studio_root() / "docs" / "batch-generation-guideline.md"
PROFILE_PLACEHOLDER = "保存プロファイルを選ぶ…"
HISTORY_PLACEHOLDER = "履歴を選ぶ…"

LORA_TAB_TITLE = "LoRA 作成（学習）"

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
        root.minsize(760, 680)
        root.geometry("920x760")

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
        self._studio_settings = studio_settings.load()
        self.var_mp3_convert = BooleanVar(
            value=studio_settings.resolve_ffmpeg_path() is not None
        )
        self.var_mp3_bitrate = StringVar(value=str(self._studio_settings.get("mp3_bitrate", "192k")))
        self.var_infer_checkpoint = StringVar(value="")
        self.var_body_preset_label = StringVar(value=PRESET_LABELS[0])
        self.status = StringVar(value="準備OK")
        self._last_output_wav: str | None = None
        self._last_output_mp3: str | None = None

        _tts0 = Path(self.var_tts_dir.get().strip() or default_irodori_tts_dir())
        self.var_lora_config = StringVar(
            value=str(_tts0 / "configs" / "train_500m_v2_lora.yaml")
        )
        self.var_lora_manifest = StringVar(
            value=str(_tts0 / "data" / "train_manifest.jsonl")
        )
        self.var_lora_output = StringVar(
            value=str(_tts0 / "outputs" / "irodori_tts_lora")
        )
        self.var_lora_init = StringVar(value="")
        self.var_lora_resume = StringVar(value="")
        self.var_lora_train_device = StringVar(value="cpu")
        self._job_running = False

        self._build_menu(root)
        self._build_main_paned(root)

    def _build_menu(self, root: Tk) -> None:
        menubar = Menu(root)
        root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(
            label="WAV→MP3 一括変換（別ウィンドウ）…",
            command=self._open_wav_mp3_batch_tool,
        )
        file_menu.add_separator()
        file_menu.add_command(label="台本テキストを開く…", command=self._load_text_file)
        file_menu.add_command(label="クリップボードを貼り付け", command=self._paste_clipboard)
        file_menu.add_command(
            label="台本サンプルを挿入",
            command=self._insert_sample_text,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="連続生成（本文を行ごと）…",
            command=self._run_batch_generate,
        )
        file_menu.add_command(
            label="参照 WAV を切り出し…",
            command=self._trim_ref_wav_dialog,
        )
        file_menu.add_separator()
        file_menu.add_command(label="履歴をクリア", command=self._clear_history_storage)
        file_menu.add_separator()
        file_menu.add_command(label="アプリ設定…", command=self._open_app_settings)
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
            label="LoRA 作成ガイドを開く",
            command=self._open_lora_training_guide,
        )
        help_menu.add_command(
            label="一括生成ガイドを開く",
            command=self._open_batch_generation_guide,
        )
        help_menu.add_command(
            label="VoiceDesign モデルカード（HF）",
            command=self._open_voicedesign_card,
        )
        help_menu.add_command(
            label="LoRA 学習（公式 README）",
            command=self._open_upstream_lora_docs,
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
        paned.paneconfig(upper, minsize=260)
        paned.paneconfig(lower, minsize=140)

        self._build_form(upper)
        self._build_log(lower)
        self._setup_form_canvas_mousewheel()
        self._on_notebook_tab_changed()

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
        try:
            max_lines = int(self._studio_settings.get("log_max_lines", 3000))
            max_lines = max(100, min(50_000, max_lines))
        except (TypeError, ValueError):
            max_lines = 3000
        while True:
            try:
                li = int(self.txt_log.index("end-1c").split(".")[0])
            except TclError:
                break
            if li <= max_lines:
                break
            self.txt_log.delete("1.0", "2.0")
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

    @staticmethod
    def _widget_is_descendant(widget: object, ancestor: object) -> bool:
        """widget が ancestor の子孫（同一ウィジェット含む）か。"""
        w = widget
        while w is not None:
            if w == ancestor:
                return True
            w = getattr(w, "master", None)
        return False

    def _setup_form_canvas_mousewheel(self) -> None:
        """フォーム Canvas 上では子ウィジェット経由でもホイールで縦スクロールできるようにする。"""
        canvas = getattr(self, "_form_canvas", None)
        if canvas is None:
            return
        vsb = getattr(self, "_form_vsb", None)

        def _pointer_in_widget(widget: object, x_root: int, y_root: int) -> bool:
            try:
                x, y = widget.winfo_rootx(), widget.winfo_rooty()
                w, h = widget.winfo_width(), widget.winfo_height()
            except TclError:
                return False
            return x <= x_root < x + w and y <= y_root < y + h

        def _in_form_scroll_area(x_root: int, y_root: int) -> bool:
            if _pointer_in_widget(canvas, x_root, y_root):
                return True
            if vsb is not None and _pointer_in_widget(vsb, x_root, y_root):
                return True
            return False

        def _wheel_steps(event: object) -> int | None:
            delta = getattr(event, "delta", 0)
            num = getattr(event, "num", 0)
            if delta:
                steps = -int(round(delta / 120))
                if steps == 0:
                    steps = -1 if delta > 0 else 1
                return steps
            if num == 4:
                return -1
            if num == 5:
                return 1
            return None

        def _on_form_wheel(event) -> str | None:
            if not _in_form_scroll_area(event.x_root, event.y_root):
                return None
            try:
                w = self.root.winfo_containing(event.x_root, event.y_root)
            except TclError:
                return None
            if w is None:
                return None
            if hasattr(self, "txt_body") and self._widget_is_descendant(w, self.txt_body):
                return None
            if hasattr(self, "txt_log") and self._widget_is_descendant(w, self.txt_log):
                return None
            steps = _wheel_steps(event)
            if not steps:
                return None
            canvas.yview_scroll(steps, "units")
            return "break"

        self.root.bind_all("<MouseWheel>", _on_form_wheel, add="+")
        self.root.bind_all("<Button-4>", _on_form_wheel, add="+")
        self.root.bind_all("<Button-5>", _on_form_wheel, add="+")

    def _build_form(self, root: Frame) -> None:
        pad = {"padx": 8, "pady": 3}
        _wrap = 700

        canvas = Canvas(root, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        inner = Frame(canvas)
        inner_win = canvas.create_window((0, 0), window=inner, anchor="nw")
        self._form_canvas = canvas

        def _sync_inner_scroll(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_canvas_width(event) -> None:
            canvas.itemconfigure(inner_win, width=max(event.width - 4, 1))

        inner.bind("<Configure>", _sync_inner_scroll)
        canvas.bind("<Configure>", _sync_canvas_width)

        self._form_vsb = vsb
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        f0 = Frame(inner)
        self._f0_frame = f0
        f0.pack(fill="x", **pad)
        Label(f0, text="Irodori-TTS フォルダ").pack(side="left")
        Entry(f0, textvariable=self.var_tts_dir, width=72).pack(
            side="left", fill="x", expand=True, padx=4
        )
        Button(f0, text="フォルダを選択…", command=self._browse_tts_dir).pack(side="left")

        self._infer_ck_frame = Frame(inner)
        self._infer_ck_frame.pack(fill="x", **pad, after=f0)
        f_ck = Frame(self._infer_ck_frame)
        f_ck.pack(fill="x")
        Label(f_ck, text="推論チェックポイント（任意）").pack(side="left")
        Entry(f_ck, textvariable=self.var_infer_checkpoint, width=58).pack(
            side="left", fill="x", expand=True, padx=4
        )
        Button(f_ck, text="参照…", command=self._browse_infer_checkpoint).pack(side="left")
        Label(
            self._infer_ck_frame,
            text="空＝Hugging Face の既定モデル。convert 済みの .safetensors（学習後マージなど）を指定すると --checkpoint で推論します。",
            fg="#555555",
            font=("", 8),
            wraplength=_wrap,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        self._notebook = ttk.Notebook(inner)
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
            wraplength=_wrap,
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
            wraplength=_wrap,
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

        tab_lora = Frame(self._notebook, padx=6, pady=6)
        self._notebook.add(tab_lora, text=LORA_TAB_TITLE)
        Label(
            tab_lora,
            text="Irodori-TTS の train.py で PEFT LoRA を学習します。事前に manifest（JSONL）と "
            "DACVAE 前処理が必要です（公式 README の Dataset / prepare_manifest を参照）。",
            fg="#555555",
            anchor="w",
            wraplength=_wrap,
            justify="left",
        ).pack(fill="x", pady=(0, 6))
        row_lc = Frame(tab_lora)
        row_lc.pack(fill="x", pady=(0, 2))
        Label(row_lc, text="設定プリセット").pack(side="left")
        _lora_yaml_labels = (
            "train_500m_v2_lora.yaml（基本モデル LoRA）",
            "train_500m_v2_voice_design_lora.yaml（VoiceDesign LoRA）",
        )
        self._combo_lora_yaml = ttk.Combobox(
            row_lc,
            state="readonly",
            width=42,
            values=_lora_yaml_labels,
        )
        self._combo_lora_yaml.current(0)
        self._combo_lora_yaml.pack(side="left", padx=4)
        Button(row_lc, text="パスに反映", command=self._apply_lora_yaml_preset).pack(
            side="left"
        )

        def row_file(parent: Frame, label: str, var: StringVar, browse_cmd: object) -> None:
            r = Frame(parent)
            r.pack(fill="x", pady=2)
            Label(r, text=label, width=18, anchor="w").pack(side="left")
            Entry(r, textvariable=var, width=62).pack(
                side="left", fill="x", expand=True, padx=4
            )
            Button(r, text="参照…", command=browse_cmd).pack(side="left")

        row_file(tab_lora, "学習 YAML", self.var_lora_config, self._browse_lora_config)
        row_file(tab_lora, "manifest（JSONL）", self.var_lora_manifest, self._browse_lora_manifest)
        row_file(tab_lora, "出力フォルダ", self.var_lora_output, self._browse_lora_output)
        row_file(tab_lora, "初期重み（.safetensors）", self.var_lora_init, self._browse_lora_init)
        r_res = Frame(tab_lora)
        r_res.pack(fill="x", pady=2)
        Label(r_res, text="再開（LoRA フォルダ）", width=18, anchor="w").pack(side="left")
        Entry(r_res, textvariable=self.var_lora_resume, width=62).pack(
            side="left", fill="x", expand=True, padx=4
        )
        Button(r_res, text="参照…", command=self._browse_lora_resume).pack(side="left")
        Label(
            tab_lora,
            text="新規学習: 初期重みに公式の .safetensors を指定。中断からの再開のみなら「再開」にアダプタフォルダを指定（任意で初期重みも上書き可）。",
            fg="#555555",
            font=("", 8),
            wraplength=_wrap,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))
        row_dev = Frame(tab_lora)
        row_dev.pack(fill="x", pady=(0, 6))
        Label(row_dev, text="学習デバイス").pack(side="left")
        self._lora_device_combo = ttk.Combobox(
            row_dev,
            textvariable=self.var_lora_train_device,
            values=_device_choices(),
            width=8,
            state="readonly",
        )
        self._lora_device_combo.pack(side="left", padx=8)
        row_btns = Frame(tab_lora)
        row_btns.pack(fill="x", pady=(4, 0))
        self.btn_lora_copy = Button(
            row_btns,
            text="コマンドをコピー",
            command=self._copy_lora_command,
        )
        self.btn_lora_copy.pack(side="left", padx=(0, 8))
        self.btn_lora_run = Button(
            row_btns,
            text="学習を開始（ログに出力）",
            command=self._run_lora_training,
        )
        self.btn_lora_run.pack(side="left")

        self._notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)
        self._combo_vd_char.bind("<<ComboboxSelected>>", self._on_voice_character_selected)

        fp = Frame(inner)
        fp.pack(fill="x", **pad)
        fp_top = Frame(fp)
        fp_top.pack(fill="x")
        Label(fp_top, text="保存プロファイル").pack(side="left")
        self._combo_preset = ttk.Combobox(
            fp_top,
            textvariable=self.var_preset_pick,
            values=self._preset_names(),
            width=28,
            state="readonly",
        )
        self._combo_preset.pack(side="left", padx=4)
        self.btn_apply_profile = Button(fp_top, text="読み込む", command=self._apply_preset)
        self.btn_apply_profile.pack(side="left")
        Button(fp_top, text="一覧更新", command=self._refresh_preset_combo).pack(
            side="left", padx=4
        )
        Label(
            fp,
            text="自分で保存した設定・本文・キャプションをまとめて呼び出します。",
            fg="#555555",
            wraplength=_wrap,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        fh = Frame(inner)
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

        self._history_row_frame = fh
        self._infer_form_pack = {"fill": "x", "padx": 8, "pady": 3}
        self._infer_only_frame = Frame(inner)
        self._infer_only_frame.pack(after=fh, **self._infer_form_pack)

        f2 = Frame(self._infer_only_frame)
        f2.pack(fill="x", pady=(0, 3))
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
        row_tp_top = Frame(row_tp)
        row_tp_top.pack(fill="x")
        Label(row_tp_top, text="台本サンプル:").pack(side="left")
        self._combo_body_preset = ttk.Combobox(
            row_tp_top,
            textvariable=self.var_body_preset_label,
            values=PRESET_LABELS,
            state="readonly",
            width=26,
        )
        self._combo_body_preset.pack(side="left", padx=4)
        Button(
            row_tp_top,
            text="このサンプルに差し替え",
            command=self._apply_selected_body_preset,
        ).pack(side="left", padx=2)
        Label(
            row_tp,
            text="最初から入っている台本サンプルです。保存プロファイルとは別物です。",
            fg="#555555",
            wraplength=_wrap,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        self.txt_body = scrolledtext.ScrolledText(f2, height=7, wrap="word")
        self.txt_body.pack(fill="x", pady=(2, 0))
        self.txt_body.insert("1.0", DEFAULT_SAMPLE_TEXT)

        f_emoji = Frame(self._infer_only_frame)
        f_emoji.pack(fill="x", pady=(0, 3))
        Label(f_emoji, text="絵文字:").pack(side="left")
        for em in EMOJI_PRESETS[:8]:
            Button(f_emoji, text=em, width=3, command=lambda e=em: self._insert_emoji(e)).pack(
                side="left", padx=1
            )
        f_emoji2 = Frame(self._infer_only_frame)
        f_emoji2.pack(fill="x", pady=(0, 3))
        Label(f_emoji2, text="").pack(side="left", padx=32)
        for em in EMOJI_PRESETS[8:]:
            Button(f_emoji2, text=em, width=3, command=lambda e=em: self._insert_emoji(e)).pack(
                side="left", padx=1
            )

        f5 = Frame(self._infer_only_frame)
        f5.pack(fill="x", pady=(0, 3))
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
        f5_mp3 = Frame(self._infer_only_frame)
        f5_mp3.pack(fill="x", pady=(0, 2))
        Checkbutton(
            f5_mp3,
            text="MP3 も作成",
            variable=self.var_mp3_convert,
        ).pack(side="left")
        Label(f5_mp3, text="ビットレート").pack(side="left", padx=(8, 0))
        self._combo_mp3_bitrate = ttk.Combobox(
            f5_mp3,
            textvariable=self.var_mp3_bitrate,
            values=studio_settings.MP3_BITRATES,
            width=6,
            state="readonly",
        )
        self._combo_mp3_bitrate.pack(side="left", padx=2)
        self._combo_mp3_bitrate.bind("<<ComboboxSelected>>", self._persist_mp3_bitrate)
        f5h = Frame(self._infer_only_frame)
        f5h.pack(fill="x", pady=(0, 2))
        _has_ffmpeg = studio_settings.resolve_ffmpeg_path() is not None
        Label(
            f5h,
            text="自動ON: outputs/generated に「声の種類_セリフ先頭_連番.wav」。"
            "MP3 は同名 .mp3（要 ffmpeg。未検出時はヘルプ→アプリ設定でパス指定）"
            + ("" if _has_ffmpeg else "  ⚠ ffmpeg 未検出"),
            fg="#555555" if _has_ffmpeg else "#cc6600",
            font=("", 8),
            wraplength=_wrap,
            justify="left",
        ).pack(anchor="w")

        f6 = Frame(self._infer_only_frame)
        f6.pack(fill="x", pady=(0, 0))
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
        if idx == 2:
            return BASE_CHECKPOINT_LABEL
        if idx == 1:
            return vd_presets.VOICEDESIGN_CHECKPOINT_LABEL
        return BASE_CHECKPOINT_LABEL

    def _sync_infer_form_visibility(self) -> None:
        """LoRA タブでは推論専用ブロック（チェックポイント行・本文・絵文字等）を隠す。"""
        try:
            icf = self._infer_ck_frame
            fr = self._infer_only_frame
            anchor = self._history_row_frame
            f0 = self._f0_frame
            kw = self._infer_form_pack
        except AttributeError:
            return
        try:
            idx = self._notebook.index(self._notebook.select())
        except TclError:
            idx = 0
        if idx == 2:
            if icf.winfo_ismapped():
                icf.pack_forget()
            if fr.winfo_ismapped():
                fr.pack_forget()
        else:
            if not icf.winfo_ismapped():
                icf.pack(after=f0, **kw)
            if not fr.winfo_ismapped():
                fr.pack(after=anchor, **kw)

    def _on_notebook_tab_changed(self, _event: object = None) -> None:
        try:
            title = self._notebook.tab(self._notebook.select(), "text")
        except TclError:
            return
        if title == LORA_TAB_TITLE:
            self.status.set(f"モード: {title}（推論は他タブへ／学習は下のボタン）")
            if not self._job_running:
                self.btn_gen.configure(state="disabled")
        elif "VoiceDesign" in title:
            self.status.set(f"モード: {title}（キャプション必須）")
            if not self._job_running:
                self.btn_gen.configure(state="normal")
        else:
            self.status.set(f"モード: {title}")
            if not self._job_running:
                self.btn_gen.configure(state="normal")
        self._sync_infer_form_visibility()

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
        if mode_idx == 2:
            return "LORA"
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

    def _compute_auto_output_wav_path_from_line(self, line: str) -> Path:
        """連続生成用: 1 行分のテキストでファイル名を組み立てる。"""
        out_dir = studio_root() / "outputs" / "generated"
        voice = self._voice_slug_for_filename()
        first_line = line.strip().split("\n")[0] if line else ""
        text_part = _sanitize_filename_segment(first_line, 42)
        base = f"{voice}_{text_part}"
        if len(base) > 115:
            base = base[:115]
        return _next_numbered_wav_path(out_dir, base)

    def _try_mp3_convert_with(
        self,
        wav_p: Path,
        do_mp3: bool,
        bitrate: str,
    ) -> tuple[str | None, str]:
        """MP3 変換（スレッドからも可）。戻り値: (mp3 パス or None, ログ行)。"""
        if not do_mp3:
            return None, ""
        ff = studio_settings.resolve_ffmpeg_path()
        if not ff:
            return None, "MP3: ffmpeg が見つかりません（アプリ設定でパス指定可）\n"
        br = (bitrate or "192k").strip()
        if br not in studio_settings.MP3_BITRATES:
            br = "192k"
        mp3, err = convert_wav_to_mp3(wav_p, ffmpeg=ff, bitrate=br)
        if mp3 is not None:
            return str(mp3), f"MP3 変換: {mp3}\n"
        return None, f"MP3 変換失敗: {err}\n"

    def _try_mp3_convert(self, wav_p: Path) -> tuple[str | None, str]:
        return self._try_mp3_convert_with(
            wav_p,
            self.var_mp3_convert.get(),
            self.var_mp3_bitrate.get(),
        )

    def _persist_mp3_bitrate(self, _event: object = None) -> None:
        br = self.var_mp3_bitrate.get().strip()
        if br not in studio_settings.MP3_BITRATES:
            br = "192k"
            self.var_mp3_bitrate.set(br)
        self._studio_settings["mp3_bitrate"] = br
        studio_settings.save(self._studio_settings)

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

    def _open_lora_training_guide(self) -> None:
        if not LORA_GUIDE_PATH.is_file():
            messagebox.showerror("ガイド", f"ガイドが見つかりません:\n{LORA_GUIDE_PATH}")
            return
        _open_path_windows(LORA_GUIDE_PATH)

    def _open_batch_generation_guide(self) -> None:
        if not BATCH_GUIDE_PATH.is_file():
            messagebox.showerror("ガイド", f"ガイドが見つかりません:\n{BATCH_GUIDE_PATH}")
            return
        _open_path_windows(BATCH_GUIDE_PATH)

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
                if "mp3_convert" in p:
                    self.var_mp3_convert.set(bool(p["mp3_convert"]))
                if p.get("mp3_bitrate") in studio_settings.MP3_BITRATES:
                    self.var_mp3_bitrate.set(str(p["mp3_bitrate"]))
                if "infer_local_checkpoint" in p and p["infer_local_checkpoint"] is not None:
                    self.var_infer_checkpoint.set(str(p["infer_local_checkpoint"]))
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
            "mp3_convert": self.var_mp3_convert.get(),
            "mp3_bitrate": self.var_mp3_bitrate.get().strip(),
            "infer_local_checkpoint": self.var_infer_checkpoint.get().strip(),
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

    def _open_wav_mp3_batch_tool(self) -> None:
        """WAV 一括 MP3 ツールを別プロセスで起動。"""
        script = studio_root() / "wav_to_mp3_gui.py"
        if not script.is_file():
            messagebox.showerror("エラー", f"スクリプトが見つかりません:\n{script}")
            return
        try:
            subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(studio_root()),
                env=_env_with_uv_on_path(),
                close_fds=False,
            )
        except OSError as e:
            messagebox.showerror("起動", str(e))

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
        if not text:
            if cur and messagebox.askyesno("本文クリア", "本文を空にしますか？"):
                self.txt_body.delete("1.0", END)
                self.var_body_preset_label.set(label)
                self.status.set("本文をクリアしました")
            return
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
        """メニュー・サンプルボタン: 先頭のサンプル本文（「指定なし」は飛ばす）。"""
        for label in PRESET_LABELS:
            if body_text_for_label(label):
                self._replace_body_with_preset(label)
                return

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
            self._sync_lora_paths_from_tts()

    def _sync_lora_paths_from_tts(self) -> None:
        """TTS ルート変更時に LoRA 用の既定パスを更新（空欄のときのみ補完）。"""
        tts = Path(self.var_tts_dir.get().strip())
        if not tts.is_dir():
            return
        cfg = tts / "configs" / "train_500m_v2_lora.yaml"
        if cfg.is_file():
            cur = self.var_lora_config.get().strip()
            if not cur or not Path(cur).is_file():
                self.var_lora_config.set(str(cfg))
        man = tts / "data" / "train_manifest.jsonl"
        if man.is_file():
            cur_m = self.var_lora_manifest.get().strip()
            if not cur_m or not Path(cur_m).is_file():
                self.var_lora_manifest.set(str(man))
        out = tts / "outputs" / "irodori_tts_lora"
        if not self.var_lora_output.get().strip():
            self.var_lora_output.set(str(out))

    def _apply_lora_yaml_preset(self) -> None:
        tts = Path(self.var_tts_dir.get().strip())
        choice = self._combo_lora_yaml.get()
        if "voice_design" in choice.lower():
            name = "train_500m_v2_voice_design_lora.yaml"
        else:
            name = "train_500m_v2_lora.yaml"
        self.var_lora_config.set(str(tts / "configs" / name))

    def _browse_lora_config(self) -> None:
        p = filedialog.askopenfilename(
            title="学習用 YAML",
            filetypes=[("YAML", "*.yaml;*.yml"), ("すべて", "*.*")],
        )
        if p:
            self.var_lora_config.set(p)

    def _browse_lora_manifest(self) -> None:
        p = filedialog.askopenfilename(
            title="manifest（JSONL）",
            filetypes=[("JSONL", "*.jsonl"), ("すべて", "*.*")],
        )
        if p:
            self.var_lora_manifest.set(p)

    def _browse_lora_output(self) -> None:
        p = filedialog.askdirectory(title="学習の出力フォルダ")
        if p:
            self.var_lora_output.set(p)

    def _browse_lora_init(self) -> None:
        p = filedialog.askopenfilename(
            title="初期チェックポイント（.safetensors）",
            filetypes=[("Safetensors", "*.safetensors"), ("すべて", "*.*")],
        )
        if p:
            self.var_lora_init.set(p)

    def _browse_lora_resume(self) -> None:
        p = filedialog.askdirectory(title="再開する LoRA アダプタのフォルダ")
        if p:
            self.var_lora_resume.set(p)

    def _build_lora_train_command(self) -> tuple[list[str], Path] | tuple[None, None]:
        """検証済みの train コマンドと cwd を返す。失敗時は (None, None)。"""
        tts = Path(self.var_tts_dir.get().strip())
        train_py = tts / "train.py"
        if not train_py.is_file():
            messagebox.showerror(
                "エラー",
                f"train.py が見つかりません。\n{train_py}\n\n"
                "Irodori-TTS のルートを「Irodori-TTS フォルダ」に指定してください。",
            )
            return None, None

        cfg_path = Path(self.var_lora_config.get().strip())
        if not cfg_path.is_file():
            messagebox.showerror("エラー", f"学習 YAML が見つかりません:\n{cfg_path}")
            return None, None

        manifest = Path(self.var_lora_manifest.get().strip())
        if not manifest.is_file():
            messagebox.showerror(
                "エラー",
                f"manifest が見つかりません:\n{manifest}\n\n"
                "prepare_manifest.py で JSONL を作成してください（公式 README）。",
            )
            return None, None

        out_dir = self.var_lora_output.get().strip()
        if not out_dir:
            messagebox.showerror("エラー", "出力フォルダを指定してください。")
            return None, None

        init_s = self.var_lora_init.get().strip()
        resume_s = self.var_lora_resume.get().strip()
        if resume_s:
            rp = Path(resume_s)
            if not rp.is_dir():
                messagebox.showerror("エラー", f"再開パスはフォルダである必要があります:\n{rp}")
                return None, None
        if not resume_s and not init_s:
            messagebox.showerror(
                "エラー",
                "「初期重み（.safetensors）」を指定するか、"
                "「再開」に LoRA アダプタフォルダを指定してください。",
            )
            return None, None
        if init_s:
            ip = Path(init_s)
            if not ip.is_file():
                messagebox.showerror("エラー", f"初期重みが見つかりません:\n{ip}")
                return None, None

        uv_bin = _which_uv()
        if not uv_bin:
            messagebox.showerror(
                "エラー",
                "uv が見つかりません。\n"
                "PowerShell: irm https://astral.sh/uv/install.ps1 | iex",
            )
            return None, None

        dev = self.var_lora_train_device.get().strip()
        if dev == "cuda" and not torch_cuda_available(tts, uv_bin):
            messagebox.showerror(
                "CUDA が使えません",
                "学習デバイスに cuda を選んでいますが、この環境では "
                "CUDA が利用できません。cpu に変更するか、GPU 環境を確認してください。",
            )
            return None, None

        cmd: list[str] = [
            uv_bin,
            "run",
            "python",
            "train.py",
            "--config",
            str(cfg_path),
            "--manifest",
            str(manifest),
            "--output-dir",
            out_dir,
            "--device",
            dev,
        ]
        if resume_s:
            cmd.extend(["--resume", resume_s])
        if init_s:
            cmd.extend(["--init-checkpoint", init_s])

        return cmd, tts

    def _copy_lora_command(self) -> None:
        if self._job_running:
            return
        cmd_t = self._build_lora_train_command()
        if cmd_t[0] is None:
            return
        cmd, _cwd = cmd_t
        line = subprocess.list2cmdline(cmd)
        self.root.clipboard_clear()
        self.root.clipboard_append(line)
        self.status.set("学習コマンドをクリップボードにコピーしました")

    def _run_lora_training(self) -> None:
        if self._job_running:
            return
        cmd_t = self._build_lora_train_command()
        if cmd_t[0] is None:
            return
        cmd, cwd = cmd_t
        self._job_running = True
        self.btn_lora_run.configure(state="disabled")
        self.btn_lora_copy.configure(state="disabled")
        self.btn_gen.configure(state="disabled")
        self.status.set("学習中…（ログを確認。中断はコンソール相当）")
        self._log_append("\n----------\n")
        self._log_append("LoRA 学習: " + subprocess.list2cmdline(cmd[:10]) + " ...\n")

        def work() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=_env_with_uv_on_path(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                if proc.stdout is None:
                    self.root.after(
                        0,
                        lambda: self._lora_train_finished(
                            -1,
                            "stdout を開けませんでした。",
                        ),
                    )
                    return
                for line in proc.stdout:
                    self.root.after(0, lambda l=line: self._log_append(l))
                code = proc.wait()
                self.root.after(0, lambda: self._lora_train_finished(code, None))
            except OSError as e:
                self.root.after(0, lambda: self._lora_train_finished(-1, str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _lora_train_finished(self, code: int, err: str | None) -> None:
        self._job_running = False
        self.btn_lora_run.configure(state="normal")
        self.btn_lora_copy.configure(state="normal")
        self._on_notebook_tab_changed()
        if err:
            self.status.set("学習エラー")
            self._log_append(err + "\n")
            messagebox.showerror("LoRA 学習", err)
            return
        if code == 0:
            self.status.set("学習プロセスが終了しました（成功）")
            self._log_append("\n（プロセス終了コード 0）\n")
        else:
            self.status.set(f"学習終了（コード {code}）")
            self._log_append(f"\n（プロセス終了コード {code}）\n")
            messagebox.showwarning(
                "LoRA 学習",
                f"終了コード {code} です。ログを確認してください。",
            )

    def _open_upstream_lora_docs(self) -> None:
        webbrowser.open(
            "https://github.com/Aratako/Irodori-TTS/blob/main/README.md#fine-tuning-from-released-weights"
        )

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

    def _validate(
        self,
        *,
        body_override: str | None = None,
        output_path_override: Path | None = None,
        skip_overwrite_prompt: bool = False,
    ) -> tuple[list[str], Path] | None:
        try:
            mode_idx = self._notebook.index(self._notebook.select())
        except TclError:
            mode_idx = 0
        if mode_idx == 2:
            messagebox.showinfo(
                "モード",
                "LoRA 作成タブでは音声は生成できません。\n"
                "「参照音声」または「VoiceDesign」タブに切り替えてください。\n\n"
                "学習を始める場合は、同タブの「学習を開始」を使ってください。",
            )
            return None

        tts = Path(self.var_tts_dir.get().strip())
        infer = tts / "infer.py"
        if not infer.is_file():
            messagebox.showerror(
                "エラー",
                f"infer.py が見つかりません。\n{infer}\n\n"
                "Irodori-TTS を git clone し、そのフォルダを指定してください。",
            )
            return None

        if body_override is not None:
            body = body_override.strip()
        else:
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

        local_ckpt = self.var_infer_checkpoint.get().strip()
        if local_ckpt:
            lcp = Path(local_ckpt)
            if not lcp.is_file():
                messagebox.showerror(
                    "エラー",
                    f"推論チェックポイントが見つかりません:\n{lcp}",
                )
                return None

        if output_path_override is not None:
            out = output_path_override
            self.var_output_wav.set(str(out))
        elif self.var_auto_filename.get():
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
            if out.exists() and not skip_overwrite_prompt:
                if not messagebox.askyesno(
                    "上書き確認",
                    f"既存ファイルを上書きしますか？\n{out}",
                ):
                    return None
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("エラー", f"出力フォルダを作成できません: {e}")
            return None

        cap = ""
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
        ]
        if local_ckpt:
            cmd.extend(["--checkpoint", local_ckpt])
        else:
            cmd.extend(["--hf-checkpoint", ckpt_id])
        cmd.extend(
            [
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
        )

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

    def _try_build_studio_batch_config(
        self, lines: list[str]
    ) -> tuple[dict, Path, list[Path]] | None:
        """連続生成（1プロセスバッチ）用の設定 JSON。検証はメインスレッドで行う。"""
        try:
            mode_idx = self._notebook.index(self._notebook.select())
        except TclError:
            mode_idx = 0
        if mode_idx == 2:
            messagebox.showinfo(
                "モード",
                "LoRA 作成タブでは音声は生成できません。\n"
                "「参照音声」または「VoiceDesign」タブに切り替えてください。",
            )
            return None

        tts = Path(self.var_tts_dir.get().strip())
        infer = tts / "infer.py"
        if not infer.is_file():
            messagebox.showerror(
                "エラー",
                f"infer.py が見つかりません。\n{infer}\n\n"
                "Irodori-TTS を git clone し、そのフォルダを指定してください。",
            )
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

        local_ckpt = self.var_infer_checkpoint.get().strip()
        if local_ckpt:
            lcp = Path(local_ckpt)
            if not lcp.is_file():
                messagebox.showerror(
                    "エラー",
                    f"推論チェックポイントが見つかりません:\n{lcp}",
                )
                return None

        cap = ""
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

        items: list[dict[str, str]] = []
        out_paths: list[Path] = []
        for line in lines:
            outp = self._compute_auto_output_wav_path_from_line(line)
            out_paths.append(outp)
            items.append({"text": line, "output_wav": str(outp)})
        try:
            out_paths[0].parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("エラー", f"出力フォルダを作成できません: {e}")
            return None

        if local_ckpt:
            ck: str | None = local_ckpt
            hf_ck: str | None = None
        else:
            ck = None
            hf_ck = ckpt_id

        ref_wav: str | None = None
        if mode_idx == 0 and not no_ref:
            ref_wav = ref

        config: dict = {
            "checkpoint": ck,
            "hf_checkpoint": hf_ck,
            "model_device": device,
            "codec_device": device,
            "codec_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
            "model_precision": "fp32",
            "codec_precision": "fp32",
            "codec_deterministic_encode": True,
            "codec_deterministic_decode": True,
            "enable_watermark": False,
            "compile_model": False,
            "compile_dynamic": False,
            "num_steps": steps,
            "seed": int(seed_str) if seed_str else None,
            "caption": cap if mode_idx == 1 else None,
            "no_ref": bool(no_ref) if mode_idx == 0 else True,
            "ref_wav": ref_wav,
            "items": items,
        }
        return config, tts, out_paths

    def _on_generate(self) -> None:
        if self._job_running:
            messagebox.showinfo(
                "実行中",
                "バックグラウンド処理（LoRA 学習・連続生成など）が進行中です。完了を待ってください。",
            )
            return
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

        mp3_path_s: str | None = None
        if out:
            wav_p = Path(out)
            if wav_p.is_file():
                mp3_path_s, mp3_log = self._try_mp3_convert(wav_p)
                if mp3_log:
                    self._log_append(mp3_log)
        self._last_output_wav = out or None
        self._last_output_mp3 = mp3_path_s

        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

        if self.var_notify_done.get():
            self._show_completion_dialog(out, mp3_path_s)

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
            "・VoiceDesign … キャプションで話者指定（キャラプリセット利用可）\n"
            "・LoRA 作成 … train.py で学習（manifest 準備が必要。ヘルプの LoRA ガイド参照）\n\n"
            "・保存プロファイルは、自分で保存した設定と本文の呼び出し用です。\n"
            "・台本サンプルは、最初から入っている例文の差し替え用です。\n"
            "・起動時は先頭プリセットが入っています。\n"
            "・ヘルプからテキスト作成ガイドを開けます。\n"
            "・ログ・履歴・保存プロファイルは config に保存されます。\n"
            "・初回は Hugging Face からモデルがダウンロードされます。",
        )

    def _show_completion_dialog(self, wav_s: str | None, mp3_s: str | None) -> None:
        """単発生成完了時: フォルダ／ファイルを開く。"""
        top = Toplevel(self.root)
        top.title("生成完了")
        top.transient(self.root)
        top.grab_set()
        Label(top, text="生成が完了しました。").pack(padx=16, pady=(12, 8))
        wav_p = Path(wav_s) if wav_s else None
        mp3_p = Path(mp3_s) if mp3_s else None
        fr = Frame(top)
        fr.pack(pady=4)

        def close() -> None:
            try:
                top.grab_release()
            except TclError:
                pass
            top.destroy()

        if wav_p is not None and wav_p.is_file():

            def open_dir() -> None:
                _open_folder_windows(wav_p.parent)
                close()

            def open_wav() -> None:
                _open_path_windows(wav_p)
                close()

            Button(fr, text="出力フォルダを開く", command=open_dir).pack(side="left", padx=4)
            Button(fr, text="WAV を開く", command=open_wav).pack(side="left", padx=4)
        if mp3_p is not None and mp3_p.is_file():

            def open_mp3() -> None:
                _open_path_windows(mp3_p)
                close()

            Button(fr, text="MP3 を開く", command=open_mp3).pack(side="left", padx=4)
        Button(top, text="閉じる", command=close).pack(pady=(4, 12))

    def _browse_infer_checkpoint(self) -> None:
        p = filedialog.askopenfilename(
            title="推論チェックポイントを選択（.safetensors）",
            filetypes=[
                ("Safetensors", "*.safetensors"),
                ("すべて", "*.*"),
            ],
        )
        if p:
            self.var_infer_checkpoint.set(p)

    def _open_app_settings(self) -> None:
        top = Toplevel(self.root)
        top.title("アプリ設定")
        top.transient(self.root)
        var_ff = StringVar(value=str(self._studio_settings.get("ffmpeg_path", "") or ""))
        var_log = StringVar(value=str(self._studio_settings.get("log_max_lines", 3000)))

        Label(
            top,
            text="ffmpeg のフルパス（空のときは PATH から検索）",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 2))
        Entry(top, textvariable=var_ff, width=72).pack(fill="x", padx=12)
        f_btn = Frame(top)
        f_btn.pack(fill="x", padx=12, pady=4)

        def browse_ff() -> None:
            fp = filedialog.askopenfilename(
                title="ffmpeg を選択",
                filetypes=[("実行ファイル", "*.exe"), ("すべて", "*.*")],
            )
            if fp:
                var_ff.set(fp)

        Button(f_btn, text="参照…", command=browse_ff).pack(side="left")

        Label(
            top,
            text="ログの最大行数（超えた分は先頭から削除）",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))
        Entry(top, textvariable=var_log, width=12).pack(anchor="w", padx=12)

        def save_settings() -> None:
            self._studio_settings["ffmpeg_path"] = var_ff.get().strip()
            try:
                lm = int(var_log.get().strip())
            except ValueError:
                messagebox.showerror("設定", "ログ行数は整数で入力してください。", parent=top)
                return
            self._studio_settings["log_max_lines"] = max(500, min(50_000, lm))
            studio_settings.save(self._studio_settings)
            messagebox.showinfo("設定", "保存しました。", parent=top)
            top.destroy()

        Button(top, text="保存", command=save_settings).pack(pady=12)

    def _log_batch_line(self, idx: int, total: int, msg: str) -> None:
        self._log_append(f"\n----- 連続生成 [{idx}/{total}] -----\n")
        tail = msg[-12_000:] if len(msg) > 12_000 else msg
        self._log_append(tail)
        if not tail.endswith("\n"):
            self._log_append("\n")
        self.status.set(f"連続生成中…（{idx}/{total}）")

    def _batch_finished(
        self,
        ok: bool,
        total: int,
        failed_at: int | None,
        err: str | None = None,
    ) -> None:
        self._job_running = False
        try:
            title = self._notebook.tab(self._notebook.select(), "text")
        except TclError:
            title = ""
        if title == LORA_TAB_TITLE:
            self.btn_gen.configure(state="disabled")
        else:
            self.btn_gen.configure(state="normal")

        if ok:
            self.status.set("連続生成 完了")
            self._log_append(f"\n連続生成: {total} 件すべて完了。\n")
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            except Exception:
                pass
            if self.var_notify_done.get():
                gen_dir = studio_root() / "outputs" / "generated"
                if gen_dir.is_dir():
                    _open_folder_windows(gen_dir)
            messagebox.showinfo("連続生成", f"{total} 件すべて完了しました。")
            return

        self.status.set("連続生成 エラー")
        if err:
            self._log_append(err)
            if not err.endswith("\n"):
                self._log_append("\n")
        hint = ""
        if failed_at is not None:
            hint = f"{failed_at}/{total} 件目で失敗しました。\n\n"
        messagebox.showerror(
            "連続生成",
            hint + (err[-6000:] if err else "不明なエラー"),
        )

    def _run_batch_generate(self) -> None:
        if self._job_running:
            messagebox.showinfo(
                "実行中",
                "バックグラウンド処理が進行中です。完了を待ってください。",
            )
            return
        try:
            if self._notebook.index(self._notebook.select()) == 2:
                messagebox.showinfo(
                    "連続生成",
                    "LoRA 作成タブでは連続生成できません。\n"
                    "「参照音声」または「VoiceDesign」に切り替えてください。",
                )
                return
        except TclError:
            pass

        raw = self.txt_body.get("1.0", END)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            messagebox.showerror("連続生成", "本文に空でない行がありません。")
            return
        nlines = len(lines)
        if nlines > 300:
            if not messagebox.askyesno(
                "確認",
                f"{nlines} 行あります。連続生成を続行しますか？",
            ):
                return

        tts_root = Path(self.var_tts_dir.get().strip())
        batch_script = tts_root / "scripts" / "studio_batch_infer.py"
        use_batch_infer = batch_script.is_file()

        prepared: tuple[dict, Path, list[Path]] | None = None
        planned: list[tuple[list[str], Path, Path]] = []

        if use_batch_infer:
            p = self._try_build_studio_batch_config(lines)
            if not p:
                return
            prepared = p
        else:
            for line in lines:
                out_path = self._compute_auto_output_wav_path_from_line(line)
                validated = self._validate(
                    body_override=line,
                    output_path_override=out_path,
                    skip_overwrite_prompt=True,
                )
                if not validated:
                    return
                cmd, tts = validated
                planned.append((cmd, tts, out_path))

        self._job_running = True
        self.btn_gen.configure(state="disabled")
        self.status.set(f"連続生成中（0/{nlines}）…")
        if use_batch_infer:
            self._log_append(
                f"\n========== 連続生成開始: {nlines} 行（モデルは1回だけロード）==========\n"
            )
        else:
            self._log_append(f"\n========== 連続生成開始: {nlines} 行 ==========\n")
            self._log_append(
                "※ Irodori-TTS に scripts/studio_batch_infer.py が無いため、"
                "行ごとに infer を起動します（時間がかかります）。\n"
            )

        do_mp3 = self.var_mp3_convert.get()
        br = self.var_mp3_bitrate.get().strip()

        def work() -> None:
            total = nlines
            if prepared is not None:
                config, cwd, out_paths = prepared
                tmp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        suffix=".json",
                        delete=False,
                    ) as tf:
                        json.dump(config, tf, ensure_ascii=False)
                        tmp_path = Path(tf.name)
                    uv_bin = _which_uv()
                    if not uv_bin:
                        self.root.after(
                            0,
                            lambda: self._batch_finished(
                                False, total, 1, "uv が見つかりません。"
                            ),
                        )
                        return
                    proc = subprocess.run(
                        [
                            uv_bin,
                            "run",
                            "python",
                            "scripts/studio_batch_infer.py",
                            "--config",
                            str(tmp_path),
                        ],
                        cwd=str(cwd),
                        env=_env_with_uv_on_path(),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError as e:
                    self.root.after(
                        0,
                        lambda: self._batch_finished(False, total, 1, str(e)),
                    )
                    return
                finally:
                    if tmp_path is not None:
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass

                msg = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")

                def _log_one_shot() -> None:
                    self._log_append(
                        f"\n----- 連続生成（1プロセス・全 {total} 行）-----\n"
                    )
                    tail = msg[-12_000:] if len(msg) > 12_000 else msg
                    self._log_append(tail)
                    if not tail.endswith("\n"):
                        self._log_append("\n")
                    self.status.set(f"連続生成中…（{total}/{total}）")

                self.root.after(0, _log_one_shot)

                if proc.returncode != 0:
                    err = f"終了コード {proc.returncode}\n\n{msg[-12_000:]}"
                    self.root.after(
                        0,
                        lambda e=err: self._batch_finished(False, total, 1, e),
                    )
                    return

                for out_path in out_paths:
                    _mp3_p, mp3_log = self._try_mp3_convert_with(out_path, do_mp3, br)
                    if mp3_log:
                        self.root.after(0, lambda log=mp3_log: self._log_append(log))

                self.root.after(0, lambda: self._batch_finished(True, total, None, None))
                return

            for i, (cmd, cwd, out_path) in enumerate(planned):
                cur = i + 1
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
                except OSError as e:
                    self.root.after(
                        0,
                        lambda: self._batch_finished(False, total, cur, str(e)),
                    )
                    return
                msg = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
                self.root.after(
                    0,
                    lambda m=msg, c=cur, t=total: self._log_batch_line(c, t, m),
                )
                if proc.returncode != 0:
                    err = f"終了コード {proc.returncode}\n\n{msg[-12_000:]}"
                    self.root.after(
                        0,
                        lambda e=err, c=cur, t=total: self._batch_finished(
                            False, t, c, e
                        ),
                    )
                    return
                _mp3_p, mp3_log = self._try_mp3_convert_with(out_path, do_mp3, br)
                if mp3_log:
                    self.root.after(0, lambda log=mp3_log: self._log_append(log))

            self.root.after(0, lambda: self._batch_finished(True, total, None, None))

        threading.Thread(target=work, daemon=True).start()

    def _trim_ref_wav_dialog(self) -> None:
        ff = studio_settings.resolve_ffmpeg_path()
        if not ff:
            messagebox.showerror(
                "参照 WAV を切り出し",
                "ffmpeg が見つかりません。\n"
                "ファイル → アプリ設定 でパスを指定するか、PATH に追加してください。",
            )
            return
        src = filedialog.askopenfilename(
            title="元の WAV を選択",
            filetypes=[("WAV", "*.wav"), ("すべて", "*.*")],
        )
        if not src:
            return
        start_s = askstring("切り出し", "開始位置（秒。例: 1.5）:")
        if start_s is None:
            return
        dur_s = askstring("切り出し", "長さ（秒。例: 10）:")
        if dur_s is None:
            return
        try:
            ss = float(start_s.strip().replace(",", "."))
            dur = float(dur_s.strip().replace(",", "."))
        except ValueError:
            messagebox.showerror("切り出し", "開始・長さは数値で指定してください。")
            return
        if dur <= 0:
            messagebox.showerror("切り出し", "長さは正の数にしてください。")
            return

        default_name = Path(src).stem + "_trim.wav"
        out = filedialog.asksaveasfilename(
            title="切り出した WAV の保存先",
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV", "*.wav"), ("すべて", "*.*")],
        )
        if not out:
            return
        out_p = Path(out)
        try:
            out_p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("切り出し", f"保存先を作成できません: {e}")
            return

        self.status.set("参照 WAV 切り出し中…")
        cmd = [
            ff,
            "-y",
            "-ss",
            str(ss),
            "-i",
            src,
            "-t",
            str(dur),
            str(out_p),
        ]

        def work() -> None:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=600,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                self.root.after(
                    0,
                    lambda: self._trim_ref_done(str(e), None),
                )
                return
            msg = (proc.stderr or proc.stdout or "").strip()
            if proc.returncode != 0:
                err = msg or f"ffmpeg 終了コード {proc.returncode}"
                self.root.after(
                    0,
                    lambda: self._trim_ref_done(err, None),
                )
                return
            self.root.after(0, lambda: self._trim_ref_done(None, out_p))

        threading.Thread(target=work, daemon=True).start()

    def _trim_ref_done(self, err: str | None, out_p: Path | None) -> None:
        if err:
            self.status.set("切り出し エラー")
            self._log_append(f"参照 WAV 切り出し失敗:\n{err}\n")
            messagebox.showerror("切り出し", err[-4000:])
            return
        self.status.set("切り出し 完了")
        if out_p is not None:
            self._log_append(f"参照 WAV 切り出し: {out_p}\n")
            messagebox.showinfo("切り出し", f"保存しました:\n{out_p}")
            if messagebox.askyesno("参照 WAV", "このファイルを参照音声に設定しますか？"):
                self.var_ref_wav.set(str(out_p))


def run_app() -> None:
    root = Tk()
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = IrodoriStudioApp(root)
    app._sync_lora_paths_from_tts()
    app._refresh_preset_combo()
    app._refresh_history_combo()
    root.mainloop()
