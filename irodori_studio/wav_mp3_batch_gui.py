"""フォルダ内 WAV を MP3 に一括変換する簡易 GUI（tkinter）。"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import END, BooleanVar, Button, Checkbutton, Frame, Label, StringVar, filedialog, messagebox, scrolledtext, ttk

from irodori_studio import settings as studio_settings
from irodori_studio.wav_mp3 import convert_wav_to_mp3, list_wav_files


def run_wav_mp3_batch_gui() -> None:
    root = tk.Tk()
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root.title("WAV → MP3 一括変換")
    root.minsize(520, 420)
    root.geometry("640x480")

    _st = studio_settings.load()
    var_dir = StringVar(value="")
    var_recursive = BooleanVar(value=True)
    var_skip_existing = BooleanVar(value=True)
    var_bitrate = StringVar(value=str(_st.get("mp3_bitrate", "192k")))
    status = StringVar(value="フォルダを選んでください")
    _running = False

    pad = {"padx": 10, "pady": 4}

    top = Frame(root)
    top.pack(fill="x", **pad)
    Label(top, text="変換元フォルダ").pack(anchor="w")
    row = Frame(top)
    row.pack(fill="x")
    ent = ttk.Entry(row, textvariable=var_dir)
    ent.pack(side="left", fill="x", expand=True, padx=(0, 6))

    ff_path = studio_settings.resolve_ffmpeg_path()
    ff_label = "ffmpeg: 検出済み" if ff_path else "ffmpeg: 未検出（PATH か Irodori-Studio のアプリ設定）"
    ff_color = "#333" if ff_path else "#cc6600"
    lbl_ff = Label(top, text=ff_label, fg=ff_color, font=("", 9))
    lbl_ff.pack(anchor="w", pady=(2, 0))

    def browse() -> None:
        p = filedialog.askdirectory(title="WAV が入っているフォルダを選択")
        if p:
            var_dir.set(p)
            status.set(f"選択: {p}")

    def open_studio_settings_hint() -> None:
        messagebox.showinfo(
            "ffmpeg の場所",
            "Irodori-Studio 本体を起動し、\n"
            "「ファイル」→「アプリ設定」から ffmpeg のフルパスを指定できます。\n\n"
            f"設定ファイル: {studio_settings.SETTINGS_PATH}",
        )

    Button(row, text="フォルダを選択…", command=browse).pack(side="left")
    Button(row, text="設定のヒント", command=open_studio_settings_hint).pack(side="left", padx=(4, 0))

    opts = Frame(root)
    opts.pack(fill="x", **pad)
    Checkbutton(
        opts,
        text="サブフォルダも含める",
        variable=var_recursive,
    ).pack(side="left")
    Checkbutton(
        opts,
        text="同名の MP3 がある場合はスキップ",
        variable=var_skip_existing,
    ).pack(side="left", padx=(16, 0))
    Label(opts, text="ビットレート").pack(side="left", padx=(16, 4))

    def persist_bitrate(_event: object = None) -> None:
        br = var_bitrate.get().strip()
        if br not in studio_settings.MP3_BITRATES:
            br = "192k"
            var_bitrate.set(br)
        st = studio_settings.load()
        st["mp3_bitrate"] = br
        studio_settings.save(st)

    combo_br = ttk.Combobox(
        opts,
        textvariable=var_bitrate,
        values=studio_settings.MP3_BITRATES,
        width=7,
        state="readonly",
    )
    combo_br.pack(side="left")
    combo_br.bind("<<ComboboxSelected>>", persist_bitrate)

    prog = ttk.Progressbar(root, mode="determinate")
    prog.pack(fill="x", **pad)

    mid = Frame(root)
    mid.pack(fill="both", expand=True, **pad)
    Label(mid, text="ログ").pack(anchor="w")
    txt = scrolledtext.ScrolledText(mid, height=14, wrap="word", font=("Consolas", 9))
    txt.pack(fill="both", expand=True)

    def log(msg: str) -> None:
        txt.configure(state="normal")
        txt.insert(END, msg)
        txt.see(END)
        txt.configure(state="disabled")

    def set_status(s: str) -> None:
        status.set(s)

    btn_run = Button(root, text="一括変換を開始", height=2)

    def finish_ui(ok: int, fail: int, err: str | None) -> None:
        nonlocal _running
        _running = False
        btn_run.configure(state="normal")
        prog["value"] = prog["maximum"] or 0
        if err:
            set_status("エラー")
            messagebox.showerror("変換", err[:8000])
            return
        set_status(f"完了（成功 {ok} / 失敗 {fail}）")
        messagebox.showinfo("変換", f"成功: {ok} 件\n失敗: {fail} 件")

    def run_batch() -> None:
        nonlocal _running
        if _running:
            return
        folder_s = var_dir.get().strip()
        if not folder_s:
            messagebox.showwarning("変換", "フォルダを指定してください。")
            return
        folder = Path(folder_s)
        if not folder.is_dir():
            messagebox.showerror("変換", f"フォルダがありません:\n{folder}")
            return

        ff = studio_settings.resolve_ffmpeg_path()
        if not ff:
            messagebox.showerror(
                "変換",
                "ffmpeg が見つかりません。\n"
                "Irodori-Studio の「ファイル」→「アプリ設定」でパスを指定するか、"
                "ffmpeg を PATH に追加してください。",
            )
            return

        br = var_bitrate.get().strip()
        if br not in studio_settings.MP3_BITRATES:
            br = "192k"
            var_bitrate.set(br)

        files = list_wav_files(folder, recursive=var_recursive.get())
        if not files:
            messagebox.showinfo("変換", "WAV ファイルが見つかりませんでした。")
            return

        to_convert: list[Path] = []
        skipped = 0
        if var_skip_existing.get():
            for w in files:
                mp3 = w.with_suffix(".mp3")
                if mp3.is_file():
                    skipped += 1
                else:
                    to_convert.append(w)
        else:
            to_convert = list(files)

        n = len(to_convert)
        if n == 0:
            messagebox.showinfo(
                "変換",
                f"変換対象がありません（スキップ {skipped} 件、WAV 合計 {len(files)} 件）。",
            )
            return

        if not messagebox.askyesno(
            "確認",
            f"変換しますか？\n\n対象: {n} 件\n"
            + (f"スキップ（既存 MP3）: {skipped} 件\n" if skipped else "")
            + f"ビットレート: {br}",
        ):
            return

        _running = True
        btn_run.configure(state="disabled")
        txt.configure(state="normal")
        txt.delete("1.0", END)
        txt.configure(state="disabled")
        prog["maximum"] = n
        prog["value"] = 0
        set_status(f"変換中…（0/{n}）")

        def work() -> None:
            ok = 0
            fail = 0
            for i, w in enumerate(to_convert, start=1):
                mp3, err = convert_wav_to_mp3(w, ffmpeg=ff, bitrate=br)
                if mp3 is not None:
                    ok += 1
                    line = f"[OK] {w.name} → {mp3.name}\n"
                else:
                    fail += 1
                    line = f"[NG] {w.name}\n{err}\n"

                def _prog(v: int, total: int) -> None:
                    prog["value"] = v
                    set_status(f"変換中…（{v}/{total}）")

                root.after(0, lambda ln=line: log(ln))
                root.after(0, lambda v=i, total=n: _prog(v, total))

            root.after(0, lambda: finish_ui(ok, fail, None))

        threading.Thread(target=work, daemon=True).start()

    btn_run.configure(command=run_batch)
    btn_run.pack(fill="x", **pad)

    Label(root, textvariable=status, anchor="w", fg="#444").pack(fill="x", padx=10, pady=(0, 8))

    root.mainloop()
