import os
import sys
import json
import time
import queue
import threading
import webbrowser
import requests
from tkinter import filedialog, messagebox
import customtkinter as ctk
from faster_whisper import WhisperModel

# Evita crash su tqdm/huggingface quando compilato con --noconsole
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# Cartella dati utente scrivibile anche senza permessi Admin
DATA_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
    "SbobinaFacile",
)
os.makedirs(DATA_DIR, exist_ok=True)

CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    "ai_provider": "Gemini (Gratis Online)",
    "gemini_key": "",
    "gemini_model": "gemini-2.5-flash",
    "ollama_model": "llama3.2",
    "quality": "Bilanciata",
    "language": "Italiano",
    "appearance": "Sistema",
}

QUALITY_MAP = {
    "Veloce": "small",
    "Bilanciata": "medium",
    "Massima": "large-v3-turbo",
}

LANG_MAP = {
    "Italiano": "it",
    "Inglese": "en",
    "Spagnolo": "es",
    "Francese": "fr",
    "Tedesco": "de",
    "Rileva da solo": "auto",
}


def load_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_CONFIG, **data}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return "meno di 1 minuto"
    mins = int(round(seconds / 60))
    if mins < 60:
        return f"circa {mins} min"
    hrs = mins // 60
    rem = mins % 60
    return f"circa {hrs}h {rem}m"


def chunk_text(text: str, max_chars: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks, current, current_len = [], [], 0
    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, config: dict, on_save, first_run: bool = False):
        super().__init__(parent)
        self.title(
            "Benvenuto! Configurazione Iniziale" if first_run else "Impostazioni AI e Aspetto"
        )
        self.geometry("500x520" if first_run else "500x450")
        self.resizable(False, False)
        self.config = config
        self.on_save = on_save

        self.transient(parent)
        self.grab_set()

        frame = ctk.CTkFrame(self, corner_radius=12)
        frame.pack(fill="both", expand=True, padx=18, pady=18)

        if first_run:
            banner = ctk.CTkFrame(frame, fg_color=("#dceeff", "#1c3b57"), corner_radius=10)
            banner.pack(fill="x", padx=16, pady=(14, 10))
            ctk.CTkLabel(
                banner,
                text="Prima installazione rilevata!\n"
                "Per generare i riassunti gratis, incolla qui sotto la tua chiave API di Google Gemini.\n"
                "Puoi ottenerla gratis in 30 secondi cliccando il pulsante qui sotto.",
                font=ctk.CTkFont(size=12),
                justify="left",
            ).pack(anchor="w", padx=12, pady=10)

        ctk.CTkLabel(
            frame,
            text="Configurazione Motore Riassunti AI",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(10, 6))

        self.provider_var = ctk.StringVar(value=config["ai_provider"])
        self.provider_menu = ctk.CTkOptionMenu(
            frame,
            variable=self.provider_var,
            values=[
                "Gemini (Gratis Online)",
                "Ollama (Gratis Offline)",
                "Disattivato (Solo trascrizione)",
            ],
            width=320,
            command=self._toggle_fields,
        )
        self.provider_menu.pack(anchor="w", padx=16, pady=(0, 12))

        key_header = ctk.CTkFrame(frame, fg_color="transparent")
        key_header.pack(fill="x", padx=16)
        ctk.CTkLabel(
            key_header,
            text="Chiave API Google Gemini:",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")

        self.btn_get_key = ctk.CTkButton(
            key_header,
            text="🌐 Ottieni chiave gratis",
            width=155,
            height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: webbrowser.open("https://aistudio.google.com/apikey"),
        )
        self.btn_get_key.pack(side="right")

        self.gemini_entry = ctk.CTkEntry(
            frame, placeholder_text="Incolla qui la chiave (es. AIzaSy...)", show="*", width=430
        )
        self.gemini_entry.insert(0, config.get("gemini_key", ""))
        self.gemini_entry.pack(anchor="w", padx=16, pady=(6, 12))

        ctk.CTkLabel(
            frame, text="Nome modello AI:", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=16)
        self.model_entry = ctk.CTkEntry(frame, width=260)
        self.model_entry.pack(anchor="w", padx=16, pady=(4, 12))

        ctk.CTkLabel(
            frame, text="Tema grafico:", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=16, pady=(2, 0))
        self.theme_var = ctk.StringVar(value=config.get("appearance", "Sistema"))
        ctk.CTkSegmentedButton(
            frame,
            values=["Sistema", "Chiaro", "Scuro"],
            variable=self.theme_var,
        ).pack(anchor="w", padx=16, pady=(4, 16))

        ctk.CTkButton(
            frame,
            text="💾 Salva e Continua",
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1f8b4c",
            hover_color="#176d3b",
            command=self._save,
        ).pack(fill="x", padx=16, pady=(4, 14))

        self._toggle_fields(self.provider_var.get())

    def _toggle_fields(self, choice: str):
        self.model_entry.delete(0, "end")
        if "Gemini" in choice:
            self.gemini_entry.configure(state="normal")
            self.btn_get_key.configure(state="normal")
            self.model_entry.configure(state="normal")
            self.model_entry.insert(0, self.config.get("gemini_model", "gemini-2.5-flash"))
        elif "Ollama" in choice:
            self.gemini_entry.configure(state="disabled")
            self.btn_get_key.configure(state="disabled")
            self.model_entry.configure(state="normal")
            self.model_entry.insert(0, self.config.get("ollama_model", "llama3.2"))
        else:
            self.gemini_entry.configure(state="disabled")
            self.btn_get_key.configure(state="disabled")
            self.model_entry.configure(state="disabled")

    def _save(self):
        provider = self.provider_var.get()
        key = self.gemini_entry.get().strip()

        if "Gemini" in provider and not key:
            messagebox.showwarning(
                "Chiave mancante",
                "Incolla la chiave API di Google Gemini oppure scegli 'Disattivato' / 'Ollama' dal menu.",
                parent=self,
            )
            return

        self.config["ai_provider"] = provider
        self.config["gemini_key"] = key
        if "Gemini" in provider and self.model_entry.get().strip():
            self.config["gemini_model"] = self.model_entry.get().strip()
        elif "Ollama" in provider and self.model_entry.get().strip():
            self.config["ollama_model"] = self.model_entry.get().strip()
        self.config["appearance"] = self.theme_var.get()

        save_config(self.config)
        self.on_save()
        self.destroy()


class SbobinaFacileApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.config_data = load_config()
        self._apply_theme(self.config_data.get("appearance", "Sistema"))
        ctk.set_default_color_theme("blue")

        self.title("SbobinaFacile AI")
        self.geometry("880x740")
        self.minsize(780, 640)

        self.ui_queue = queue.Queue()
        self.selected_file = ""
        self.last_output_dir = ""
        self.stop_requested = False

        self._build_ui()
        self.after(100, self._process_queue)
        self.after(300, self._check_first_run)

    def _apply_theme(self, choice: str):
        mapping = {"Sistema": "System", "Chiaro": "Light", "Scuro": "Dark"}
        ctk.set_appearance_mode(mapping.get(choice, "System"))

    def _check_first_run(self):
        is_missing_config = not os.path.isfile(CONFIG_PATH)
        needs_key = (
            "Gemini" in self.config_data.get("ai_provider", "")
            and not self.config_data.get("gemini_key", "").strip()
        )
        if is_missing_config or needs_key:
            SettingsWindow(self, self.config_data, self._on_settings_saved, first_run=True)

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=22, pady=(16, 8))

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left")
        ctk.CTkLabel(
            title_box,
            text="🎙️ SbobinaFacile AI",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="Trasforma file audio e video lunghi in testo chiaro e riassunti automatici",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(anchor="w")

        ctk.CTkButton(
            header,
            text="⚙️ Impostazioni",
            width=130,
            height=34,
            fg_color=("gray82", "gray25"),
            text_color=("gray10", "gray90"),
            hover_color=("gray75", "gray35"),
            command=self._open_settings,
        ).pack(side="right", pady=4)

        card_file = ctk.CTkFrame(self, corner_radius=14)
        card_file.pack(fill="x", padx=22, pady=6)

        row_file = ctk.CTkFrame(card_file, fg_color="transparent")
        row_file.pack(fill="x", padx=16, pady=14)

        self.btn_pick = ctk.CTkButton(
            row_file,
            text="📁  Scegli File Audio o Video",
            height=44,
            width=230,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._pick_file,
        )
        self.btn_pick.pack(side="left")

        info_col = ctk.CTkFrame(row_file, fg_color="transparent")
        info_col.pack(side="left", fill="x", expand=True, padx=(16, 8))

        self.file_name_label = ctk.CTkLabel(
            info_col,
            text="Nessun file selezionato",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        self.file_name_label.pack(anchor="w")

        self.file_size_label = ctk.CTkLabel(
            info_col,
            text="Supporta MP3, MP4, MKV, WAV, M4A, MOV (anche oltre 400 MB)",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        )
        self.file_size_label.pack(anchor="w")

        self.btn_txt_only = ctk.CTkButton(
            row_file,
            text="📝 Riassumi un .txt",
            height=34,
            width=140,
            fg_color="transparent",
            border_width=1,
            text_color=("gray20", "gray85"),
            command=self._pick_and_summarize_txt,
        )
        self.btn_txt_only.pack(side="right")

        card_opts = ctk.CTkFrame(self, corner_radius=14)
        card_opts.pack(fill="x", padx=22, pady=6)

        opts_inner = ctk.CTkFrame(card_opts, fg_color="transparent")
        opts_inner.pack(fill="x", padx=16, pady=12)

        q_box = ctk.CTkFrame(opts_inner, fg_color="transparent")
        q_box.pack(side="left", padx=(0, 24))
        ctk.CTkLabel(
            q_box, text="Precisione trascrizione:", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(0, 4))
        self.quality_seg = ctk.CTkSegmentedButton(
            q_box, values=["Veloce", "Bilanciata", "Massima"]
        )
        self.quality_seg.set(self.config_data.get("quality", "Bilanciata"))
        self.quality_seg.pack(anchor="w")

        l_box = ctk.CTkFrame(opts_inner, fg_color="transparent")
        l_box.pack(side="left", padx=(0, 24))
        ctk.CTkLabel(
            l_box, text="Lingua parlata:", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", pady=(0, 4))
        self.lang_menu = ctk.CTkOptionMenu(
            l_box, values=list(LANG_MAP.keys()), width=140
        )
        self.lang_menu.set(self.config_data.get("language", "Italiano"))
        self.lang_menu.pack(anchor="w")

        s_box = ctk.CTkFrame(opts_inner, fg_color="transparent")
        s_box.pack(side="right")
        self.summary_switch = ctk.CTkSwitch(
            s_box,
            text="Crea anche il riassunto AI",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        if "Disattivato" not in self.config_data.get("ai_provider", ""):
            self.summary_switch.select()
        self.summary_switch.pack(pady=(16, 0))

        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(fill="x", padx=22, pady=(8, 4))

        btn_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_row.pack(fill="x")

        self.btn_start = ctk.CTkButton(
            btn_row,
            text="▶   AVVIA ELABORAZIONE",
            height=46,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#1f8b4c",
            hover_color="#176d3b",
            command=self._start_full,
        )
        self.btn_start.pack(side="left", fill="x", expand=True)

        self.btn_stop = ctk.CTkButton(
            btn_row,
            text="⏹ Ferma",
            height=46,
            width=110,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#b83b3b",
            hover_color="#8f2b2b",
            state="disabled",
            command=self._request_stop,
        )
        self.btn_stop.pack(side="right", padx=(10, 0))

        self.progress_bar = ctk.CTkProgressBar(action_frame, height=12, corner_radius=6)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", pady=(10, 4))

        self.status_label = ctk.CTkLabel(
            action_frame,
            text="Pronto. Seleziona un file per iniziare.",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        self.status_label.pack(anchor="w")

        self.tabs = ctk.CTkTabview(self, corner_radius=14)
        self.tabs.pack(fill="both", expand=True, padx=22, pady=(4, 16))

        tab_sum = self.tabs.add("✨ Riassunto")
        tab_tr = self.tabs.add("📝 Trascrizione Completa")

        sum_bar = ctk.CTkFrame(tab_sum, fg_color="transparent")
        sum_bar.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            sum_bar, text="📋 Copia Riassunto", width=150, height=30,
            command=lambda: self._copy_text(self.summary_box),
        ).pack(side="left")
        self.btn_open_folder_1 = ctk.CTkButton(
            sum_bar, text="📂 Apri cartella dei file", width=170, height=30,
            fg_color=("gray80", "gray28"), text_color=("gray10", "gray90"),
            state="disabled", command=self._open_output_folder,
        )
        self.btn_open_folder_1.pack(side="right")

        self.summary_box = ctk.CTkTextbox(tab_sum, wrap="word", font=ctk.CTkFont(size=14))
        self.summary_box.pack(fill="both", expand=True)

        tr_bar = ctk.CTkFrame(tab_tr, fg_color="transparent")
        tr_bar.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(
            tr_bar, text="📋 Copia Trascrizione", width=160, height=30,
            command=lambda: self._copy_text(self.transcript_box),
        ).pack(side="left")
        self.btn_open_folder_2 = ctk.CTkButton(
            tr_bar, text="📂 Apri cartella dei file", width=170, height=30,
            fg_color=("gray80", "gray28"), text_color=("gray10", "gray90"),
            state="disabled", command=self._open_output_folder,
        )
        self.btn_open_folder_2.pack(side="right")

        self.transcript_box = ctk.CTkTextbox(tab_tr, wrap="word", font=ctk.CTkFont(size=13))
        self.transcript_box.pack(fill="both", expand=True)

    def _open_settings(self):
        SettingsWindow(self, self.config_data, self._on_settings_saved, first_run=False)

    def _on_settings_saved(self):
        self._apply_theme(self.config_data.get("appearance", "Sistema"))
        if "Disattivato" in self.config_data.get("ai_provider", ""):
            self.summary_switch.deselect()
        else:
            self.summary_switch.select()

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Seleziona file audio o video",
            filetypes=[
                ("Audio e Video", "*.mp3 *.mp4 *.mkv *.wav *.m4a *.mov *.webm *.ogg *.flac *.avi"),
                ("Tutti i file", "*.*"),
            ],
        )
        if path:
            self.selected_file = path
            name = os.path.basename(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self.file_name_label.configure(text=name)
            self.file_size_label.configure(text=f"Dimensione: {size_mb:.1f} MB — Pronto per essere elaborato")

    def _copy_text(self, textbox: ctk.CTkTextbox):
        content = textbox.get("1.0", "end-1c").strip()
        if not content:
            messagebox.showinfo("Vuoto", "Non c'è ancora nessun testo da copiare.")
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.update()
        self.status_label.configure(text="✅ Testo copiato negli appunti! Ora puoi incollarlo dove vuoi.")

    def _open_output_folder(self):
        if self.last_output_dir and os.path.isdir(self.last_output_dir):
            os.startfile(self.last_output_dir)

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.btn_start.configure(state=state)
        self.btn_pick.configure(state=state)
        self.btn_txt_only.configure(state=state)
        self.btn_stop.configure(state="normal" if busy else "disabled")

    def _request_stop(self):
        self.stop_requested = True
        self.status_label.configure(text="Interruzione in corso, salvo ciò che è stato fatto finora...")

    def _check_ai_ready(self) -> bool:
        provider = self.config_data.get("ai_provider", "")
        if "Gemini" in provider and not self.config_data.get("gemini_key", "").strip():
            SettingsWindow(self, self.config_data, self._on_settings_saved, first_run=True)
            return False
        return True

    def _start_full(self):
        if not self.selected_file or not os.path.isfile(self.selected_file):
            messagebox.showwarning("Nessun file", "Clicca prima su 'Scegli File Audio o Video'.")
            return

        do_summary = bool(self.summary_switch.get()) and "Disattivato" not in self.config_data.get("ai_provider", "")
        if do_summary and not self._check_ai_ready():
            return

        self.config_data["quality"] = self.quality_seg.get()
        self.config_data["language"] = self.lang_menu.get()
        save_config(self.config_data)

        self.transcript_box.delete("1.0", "end")
        self.summary_box.delete("1.0", "end")
        self.progress_bar.set(0.0)
        self.stop_requested = False
        self.last_output_dir = os.path.dirname(self.selected_file)
        self._set_busy(True)
        self.tabs.set("📝 Trascrizione Completa")

        threading.Thread(
            target=self._worker_full,
            args=(self.selected_file, do_summary),
            daemon=True,
        ).start()

    def _pick_and_summarize_txt(self):
        if "Disattivato" in self.config_data.get("ai_provider", ""):
            messagebox.showinfo("AI disattivata", "Attiva Gemini o Ollama nelle Impostazioni per usare i riassunti.")
            return
        if not self._check_ai_ready():
            return

        txt_path = filedialog.askopenfilename(
            title="Scegli un file di testo (.txt) da riassumere",
            filetypes=[("File di testo", "*.txt"), ("Tutti i file", "*.*")],
        )
        if not txt_path:
            return

        self.summary_box.delete("1.0", "end")
        self.progress_bar.set(0.0)
        self.stop_requested = False
        self.last_output_dir = os.path.dirname(txt_path)
        self._set_busy(True)
        self.tabs.set("✨ Riassunto")

        threading.Thread(
            target=self._worker_txt_only,
            args=(txt_path,),
            daemon=True,
        ).start()

    def _load_whisper(self, model_size: str) -> WhisperModel:
        try:
            return WhisperModel(model_size, device="cuda", compute_type="float16", download_root=MODELS_DIR)
        except Exception:
            return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=MODELS_DIR)

    def _worker_full(self, file_path: str, do_summary: bool):
        try:
            model_size = QUALITY_MAP.get(self.quality_seg.get(), "medium")
            lang_code = LANG_MAP.get(self.lang_menu.get(), "it")
            lang = None if lang_code == "auto" else lang_code

            self.ui_queue.put(("status", "Preparazione del motore vocale (al primo avvio scarica il modello, attendi 1-2 min)..."))
            model = self._load_whisper(model_size)

            self.ui_queue.put(("status", "Analisi dell'audio in corso..."))
            segments, info = model.transcribe(
                file_path,
                language=lang,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            total_dur = info.duration or 1.0
            base_path = os.path.splitext(file_path)[0]
            txt_path = f"{base_path}_trascrizione.txt"
            parts = []
            t_start = time.time()

            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    if self.stop_requested:
                        break
                    line = f"[{format_timestamp(seg.start)}] {seg.text.strip()}\n"
                    f.write(line)
                    f.flush()
                    parts.append(line)

                    ratio = min(0.99, seg.end / total_dur)
                    eta_txt = ""
                    if ratio > 0.02:
                        elapsed = time.time() - t_start
                        eta_sec = (elapsed / ratio) - elapsed
                        eta_txt = f" — Tempo stimato rimasto: {format_eta(eta_sec)}"

                    self.ui_queue.put(("transcript", line))
                    self.ui_queue.put(("progress", ratio))
                    self.ui_queue.put((
                        "status",
                        f"Sto trascrivendo... {ratio*100:.0f}% ({format_timestamp(seg.end)} su {format_timestamp(total_dur)}){eta_txt}",
                    ))

            self.ui_queue.put(("enable_folder", None))

            if self.stop_requested:
                self.ui_queue.put(("status", f"Fermato. La parte già trascritta è salvata in {os.path.basename(txt_path)}"))
                return

            self.ui_queue.put(("progress", 1.0))

            if do_summary and parts:
                self._run_summary("".join(parts), base_path)
            else:
                self.ui_queue.put(("status", "✅ Completato! File salvato nella stessa cartella dell'originale."))

        except Exception as e:
            self.ui_queue.put(("error", self._humanize_error(e)))
        finally:
            self.ui_queue.put(("done", None))

    def _worker_txt_only(self, txt_path: str):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            base_path = os.path.splitext(txt_path)[0]
            self.ui_queue.put(("enable_folder", None))
            self._run_summary(text, base_path)
        except Exception as e:
            self.ui_queue.put(("error", self._humanize_error(e)))
        finally:
            self.ui_queue.put(("done", None))

    def _call_ai(self, prompt: str) -> str:
        provider = self.config_data.get("ai_provider", "")
        if "Ollama" in provider:
            model = self.config_data.get("ollama_model", "llama3.2")
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=300,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama ha risposto con errore {resp.status_code}")
            return resp.json().get("response", "").strip()

        elif "Gemini" in provider:
            model = self.config_data.get("gemini_model", "gemini-2.5-flash")
            key = self.config_data.get("gemini_key", "").strip()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=180,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Errore Gemini ({resp.status_code}): {resp.text}")
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        return ""

    def _run_summary(self, full_text: str, base_path: str):
        provider = self.config_data.get("ai_provider", "")
        self.ui_queue.put(("status", "Sto leggendo la trascrizione per preparare il riassunto AI..."))
        self.ui_queue.put(("switch_tab", "✨ Riassunto"))

        max_chars = 350000 if "Gemini" in provider else 10000
        chunks = chunk_text(full_text, max_chars=max_chars)

        if len(chunks) == 1:
            prompt = (
                "Analizza questa trascrizione in italiano e scrivi un riassunto chiaro, ordinato e facile da leggere "
                "(usa titoli in maiuscolo e semplici elenchi con il simbolo •, evita codici markdown complessi).\n"
                "Includi:\n"
                "1. SINTESI GENERALE\n"
                "2. PUNTI CHIAVE E ARGOMENTI TRATTATI\n"
                "3. CONCLUSIONI O DECISIONI IMPORTANTI\n\n"
                f"TRASCRIZIONE:\n{chunks[0]}"
            )
            final_summary = self._call_ai(prompt)
        else:
            partials = []
            for idx, ch in enumerate(chunks, start=1):
                if self.stop_requested:
                    return
                self.ui_queue.put(("status", f"Riassumo la parte {idx} di {len(chunks)}..."))
                self.ui_queue.put(("progress", idx / (len(chunks) + 1)))
                p = (
                    f"Riassumi in modo fedele e completo questa parte ({idx}/{len(chunks)}) "
                    f"di una lunga trascrizione in italiano, mantenendo nomi e concetti chiave:\n\n{ch}"
                )
                partials.append(self._call_ai(p))

            self.ui_queue.put(("status", "Unisco tutto nel riassunto finale..."))
            combined = "\n\n---\n\n".join(partials)
            final_prompt = (
                "Unisci questi riassunti parziali in un unico riassunto finale ordinato e facile da leggere in italiano "
                "(usa titoli in maiuscolo ed elenchi con •):\n"
                "1. SINTESI GENERALE\n"
                "2. PUNTI CHIAVE PER ARGOMENTO\n"
                "3. CONCLUSIONI\n\n"
                f"{combined}"
            )
            final_summary = self._call_ai(final_prompt)

        summary_path = f"{base_path}_riassunto.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(final_summary)

        self.ui_queue.put(("progress", 1.0))
        self.ui_queue.put(("summary", final_summary))
        self.ui_queue.put(("status", "✅ Tutto pronto! Trascrizione e riassunto salvati nella cartella del file."))

    def _humanize_error(self, err: Exception) -> str:
        msg = str(err)
        if "11434" in msg or "ConnectionError" in msg:
            return "Non riesco a collegarmi a Ollama. Assicurati che Ollama sia aperto sul PC (o passa a Gemini nelle Impostazioni)."
        if "API_KEY_INVALID" in msg or "400" in msg or "403" in msg:
            return "La chiave API di Google Gemini sembra non valida. Controllala cliccando su ⚙️ Impostazioni."
        return f"Si è verificato un imprevisto:\n{msg}"

    def _process_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "transcript":
                    self.transcript_box.insert("end", payload)
                    self.transcript_box.see("end")
                elif kind == "summary":
                    self.summary_box.delete("1.0", "end")
                    self.summary_box.insert("1.0", payload)
                elif kind == "progress":
                    self.progress_bar.set(payload)
                elif kind == "status":
                    self.status_label.configure(text=payload)
                elif kind == "switch_tab":
                    self.tabs.set(payload)
                elif kind == "enable_folder":
                    self.btn_open_folder_1.configure(state="normal")
                    self.btn_open_folder_2.configure(state="normal")
                elif kind == "error":
                    self.status_label.configure(text="❌ Operazione interrotta per un errore.")
                    messagebox.showerror("Attenzione", payload)
                elif kind == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self._process_queue)


if __name__ == "__main__":
    app = SbobinaFacileApp()
    app.mainloop()
