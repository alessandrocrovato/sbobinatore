import os
import sys
import json
import time
import queue
import socket
import shutil
import ctypes
import zipfile
import itertools
import threading
import subprocess
import webbrowser
import requests
from tkinter import filedialog, messagebox
import customtkinter as ctk

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from huggingface_hub import hf_hub_download

if getattr(sys, "frozen", False):
    APP_BIN_DIR = os.path.dirname(sys.executable)
else:
    APP_BIN_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "SbobinaFacile")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
MODELS_DIR = os.path.join(DATA_DIR, "models")
CUDA_LIBS_DIR = os.path.join(DATA_DIR, "cuda_libs")
os.makedirs(MODELS_DIR, exist_ok=True)

LOCAL_LLM_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LOCAL_LLM_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"

DEFAULT_CONFIG = {
    "ai_provider": "Gemini (Gratis Online)",
    "gemini_key": "",
    "gemini_model": "gemini-2.5-flash",
    "quality": "Bilanciata",
    "language": "Italiano",
    "appearance": "Sistema",
    "skip_cuda_download": False,
}
QUALITY_MAP = {"Veloce": "small", "Bilanciata": "medium", "Massima": "large-v3-turbo"}
LANG_MAP = {"Italiano": "it", "Inglese": "en", "Spagnolo": "es", "Francese": "fr", "Tedesco": "de", "Rileva da solo": "auto"}


def register_cuda_dll_dir():
    if os.path.isdir(CUDA_LIBS_DIR):
        try:
            os.add_dll_directory(CUDA_LIBS_DIR)
        except Exception:
            pass
        if CUDA_LIBS_DIR not in os.environ.get("PATH", ""):
            os.environ["PATH"] = CUDA_LIBS_DIR + os.pathsep + os.environ.get("PATH", "")


register_cuda_dll_dir()


def has_nvidia_gpu() -> bool:
    for p in (shutil.which("nvidia-smi"), r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if p and os.path.isfile(p):
            return True
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name -join ';'"],
            creationflags=flags, timeout=5, text=True,
        )
        return "nvidia" in out.lower()
    except Exception:
        return False


def cuda_libs_available() -> bool:
    has_cublas = os.path.isfile(os.path.join(CUDA_LIBS_DIR, "cublas64_12.dll"))
    has_cudnn = any(os.path.isfile(os.path.join(CUDA_LIBS_DIR, n)) for n in ("cudnn_ops64_9.dll", "cudnn64_9.dll", "cudnn64_8.dll"))
    if has_cublas and has_cudnn:
        return True
    try:
        ctypes.WinDLL("cublas64_12.dll")
        for cdn in ("cudnn_ops64_9.dll", "cudnn64_9.dll", "cudnn64_8.dll"):
            try:
                ctypes.WinDLL(cdn)
                return True
            except OSError:
                pass
    except OSError:
        pass
    return False


def get_pypi_wheel_url(package_name: str, prefer_major: str | None = None) -> tuple[str, str]:
    r = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=20)
    r.raise_for_status()
    data = r.json()
    urls = data.get("urls", [])
    if prefer_major and not data.get("info", {}).get("version", "").startswith(prefer_major):
        vers = [v for v in data.get("releases", {}) if v.startswith(prefer_major) and not any(x in v for x in ("a", "b", "rc"))]
        if vers:
            best = max(vers, key=lambda v: tuple(int(x) for x in v.split(".") if x.isdigit()))
            urls = data["releases"][best]
    for item in urls:
        fname = item.get("filename", "")
        if item.get("packagetype") == "bdist_wheel" and "win_amd64" in fname:
            return item["url"], fname
    raise RuntimeError(f"Pacchetto Windows non trovato per {package_name}")


def download_file_resumable(url: str, dest_path: str, progress_cb=None, max_retries: int = 6):
    temp_path = dest_path + ".part"
    for attempt in range(1, max_retries + 1):
        try:
            existing = os.path.getsize(temp_path) if os.path.isfile(temp_path) else 0
            headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                if r.status_code == 416:
                    break
                if existing > 0 and r.status_code == 200:
                    existing, mode = 0, "wb"
                elif r.status_code in (200, 206):
                    mode = "ab" if r.status_code == 206 else "wb"
                else:
                    r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + existing
                downloaded = existing
                with open(temp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_cb and total > 0:
                                progress_cb(downloaded, total)
            break
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(10, attempt * 2))
    if os.path.isfile(temp_path):
        os.replace(temp_path, dest_path)


def hf_download_with_retry(repo_id: str, filename: str, local_dir: str, max_retries: int = 6) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(12, attempt * 2))
    raise RuntimeError("Download interrotto.")


def load_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def format_timestamp(seconds: float) -> str:
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return "meno di 1 minuto"
    mins = int(round(seconds / 60))
    return f"circa {mins} min" if mins < 60 else f"circa {mins // 60}h {mins % 60}m"


def chunk_text(text: str, max_chars: int) -> list[str]:
    chunks, current, current_len = [], [], 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > max_chars and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def find_llama_server() -> str | None:
    root = os.path.join(APP_BIN_DIR, "llama_bin")
    if not os.path.isdir(root):
        return None
    for rdir, _dirs, files in os.walk(root):
        if "llama-server.exe" in files:
            return os.path.join(rdir, "llama-server.exe")
    return None


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, config: dict, on_save, first_run: bool = False):
        super().__init__(parent)
        self.title("Benvenuto! Configurazione Iniziale" if first_run else "Impostazioni AI e Aspetto")
        self.geometry("510x580" if first_run else "510x495")
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
                "• Con Google Gemini: veloce e gratis online (incolla la tua chiave API).\n"
                "• Con AI Locale Integrata: tutto offline sul tuo PC usando la scheda video\n"
                "  (nessuna chiave richiesta; scaricherà ~2 GB al primo riassunto).",
                font=ctk.CTkFont(size=12), justify="left",
            ).pack(anchor="w", padx=12, pady=10)

        ctk.CTkLabel(frame, text="Motore per i Riassunti AI", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=16, pady=(10, 6))
        self.provider_var = ctk.StringVar(value=config["ai_provider"])
        ctk.CTkOptionMenu(
            frame, variable=self.provider_var, width=340, command=self._toggle_fields,
            values=["Gemini (Gratis Online)", "AI Locale Integrata (Offline, Gratis)", "Disattivato (Solo trascrizione)"],
        ).pack(anchor="w", padx=16, pady=(0, 12))

        key_header = ctk.CTkFrame(frame, fg_color="transparent")
        key_header.pack(fill="x", padx=16)
        ctk.CTkLabel(key_header, text="Chiave API Google Gemini:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.btn_get_key = ctk.CTkButton(
            key_header, text="🌐 Ottieni chiave gratis", width=155, height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: webbrowser.open("https://aistudio.google.com/apikey"),
        )
        self.btn_get_key.pack(side="right")

        self.gemini_entry = ctk.CTkEntry(frame, placeholder_text="Incolla qui la chiave (es. AIzaSy...)", show="*", width=440)
        self.gemini_entry.insert(0, config.get("gemini_key", ""))
        self.gemini_entry.pack(anchor="w", padx=16, pady=(6, 12))

        ctk.CTkLabel(frame, text="Modello Gemini (opzionale):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=16)
        self.model_entry = ctk.CTkEntry(frame, width=260)
        self.model_entry.pack(anchor="w", padx=16, pady=(4, 12))

        ctk.CTkLabel(frame, text="Tema grafico:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=16, pady=(2, 0))
        self.theme_var = ctk.StringVar(value=config.get("appearance", "Sistema"))
        ctk.CTkSegmentedButton(frame, values=["Sistema", "Chiaro", "Scuro"], variable=self.theme_var).pack(anchor="w", padx=16, pady=(4, 12))

        self.cuda_switch = ctk.CTkSwitch(frame, text="Usa accelerazione GPU NVIDIA (se presente nel PC)", font=ctk.CTkFont(size=12))
        if not config.get("skip_cuda_download", False):
            self.cuda_switch.select()
        self.cuda_switch.pack(anchor="w", padx=16, pady=(2, 12))

        ctk.CTkButton(
            frame, text="💾 Salva e Continua", height=40, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1f8b4c", hover_color="#176d3b", command=self._save,
        ).pack(fill="x", padx=16, pady=(4, 14))
        self._toggle_fields(self.provider_var.get())

    def _toggle_fields(self, choice: str):
        self.model_entry.delete(0, "end")
        if "Gemini" in choice:
            self.gemini_entry.configure(state="normal")
            self.btn_get_key.configure(state="normal")
            self.model_entry.configure(state="normal")
            self.model_entry.insert(0, self.config.get("gemini_model", "gemini-2.5-flash"))
        elif "Locale" in choice:
            self.gemini_entry.configure(state="disabled")
            self.btn_get_key.configure(state="disabled")
            self.model_entry.insert(0, "Qwen 2.5 3B (Integrato)")
            self.model_entry.configure(state="disabled")
        else:
            self.gemini_entry.configure(state="disabled")
            self.btn_get_key.configure(state="disabled")
            self.model_entry.configure(state="disabled")

    def _save(self):
        provider = self.provider_var.get()
        key = self.gemini_entry.get().strip()
        if "Gemini" in provider and not key:
            messagebox.showwarning("Chiave mancante", "Incolla la chiave API di Google Gemini oppure scegli 'AI Locale Integrata'.", parent=self)
            return
        self.config["ai_provider"] = provider
        self.config["gemini_key"] = key
        if "Gemini" in provider and self.model_entry.get().strip():
            self.config["gemini_model"] = self.model_entry.get().strip()
        self.config["appearance"] = self.theme_var.get()
        self.config["skip_cuda_download"] = not bool(self.cuda_switch.get())
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
        self.llama_proc = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self.after(100, self._process_queue)
        self.after(300, self._check_first_run)

    def _on_close(self):
        self._stop_local_llm()
        self.destroy()

    def _apply_theme(self, choice: str):
        ctk.set_appearance_mode({"Sistema": "System", "Chiaro": "Light", "Scuro": "Dark"}.get(choice, "System"))

    def _check_first_run(self):
        if not os.path.isfile(CONFIG_PATH) or ("Gemini" in self.config_data.get("ai_provider", "") and not self.config_data.get("gemini_key", "").strip()):
            SettingsWindow(self, self.config_data, self._on_settings_saved, first_run=True)

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=22, pady=(16, 8))
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left")
        ctk.CTkLabel(title_box, text="🎙️ SbobinaFacile AI", font=ctk.CTkFont(size=24, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(title_box, text="Trasforma file audio e video lunghi in testo chiaro e riassunti automatici", font=ctk.CTkFont(size=13), text_color="gray").pack(anchor="w")
        ctk.CTkButton(
            header, text="⚙️ Impostazioni", width=130, height=34,
            fg_color=("gray82", "gray25"), text_color=("gray10", "gray90"), hover_color=("gray75", "gray35"),
            command=self._open_settings,
        ).pack(side="right", pady=4)

        card_file = ctk.CTkFrame(self, corner_radius=14)
        card_file.pack(fill="x", padx=22, pady=6)
        row_file = ctk.CTkFrame(card_file, fg_color="transparent")
        row_file.pack(fill="x", padx=16, pady=14)
        self.btn_pick = ctk.CTkButton(row_file, text="📁  Scegli File Audio o Video", height=44, width=230, font=ctk.CTkFont(size=14, weight="bold"), command=self._pick_file)
        self.btn_pick.pack(side="left")
        info_col = ctk.CTkFrame(row_file, fg_color="transparent")
        info_col.pack(side="left", fill="x", expand=True, padx=(16, 8))
        self.file_name_label = ctk.CTkLabel(info_col, text="Nessun file selezionato", font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
        self.file_name_label.pack(anchor="w")
        self.file_size_label = ctk.CTkLabel(info_col, text="Supporta MP3, MP4, MKV, WAV, M4A, MOV (anche oltre 400 MB)", font=ctk.CTkFont(size=12), text_color="gray", anchor="w")
        self.file_size_label.pack(anchor="w")
        self.btn_txt_only = ctk.CTkButton(row_file, text="📝 Riassumi un .txt", height=34, width=140, fg_color="transparent", border_width=1, text_color=("gray20", "gray85"), command=self._pick_and_summarize_txt)
        self.btn_txt_only.pack(side="right")

        card_opts = ctk.CTkFrame(self, corner_radius=14)
        card_opts.pack(fill="x", padx=22, pady=6)
        opts_inner = ctk.CTkFrame(card_opts, fg_color="transparent")
        opts_inner.pack(fill="x", padx=16, pady=12)

        q_box = ctk.CTkFrame(opts_inner, fg_color="transparent")
        q_box.pack(side="left", padx=(0, 24))
        ctk.CTkLabel(q_box, text="Precisione trascrizione:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(0, 4))
        self.quality_seg = ctk.CTkSegmentedButton(q_box, values=["Veloce", "Bilanciata", "Massima"])
        self.quality_seg.set(self.config_data.get("quality", "Bilanciata"))
        self.quality_seg.pack(anchor="w")

        l_box = ctk.CTkFrame(opts_inner, fg_color="transparent")
        l_box.pack(side="left", padx=(0, 24))
        ctk.CTkLabel(l_box, text="Lingua parlata:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(0, 4))
        self.lang_menu = ctk.CTkOptionMenu(l_box, values=list(LANG_MAP.keys()), width=140)
        self.lang_menu.set(self.config_data.get("language", "Italiano"))
        self.lang_menu.pack(anchor="w")

        s_box = ctk.CTkFrame(opts_inner, fg_color="transparent")
        s_box.pack(side="right")
        self.summary_switch = ctk.CTkSwitch(s_box, text="Crea anche il riassunto AI", font=ctk.CTkFont(size=13, weight="bold"))
        if "Disattivato" not in self.config_data.get("ai_provider", ""):
            self.summary_switch.select()
        self.summary_switch.pack(pady=(16, 0))

        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(fill="x", padx=22, pady=(8, 4))
        btn_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_row.pack(fill="x")
        self.btn_start = ctk.CTkButton(btn_row, text="▶   AVVIA ELABORAZIONE", height=46, font=ctk.CTkFont(size=15, weight="bold"), fg_color="#1f8b4c", hover_color="#176d3b", command=self._start_full)
        self.btn_start.pack(side="left", fill="x", expand=True)
        self.btn_stop = ctk.CTkButton(btn_row, text="⏹ Ferma", height=46, width=110, font=ctk.CTkFont(size=14, weight="bold"), fg_color="#b83b3b", hover_color="#8f2b2b", state="disabled", command=self._request_stop)
        self.btn_stop.pack(side="right", padx=(10, 0))

        self.progress_bar = ctk.CTkProgressBar(action_frame, height=12, corner_radius=6)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", pady=(10, 4))
        self.status_label = ctk.CTkLabel(action_frame, text="Pronto. Seleziona un file per iniziare.", font=ctk.CTkFont(size=13), text_color="gray")
        self.status_label.pack(anchor="w")

        self.tabs = ctk.CTkTabview(self, corner_radius=14)
        self.tabs.pack(fill="both", expand=True, padx=22, pady=(4, 16))
        tab_sum = self.tabs.add("✨ Riassunto")
        tab_tr = self.tabs.add("📝 Trascrizione Completa")

        sum_bar = ctk.CTkFrame(tab_sum, fg_color="transparent")
        sum_bar.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(sum_bar, text="📋 Copia Riassunto", width=150, height=30, command=lambda: self._copy_text(self.summary_box)).pack(side="left")
        self.btn_open_folder_1 = ctk.CTkButton(sum_bar, text="📂 Apri cartella dei file", width=170, height=30, fg_color=("gray80", "gray28"), text_color=("gray10", "gray90"), state="disabled", command=self._open_output_folder)
        self.btn_open_folder_1.pack(side="right")
        self.summary_box = ctk.CTkTextbox(tab_sum, wrap="word", font=ctk.CTkFont(size=14))
        self.summary_box.pack(fill="both", expand=True)

        tr_bar = ctk.CTkFrame(tab_tr, fg_color="transparent")
        tr_bar.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(tr_bar, text="📋 Copia Trascrizione", width=160, height=30, command=lambda: self._copy_text(self.transcript_box)).pack(side="left")
        self.btn_open_folder_2 = ctk.CTkButton(tr_bar, text="📂 Apri cartella dei file", width=170, height=30, fg_color=("gray80", "gray28"), text_color=("gray10", "gray90"), state="disabled", command=self._open_output_folder)
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
        path = filedialog.askopenfilename(title="Seleziona file audio o video", filetypes=[("Audio e Video", "*.mp3 *.mp4 *.mkv *.wav *.m4a *.mov *.webm *.ogg *.flac *.avi"), ("Tutti i file", "*.*")])
        if path:
            self.selected_file = path
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self.file_name_label.configure(text=os.path.basename(path))
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
        if "Gemini" in self.config_data.get("ai_provider", "") and not self.config_data.get("gemini_key", "").strip():
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

        download_cuda_now = False
        if not self.config_data.get("skip_cuda_download", False) and not cuda_libs_available() and has_nvidia_gpu():
            ans = messagebox.askyesnocancel(
                "Scheda grafica NVIDIA rilevata!",
                "Questo PC ha una scheda grafica NVIDIA.\n\n"
                "Vuoi scaricare i componenti per l'accelerazione GPU (~650 MB, una sola volta)?\n"
                "Renderà le trascrizioni fino a 10 volte più veloci.\n\n"
                "• Sì = Scarica ora e usa la GPU NVIDIA\n"
                "• No = Usa solo la CPU (potrai riattivarla dalle Impostazioni)",
            )
            if ans is None:
                return
            if ans is True:
                download_cuda_now = True
            else:
                self.config_data["skip_cuda_download"] = True

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
        threading.Thread(target=self._worker_full, args=(self.selected_file, do_summary, download_cuda_now), daemon=True).start()

    def _pick_and_summarize_txt(self):
        if "Disattivato" in self.config_data.get("ai_provider", ""):
            messagebox.showinfo("AI disattivata", "Attiva Gemini o AI Locale nelle Impostazioni per i riassunti.")
            return
        if not self._check_ai_ready():
            return
        txt_path = filedialog.askopenfilename(title="Scegli un file di testo (.txt) da riassumere", filetypes=[("File di testo", "*.txt"), ("Tutti i file", "*.*")])
        if not txt_path:
            return
        self.summary_box.delete("1.0", "end")
        self.progress_bar.set(0.0)
        self.stop_requested = False
        self.last_output_dir = os.path.dirname(txt_path)
        self._set_busy(True)
        self.tabs.set("✨ Riassunto")
        threading.Thread(target=self._worker_txt_only, args=(txt_path,), daemon=True).start()

    def _download_nvidia_cuda_libs(self):
        os.makedirs(CUDA_LIBS_DIR, exist_ok=True)
        for pkg_name, pref_ver, lbl in (("nvidia-cublas-cu12", None, "1/2"), ("nvidia-cudnn-cu12", "9.", "2/2")):
            self.ui_queue.put(("status", f"Ricerca componenti GPU NVIDIA ({lbl})..."))
            url, fname = get_pypi_wheel_url(pkg_name, prefer_major=pref_ver)
            whl_path = os.path.join(CUDA_LIBS_DIR, fname)

            def on_prog(done, total, step=lbl):
                pct = done / total
                self.ui_queue.put(("progress", pct))
                self.ui_queue.put(("status", f"Scarico accelerazione NVIDIA ({step}): {pct*100:.0f}% ({done/(1024*1024):.0f} / {total/(1024*1024):.0f} MB)"))

            download_file_resumable(url, whl_path, progress_cb=on_prog)
            self.ui_queue.put(("status", f"Installazione componenti GPU ({lbl})..."))
            with zipfile.ZipFile(whl_path, "r") as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".dll") and os.path.basename(member):
                        with zf.open(member) as src, open(os.path.join(CUDA_LIBS_DIR, os.path.basename(member)), "wb") as dst:
                            shutil.copyfileobj(src, dst)
            try:
                os.remove(whl_path)
            except Exception:
                pass
        register_cuda_dll_dir()
        self.ui_queue.put(("progress", 0.0))

    def _start_whisper_with_fallback(self, file_path: str, model_size: str, lang: str | None):
        from faster_whisper import WhisperModel
        use_cuda = not self.config_data.get("skip_cuda_download", False) and cuda_libs_available()
        if use_cuda:
            for attempt in range(1, 4):
                try:
                    self.ui_queue.put(("status", "Avvio motore vocale su GPU NVIDIA..."))
                    model = WhisperModel(model_size, device="cuda", compute_type="float16", download_root=MODELS_DIR)
                    segments, info = model.transcribe(file_path, language=lang, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
                    seg_iter = iter(segments)
                    return "🚀 GPU NVIDIA", info, next(seg_iter, None), seg_iter
                except Exception as e:
                    if attempt < 3 and any(x in str(e) for x in ("Connection", "10054", "ReadTimeout")):
                        time.sleep(attempt * 2)
                        continue
                    break

        for attempt in range(1, 4):
            try:
                self.ui_queue.put(("status", "Avvio motore vocale su processore (CPU)..."))
                model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=MODELS_DIR)
                segments, info = model.transcribe(file_path, language=lang, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
                seg_iter = iter(segments)
                return "💻 CPU", info, next(seg_iter, None), seg_iter
            except Exception as e:
                if attempt < 3 and any(x in str(e) for x in ("Connection", "10054", "ReadTimeout")):
                    time.sleep(attempt * 2)
                    continue
                raise

    def _worker_full(self, file_path: str, do_summary: bool, download_cuda_now: bool):
        try:
            if download_cuda_now:
                try:
                    self._download_nvidia_cuda_libs()
                except Exception:
                    self.ui_queue.put(("status", "Download NVIDIA non riuscito, proseguo comunque con la CPU..."))

            model_size = QUALITY_MAP.get(self.quality_seg.get(), "medium")
            lang_code = LANG_MAP.get(self.lang_menu.get(), "it")
            lang = None if lang_code == "auto" else lang_code

            self.ui_queue.put(("status", "Preparazione del motore vocale (al primo avvio scarica il modello)..."))
            hw_label, info, first_seg, seg_iter = self._start_whisper_with_fallback(file_path, model_size, lang)

            all_segs = itertools.chain([first_seg], seg_iter) if first_seg is not None else []
            total_dur = info.duration or 1.0
            base_path = os.path.splitext(file_path)[0]
            txt_path = f"{base_path}_trascrizione.txt"
            parts, t_start = [], time.time()

            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in all_segs:
                    if self.stop_requested:
                        break
                    line = f"[{format_timestamp(seg.start)}] {seg.text.strip()}\n"
                    f.write(line)
                    f.flush()
                    parts.append(line)
                    ratio = min(0.99, seg.end / total_dur)
                    eta_txt = f" — Rimasti {format_eta((time.time() - t_start) / ratio - (time.time() - t_start))}" if ratio > 0.02 else ""
                    self.ui_queue.put(("transcript", line))
                    self.ui_queue.put(("progress", ratio))
                    self.ui_queue.put(("status", f"[{hw_label}] Trascrizione: {ratio*100:.0f}% ({format_timestamp(seg.end)} / {format_timestamp(total_dur)}){eta_txt}"))

            self.ui_queue.put(("enable_folder", None))
            if self.stop_requested:
                self.ui_queue.put(("status", f"Fermato. La parte già trascritta è salvata in {os.path.basename(txt_path)}"))
                return
            self.ui_queue.put(("progress", 1.0))

            if do_summary and parts:
                self._run_summary("".join(parts), base_path)
            else:
                self.ui_queue.put(("status", f"✅ Completato ({hw_label})! File salvato nella stessa cartella."))
        except Exception as e:
            self.ui_queue.put(("error", self._humanize_error(e)))
        finally:
            self.ui_queue.put(("done", None))

    def _worker_txt_only(self, txt_path: str):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.ui_queue.put(("enable_folder", None))
            self._run_summary(text, os.path.splitext(txt_path)[0])
        except Exception as e:
            self.ui_queue.put(("error", self._humanize_error(e)))
        finally:
            self.ui_queue.put(("done", None))

    def _ensure_local_llm(self) -> int:
        server_exe = find_llama_server()
        if not server_exe:
            raise RuntimeError("Motore AI locale (llama-server.exe) non trovato nel pacchetto.")
        model_path = os.path.join(MODELS_DIR, LOCAL_LLM_FILE)
        if not os.path.isfile(model_path):
            self.ui_queue.put(("status", "Primo avvio offline: scarico il modello AI locale (~2 GB) con ripresa automatica..."))
            model_path = hf_download_with_retry(LOCAL_LLM_REPO, LOCAL_LLM_FILE, MODELS_DIR)

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for ngl in (99, 0):
            port = get_free_port()
            proc = subprocess.Popen(
                [server_exe, "-m", model_path, "--host", "127.0.0.1", "--port", str(port), "-c", "8192", "-ngl", str(ngl)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
            )
            self.llama_proc = proc
            for _ in range(60):
                if proc.poll() is not None:
                    break
                try:
                    if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                        self.ui_queue.put(("status", f"AI locale pronta ({'GPU Vulkan' if ngl > 0 else 'CPU'}). Generazione riassunto..."))
                        return port
                except Exception:
                    pass
                time.sleep(0.5)
            self._stop_local_llm()
        raise RuntimeError("Impossibile avviare il motore AI locale integrato.")

    def _stop_local_llm(self):
        if self.llama_proc is not None:
            try:
                self.llama_proc.terminate()
                self.llama_proc.wait(timeout=5)
            except Exception:
                try:
                    self.llama_proc.kill()
                except Exception:
                    pass
            self.llama_proc = None

    def _call_ai(self, prompt: str, local_port: int | None = None) -> str:
        provider = self.config_data.get("ai_provider", "")
        if "Locale" in provider:
            url = f"http://127.0.0.1:{local_port}/v1/chat/completions"
            resp = requests.post(url, json={"messages": [{"role": "user", "content": prompt}], "temperature": 0.3}, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(f"Errore AI locale ({resp.status_code}): {resp.text}")
            return resp.json()["choices"][0]["message"]["content"].strip()
        elif "Gemini" in provider:
            model = self.config_data.get("gemini_model", "gemini-2.5-flash")
            key = self.config_data.get("gemini_key", "").strip()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            for attempt in range(1, 4):
                try:
                    resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=180)
                    if resp.status_code != 200:
                        raise RuntimeError(f"Errore Gemini ({resp.status_code}): {resp.text}")
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                except requests.exceptions.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(attempt * 2)
        return ""

    def _run_summary(self, full_text: str, base_path: str):
        provider = self.config_data.get("ai_provider", "")
        self.ui_queue.put(("status", "Preparazione del riassunto AI..."))
        self.ui_queue.put(("switch_tab", "✨ Riassunto"))
        local_port = None
        try:
            if "Locale" in provider:
                local_port = self._ensure_local_llm()
            chunks = chunk_text(full_text, max_chars=350000 if "Gemini" in provider else 10000)
            if len(chunks) == 1:
                prompt = (
                    "Analizza questa trascrizione in italiano e scrivi un riassunto chiaro, ordinato e facile da leggere "
                    "(usa titoli in maiuscolo e semplici elenchi con il simbolo •, evita codici markdown complessi).\n"
                    "Includi:\n1. SINTESI GENERALE\n2. PUNTI CHIAVE E ARGOMENTI TRATTATI\n3. CONCLUSIONI O DECISIONI IMPORTANTI\n\n"
                    f"TRASCRIZIONE:\n{chunks[0]}"
                )
                final_summary = self._call_ai(prompt, local_port)
            else:
                partials = []
                for idx, ch in enumerate(chunks, start=1):
                    if self.stop_requested:
                        return
                    self.ui_queue.put(("status", f"Riassumo la parte {idx} di {len(chunks)}..."))
                    self.ui_queue.put(("progress", idx / (len(chunks) + 1)))
                    p = f"Riassumi in modo fedele e completo questa parte ({idx}/{len(chunks)}) di una lunga trascrizione in italiano, mantenendo nomi e concetti chiave:\n\n{ch}"
                    partials.append(self._call_ai(p, local_port))
                self.ui_queue.put(("status", "Unisco tutto nel riassunto finale..."))
                final_prompt = (
                    "Unisci questi riassunti parziali in un unico riassunto finale ordinato e facile da leggere in italiano "
                    "(usa titoli in maiuscolo ed elenchi con •):\n1. SINTESI GENERALE\n2. PUNTI CHIAVE PER ARGOMENTO\n3. CONCLUSIONI\n\n"
                    + "\n\n---\n\n".join(partials)
                )
                final_summary = self._call_ai(final_prompt, local_port)

            with open(f"{base_path}_riassunto.txt", "w", encoding="utf-8") as f:
                f.write(final_summary)
            self.ui_queue.put(("progress", 1.0))
            self.ui_queue.put(("summary", final_summary))
            self.ui_queue.put(("status", "✅ Tutto pronto! Trascrizione e riassunto salvati nella cartella del file."))
        finally:
            self._stop_local_llm()

    def _humanize_error(self, err: Exception) -> str:
        msg = str(err)
        if "API_KEY_INVALID" in msg or "400" in msg or "403" in msg:
            return "La chiave API di Google Gemini sembra non valida. Controllala cliccando su ⚙️ Impostazioni."
        if "10054" in msg or "Connection" in msg:
            return "La connessione internet si è interrotta durante il download o l'invio all'AI. Riprova tra poco (se avevi già trascorso del testo, usa '📝 Riassumi un .txt')."
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
