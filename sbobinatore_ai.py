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

DATA_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
    "SbobinaFacile",
)
os.makedirs(DATA_DIR, exist_ok=True)

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
    smi_paths = [
        shutil.which("nvidia-smi"),
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for p in smi_paths:
        if p and os.path.isfile(p):
            return True
    try:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name -join ';'",
            ],
            creationflags=creation_flags,
            timeout=5,
            text=True,
        )
        return "nvidia" in out.lower()
    except Exception:
        return False


def cuda_libs_available() -> bool:
    has_cublas = os.path.isfile(os.path.join(CUDA_LIBS_DIR, "cublas64_12.dll"))
    has_cudnn = any(
        os.path.isfile(os.path.join(CUDA_LIBS_DIR, name))
        for name in ("cudnn_ops64_9.dll", "cudnn64_9.dll", "cudnn64_8.dll")
    )
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
        matching_vers = [
            v
            for v in data.get("releases", {}).keys()
            if v.startswith(prefer_major) and "a" not in v and "b" not in v and "rc" not in v
        ]
        if matching_vers:
            def ver_key(v):
                return tuple(int(x) for x in v.split(".") if x.isdigit())

            best_ver = max(matching_vers, key=ver_key)
            urls = data["releases"][best_ver]

    for item in urls:
        fname = item.get("filename", "")
        if item.get("packagetype") == "bdist_wheel" and "win_amd64" in fname:
            return item["url"], fname
    raise RuntimeError(f"Pacchetto Windows non trovato su PyPI per {package_name}")


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
                    existing = 0
                    mode = "wb"
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
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(12, attempt * 2))
    raise RuntimeError("Download interrotto.")


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


def find_llama_server() -> str | None:
    search_root = os.path.join(APP_BIN_DIR, "llama_bin")
    if not os.path.isdir(search_root):
        return None
    for root_dir, _dirs, files in os.walk(search_root):
        if "llama-server.exe" in files:
            return os.path.join(root_dir, "llama-server.exe")
    return None


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, config: dict, on_save, first_run: bool = False):
        super().__init__(parent)
        self.title(
            "Benvenuto! Configurazione Iniziale" if first_run else "Impostazioni AI e Aspetto"
        )
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
                font=ctk.CTkFont(size=12),
                justify="left",
            ).pack(anchor="w", padx=12, pady=10)

        ctk.CTkLabel(
            frame,
            text="Motore per i Riassunti AI",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(10, 6))

        self.provider_var = ctk.StringVar(value=config["ai_provider"])
        self.provider_menu = ctk.CTkOptionMenu(
            frame,
            variable=self.provider_var,
            values=[
                "Gemini (Gratis Online)",
                "AI Locale Integrata (Offline, Gratis)",
                "Disattivato (Solo trascrizione)",
            ],
            width=340,
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
            frame, placeholder_text="Incolla qui la chiave (es. AIzaSy...)", show="*", width=440
        )
        self.gemini_entry.insert(0, config.get("gemini_key", ""))
        self.gemini_entry.pack(anchor="w", padx=16, pady=(6, 12))

        ctk.CTkLabel(
            frame, text="Modello Gemini (opzionale):", font=ctk.CTkFont(size=12, weight="bold")
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
        ).pack(anchor="w", padx=16, pady=(4, 12))

        self.cuda_switch = ctk.CTkSwitch(
            frame,
            text="Usa accelerazione GPU NVIDIA (se presente nel PC)",
            font=ctk.CTkFont(size=12),
        )
        if not config.get("skip_cuda_download", False):
            self.cuda_switch.select()
        self.cuda_switch.pack(anchor="w", padx=16, pady=(2, 12))

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
            messagebox.showwarning(
                "Chiave mancante",
                "Incolla la chiave API di Google Gemini oppure scegli 'AI Locale Integrata' dal menu.",
                parent=self,
            )
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
