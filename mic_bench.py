"""
Mic-Bench: Multi-Microphone Accuracy Comparison Tool
"""

import sys
import csv
import queue
import threading
import time
from collections import deque
from datetime import datetime
from math import gcd

import numpy as np
import pyaudio
from scipy.signal import resample_poly
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QProgressBar, QFrame,
    QFileDialog, QMessageBox, QScrollArea, QPlainTextEdit,
)

# ── Audio constants ───────────────────────────────────────────────────────────
CHUNK = 1024
FALLBACK_RATES = [44100, 48000, 22050, 16000, 8000]
FORMAT = pyaudio.paInt16
CHANNELS = 1
MAX_DEVICES = 4
UPDATE_INTERVAL_MS = 50

# ── Transcription constants ───────────────────────────────────────────────────
WHISPER_RATE = 16_000
MAX_SEGMENT_SECS = 3.0   # flush buffer after this many seconds regardless
MIN_SEGMENT_SECS = 0.5   # discard segments shorter than this
SILENCE_RMS = 0.008      # normalised RMS below which a chunk is "silent"
SILENCE_SECS = 0.6       # flush after this many seconds of continuous silence
TRANSCRIPT_MAX_LINES = 80

# ── S/N calibration constants ─────────────────────────────────────────────────
NOISE_BUF_LEN = 200      # max silent-chunk RMS values to retain
NOISE_MIN_SAMPLES = 20   # require at least this many silent chunks before SNR is valid

# ── Global Whisper model (lazy, shared across all panels) ─────────────────────
_WHISPER_MODEL = None
_WHISPER_LOADED_SIZE = ""
_WHISPER_LOCK = threading.Lock()


def _get_model(size: str):
    """Load (or return cached) WhisperModel. Thread-safe."""
    global _WHISPER_MODEL, _WHISPER_LOADED_SIZE
    from faster_whisper import WhisperModel  # deferred import
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None or _WHISPER_LOADED_SIZE != size:
            _WHISPER_LOADED_SIZE = size
            try:
                _WHISPER_MODEL = WhisperModel(
                    size, device="cuda", compute_type="float16"
                )
            except Exception:
                _WHISPER_MODEL = WhisperModel(
                    size, device="cpu", compute_type="int8"
                )
        return _WHISPER_MODEL


def _resample_to_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == WHISPER_RATE:
        return samples.astype(np.float32)
    g = gcd(src_rate, WHISPER_RATE)
    return resample_poly(samples, WHISPER_RATE // g, src_rate // g).astype(np.float32)


# ── AudioWorker ───────────────────────────────────────────────────────────────

class AudioWorker(QObject):
    error_occurred = pyqtSignal(int, str)

    def __init__(self, slot_index: int, device_index: int, rate: int,
                 seg_queue: "queue.Queue[np.ndarray]"):
        super().__init__()
        self.slot_index = slot_index
        self.device_index = device_index
        self.rate = rate
        self._seg_queue = seg_queue
        self._running = False
        self._lock = threading.Lock()
        self._latest_rms = 0.0
        self._current_rms_raw = 0.0
        self._noise_buf: deque[float] = deque(maxlen=NOISE_BUF_LEN)

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False

    def get_rms(self) -> float:
        with self._lock:
            return self._latest_rms

    def get_snr_stats(self) -> tuple[float | None, float]:
        """Return (noise_floor_linear | None, current_rms_linear).

        noise_floor is the 10th-percentile of silent-chunk RMS values.
        None is returned until NOISE_MIN_SAMPLES silent chunks are collected.
        """
        with self._lock:
            cur = self._current_rms_raw
            if len(self._noise_buf) < NOISE_MIN_SAMPLES:
                return None, cur
            nf = float(np.percentile(list(self._noise_buf), 10))
            return nf, cur

    def reset_noise_floor(self):
        with self._lock:
            self._noise_buf.clear()

    def _run(self):
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=FORMAT, channels=CHANNELS, rate=self.rate,
                input=True, input_device_index=self.device_index,
                frames_per_buffer=CHUNK,
            )
        except OSError as e:
            self.error_occurred.emit(self.slot_index, str(e))
            pa.terminate()
            return

        max_buf = int(self.rate * MAX_SEGMENT_SECS)
        min_buf = int(self.rate * MIN_SEGMENT_SECS)
        sil_thresh = max(1, int(SILENCE_SECS * self.rate / CHUNK))

        buf: list[np.ndarray] = []
        buf_n = 0
        sil_cnt = 0

        while self._running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except OSError:
                break

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0

            with self._lock:
                self._latest_rms = min(rms * 5.0, 1.0)
                self._current_rms_raw = rms
                if rms < SILENCE_RMS:
                    self._noise_buf.append(rms)

            buf.append(samples)
            buf_n += len(samples)
            sil_cnt = sil_cnt + 1 if rms < SILENCE_RMS else 0

            should_flush = (
                (sil_cnt >= sil_thresh and buf_n >= min_buf)
                or buf_n >= max_buf
            )
            if should_flush:
                seg = np.concatenate(buf)
                buf, buf_n, sil_cnt = [], 0, 0
                # Resample to Whisper's 16kHz and normalise to float32 [-1, 1]
                self._seg_queue.put(_resample_to_16k(seg, self.rate) / 32768.0)

        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── TranscriptionWorker ───────────────────────────────────────────────────────

class TranscriptionWorker(QObject):
    transcription_ready = pyqtSignal(int, str)
    # (slot, confidence_pct, latency_secs)
    # confidence = clamp((avg_logprob + 1.0) * 100, 0, 100)
    metrics_ready = pyqtSignal(int, float, float)
    status_changed = pyqtSignal(int, str)

    _STOP = object()  # sentinel value

    def __init__(self, slot_index: int, seg_queue: "queue.Queue[np.ndarray]",
                 model_size: str):
        super().__init__()
        self.slot_index = slot_index
        self._seg_queue = seg_queue
        self._model_size = model_size
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False
        self._seg_queue.put(self._STOP)

    def _run(self):
        self.status_changed.emit(self.slot_index, "Loading model…")
        try:
            model = _get_model(self._model_size)
        except Exception as e:
            self.status_changed.emit(self.slot_index, f"Model error: {e}")
            return

        self.status_changed.emit(self.slot_index, "Listening…")

        while self._running:
            try:
                seg = self._seg_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if seg is self._STOP:
                break

            self.status_changed.emit(self.slot_index, "Transcribing…")
            t0 = time.perf_counter()
            try:
                # list() forces the lazy generator inside the lock so all inference
                # is serialised and latency is measured correctly.
                with _WHISPER_LOCK:
                    segs_gen, _ = model.transcribe(
                        seg,
                        beam_size=1,
                        vad_filter=True,
                        language=None,  # auto-detect
                    )
                    collected = list(segs_gen)
                latency = time.perf_counter() - t0

                text = " ".join(s.text.strip() for s in collected).strip()

                if collected:
                    avg_lp = sum(s.avg_logprob for s in collected) / len(collected)
                    confidence = max(0.0, min(100.0, (avg_lp + 1.0) * 100.0))
                    self.metrics_ready.emit(self.slot_index, confidence, latency)
            except Exception as e:
                self.status_changed.emit(self.slot_index, f"Error: {e}")
                continue

            if text:
                self.transcription_ready.emit(self.slot_index, text)
            self.status_changed.emit(self.slot_index, "Listening…")


# ── Single device panel ───────────────────────────────────────────────────────

class DevicePanel(QFrame):
    def __init__(self, slot_index: int, pa: pyaudio.PyAudio,
                 model_size_getter, parent=None):
        super().__init__(parent)
        self.slot_index = slot_index
        self.pa = pa
        self._model_size_getter = model_size_getter  # callable → str
        self.worker: AudioWorker | None = None
        self.tx_worker: TranscriptionWorker | None = None
        self._active = False

        # Per-session metric accumulators (reset on each Start, kept after Stop)
        self._confidence_history: list[float] = []
        self._latency_history: list[float] = []
        self._snr_history: list[float] = []

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setMinimumWidth(260)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Slot header ──────────────────────────────────────────────────
        header = QLabel(f"Device {self.slot_index + 1}")
        header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # ── Device selector ──────────────────────────────────────────────
        self.combo = QComboBox()
        self.combo.setFont(QFont("Segoe UI", 9))
        self._populate_combo()
        layout.addWidget(self.combo)

        # ── System device name ───────────────────────────────────────────
        self.name_label = QLabel("")
        self.name_label.setFont(QFont("Segoe UI", 8))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("color: #888;")
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self._on_combo_changed()
        layout.addWidget(self.name_label)

        # ── Level meter ──────────────────────────────────────────────────
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(22)
        self.meter.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555; border-radius: 4px;
                background: #1a1a2e;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00c853, stop:0.7 #ffeb3b, stop:1 #f44336
                );
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.meter)

        # ── dBFS numeric label ───────────────────────────────────────────
        self.db_label = QLabel("--- dBFS")
        self.db_label.setFont(QFont("Consolas", 9))
        self.db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.db_label)

        # ── Metrics grid ─────────────────────────────────────────────────
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(
            "background: #161616; border: 1px solid #2a2a2a; border-radius: 5px;"
        )
        mg = QGridLayout(metrics_frame)
        mg.setContentsMargins(8, 5, 8, 5)
        mg.setHorizontalSpacing(6)
        mg.setVerticalSpacing(3)
        mg.setColumnStretch(1, 1)

        def _metric_label(text: str) -> QLabel:
            w = QLabel(text)
            w.setFont(QFont("Segoe UI", 7))
            w.setStyleSheet("color: #607d8b; background: transparent; border: none;")
            return w

        def _metric_value(init: str = "---") -> QLabel:
            w = QLabel(init)
            w.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            w.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            w.setStyleSheet("color: #e0e0e0; background: transparent; border: none;")
            return w

        self.conf_val  = _metric_value()
        self.lat_val   = _metric_value()
        self.noise_val = _metric_value()
        self.snr_val   = _metric_value()

        mg.addWidget(_metric_label("Confidence:"), 0, 0)
        mg.addWidget(self.conf_val,                0, 1)
        mg.addWidget(_metric_label("Latency:"),    1, 0)
        mg.addWidget(self.lat_val,                 1, 1)
        mg.addWidget(_metric_label("Base Noise:"), 2, 0)
        mg.addWidget(self.noise_val,               2, 1)
        mg.addWidget(_metric_label("SNR:"),        3, 0)
        mg.addWidget(self.snr_val,                 3, 1)

        layout.addWidget(metrics_frame)

        # ── Start / Stop ─────────────────────────────────────────────────
        self.btn = QPushButton("Start")
        self.btn.setCheckable(True)
        self.btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.btn.setFixedHeight(30)
        self.btn.setStyleSheet("""
            QPushButton {
                background: #2979ff; color: white;
                border: none; border-radius: 5px;
            }
            QPushButton:checked { background: #d32f2f; }
        """)
        self.btn.toggled.connect(self._on_toggle)
        layout.addWidget(self.btn)

        # ── Status ───────────────────────────────────────────────────────
        self.status_label = QLabel("Idle")
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.status_label)

        # ── Separator ────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        # ── Transcript header row ─────────────────────────────────────────
        tx_row = QHBoxLayout()
        tx_title = QLabel("Transcript")
        tx_title.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        tx_title.setStyleSheet("color: #90caf9;")
        tx_row.addWidget(tx_title)
        tx_row.addStretch()
        self.tx_status = QLabel("")
        self.tx_status.setFont(QFont("Segoe UI", 7))
        self.tx_status.setStyleSheet("color: #78909c;")
        tx_row.addWidget(self.tx_status)
        clear_btn = QPushButton("Clear Text")
        clear_btn.setFixedHeight(18)
        clear_btn.setFont(QFont("Segoe UI", 7))
        clear_btn.setStyleSheet(
            "QPushButton { background:#37474f; color:white; border:none;"
            " border-radius:3px; padding:0 6px; }"
            "QPushButton:hover { background:#546e7a; }"
        )
        clear_btn.clicked.connect(lambda: self.transcript.clear())
        tx_row.addWidget(clear_btn)
        layout.addLayout(tx_row)

        # ── Transcript text area ─────────────────────────────────────────
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMaximumBlockCount(TRANSCRIPT_MAX_LINES)
        self.transcript.setMinimumHeight(140)
        self.transcript.setFont(QFont("Consolas", 8))
        self.transcript.setStyleSheet("""
            QPlainTextEdit {
                background: #0d0d0d; color: #c8e6c9;
                border: 1px solid #333; border-radius: 4px;
            }
        """)
        self.transcript.setPlaceholderText("Transcript will appear here…")
        layout.addWidget(self.transcript, stretch=1)

    # ── Metric helpers ────────────────────────────────────────────────────────

    def _reset_metric_labels(self):
        for lbl in (self.conf_val, self.lat_val, self.noise_val, self.snr_val):
            lbl.setText("---")
            lbl.setStyleSheet("color: #e0e0e0; background: transparent; border: none;")

    # ── Device helpers ────────────────────────────────────────────────────────

    def _populate_combo(self):
        self.combo.clear()
        self.combo.addItem("-- Select device --", userData=-1)
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                self.combo.addItem(info["name"], userData=i)

    def _on_combo_changed(self):
        idx = self.combo.currentData()
        if idx is None or idx < 0:
            self.name_label.setText("")
            return
        try:
            info = self.pa.get_device_info_by_index(idx)
            self.name_label.setText(
                f"{info['name']}\n({int(info['defaultSampleRate'])} Hz)"
            )
        except Exception:
            self.name_label.setText("")

    def _resolve_rate(self, device_index: int) -> int | None:
        for rate in FALLBACK_RATES:
            try:
                if self.pa.is_format_supported(
                    rate, input_device=device_index,
                    input_channels=CHANNELS, input_format=FORMAT,
                ):
                    return rate
            except ValueError:
                continue
        return None

    # ── Toggle recording + transcription ─────────────────────────────────────

    def _on_toggle(self, checked: bool):
        if checked:
            device_index = self.combo.currentData()
            if device_index is None or device_index < 0:
                QMessageBox.warning(self, "No device", "Please select an input device.")
                self.btn.setChecked(False)
                return

            rate = self._resolve_rate(device_index)
            if rate is None:
                QMessageBox.critical(
                    self, "Unsupported device",
                    f"No supported sample rate found.\nTried: {FALLBACK_RATES}",
                )
                self.btn.setChecked(False)
                return

            seg_queue: queue.Queue = queue.Queue()

            self.worker = AudioWorker(self.slot_index, device_index, rate, seg_queue)
            self.worker.error_occurred.connect(self._on_worker_error)
            self.worker.start()

            self.tx_worker = TranscriptionWorker(
                self.slot_index, seg_queue, self._model_size_getter()
            )
            self.tx_worker.transcription_ready.connect(self._on_transcript)
            self.tx_worker.metrics_ready.connect(self._on_metrics)
            self.tx_worker.status_changed.connect(self._on_tx_status)
            self.tx_worker.start()

            # Reset accumulators for the new session
            self._confidence_history.clear()
            self._latency_history.clear()
            self._snr_history.clear()

            self._active = True
            self.btn.setText("Stop")
            self.status_label.setText(f"Recording @ {rate} Hz")
            self.status_label.setStyleSheet("color: #69f0ae;")
            self.combo.setEnabled(False)
            self.noise_val.setText("Calibrating…")
        else:
            self._stop_workers()
            self.btn.setText("Start")
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet("color: #aaa;")
            self.tx_status.setText("")
            self.combo.setEnabled(True)
            self.meter.setValue(0)
            self.db_label.setText("--- dBFS")
            self._reset_metric_labels()

    def _stop_workers(self):
        # Stop audio first so no more segments are enqueued, then signal tx_worker.
        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.tx_worker:
            self.tx_worker.stop()
            self.tx_worker = None
        self._active = False

    def _on_worker_error(self, slot: int, msg: str):
        self._stop_workers()
        self.btn.setChecked(False)
        self.btn.setText("Start")
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: #f44336;")
        self.combo.setEnabled(True)
        QMessageBox.critical(self, "Stream error", f"Device {slot + 1}: {msg}")

    def _on_transcript(self, _slot: int, text: str):
        self.transcript.appendPlainText(text)

    def _on_metrics(self, _slot: int, confidence: float, latency: float):
        # Accumulate for report export
        self._confidence_history.append(confidence)
        self._latency_history.append(latency)
        if self.worker:
            nf, cur = self.worker.get_snr_stats()
            if nf is not None and nf > 0 and cur > nf:
                self._snr_history.append(20 * np.log10(cur / nf))

        self.conf_val.setText(f"{confidence:.1f} %")
        self.lat_val.setText(f"{latency:.2f} s")
        # Green ≥70 %, yellow ≥40 %, red <40 %
        if confidence >= 70:
            color = "#69f0ae"
        elif confidence >= 40:
            color = "#ffeb3b"
        else:
            color = "#f44336"
        self.conf_val.setStyleSheet(
            f"color: {color}; background: transparent; border: none;"
        )

    def _on_tx_status(self, _slot: int, msg: str):
        self.tx_status.setText(msg)

    # ── Meter + SNR refresh (called from main QTimer) ─────────────────────────

    def refresh_meter(self):
        if not self._active or self.worker is None:
            return
        rms = self.worker.get_rms()
        self.meter.setValue(int(rms * 100))
        if rms > 0:
            self.db_label.setText(f"{20 * np.log10(rms + 1e-9):+.1f} dBFS")
        else:
            self.db_label.setText("--- dBFS")

        nf, cur = self.worker.get_snr_stats()
        if nf is None:
            # Keep the "Calibrating…" text set at start; don't overwrite yet.
            self.snr_val.setText("---")
        else:
            nf_db = 20 * np.log10(max(nf, 1e-9))
            self.noise_val.setText(f"{nf_db:+.1f} dBFS")
            if cur > nf > 0:
                snr_db = 20 * np.log10(cur / nf)
                self.snr_val.setText(f"{snr_db:+.1f} dB")
            else:
                self.snr_val.setText("0.0 dB")

    def get_report_data(self) -> dict:
        """Return accumulated session metrics for CSV export."""
        dev_idx = self.combo.currentData()
        device_name = self.combo.currentText() if (dev_idx is not None and dev_idx >= 0) else ""

        def _avg(lst: list[float]) -> float | None:
            return sum(lst) / len(lst) if lst else None

        return {
            "slot": self.slot_index + 1,
            "device_name": device_name,
            "avg_confidence": _avg(self._confidence_history),
            "total_latency": sum(self._latency_history) if self._latency_history else None,
            "avg_snr": _avg(self._snr_history),
            "segments": len(self._confidence_history),
            "transcript": self.transcript.toPlainText().strip(),
        }

    def closedown(self):
        self._stop_workers()


# ── Main window ───────────────────────────────────────────────────────────────

class MicBench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mic-Bench")
        self.pa = pyaudio.PyAudio()
        self.panels: list[DevicePanel] = []
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_all_meters)
        self._timer.start(UPDATE_INTERVAL_MS)

    def _get_model_size(self) -> str:
        return self.model_combo.currentText()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Header ──────────────────────────────────────────────────────
        header_frame = QFrame()
        header_frame.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #0d1b2a, stop:1 #1b2838); border-radius: 8px;"
        )
        hl = QVBoxLayout(header_frame)
        hl.setContentsMargins(12, 10, 12, 10)
        title = QLabel("Mic-Bench")
        title.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #64b5f6; letter-spacing: 4px;")
        hl.addWidget(title)
        subtitle = QLabel("Multi-Microphone Accuracy Comparison Tool")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #90a4ae;")
        hl.addWidget(subtitle)
        root.addWidget(header_frame)

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = QHBoxLayout()

        refresh_btn = QPushButton("Refresh Device List")
        refresh_btn.setFont(QFont("Segoe UI", 9))
        refresh_btn.setFixedHeight(28)
        refresh_btn.setStyleSheet(
            "QPushButton { background:#37474f; color:white; border:none;"
            " border-radius:4px; padding:0 12px; }"
            "QPushButton:hover { background:#546e7a; }"
        )
        refresh_btn.clicked.connect(self._refresh_devices)
        toolbar.addWidget(refresh_btn)

        toolbar.addSpacing(8)

        save_btn = QPushButton("Save Report (CSV)")
        save_btn.setFont(QFont("Segoe UI", 9))
        save_btn.setFixedHeight(28)
        save_btn.setStyleSheet(
            "QPushButton { background:#1b5e20; color:white; border:none;"
            " border-radius:4px; padding:0 12px; }"
            "QPushButton:hover { background:#2e7d32; }"
        )
        save_btn.clicked.connect(self._save_report_csv)
        toolbar.addWidget(save_btn)

        toolbar.addSpacing(16)

        model_label = QLabel("Whisper model:")
        model_label.setFont(QFont("Segoe UI", 9))
        model_label.setStyleSheet("color: #90a4ae;")
        toolbar.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small"])
        self.model_combo.setCurrentText("base")
        self.model_combo.setFont(QFont("Segoe UI", 9))
        self.model_combo.setFixedWidth(80)
        self.model_combo.setFixedHeight(28)
        toolbar.addWidget(self.model_combo)

        toolbar.addStretch()

        n_in = sum(
            1 for i in range(self.pa.get_device_count())
            if self.pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        )
        self._device_count_label = QLabel(f"{n_in} input device(s) found")
        self._device_count_label.setFont(QFont("Segoe UI", 9))
        self._device_count_label.setStyleSheet("color: #78909c;")
        toolbar.addWidget(self._device_count_label)

        root.addLayout(toolbar)

        # ── Device panels ────────────────────────────────────────────────
        panels_widget = QWidget()
        self.panels_layout = QHBoxLayout(panels_widget)
        self.panels_layout.setSpacing(12)

        for i in range(MAX_DEVICES):
            panel = DevicePanel(i, self.pa, self._get_model_size)
            self.panels.append(panel)
            self.panels_layout.addWidget(panel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panels_widget)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll, stretch=1)

        # ── Footer ───────────────────────────────────────────────────────
        footer = QLabel("© 2025 Mic-Bench  |  PyAudio + faster-whisper + PyQt6")
        footer.setFont(QFont("Segoe UI", 8))
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #546e7a;")
        root.addWidget(footer)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #121212; color: #e0e0e0; }
            QComboBox {
                background: #1e1e1e; color: #e0e0e0;
                border: 1px solid #444; border-radius: 4px;
                padding: 3px 8px;
            }
            QComboBox QAbstractItemView {
                background: #1e1e1e; color: #e0e0e0;
                selection-background-color: #2979ff;
            }
            QScrollArea { background: transparent; }
            QFrame[frameShape="1"] {
                background: #1e1e1e;
                border: 1px solid #333;
                border-radius: 8px;
            }
        """)
        self.resize(1100, 760)

    def _refresh_all_meters(self):
        for panel in self.panels:
            panel.refresh_meter()

    def _save_report_csv(self):
        rows = [p.get_report_data() for p in self.panels]
        active_rows = [r for r in rows if r["segments"] > 0 or r["transcript"]]
        if not active_rows:
            QMessageBox.information(
                self, "No Data",
                "No transcription data has been collected yet.\n"
                "Start recording and speak into at least one microphone first.",
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        default_name = f"mic_bench_report_{ts}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default_name, "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        def _fmt(v: float | None, spec: str) -> str:
            return format(v, spec) if v is not None else ""

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)

                # ── File header ───────────────────────────────────────────
                w.writerow(["# Mic-Bench Report"])
                w.writerow([f"# Generated: {now_str}"])
                w.writerow([f"# Whisper model: {self._get_model_size()}"])
                w.writerow([])

                # ── Column headers ────────────────────────────────────────
                w.writerow([
                    "Slot", "Device Name",
                    "Avg Confidence (%)", "Total Latency (s)",
                    "Avg SNR (dB)", "Segments", "Transcript",
                ])

                # ── One row per panel ─────────────────────────────────────
                for r in rows:
                    transcript_flat = r["transcript"].replace("\n", " | ")
                    w.writerow([
                        f"Device {r['slot']}",
                        r["device_name"],
                        _fmt(r["avg_confidence"], ".1f"),
                        _fmt(r["total_latency"], ".2f"),
                        _fmt(r["avg_snr"], ".1f"),
                        r["segments"],
                        transcript_flat,
                    ])

                w.writerow([])

                # ── Summary ───────────────────────────────────────────────
                w.writerow(["# Summary"])

                conf_rows = [(r, r["avg_confidence"]) for r in rows
                             if r["avg_confidence"] is not None]
                if conf_rows:
                    best_c = max(conf_rows, key=lambda x: x[1])
                    w.writerow([
                        f"# Best Confidence: Device {best_c[0]['slot']}"
                        f" - {best_c[0]['device_name']}"
                        f" ({best_c[1]:.1f} %)"
                    ])

                snr_rows = [(r, r["avg_snr"]) for r in rows
                            if r["avg_snr"] is not None]
                if snr_rows:
                    best_s = max(snr_rows, key=lambda x: x[1])
                    w.writerow([
                        f"# Best SNR:        Device {best_s[0]['slot']}"
                        f" - {best_s[0]['device_name']}"
                        f" ({best_s[1]:.1f} dB)"
                    ])

                # Lowest latency (most responsive)
                lat_rows = [(r, r["total_latency"]) for r in rows
                            if r["total_latency"] is not None and r["segments"] > 0]
                if lat_rows:
                    # Normalise by segment count for fair comparison
                    best_l = min(
                        lat_rows,
                        key=lambda x: x[1] / max(x[0]["segments"], 1),
                    )
                    avg_lat = best_l[1] / max(best_l[0]["segments"], 1)
                    w.writerow([
                        f"# Fastest Avg Latency: Device {best_l[0]['slot']}"
                        f" - {best_l[0]['device_name']}"
                        f" ({avg_lat:.2f} s/segment)"
                    ])

        except OSError as e:
            QMessageBox.critical(self, "Save Failed", f"Could not write file:\n{e}")
            return

        QMessageBox.information(
            self, "Report Saved", f"Report saved successfully:\n{path}"
        )

    def _refresh_devices(self):
        for panel in self.panels:
            panel.closedown()
        self.pa.terminate()
        self.pa = pyaudio.PyAudio()
        for panel in self.panels:
            panel.pa = self.pa
            panel._populate_combo()
            panel._on_combo_changed()
        n_in = sum(
            1 for i in range(self.pa.get_device_count())
            if self.pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        )
        self._device_count_label.setText(f"{n_in} input device(s) found")

    def closeEvent(self, event):
        self._timer.stop()
        for panel in self.panels:
            panel.closedown()
        self.pa.terminate()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Mic-Bench")
    app.setOrganizationName("Mic-Bench")
    window = MicBench()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
