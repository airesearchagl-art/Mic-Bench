"""
Mic-Bench: Multi-Microphone Accuracy Comparison Tool
"""

import sys
import threading
import numpy as np

import pyaudio
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QProgressBar, QFrame, QGridLayout,
    QMessageBox, QScrollArea, QSizePolicy,
)

CHUNK = 1024
DEFAULT_RATE = 44100
FALLBACK_RATES = [44100, 48000, 22050, 16000, 8000]
FORMAT = pyaudio.paInt16
CHANNELS = 1
MAX_DEVICES = 4
UPDATE_INTERVAL_MS = 50  # UI refresh rate


# ---------------------------------------------------------------------------
# Audio worker
# ---------------------------------------------------------------------------

class AudioWorker(QObject):
    level_changed = pyqtSignal(int, float)  # (slot_index, rms 0.0–1.0)
    error_occurred = pyqtSignal(int, str)

    def __init__(self, slot_index: int, device_index: int, rate: int):
        super().__init__()
        self.slot_index = slot_index
        self.device_index = device_index
        self.rate = rate
        self._running = False
        self._stream = None
        self._pa = None
        self._lock = threading.Lock()
        self._latest_rms = 0.0

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def get_rms(self) -> float:
        with self._lock:
            return self._latest_rms

    def _run(self):
        self._pa = pyaudio.PyAudio()
        try:
            stream = self._pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK,
            )
        except OSError as e:
            self.error_occurred.emit(self.slot_index, str(e))
            self._pa.terminate()
            return

        while self._running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
                with self._lock:
                    self._latest_rms = min(rms * 5.0, 1.0)  # amplify for visibility
            except OSError:
                break

        stream.stop_stream()
        stream.close()
        self._pa.terminate()


# ---------------------------------------------------------------------------
# Single device panel
# ---------------------------------------------------------------------------

class DevicePanel(QFrame):
    def __init__(self, slot_index: int, pa: pyaudio.PyAudio, parent=None):
        super().__init__(parent)
        self.slot_index = slot_index
        self.pa = pa
        self.worker: AudioWorker | None = None
        self._active = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setMinimumWidth(260)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header row: slot label
        header = QLabel(f"Device {self.slot_index + 1}")
        header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Device selector
        self.combo = QComboBox()
        self.combo.setFont(QFont("Segoe UI", 9))
        self._populate_combo()
        layout.addWidget(self.combo)

        # Device name label (system name)
        self.name_label = QLabel("")
        self.name_label.setFont(QFont("Segoe UI", 8))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("color: #888;")
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self._on_combo_changed()
        layout.addWidget(self.name_label)

        # Level meter
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(22)
        self.meter.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
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

        # dB-style numeric label
        self.db_label = QLabel("--- dBFS")
        self.db_label.setFont(QFont("Consolas", 9))
        self.db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.db_label)

        # Start / Stop button
        self.btn = QPushButton("Start")
        self.btn.setCheckable(True)
        self.btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.btn.setFixedHeight(30)
        self.btn.setStyleSheet("""
            QPushButton {
                background: #2979ff; color: white;
                border: none; border-radius: 5px;
            }
            QPushButton:checked {
                background: #d32f2f;
            }
            QPushButton:hover { opacity: 0.85; }
        """)
        self.btn.toggled.connect(self._on_toggle)
        layout.addWidget(self.btn)

        # Status label
        self.status_label = QLabel("Idle")
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    def _populate_combo(self):
        self.combo.clear()
        self.combo.addItem("-- Select device --", userData=-1)
        pa = self.pa
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"]
                self.combo.addItem(name, userData=i)

    def _on_combo_changed(self):
        idx = self.combo.currentData()
        if idx is None or idx < 0:
            self.name_label.setText("")
            return
        try:
            info = self.pa.get_device_info_by_index(idx)
            full_name = info["name"]
            rate = int(info["defaultSampleRate"])
            self.name_label.setText(f"{full_name}\n({rate} Hz)")
        except Exception:
            self.name_label.setText("")

    # ------------------------------------------------------------------
    def _resolve_rate(self, device_index: int) -> int | None:
        pa = self.pa
        for rate in FALLBACK_RATES:
            try:
                supported = pa.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=CHANNELS,
                    input_format=FORMAT,
                )
                if supported:
                    return rate
            except ValueError:
                continue
        return None

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
                    f"Could not find a supported sample rate for this device.\n"
                    f"Tried: {FALLBACK_RATES}"
                )
                self.btn.setChecked(False)
                return

            self.worker = AudioWorker(self.slot_index, device_index, rate)
            self.worker.error_occurred.connect(self._on_worker_error)
            self.worker.start()
            self._active = True
            self.btn.setText("Stop")
            self.status_label.setText(f"Recording @ {rate} Hz")
            self.status_label.setStyleSheet("color: #69f0ae;")
            self.combo.setEnabled(False)
        else:
            self._stop_worker()
            self.btn.setText("Start")
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet("color: #aaa;")
            self.combo.setEnabled(True)
            self.meter.setValue(0)
            self.db_label.setText("--- dBFS")

    def _stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self._active = False

    def _on_worker_error(self, slot: int, msg: str):
        self._stop_worker()
        self.btn.setChecked(False)
        self.btn.setText("Start")
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: #f44336;")
        self.combo.setEnabled(True)
        QMessageBox.critical(self, "Stream error", f"Device {slot + 1}: {msg}")

    def refresh_meter(self):
        if not self._active or self.worker is None:
            return
        rms = self.worker.get_rms()
        pct = int(rms * 100)
        self.meter.setValue(pct)
        if rms > 0:
            db = 20 * np.log10(rms + 1e-9)
            self.db_label.setText(f"{db:+.1f} dBFS")
        else:
            self.db_label.setText("--- dBFS")

    def closedown(self):
        self._stop_worker()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

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
            "stop:0 #0d1b2a, stop:1 #1b2838);"
            "border-radius: 8px;"
        )
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel("Mic-Bench")
        title.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #64b5f6; letter-spacing: 4px;")
        header_layout.addWidget(title)

        subtitle = QLabel("Multi-Microphone Accuracy Comparison Tool")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #90a4ae;")
        header_layout.addWidget(subtitle)

        root.addWidget(header_frame)

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Device List")
        refresh_btn.setFont(QFont("Segoe UI", 9))
        refresh_btn.setFixedHeight(28)
        refresh_btn.setStyleSheet(
            "QPushButton { background:#37474f; color:white; border:none; border-radius:4px; padding:0 12px; }"
            "QPushButton:hover { background:#546e7a; }"
        )
        refresh_btn.clicked.connect(self._refresh_devices)
        toolbar.addWidget(refresh_btn)
        toolbar.addStretch()

        device_count = QLabel()
        n_in = sum(
            1 for i in range(self.pa.get_device_count())
            if self.pa.get_device_info_by_index(i)["maxInputChannels"] > 0
        )
        device_count.setText(f"{n_in} input device(s) found")
        device_count.setFont(QFont("Segoe UI", 9))
        device_count.setStyleSheet("color: #78909c;")
        self._device_count_label = device_count
        toolbar.addWidget(device_count)

        root.addLayout(toolbar)

        # ── Device panels ────────────────────────────────────────────────
        panels_widget = QWidget()
        self.panels_layout = QHBoxLayout(panels_widget)
        self.panels_layout.setSpacing(12)

        for i in range(MAX_DEVICES):
            panel = DevicePanel(i, self.pa)
            self.panels.append(panel)
            self.panels_layout.addWidget(panel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panels_widget)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll, stretch=1)

        # ── Footer ───────────────────────────────────────────────────────
        footer = QLabel("© 2025 Mic-Bench  |  PyAudio + PyQt6")
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
        self.resize(1100, 480)

    def _refresh_all_meters(self):
        for panel in self.panels:
            panel.refresh_meter()

    def _refresh_devices(self):
        # Terminate old PA instance and reinitialise
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Mic-Bench")
    app.setOrganizationName("Mic-Bench")

    window = MicBench()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
