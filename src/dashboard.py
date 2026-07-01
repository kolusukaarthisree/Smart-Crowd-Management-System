"""
Smart Crowd Management System — Dashboard (HMI)

A pure presentation layer built with Tkinter + OpenCV.
It performs NO analytics, detection, counting, risk evaluation, or logging.
It only renders state objects produced by the backend and draws bounding
boxes / track IDs on the live camera feed.

Design philosophy: Minimal. Professional. Operational.
Theme: charcoal + amber.

Public API
----------
    dashboard = Dashboard()
    dashboard.run()                         # blocks; shows startup screen first
    # From the backend loop, on every new frame:
    dashboard.update(
        frame=bgr_ndarray,
        people=[Person(...), ...],
        counter=CounterState(...),
        occupancy=OccupancyState(...),
        risk=RiskState(...),
        alert=AlertState(...),
        camera=CameraHealth(...),
        perf=PerformanceMetrics(...),
    )

All state dataclasses below are duck-typed: the dashboard only reads the
attributes it documents. Backend modules may supply their own equivalent
objects as long as the attribute names match.
"""

from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass, field
from queue import Empty, Queue
from tkinter import ttk
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image, ImageTk  # type: ignore
except ImportError:  # pragma: no cover
    Image = None
    ImageTk = None


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────

BG          = "#181818"   # background — almost black
PANEL       = "#232323"   # panels — dark gray
ACCENT      = "#F5A623"   # amber
TEXT        = "#EAEAEA"   # light gray
TEXT_DIM    = "#8A8A8A"
GREEN       = "#3FB950"
AMBER       = "#F5A623"
RED         = "#E5484D"
GRAY        = "#5A5A5A"

FONT_FAMILY = "Helvetica"
F_TITLE     = (FONT_FAMILY, 14, "bold")
F_LABEL     = (FONT_FAMILY, 9)
F_METRIC    = (FONT_FAMILY, 28, "bold")
F_SECTION   = (FONT_FAMILY, 10, "bold")
F_BODY      = (FONT_FAMILY, 10)
F_CLOCK     = (FONT_FAMILY, 12, "bold")


# ─────────────────────────────────────────────────────────────────────────────
# Duck-typed state objects (the backend may supply its own equivalents)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Person:
    track_id: int
    bbox: tuple                       # (x1, y1, x2, y2)
    confidence: float = 0.0


@dataclass
class CounterState:
    people: int = 0


@dataclass
class OccupancyState:
    current: int = 0
    capacity: int = 0
    percent: float = 0.0


@dataclass
class RiskState:
    level: str = "NORMAL"             # NORMAL | HIGH | CRITICAL


@dataclass
class AlertState:
    active: int = 0
    severity: str = "NORMAL"          # NORMAL | WARNING | CRITICAL
    message: str = ""


@dataclass
class CameraHealth:
    online: bool = True
    quality: str = "GOOD"             # GOOD | BLUR | OFFLINE
    resolution: tuple = (0, 0)


@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    detector_ready: bool = True
    tracker_ready: bool = True
    logger_active: bool = True


@dataclass
class _Snapshot:
    frame: object = None
    people: Sequence[Person] = field(default_factory=list)
    counter: CounterState = field(default_factory=CounterState)
    occupancy: OccupancyState = field(default_factory=OccupancyState)
    risk: RiskState = field(default_factory=RiskState)
    alert: AlertState = field(default_factory=AlertState)
    camera: CameraHealth = field(default_factory=CameraHealth)
    perf: PerformanceMetrics = field(default_factory=PerformanceMetrics)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class Dashboard:
    """Smart Crowd Management — operator HMI."""

    def __init__(self) -> None:
        self.venue_name: str = ""
        self.max_capacity: int = 0

        self._queue: "Queue[_Snapshot]" = Queue(maxsize=2)
        self._latest: _Snapshot = _Snapshot()
        self._photo = None              # keep ref so Tk doesn't GC the image
        self._photo_refs = []          # retain recent PhotoImage objects
        self._pulse_on = True

        self.root = tk.Tk()
        self.root.title("Smart Crowd Management System")
        self.root.configure(bg=BG)
        self.root.geometry("1280x800")
        self.root.minsize(1100, 720)

        self._monitoring = False
        self._start_callback = None
        self._build_startup()

    # ── Public API ──────────────────────────────────────────────────────────

    def set_start_callback(self, callback) -> None:
        """Set callback to be called when user clicks Start Monitoring."""
        self._start_callback = callback

    def update(
        self,
        frame=None,
        people: Iterable[Person] = (),
        counter: Optional[CounterState] = None,
        occupancy: Optional[OccupancyState] = None,
        risk: Optional[RiskState] = None,
        alert: Optional[AlertState] = None,
        camera: Optional[CameraHealth] = None,
        perf: Optional[PerformanceMetrics] = None,
    ) -> None:
        """Push a new backend snapshot. Thread-safe; latest-wins."""
        snap = _Snapshot(
            frame=frame,
            people=list(people),
            counter=counter or CounterState(),
            occupancy=occupancy or OccupancyState(),
            risk=risk or RiskState(),
            alert=alert or AlertState(),
            camera=camera or CameraHealth(),
            perf=perf or PerformanceMetrics(),
        )
        # drop stale snapshot if queue is full — UI only needs the latest
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        self._queue.put_nowait(snap)

    def run(self) -> None:
        """Start the Tk main loop (blocks)."""
        self.root.mainloop()

    # ── Startup screen ──────────────────────────────────────────────────────

    def _build_startup(self) -> None:
        self._startup = tk.Frame(self.root, bg=BG)
        self._startup.pack(expand=True, fill="both")

        card = tk.Frame(self._startup, bg=PANEL, padx=40, pady=36)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            card, text="SMART CROWD MANAGEMENT SYSTEM",
            font=(FONT_FAMILY, 16, "bold"), fg=ACCENT, bg=PANEL,
        ).grid(row=0, column=0, columnspan=2, pady=(0, 24), sticky="w")

        tk.Label(card, text="VENUE NAME", font=F_LABEL, fg=TEXT_DIM, bg=PANEL)\
            .grid(row=1, column=0, sticky="w", pady=(0, 4))
        venue_var = tk.StringVar(value="Main Hall")
        venue_entry = tk.Entry(
            card, textvariable=venue_var, font=F_BODY, width=28,
            bg=BG, fg=TEXT, insertbackground=TEXT, relief="flat",
            highlightthickness=1, highlightbackground=GRAY, highlightcolor=ACCENT,
        )
        venue_entry.grid(row=2, column=0, columnspan=2, sticky="we", ipady=6, pady=(0, 18))

        tk.Label(card, text="MAXIMUM CAPACITY", font=F_LABEL, fg=TEXT_DIM, bg=PANEL)\
            .grid(row=3, column=0, sticky="w", pady=(0, 4))
        cap_var = tk.StringVar(value="120")
        cap_entry = tk.Entry(
            card, textvariable=cap_var, font=F_BODY, width=28,
            bg=BG, fg=TEXT, insertbackground=TEXT, relief="flat",
            highlightthickness=1, highlightbackground=GRAY, highlightcolor=ACCENT,
        )
        cap_entry.grid(row=4, column=0, columnspan=2, sticky="we", ipady=6, pady=(0, 22))

        tk.Label(card, text="CAMERA", font=F_LABEL, fg=TEXT_DIM, bg=PANEL)\
            .grid(row=5, column=0, sticky="w")
        tk.Label(card, text="●  Connected", font=F_BODY, fg=GREEN, bg=PANEL)\
            .grid(row=5, column=1, sticky="e")

        tk.Label(card, text="RESOLUTION", font=F_LABEL, fg=TEXT_DIM, bg=PANEL)\
            .grid(row=6, column=0, sticky="w", pady=(6, 22))
        tk.Label(card, text="1280 × 720", font=F_BODY, fg=TEXT, bg=PANEL)\
            .grid(row=6, column=1, sticky="e", pady=(6, 22))

        start_btn = tk.Button(
            card, text="START MONITORING",
            font=(FONT_FAMILY, 11, "bold"),
            bg=ACCENT, fg=BG, activebackground="#d98e1c", activeforeground=BG,
            relief="flat", cursor="hand2", padx=20, pady=10, borderwidth=0,
            command=lambda: self._start_monitoring(venue_var.get().strip(), cap_var.get().strip()),
        )
        start_btn.grid(row=7, column=0, columnspan=2, sticky="we")

        self._startup_error = tk.Label(card, text="", font=F_LABEL, fg=RED, bg=PANEL)
        self._startup_error.grid(row=8, column=0, columnspan=2, pady=(10, 0))

        venue_entry.focus_set()

    def _start_monitoring(self, venue: str, capacity_str: str) -> None:
        if not venue:
            self._startup_error.config(text="Venue name required.")
            return
        try:
            capacity = int(capacity_str)
            if capacity <= 0:
                raise ValueError
        except ValueError:
            self._startup_error.config(text="Capacity must be a positive integer.")
            return

        self.venue_name = venue
        self.max_capacity = capacity
        self._startup.destroy()
        self._build_monitor()
        self._monitoring = True
        
        print("1 Dashboard button pressed")

        # Call the callback if set
        if self._start_callback:
            self._start_callback(venue, capacity)

        self._tick_clock()
        self._tick_pulse()
        self._drain_queue()

    # ── Monitoring screen ───────────────────────────────────────────────────

    def _build_monitor(self) -> None:
        root = self.root
        root.configure(bg=BG)

        # Row weights: header / body / metrics / alert
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=0)
        root.grid_rowconfigure(3, weight=0)
        root.grid_columnconfigure(0, weight=1)

        # ── Header ──
        header = tk.Frame(root, bg=PANEL, height=56)
        header.grid(row=0, column=0, sticky="we")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1, uniform="h")
        header.grid_columnconfigure(1, weight=1, uniform="h")
        header.grid_columnconfigure(2, weight=1, uniform="h")

        tk.Label(
            header, text="SMART CROWD MANAGEMENT SYSTEM",
            font=(FONT_FAMILY, 12, "bold"), fg=ACCENT, bg=PANEL,
        ).grid(row=0, column=0, sticky="w", padx=20, pady=14)

        tk.Label(
            header, text=self.venue_name,
            font=(FONT_FAMILY, 13, "bold"), fg=TEXT, bg=PANEL,
        ).grid(row=0, column=1, pady=14)

        right = tk.Frame(header, bg=PANEL)
        right.grid(row=0, column=2, sticky="e", padx=20, pady=10)
        self._clock_lbl = tk.Label(right, text="--:--:--", font=F_CLOCK, fg=TEXT, bg=PANEL)
        self._clock_lbl.pack(side="left", padx=(0, 16))
        self._live_dot = tk.Label(right, text="●", font=(FONT_FAMILY, 14), fg=RED, bg=PANEL)
        self._live_dot.pack(side="left")
        tk.Label(right, text="LIVE", font=(FONT_FAMILY, 10, "bold"),
                 fg=TEXT, bg=PANEL).pack(side="left", padx=(4, 0))

        # ── Body: status panel + camera feed ──
        body = tk.Frame(root, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=220)
        body.grid_columnconfigure(1, weight=1)

        # Status panel
        status = tk.Frame(body, bg=PANEL)
        status.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(status, text="SYSTEM STATUS", font=F_SECTION,
                 fg=ACCENT, bg=PANEL).pack(anchor="w", padx=16, pady=(14, 10))

        self._status_lbls: dict[str, tk.Label] = {}
        for key, label in [
            ("camera",   "Camera"),
            ("detector", "Detector"),
            ("tracker",  "Tracker"),
            ("logger",   "Logger"),
        ]:
            row = tk.Frame(status, bg=PANEL)
            row.pack(fill="x", padx=16, pady=3)
            dot = tk.Label(row, text="●", font=(FONT_FAMILY, 12), fg=GRAY, bg=PANEL)
            dot.pack(side="left")
            tk.Label(row, text="  " + label, font=F_BODY, fg=TEXT, bg=PANEL)\
                .pack(side="left")
            self._status_lbls[key] = dot

        tk.Frame(status, bg=BG, height=1).pack(fill="x", padx=16, pady=14)

        tk.Label(status, text="RESOLUTION", font=F_LABEL,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w", padx=16)
        self._res_lbl = tk.Label(status, text="—", font=F_BODY, fg=TEXT, bg=PANEL)
        self._res_lbl.pack(anchor="w", padx=16, pady=(0, 10))

        tk.Label(status, text="FPS", font=F_LABEL,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w", padx=16)
        self._status_fps_lbl = tk.Label(status, text="—", font=F_BODY, fg=TEXT, bg=PANEL)
        self._status_fps_lbl.pack(anchor="w", padx=16, pady=(0, 10))

        tk.Label(status, text="CAMERA QUALITY", font=F_LABEL,
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w", padx=16)
        self._quality_lbl = tk.Label(status, text="—", font=(FONT_FAMILY, 11, "bold"),
                                     fg=TEXT, bg=PANEL)
        self._quality_lbl.pack(anchor="w", padx=16, pady=(0, 16))

        # Camera feed
        feed = tk.Frame(body, bg=PANEL)
        feed.grid(row=0, column=1, sticky="nsew")
        self._video_lbl = tk.Label(feed, bg="#0e0e0e", text="WAITING FOR SIGNAL",
                                   fg=TEXT_DIM, font=F_BODY)
        self._video_lbl.pack(expand=True, fill="both", padx=2, pady=2)

        # ── Metric cards ──
        metrics = tk.Frame(root, bg=BG)
        metrics.grid(row=2, column=0, sticky="we", padx=10, pady=(0, 10))
        cards = ["PEOPLE", "OCCUPANCY", "CAPACITY", "RISK", "FPS", "CAMERA", "ALERTS"]
        for i, _ in enumerate(cards):
            metrics.grid_columnconfigure(i, weight=1, uniform="m")

        self._metric_value: dict[str, tk.Label] = {}
        for i, name in enumerate(cards):
            card = tk.Frame(metrics, bg=PANEL)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            val = tk.Label(card, text="—", font=F_METRIC, fg=TEXT, bg=PANEL)
            val.pack(pady=(14, 0))
            tk.Label(card, text=name, font=F_LABEL, fg=TEXT_DIM, bg=PANEL)\
                .pack(pady=(2, 14))
            self._metric_value[name] = val

        # ── Alert bar ──
        self._alert_bar = tk.Frame(root, bg=GREEN, height=40)
        self._alert_bar.grid(row=3, column=0, sticky="we")
        self._alert_bar.grid_propagate(False)
        self._alert_lbl = tk.Label(
            self._alert_bar, text="STATUS: SYSTEM NORMAL",
            font=(FONT_FAMILY, 11, "bold"), fg="#0d0d0d", bg=GREEN,
        )
        self._alert_lbl.pack(side="left", padx=20, pady=8)

    # ── Periodic UI tasks ───────────────────────────────────────────────────

    def _tick_clock(self) -> None:
        if not self._monitoring:
            return
        self._clock_lbl.config(text=time.strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _tick_pulse(self) -> None:
        if not self._monitoring:
            return
        self._pulse_on = not self._pulse_on
        self._live_dot.config(fg=RED if self._pulse_on else PANEL)
        self.root.after(700, self._tick_pulse)

    def _drain_queue(self) -> None:
        if not self._monitoring:
            return
        try:
            while True:
                self._latest = self._queue.get_nowait()
        except Empty:
            pass
        self._render(self._latest)
        self.root.after(33, self._drain_queue)

    # ── Rendering ───────────────────────────────────────────────────────────

    def _render(self, s: _Snapshot) -> None:
        # Camera feed
        self._render_frame(s.frame, s.people)

        # Status panel dots
        self._set_dot("camera",   GREEN if s.camera.online else RED)
        self._set_dot("detector", GREEN if s.perf.detector_ready else GRAY)
        self._set_dot("tracker",  GREEN if s.perf.tracker_ready else GRAY)
        self._set_dot("logger",   GREEN if s.perf.logger_active else GRAY)

        w, h = s.camera.resolution if s.camera.resolution else (0, 0)
        self._res_lbl.config(text=f"{w}×{h}" if w and h else "—")
        self._status_fps_lbl.config(text=f"{s.perf.fps:.0f}" if s.perf.fps else "—")

        q_color = {"GOOD": GREEN, "BLUR": AMBER, "OFFLINE": RED}.get(s.camera.quality, TEXT)
        self._quality_lbl.config(text=s.camera.quality, fg=q_color)

        # Metric cards
        self._metric_value["PEOPLE"].config(text=str(s.counter.people), fg=TEXT)
        pct = int(round(s.occupancy.percent))
        self._metric_value["OCCUPANCY"].config(text=f"{pct}%", fg=TEXT)
        cap = s.occupancy.capacity or self.max_capacity
        self._metric_value["CAPACITY"].config(text=f"{s.occupancy.current}/{cap}", fg=TEXT)

        risk_color = {"NORMAL": GREEN, "HIGH": AMBER, "CRITICAL": RED}.get(s.risk.level, TEXT)
        self._metric_value["RISK"].config(text=s.risk.level, fg=risk_color)

        self._metric_value["FPS"].config(text=f"{s.perf.fps:.0f}" if s.perf.fps else "—", fg=TEXT)
        self._metric_value["CAMERA"].config(text=s.camera.quality, fg=q_color)
        self._metric_value["ALERTS"].config(
            text=f"{s.alert.active}",
            fg=AMBER if s.alert.active else TEXT,
        )

        # Alert bar
        self._render_alert(s.alert)

    def _set_dot(self, key: str, color: str) -> None:
        self._status_lbls[key].config(fg=color)

    def _render_alert(self, a: AlertState) -> None:
        sev = (a.severity or "NORMAL").upper()
        if sev == "CRITICAL":
            bg, fg = RED, "#ffffff"
            text = f"CRITICAL: {a.message or 'Maximum Capacity Exceeded'}"
        elif sev in ("WARNING", "HIGH"):
            bg, fg = AMBER, "#0d0d0d"
            text = f"WARNING: {a.message or 'High Occupancy Detected'}"
        else:
            bg, fg = GREEN, "#0d0d0d"
            text = "STATUS: SYSTEM NORMAL"
        self._alert_bar.config(bg=bg)
        self._alert_lbl.config(bg=bg, fg=fg, text=text)

    # ── Frame drawing ───────────────────────────────────────────────────────

    def _render_frame(self, frame, people: Sequence[Person]) -> None:
        if frame is None or cv2 is None or Image is None:
            return

        try:
            frame_array = np.asarray(frame)
        except Exception:
            return

        if frame_array.ndim != 3 or frame_array.shape[2] != 3:
            return

        img = frame_array.copy()
        for p in people:
            try:
                x1, y1, x2, y2 = (int(v) for v in p.bbox)
            except Exception:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), (35, 166, 245), 2)
            label = f"#{p.track_id}"
            if getattr(p, "confidence", 0):
                label += f"  {p.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), (35, 166, 245), -1)
            cv2.putText(img, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (24, 24, 24), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._photo_refs.append(self._photo)
        if len(self._photo_refs) > 3:
            self._photo_refs.pop(0)
        self._video_lbl.config(image=self._photo, text="")
        self._video_lbl.image = self._photo


# ─────────────────────────────────────────────────────────────────────────────
# Manual smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Dashboard().run()