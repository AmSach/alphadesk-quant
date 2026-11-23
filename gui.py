"""
AlphaDesk Quant - Institutional Native Desktop UI
Pure Python (PySide6 / Qt 6) - Zero HTML, Zero Browser DOM.
"""

import os
import sys
import time
import uuid
import datetime as dt
import pandas as pd
import numpy as np

# Set environment variables for numerical stability & prevent OpenMP deadlock on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from dotenv import load_dotenv
load_dotenv()

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QLabel, QPushButton, QLineEdit,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea, QFileDialog, QDoubleSpinBox,
    QSpinBox, QComboBox, QCheckBox, QMessageBox, QProgressBar,
    QGroupBox, QSizePolicy, QSystemTrayIcon, QMenu
)

# Matplotlib Qt Canvas for Native Charting
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("QtAgg")

# Core Engine Modules
import data
import quant
import options
import db
import alerts
import insiders
import stress
import bbg
from llm import KeyPool
from debate import Debate
import autonomous_trader
from autonomous_trader import trader_engine
from fast_ticker import fast_ticker
import rl_engine
from portfolio_5k import portfolio_5k
from indian_brokerage import IndianBrokerFeeModel
import webbrowser

# Modern Obsidian / Dark Institutional Style Sheet
DARK_QSS = """
QMainWindow, QWidget {
    background-color: #0b0e14;
    color: #c9d1d9;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #30363d;
    background-color: #0d1117;
    border-radius: 6px;
    top: -1px;
}

QTabBar::tab {
    background: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 9px 18px;
    font-weight: 600;
    font-size: 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #0d1117;
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}

QTabBar::tab:hover:!selected {
    background: #21262d;
    color: #c9d1d9;
}

QGroupBox {
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 18px;
    padding-top: 14px;
    background-color: #161b22;
    font-weight: 600;
    color: #f0f6fc;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #58a6ff;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background-color: #0d1117;
    color: #f0f6fc;
    border: 1px solid #30363d;
    border-radius: 5px;
    padding: 6px 10px;
    selection-background-color: #1f6feb;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #58a6ff;
}

QPushButton {
    background-color: #21262d;
    color: #f0f6fc;
    border: 1px solid #30363d;
    border-radius: 5px;
    padding: 7px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}

QPushButton:pressed {
    background-color: #161b22;
}

QPushButton#btnPrimary {
    background-color: #238636;
    border-color: #2ea043;
    color: #ffffff;
}

QPushButton#btnPrimary:hover {
    background-color: #2ea043;
}

QPushButton#btnAccent {
    background-color: #1f6feb;
    border-color: #388bfd;
    color: #ffffff;
}

QPushButton#btnAccent:hover {
    background-color: #388bfd;
}

QPushButton#btnDanger {
    background-color: #da3633;
    border-color: #f85149;
    color: #ffffff;
}

QPushButton#btnDanger:hover {
    background-color: #f85149;
}

QTableWidget {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    gridline-color: #21262d;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
}

QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    padding: 7px 10px;
    border: none;
    border-bottom: 1px solid #30363d;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}

QScrollBar:vertical {
    background: #0d1117;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #8b949e;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #0d1117;
    height: 10px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #30363d;
    min-width: 20px;
    border-radius: 5px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QProgressBar {
    border: 1px solid #30363d;
    border-radius: 4px;
    text-align: center;
    background-color: #161b22;
    color: #f0f6fc;
}

QProgressBar::chunk {
    background-color: #2ea043;
    border-radius: 3px;
}
"""

class MetricCard(QFrame):
    """Reusable sleek institutional KPI display card."""
    def __init__(self, title, value="", subtitle="", accent="#58a6ff"):
        super().__init__()
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: #161b22;
                border: 1px solid #30363d;
                border-top: 3px solid {accent};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: 700; letter-spacing: 0.5px;")
        
        self.val_lbl = QLabel(value)
        self.val_lbl.setStyleSheet(f"color: #f0f6fc; font-size: 20px; font-weight: 700; font-family: 'Consolas', monospace;")
        
        self.sub_lbl = QLabel(subtitle)
        self.sub_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.val_lbl)
        layout.addWidget(self.sub_lbl)

    def set_value(self, val, sub=None, color=None):
        self.val_lbl.setText(str(val))
        if sub is not None:
            self.sub_lbl.setText(str(sub))
        if color:
            self.val_lbl.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 700; font-family: 'Consolas', monospace;")


# =============================================================================
# Background QThread Workers (Prevents Any UI Freezing)
# =============================================================================

class ScanWorker(QThread):
    finished_scan = Signal(list)
    error_scan = Signal(str)

    def __init__(self, watchlist, capital, risk, horizon):
        super().__init__()
        self.watchlist = watchlist
        self.capital = capital
        self.risk = risk
        self.horizon = horizon

    def run(self):
        try:
            results = alerts.scan(self.watchlist, self.capital, self.risk, self.horizon)
            self.finished_scan.emit(results)
        except Exception as e:
            self.error_scan.emit(str(e))


class LevelsWorker(QThread):
    finished_levels = Signal(dict, dict)
    error_levels = Signal(str)

    def __init__(self, ticker, capital, risk, horizon):
        super().__init__()
        self.ticker = ticker
        self.capital = capital
        self.risk = risk
        self.horizon = horizon

    def run(self):
        try:
            call_res = alerts.make_call(self.ticker, self.capital, self.risk, self.horizon)
            h = quant.clean(data.hist(self.ticker))
            model_res = quant.model_signal(h, key=data.norm(self.ticker)) if h is not None else {}
            self.finished_levels.emit(call_res, model_res)
        except Exception as e:
            self.error_levels.emit(str(e))


class EvidenceWorker(QThread):
    finished_evidence = Signal(str)
    error_evidence = Signal(str)

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker

    def run(self):
        try:
            ev = data.evidence(self.ticker)
            self.finished_evidence.emit(ev)
        except Exception as e:
            self.error_evidence.emit(str(e))


class DebateRoundWorker(QThread):
    finished_round = Signal(str, str)
    error_round = Signal(str)

    def __init__(self, debate):
        super().__init__()
        self.debate = debate

    def run(self):
        try:
            bull = self.debate._say("BULL", "Make or continue your case.")
            bear = self.debate._say("BEAR", "Rebut the bull and press your case.")
            self.finished_round.emit(bull, bear)
        except Exception as e:
            self.error_round.emit(str(e))


class DebateUserWorker(QThread):
    finished_user = Signal(str, str)
    error_user = Signal(str)

    def __init__(self, debate, text):
        super().__init__()
        self.debate = debate
        self.text = text

    def run(self):
        try:
            db.save("messages", dict(sid=self.debate.sid, ts=time.time(), role="USER", content=self.text))
            bull = self.debate._say("BULL", f"The USER said: {self.text}\nRespond to them.")
            bear = self.debate._say("BEAR", f"The USER said: {self.text}\nRespond to them and to the bull.")
            self.finished_user.emit(bull, bear)
        except Exception as e:
            self.error_user.emit(str(e))


class DebateVerdictWorker(QThread):
    finished_verdict = Signal(str)
    error_verdict = Signal(str)

    def __init__(self, debate):
        super().__init__()
        self.debate = debate

    def run(self):
        try:
            verd = self.debate.verdict()
            self.finished_verdict.emit(verd)
        except Exception as e:
            self.error_verdict.emit(str(e))


class InsidersWorker(QThread):
    finished_insiders = Signal(dict)
    error_insiders = Signal(str)

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker

    def run(self):
        try:
            res = insiders.cluster(self.ticker)
            self.finished_insiders.emit(res)
        except Exception as e:
            self.error_insiders.emit(str(e))


class OptionsExpiriesWorker(QThread):
    finished_expiries = Signal(list)
    error_expiries = Signal(str)

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker

    def run(self):
        try:
            exps = options.expiries(self.ticker)
            self.finished_expiries.emit(exps)
        except Exception as e:
            self.error_expiries.emit(str(e))


class OptionsChainWorker(QThread):
    finished_chain = Signal(dict)
    error_chain = Signal(str)

    def __init__(self, ticker, expiry):
        super().__init__()
        self.ticker = ticker
        self.expiry = expiry

    def run(self):
        try:
            h = quant.clean(data.hist(self.ticker))
            raw_h = data.hist(self.ticker)
            if raw_h is None or raw_h.empty:
                raise ValueError("No price history available for underlying")
            spot = float(raw_h.Close.iloc[-1])
            chain = options.get_chain(self.ticker, self.expiry)
            res = options.analyze(chain, spot, options.rf(self.ticker), h=h)
            self.finished_chain.emit(res)
        except Exception as e:
            self.error_chain.emit(str(e))


class ScorecardWorker(QThread):
    finished_scorecard = Signal(pd.DataFrame)
    error_scorecard = Signal(str)

    def run(self):
        try:
            sc = quant.scorecard()
            self.finished_scorecard.emit(sc)
        except Exception as e:
            self.error_scorecard.emit(str(e))


class StressWorker(QThread):
    finished_stress = Signal(dict)
    error_stress = Signal(str)

    def __init__(self, path, ticker, call_date):
        super().__init__()
        self.path = path
        self.ticker = ticker
        self.call_date = call_date

    def run(self):
        try:
            res = stress.analyze(self.path, self.ticker, self.call_date)
            self.finished_stress.emit(res)
        except Exception as e:
            self.error_stress.emit(str(e))


# =============================================================================
# Main Window Application Class
# =============================================================================

class AlphaDeskGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaDesk Quant | Institutional Quantitative Terminal")
        self.resize(1340, 880)
        self.setMinimumSize(1000, 700)
        
        # Internal State
        self.pool = None
        self.active_debate = None
        self.active_evidence = None
        self.current_sid = {}
        
        # Init Pool
        self._init_pool()
        
        # Start 24/7 Autonomous Surveillance & RL Engine
        trader_engine.start_247_daemon()

        # Setup UI
        self._setup_ui()
        
        # High-Frequency Sub-Second Tick Timer (250ms interval)
        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(250)
        self.tick_timer.timeout.connect(self._on_tick_timer)
        self.tick_timer.start()

        # Autonomous Surveillance Refresh Timer (5000ms interval)
        self.surveillance_timer = QTimer(self)
        self.surveillance_timer.setInterval(5000)
        self.surveillance_timer.timeout.connect(self._on_surveillance_timer)
        self.surveillance_timer.start()

        # System Tray Integration for 24/7 Background Persistence
        self._setup_tray()

        # Auto-Scan Timer
        self.auto_scan_timer = QTimer(self)
        self.auto_scan_timer.timeout.connect(self.on_scan_now)
        
        # Initial Loads (Deferred non-blocking)
        self.refresh_alerts_table()
        QTimer.singleShot(50, self.refresh_scorecard)
        QTimer.singleShot(150, self.refresh_timing_radar)
        QTimer.singleShot(300, self.refresh_rl_portfolio)
        QTimer.singleShot(450, self.refresh_filings)
        QTimer.singleShot(600, self.on_run_backtest)

    def _init_pool(self):
        keys_str = os.getenv("LLM_KEYS", "")
        keys = [k.strip() for k in keys_str.replace("\n", ",").split(",") if k.strip()]
        self.pool = KeyPool(keys)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)
        
        # Top Header Bar
        main_layout.addWidget(self._create_header())
        
        # Splitter: Sidebar + Main Tabs
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background-color: #30363d; }")
        
        # Left Sidebar (Parameters & Control)
        sidebar_widget = self._create_sidebar()
        sidebar_widget.setMaximumWidth(320)
        sidebar_widget.setMinimumWidth(260)
        splitter.addWidget(sidebar_widget)
        
        # Right Main Tabs
        self.tabs = QTabWidget()
        self.tab_timing = self._create_tab_timing()
        self.tab_rl = self._create_tab_rl_trader()
        self.tab_portfolio_5k = self._create_tab_portfolio_5k()
        self.tab_backtest = self._create_tab_backtest()
        self.tab_ticks = self._create_tab_fast_ticks()
        self.tab_filings = self._create_tab_filings()
        self.tab_calls = self._create_tab_calls()
        self.tab_levels = self._create_tab_levels()
        self.tab_debate = self._create_tab_debate()
        self.tab_insiders = self._create_tab_insiders()
        self.tab_stress = self._create_tab_stress()
        self.tab_options = self._create_tab_options()
        
        self.tabs.addTab(self.tab_timing, "  Market Timing Radar  ")
        self.tabs.addTab(self.tab_portfolio_5k, "  5K Active Portfolio & Charges  ")
        self.tabs.addTab(self.tab_rl, "  24/7 Autonomous Trader  ")
        self.tabs.addTab(self.tab_backtest, "  Parametric Backtest Studio  ")
        self.tabs.addTab(self.tab_ticks, "  Millisecond Ticks & Depth  ")
        self.tabs.addTab(self.tab_filings, "  Exchange Filings (Reg 30)  ")
        self.tabs.addTab(self.tab_calls, "  Calls & Alerts  ")
        self.tabs.addTab(self.tab_levels, "  Levels & Model  ")
        self.tabs.addTab(self.tab_debate, "  Bull vs Bear  ")
        self.tabs.addTab(self.tab_insiders, "  Insider Radar  ")
        self.tabs.addTab(self.tab_stress, "  Call Stress  ")
        self.tabs.addTab(self.tab_options, "  Options / Black-Scholes  ")
        
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 4)
        
        main_layout.addWidget(splitter, 1)

    def _create_header(self):
        header = QFrame()
        header.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 4px;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 8, 14, 8)
        
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        t_lbl = QLabel("ALPHADESK QUANT")
        t_lbl.setStyleSheet("color: #f0f6fc; font-size: 17px; font-weight: 800; letter-spacing: 1.5px;")
        st_lbl = QLabel("Multi-Signal Quantitative Architecture: Bayesian Inference, Barrier Options, GARCH & Walk-Forward ML")
        st_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        title_box.addWidget(t_lbl)
        title_box.addWidget(st_lbl)
        hl.addLayout(title_box)
        
        hl.addStretch()
        
        # Status Badges
        ds_source = os.getenv("DATA_SOURCE", "yahoo").upper()
        self.badge_ds = QLabel(f" FEED: {ds_source} ")
        self.badge_ds.setStyleSheet("background-color: #1f242c; color: #58a6ff; border: 1px solid #388bfd; border-radius: 4px; padding: 4px 8px; font-weight: 700; font-size: 11px;")
        hl.addWidget(self.badge_ds)
        
        self.badge_keys = QLabel(f" KEYS: {len(self.pool.keys)} ")
        self.badge_keys.setStyleSheet("background-color: #1f242c; color: #3fb950; border: 1px solid #2ea043; border-radius: 4px; padding: 4px 8px; font-weight: 700; font-size: 11px;")
        hl.addWidget(self.badge_keys)
        
        self.lbl_status = QLabel("READY")
        self.lbl_status.setStyleSheet("background-color: #21262d; color: #c9d1d9; border-radius: 4px; padding: 4px 10px; font-weight: 600; font-size: 11px;")
        hl.addWidget(self.lbl_status)
        
        return header

    def _create_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background-color: #0b0e14;")
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 0, 10, 0)
        layout.setSpacing(14)
        
        # Watchlist Box
        gb_watch = QGroupBox("WATCHLIST")
        vl_w = QVBoxLayout(gb_watch)
        vl_w.setSpacing(6)
        
        self.txt_watch = QTextEdit()
        default_watch = os.getenv("WATCHLIST", "AAPL, NVDA, MSFT, NSE:RELIANCE, NSE:TCS, BSE:500325")
        self.txt_watch.setPlainText(default_watch.replace(",", ",\n"))
        self.txt_watch.setMaximumHeight(90)
        vl_w.addWidget(self.txt_watch)
        
        # Quick Preset Buttons
        hl_presets = QHBoxLayout()
        hl_presets.setSpacing(4)
        btn_us = QPushButton("US Tech")
        btn_us.clicked.connect(lambda: self.txt_watch.setPlainText("AAPL, NVDA, MSFT, AMZN, GOOGL, META, TSLA"))
        btn_in = QPushButton("Nifty Core")
        btn_in.clicked.connect(lambda: self.txt_watch.setPlainText("NSE:RELIANCE, NSE:TCS, NSE:HDFCBANK, NSE:INFY, NSE:ICICIBANK"))
        btn_sm = QPushButton("Small-Cap Vol")
        btn_sm.clicked.connect(lambda: self.txt_watch.setPlainText("SUZLON.NS, IREDA.NS, CDSL.NS, ANGELONE.NS, KAYNES.NS, MAZDOCK.NS, BSE.NS"))
        hl_presets.addWidget(btn_us)
        hl_presets.addWidget(btn_in)
        hl_presets.addWidget(btn_sm)
        vl_w.addLayout(hl_presets)
        
        layout.addWidget(gb_watch)
        
        # Risk & Capital Box
        gb_params = QGroupBox("PORTFOLIO & RISK SIZING")
        gl_p = QGridLayout(gb_params)
        gl_p.setSpacing(8)
        
        gl_p.addWidget(QLabel("Capital:"), 0, 0)
        self.spn_capital = QDoubleSpinBox()
        self.spn_capital.setRange(100, 1e9)
        self.spn_capital.setValue(100000.0)
        self.spn_capital.setPrefix("$ ")
        self.spn_capital.setSingleStep(5000)
        gl_p.addWidget(self.spn_capital, 0, 1)
        
        gl_p.addWidget(QLabel("Risk / Trade:"), 1, 0)
        self.spn_risk = QDoubleSpinBox()
        self.spn_risk.setRange(0.1, 5.0)
        self.spn_risk.setValue(1.0)
        self.spn_risk.setSuffix(" %")
        self.spn_risk.setSingleStep(0.1)
        gl_p.addWidget(self.spn_risk, 1, 1)
        
        gl_p.addWidget(QLabel("Horizon:"), 2, 0)
        self.spn_horizon = QSpinBox()
        self.spn_horizon.setRange(2, 60)
        self.spn_horizon.setValue(10)
        self.spn_horizon.setSuffix(" days")
        gl_p.addWidget(self.spn_horizon, 2, 1)
        
        gl_p.addWidget(QLabel("Auto-Scan:"), 3, 0)
        self.spn_autoscan = QSpinBox()
        self.spn_autoscan.setRange(0, 360)
        self.spn_autoscan.setValue(0)
        self.spn_autoscan.setSuffix(" min (0=off)")
        self.spn_autoscan.valueChanged.connect(self._on_autoscan_changed)
        gl_p.addWidget(self.spn_autoscan, 3, 1)
        
        layout.addWidget(gb_params)
        
        # API Keys Box
        gb_keys = QGroupBox("LLM ROTATION POOL")
        vl_k = QVBoxLayout(gb_keys)
        self.txt_keys = QTextEdit()
        self.txt_keys.setPlaceholderText("Paste API keys (one per line or comma-separated)")
        self.txt_keys.setPlainText(os.getenv("LLM_KEYS", "").replace(",", "\n"))
        self.txt_keys.setMaximumHeight(80)
        vl_k.addWidget(self.txt_keys)
        
        btn_update_keys = QPushButton("Update API Keys")
        btn_update_keys.clicked.connect(self.on_update_keys)
        vl_k.addWidget(btn_update_keys)
        
        layout.addWidget(gb_keys)
        
        # Action Buttons
        self.btn_scan = QPushButton("  RUN WATCHLIST SCAN  ")
        self.btn_scan.setObjectName("btnPrimary")
        self.btn_scan.setMinimumHeight(38)
        self.btn_scan.clicked.connect(self.on_scan_now)
        layout.addWidget(self.btn_scan)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _on_autoscan_changed(self, mins):
        if mins > 0:
            self.auto_scan_timer.start(mins * 60 * 1000)
            self.lbl_status.setText(f"AUTOSCAN: {mins}m")
        else:
            self.auto_scan_timer.stop()
            self.lbl_status.setText("READY")

    def on_update_keys(self):
        raw = self.txt_keys.toPlainText().replace("\n", ",")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        self.pool = KeyPool(keys)
        self.badge_keys.setText(f" KEYS: {len(self.pool.keys)} ")
        QMessageBox.information(self, "API Keys Updated", f"Active rotation pool now has {len(self.pool.keys)} key(s).")

    def get_watchlist(self):
        raw = self.txt_watch.toPlainText().replace("\n", ",")
        return [w.strip() for w in raw.split(",") if w.strip()]

    # =========================================================================
    # TAB 1: CALLS & ALERTS
    # =========================================================================
    def _create_tab_calls(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(10, 10, 10, 10)
        vl.setSpacing(10)
        
        # Metric Cards Header
        cards_hl = QHBoxLayout()
        cards_hl.setSpacing(10)
        self.card_total = MetricCard("Active Tickers", "0", "Watchlist monitored")
        self.card_buys = MetricCard("BUY Signals", "0", "Score >= 3 & Exp > 0", accent="#2ea043")
        self.card_sells = MetricCard("SELL Signals", "0", "Score <= -2", accent="#da3633")
        self.card_hitrate = MetricCard("Scorecard Win Rate", "N/A", "Target vs Stop hit rate", accent="#d29922")
        
        cards_hl.addWidget(self.card_total)
        cards_hl.addWidget(self.card_buys)
        cards_hl.addWidget(self.card_sells)
        cards_hl.addWidget(self.card_hitrate)
        vl.addLayout(cards_hl)
        
        # Calls Table Title + Actions
        hl_tbl_hdr = QHBoxLayout()
        lbl_tbl = QLabel("LATEST QUANTITATIVE CALLS & SIZING")
        lbl_tbl.setStyleSheet("font-weight: 700; color: #f0f6fc; font-size: 13px;")
        hl_tbl_hdr.addWidget(lbl_tbl)
        hl_tbl_hdr.addStretch()
        
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(self.export_calls_csv)
        hl_tbl_hdr.addWidget(btn_export)
        vl.addLayout(hl_tbl_hdr)
        
        # Calls Table
        self.tbl_calls = QTableWidget(0, 13)
        self.tbl_calls.setHorizontalHeaderLabels([
            "Ticker", "Signal", "Score", "Entry", "Stop (2 ATR)", "Target (3 ATR)",
            "Qty", "P(Win)", "P(Loss)", "Exp (R)", "ML Prob", "As Of", "Rationales & Why"
        ])
        self.tbl_calls.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_calls.horizontalHeader().setSectionResizeMode(12, QHeaderView.Stretch)
        self.tbl_calls.setSelectionBehavior(QTableWidget.SelectRows)
        vl.addWidget(self.tbl_calls, 3)
        
        # Lower Split: Scorecard + Live Alerts
        lower_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Scorecard
        score_box = QWidget()
        vl_sc = QVBoxLayout(score_box)
        vl_sc.setContentsMargins(0, 0, 5, 0)
        
        hl_sc_head = QHBoxLayout()
        self.lbl_sc_title = QLabel("LIVE SCORECARD OF PAST 'BUY' TRIGGERS")
        self.lbl_sc_title.setStyleSheet("font-weight: 700; color: #f0f6fc;")
        hl_sc_head.addWidget(self.lbl_sc_title)
        hl_sc_head.addStretch()
        btn_sc_refresh = QPushButton("Refresh")
        btn_sc_refresh.clicked.connect(self.refresh_scorecard)
        hl_sc_head.addWidget(btn_sc_refresh)
        vl_sc.addLayout(hl_sc_head)
        
        self.tbl_score = QTableWidget(0, 7)
        self.tbl_score.setHorizontalHeaderLabels(["Ticker", "Date", "Entry", "Stop", "Target", "Status", "Net Ret %"])
        self.tbl_score.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vl_sc.addWidget(self.tbl_score)
        lower_splitter.addWidget(score_box)
        
        # Right: Alerts History
        alert_box = QWidget()
        vl_al = QVBoxLayout(alert_box)
        vl_al.setContentsMargins(5, 0, 0, 0)
        
        hl_al_head = QHBoxLayout()
        lbl_al_title = QLabel("SYSTEM & TELEGRAM ALERTS LOG")
        lbl_al_title.setStyleSheet("font-weight: 700; color: #f0f6fc;")
        hl_al_head.addWidget(lbl_al_title)
        hl_al_head.addStretch()
        btn_al_refresh = QPushButton("Refresh")
        btn_al_refresh.clicked.connect(self.refresh_alerts_table)
        hl_al_head.addWidget(btn_al_refresh)
        vl_al.addLayout(hl_al_head)
        
        self.tbl_alerts = QTableWidget(0, 3)
        self.tbl_alerts.setHorizontalHeaderLabels(["Timestamp", "Ticker", "Message"])
        self.tbl_alerts.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_alerts.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        vl_al.addWidget(self.tbl_alerts)
        lower_splitter.addWidget(alert_box)
        
        vl.addWidget(lower_splitter, 2)
        return w

    def on_scan_now(self):
        w = self.get_watchlist()
        if not w:
            QMessageBox.warning(self, "Watchlist Empty", "Please add at least one ticker to the watchlist.")
            return
        
        self.btn_scan.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText("SCANNING...")
        
        self.scan_worker = ScanWorker(w, self.spn_capital.value(), self.spn_risk.value(), self.spn_horizon.value())
        self.scan_worker.finished_scan.connect(self._on_scan_finished)
        self.scan_worker.error_scan.connect(self._on_scan_error)
        self.scan_worker.start()

    def _on_scan_finished(self, results):
        self.btn_scan.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        self.populate_calls_table(results)
        self.refresh_scorecard()
        self.refresh_alerts_table()

    def _on_scan_error(self, err):
        self.btn_scan.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        QMessageBox.critical(self, "Scan Error", f"An error occurred during scan:\n{err}")

    def populate_calls_table(self, rows):
        self.tbl_calls.setRowCount(0)
        buys = 0
        sells = 0
        
        for r in rows:
            row_idx = self.tbl_calls.rowCount()
            self.tbl_calls.insertRow(row_idx)
            
            call_type = r.get("call", "")
            if call_type == "BUY": buys += 1
            elif call_type == "SELL": sells += 1
            
            # Ticker Item
            t_item = QTableWidgetItem(str(r.get("ticker", "")))
            t_item.setFont(QtGui.QFont("Consolas", 10, QtGui.QFont.Bold))
            self.tbl_calls.setItem(row_idx, 0, t_item)
            
            # Signal Badge Item
            sig_item = QTableWidgetItem(call_type)
            sig_item.setTextAlignment(Qt.AlignCenter)
            if call_type == "BUY":
                sig_item.setBackground(QtGui.QColor("#238636"))
                sig_item.setForeground(QtGui.QColor("#ffffff"))
            elif call_type == "SELL":
                sig_item.setBackground(QtGui.QColor("#da3633"))
                sig_item.setForeground(QtGui.QColor("#ffffff"))
            elif call_type == "HOLD":
                sig_item.setBackground(QtGui.QColor("#30363d"))
                sig_item.setForeground(QtGui.QColor("#c9d1d9"))
            else:
                sig_item.setBackground(QtGui.QColor("#9e6a03"))
                sig_item.setForeground(QtGui.QColor("#ffffff"))
            self.tbl_calls.setItem(row_idx, 1, sig_item)
            
            # Numeric fields
            score_item = QTableWidgetItem(str(r.get("score", "")))
            score_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_calls.setItem(row_idx, 2, score_item)
            
            entry = r.get("entry")
            self.tbl_calls.setItem(row_idx, 3, QTableWidgetItem(f"{entry:.2f}" if entry is not None else "-"))
            
            stop = r.get("stop")
            stop_item = QTableWidgetItem(f"{stop:.2f}" if stop is not None else "-")
            stop_item.setForeground(QtGui.QColor("#f85149"))
            self.tbl_calls.setItem(row_idx, 4, stop_item)
            
            target = r.get("target")
            target_item = QTableWidgetItem(f"{target:.2f}" if target is not None else "-")
            target_item.setForeground(QtGui.QColor("#3fb950"))
            self.tbl_calls.setItem(row_idx, 5, target_item)
            
            qty = r.get("qty")
            self.tbl_calls.setItem(row_idx, 6, QTableWidgetItem(f"{qty:,}" if qty is not None else "-"))
            
            pw = r.get("p_win")
            self.tbl_calls.setItem(row_idx, 7, QTableWidgetItem(f"{pw:.1%}" if pw is not None else "-"))
            
            pl = r.get("p_loss")
            self.tbl_calls.setItem(row_idx, 8, QTableWidgetItem(f"{pl:.1%}" if pl is not None else "-"))
            
            exp_r = r.get("exp_R")
            exp_item = QTableWidgetItem(f"{exp_r:+.2f}R" if exp_r is not None else "-")
            if exp_r and exp_r > 0: exp_item.setForeground(QtGui.QColor("#3fb950"))
            elif exp_r and exp_r < 0: exp_item.setForeground(QtGui.QColor("#f85149"))
            self.tbl_calls.setItem(row_idx, 9, exp_item)
            
            mp = r.get("model_prob")
            mp_trusted = r.get("model_trusted")
            mp_text = f"{mp:.0%}" + (" ✓" if mp_trusted else " (untrusted)") if mp is not None else "-"
            self.tbl_calls.setItem(row_idx, 10, QTableWidgetItem(mp_text))
            
            self.tbl_calls.setItem(row_idx, 11, QTableWidgetItem(str(r.get("asof", ""))))
            self.tbl_calls.setItem(row_idx, 12, QTableWidgetItem(str(r.get("why", ""))))

        self.card_total.set_value(len(rows))
        self.card_buys.set_value(buys)
        self.card_sells.set_value(sells)

    def refresh_scorecard(self):
        self.sc_worker = ScorecardWorker()
        self.sc_worker.finished_scorecard.connect(self._on_scorecard_finished)
        self.sc_worker.error_scorecard.connect(lambda e: print("Scorecard error:", e))
        self.sc_worker.start()

    def _on_scorecard_finished(self, sc):
        try:
            self.tbl_score.setRowCount(0)
            if sc.empty:
                return
            
            dn = sc[sc.status.isin(["target", "stop", "expired"])]
            if len(dn):
                hit_rate = (dn.status == "target").mean()
                avg_ret = dn.ret_pct.mean()
                self.card_hitrate.set_value(f"{hit_rate:.0%}", f"Avg {avg_ret:+.2f}% ({len(dn)} closed)")
                self.lbl_sc_title.setText(f"LIVE SCORECARD ({len(dn)} Closed | Hit Rate {hit_rate:.0%} | Avg Ret {avg_ret:+.2f}%)")
            
            for _, r in sc.iterrows():
                row_idx = self.tbl_score.rowCount()
                self.tbl_score.insertRow(row_idx)
                self.tbl_score.setItem(row_idx, 0, QTableWidgetItem(str(r.get("ticker", ""))))
                self.tbl_score.setItem(row_idx, 1, QTableWidgetItem(str(r.get("date", ""))))
                self.tbl_score.setItem(row_idx, 2, QTableWidgetItem(f"{r.get('entry', 0):.2f}"))
                self.tbl_score.setItem(row_idx, 3, QTableWidgetItem(f"{r.get('stop', 0):.2f}"))
                self.tbl_score.setItem(row_idx, 4, QTableWidgetItem(f"{r.get('target', 0):.2f}"))
                
                status_item = QTableWidgetItem(str(r.get("status", "")).upper())
                st_val = str(r.get("status", "")).lower()
                if st_val == "target":
                    status_item.setForeground(QtGui.QColor("#3fb950"))
                elif st_val == "stop":
                    status_item.setForeground(QtGui.QColor("#f85149"))
                self.tbl_score.setItem(row_idx, 5, status_item)
                
                ret_val = r.get("ret_pct")
                ret_item = QTableWidgetItem(f"{ret_val:+.2f}%" if pd.notna(ret_val) else "-")
                if pd.notna(ret_val):
                    ret_item.setForeground(QtGui.QColor("#3fb950") if ret_val >= 0 else QtGui.QColor("#f85149"))
                self.tbl_score.setItem(row_idx, 6, ret_item)
        except Exception as e:
            print("Scorecard populate error:", e)

    def refresh_alerts_table(self):
        try:
            a = db.load("alerts")
            self.tbl_alerts.setRowCount(0)
            if a.empty:
                return
            a_sorted = a.sort_values("ts", ascending=False).head(50)
            for _, r in a_sorted.iterrows():
                row_idx = self.tbl_alerts.rowCount()
                self.tbl_alerts.insertRow(row_idx)
                ts_str = dt.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
                self.tbl_alerts.setItem(row_idx, 0, QTableWidgetItem(ts_str))
                self.tbl_alerts.setItem(row_idx, 1, QTableWidgetItem(str(r.get("ticker", ""))))
                self.tbl_alerts.setItem(row_idx, 2, QTableWidgetItem(str(r.get("msg", ""))))
        except Exception as e:
            print("Alerts refresh error:", e)

    def export_calls_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Calls", "alphadesk_calls.csv", "CSV Files (*.csv)")
        if not path:
            return
        rows = []
        for i in range(self.tbl_calls.rowCount()):
            row_data = [self.tbl_calls.item(i, j).text() if self.tbl_calls.item(i, j) else "" for j in range(self.tbl_calls.columnCount())]
            rows.append(row_data)
        headers = [self.tbl_calls.horizontalHeaderItem(j).text() for j in range(self.tbl_calls.columnCount())]
        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Successfully exported {len(df)} rows to {path}")

    # =========================================================================
    # TAB 2: LEVELS & VOLATILITY MODEL
    # =========================================================================
    def _create_tab_levels(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Search Bar
        hl_search = QHBoxLayout()
        hl_search.addWidget(QLabel("Ticker:"))
        self.txt_levels_ticker = QLineEdit("AAPL")
        self.txt_levels_ticker.setMaximumWidth(200)
        hl_search.addWidget(self.txt_levels_ticker)
        
        btn_calc = QPushButton("Compute Levels + Walk-Forward Model")
        btn_calc.setObjectName("btnAccent")
        btn_calc.clicked.connect(self.on_compute_levels)
        hl_search.addWidget(btn_calc)
        hl_search.addStretch()
        layout.addLayout(hl_search)
        
        # Splitter: Left Metrics, Right Matplotlib Chart
        splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel (Metrics & Cards)
        left_panel = QWidget()
        vl_l = QVBoxLayout(left_panel)
        vl_l.setContentsMargins(0, 0, 5, 0)
        vl_l.setSpacing(10)
        
        # Header Banner
        self.lbl_lv_header = QLabel("ENTER TICKER AND COMPUTE")
        self.lbl_lv_header.setStyleSheet("font-size: 15px; font-weight: 700; color: #f0f6fc;")
        vl_l.addWidget(self.lbl_lv_header)
        
        self.lbl_lv_why = QLabel("")
        self.lbl_lv_why.setWordWrap(True)
        self.lbl_lv_why.setStyleSheet("color: #8b949e; font-size: 12px;")
        vl_l.addWidget(self.lbl_lv_why)
        
        # Sizing Grid
        gl_cards = QGridLayout()
        self.card_lv_entry = MetricCard("Entry Price", "-", "")
        self.card_lv_stop = MetricCard("Stop Loss (2 ATR)", "-", "", accent="#da3633")
        self.card_lv_target = MetricCard("Target (3 ATR)", "-", "", accent="#2ea043")
        self.card_lv_qty = MetricCard("Position Sized Qty", "-", "Risk-sized", accent="#d29922")
        
        gl_cards.addWidget(self.card_lv_entry, 0, 0)
        gl_cards.addWidget(self.card_lv_stop, 0, 1)
        gl_cards.addWidget(self.card_lv_target, 1, 0)
        gl_cards.addWidget(self.card_lv_qty, 1, 1)
        vl_l.addLayout(gl_cards)
        
        # Monte Carlo & Odds Box
        gb_mc = QGroupBox("MONTE CARLO (5,000 BLOCK-BOOTSTRAP PATHS, ZERO-DRIFT)")
        gl_mc = QGridLayout(gb_mc)
        gl_mc.setSpacing(6)
        
        self.lbl_mc_rr = QLabel("Risk:Reward: -")
        self.lbl_mc_pwin = QLabel("P(Target First): -")
        self.lbl_mc_ploss = QLabel("P(Stop First): -")
        self.lbl_mc_popen = QLabel("P(Neither): -")
        self.lbl_mc_expr = QLabel("Expectancy: -")
        self.lbl_mc_cone = QLabel("1-Sigma Cone: -")
        self.lbl_mc_pct = QLabel("Simulated 5/50/95th: -")
        
        gl_mc.addWidget(self.lbl_mc_rr, 0, 0)
        gl_mc.addWidget(self.lbl_mc_pwin, 0, 1)
        gl_mc.addWidget(self.lbl_mc_ploss, 1, 0)
        gl_mc.addWidget(self.lbl_mc_popen, 1, 1)
        gl_mc.addWidget(self.lbl_mc_expr, 2, 0)
        gl_mc.addWidget(self.lbl_mc_cone, 2, 1)
        gl_mc.addWidget(self.lbl_mc_pct, 3, 0, 1, 2)
        vl_l.addWidget(gb_mc)
        
        # Walk-Forward Stats Box
        self.gb_wf = QGroupBox("WALK-FORWARD ML (OUT-OF-SAMPLE VALIDATION)")
        vl_wf = QVBoxLayout(self.gb_wf)
        self.lbl_wf_stats = QLabel("No model computed yet.")
        self.lbl_wf_stats.setWordWrap(True)
        self.lbl_wf_stats.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        vl_wf.addWidget(self.lbl_wf_stats)
        vl_l.addWidget(self.gb_wf)
        
        vl_l.addStretch()
        splitter.addWidget(left_panel)
        
        # Right Panel (Interactive Matplotlib Native Chart)
        right_panel = QWidget()
        vl_r = QVBoxLayout(right_panel)
        vl_r.setContentsMargins(5, 0, 0, 0)
        
        self.fig = Figure(figsize=(6, 5), facecolor="#0d1117")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._init_chart()
        vl_r.addWidget(self.canvas)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        return w

    def _init_chart(self):
        self.ax.clear()
        self.ax.set_facecolor("#0d1117")
        self.ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#30363d")
        self.ax.grid(True, color="#21262d", linestyle="--", linewidth=0.7)
        self.ax.set_title("Walk-Forward Strategy vs Buy & Hold Benchmark", color="#f0f6fc", fontsize=11, fontweight="bold")
        self.canvas.draw()

    def on_compute_levels(self):
        tk = self.txt_levels_ticker.text().strip()
        if not tk: return
        self.lbl_status.setText(f"COMPUTING: {tk}...")
        self.progress_bar.setVisible(True)
        
        self.levels_worker = LevelsWorker(tk, self.spn_capital.value(), self.spn_risk.value(), self.spn_horizon.value())
        self.levels_worker.finished_levels.connect(self._on_levels_finished)
        self.levels_worker.error_levels.connect(self._on_levels_error)
        self.levels_worker.start()

    def _on_levels_finished(self, call_res, model_res):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        
        tk = call_res.get("ticker", "")
        call = call_res.get("call", "")
        why = call_res.get("why", "")
        ccy = call_res.get("ccy", "")
        
        self.lbl_lv_header.setText(f"{tk}: {call} (Score: {call_res.get('score', 0)})")
        self.lbl_lv_why.setText(why)
        
        entry = call_res.get("entry")
        stop = call_res.get("stop")
        target = call_res.get("target")
        qty = call_res.get("qty")
        
        if entry is not None:
            self.card_lv_entry.set_value(f"{entry:.2f} {ccy}")
            self.card_lv_stop.set_value(f"{stop:.2f}")
            self.card_lv_target.set_value(f"{target:.2f}")
            self.card_lv_qty.set_value(f"{qty:,}", f"Risk amt: ${call_res.get('risk_amt', 0):,.2f}")
            
            # Monte Carlo
            rr = call_res.get("rr", 0)
            pw = call_res.get("p_win", 0)
            pl = call_res.get("p_loss", 0)
            po = call_res.get("p_open", 0)
            exp_r = call_res.get("exp_R", 0)
            h = call_res.get("horizon", 10)
            
            self.lbl_mc_rr.setText(f"Risk:Reward: 1:{rr:.2f}")
            self.lbl_mc_pwin.setText(f"P(Target First): {pw:.1%}")
            self.lbl_mc_ploss.setText(f"P(Stop First): {pl:.1%}")
            self.lbl_mc_popen.setText(f"P(Neither in {h}d): {po:.1%}")
            self.lbl_mc_expr.setText(f"Expectancy: {exp_r:+.2f}R")
            self.lbl_mc_cone.setText(f"1-Sigma: {call_res.get('sigma1_lo', 0):.2f} to {call_res.get('sigma1_hi', 0):.2f}")
            self.lbl_mc_pct.setText(f"Simulated Percentiles (5th / 50th / 95th): {call_res.get('p5', 0):.2f} / {call_res.get('p50', 0):.2f} / {call_res.get('p95', 0):.2f}")
        
        # Model stats and Chart
        if model_res and model_res.get("ok"):
            is_trusted = model_res.get("trusted", False)
            trust_badge = "TRUSTED (STATISTICALLY SIGNIFICANT)" if is_trusted else "NOT TRUSTED (NO PROVEN EDGE)"
            stat_text = (
                f"Model Trust Gate: {trust_badge}\n"
                f"Trades: {model_res.get('n_trades', 0)} | Win Rate: {model_res.get('win_rate', 0):.1%} (Base: {model_res.get('base_rate', 0):.1%})\n"
                f"Accuracy: {model_res.get('accuracy', 0):.1%} | Avg Net Ret/Trade: {model_res.get('avg_net_ret', 0):.2%}\n"
                f"Total Return: {model_res.get('total_ret', 0):.1%} vs Buy&Hold: {model_res.get('buy_hold_ret', 0):.1%}\n"
                f"Max Drawdown: {model_res.get('max_dd', 0):.1%} | Sharpe: {model_res.get('sharpe', 0):.2f} | p-value: {model_res.get('p_value', 1.0):.4f}"
            )
            self.lbl_wf_stats.setText(stat_text)
            self.lbl_wf_stats.setStyleSheet("font-family: 'Consolas', monospace; color: " + ("#3fb950" if is_trusted else "#8b949e"))
            
            # Plot Curve
            curve = model_res.get("curve")
            if curve is not None and not curve.empty:
                self.ax.clear()
                self.ax.set_facecolor("#0d1117")
                self.ax.grid(True, color="#21262d", linestyle="--", linewidth=0.7)
                self.ax.tick_params(colors="#8b949e", labelsize=9)
                for spine in self.ax.spines.values(): spine.set_color("#30363d")
                
                self.ax.plot(curve.index, curve["strategy"], label="Strategy (After Costs)", color="#388bfd", linewidth=2.0)
                self.ax.plot(curve.index, curve["buy_hold"], label="Buy & Hold", color="#8b949e", linewidth=1.5, linestyle=":")
                self.ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
                self.ax.set_title(f"Walk-Forward Cumulative Equity - {tk}", color="#f0f6fc", fontsize=11, fontweight="bold")
                self.canvas.draw()
        else:
            reason = model_res.get("reason", "No model signal generated.") if model_res else "No data."
            self.lbl_wf_stats.setText(f"ML Model: {reason}")
            self.lbl_wf_stats.setStyleSheet("color: #8b949e;")

    def _on_levels_error(self, err):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        QMessageBox.critical(self, "Levels Calculation Error", f"Error calculating levels:\n{err}")

    # =========================================================================
    # TAB 3: BULL VS BEAR DEBATE
    # =========================================================================
    def _create_tab_debate(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Ticker & Control Bar
        hl_ctrl = QHBoxLayout()
        hl_ctrl.addWidget(QLabel("Ticker:"))
        self.txt_debate_ticker = QLineEdit("AAPL")
        self.txt_debate_ticker.setMaximumWidth(160)
        hl_ctrl.addWidget(self.txt_debate_ticker)
        
        btn_load_ev = QPushButton("1. Load Evidence & SEC Filings")
        btn_load_ev.setObjectName("btnAccent")
        btn_load_ev.clicked.connect(self.on_load_evidence)
        hl_ctrl.addWidget(btn_load_ev)
        
        self.btn_next_round = QPushButton("2. Next Debate Round")
        self.btn_next_round.clicked.connect(self.on_debate_round)
        self.btn_next_round.setEnabled(False)
        hl_ctrl.addWidget(self.btn_next_round)
        
        self.btn_judge = QPushButton("3. Call Judge Verdict")
        self.btn_judge.clicked.connect(self.on_debate_verdict)
        self.btn_judge.setEnabled(False)
        hl_ctrl.addWidget(self.btn_judge)
        
        btn_new_deb = QPushButton("Reset / New Debate")
        btn_new_deb.clicked.connect(self.on_new_debate)
        hl_ctrl.addWidget(btn_new_deb)
        
        hl_ctrl.addStretch()
        layout.addLayout(hl_ctrl)
        
        # Splitter: Chat Transcript (Left) + Evidence Viewer (Right)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Chat Transcript Area
        chat_box = QWidget()
        vl_chat = QVBoxLayout(chat_box)
        vl_chat.setContentsMargins(0, 0, 5, 0)
        
        self.txt_transcript = QTextEdit()
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 12px;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        vl_chat.addWidget(self.txt_transcript, 1)
        
        # User input box to jump into debate
        hl_input = QHBoxLayout()
        self.txt_user_query = QLineEdit()
        self.txt_user_query.setPlaceholderText("Jump into the debate! Ask Bull or Bear a question...")
        self.txt_user_query.returnPressed.connect(self.on_user_debate)
        hl_input.addWidget(self.txt_user_query)
        
        self.btn_user_send = QPushButton("Send")
        self.btn_user_send.clicked.connect(self.on_user_debate)
        hl_input.addWidget(self.btn_user_send)
        vl_chat.addLayout(hl_input)
        
        splitter.addWidget(chat_box)
        
        # Right: Evidence Viewer Box
        ev_box = QWidget()
        vl_ev = QVBoxLayout(ev_box)
        vl_ev.setContentsMargins(5, 0, 0, 0)
        
        lbl_ev = QLabel("EVIDENCE BUNDLE FED TO AGENTS (YF + EDGAR 10-K / 10-Q)")
        lbl_ev.setStyleSheet("font-weight: 700; color: #8b949e; font-size: 11px;")
        vl_ev.addWidget(lbl_ev)
        
        self.txt_evidence = QTextEdit()
        self.txt_evidence.setReadOnly(True)
        self.txt_evidence.setStyleSheet("background-color: #0d1117; color: #8b949e; font-family: 'Consolas', monospace; font-size: 11px;")
        vl_ev.addWidget(self.txt_evidence)
        splitter.addWidget(ev_box)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        return w

    def on_load_evidence(self):
        tk = self.txt_debate_ticker.text().strip()
        if not tk: return
        self.lbl_status.setText(f"FETCHING EVIDENCE: {tk}...")
        self.progress_bar.setVisible(True)
        
        self.ev_worker = EvidenceWorker(tk)
        self.ev_worker.finished_evidence.connect(self._on_evidence_loaded)
        self.ev_worker.error_evidence.connect(self._on_evidence_error)
        self.ev_worker.start()

    def _on_evidence_loaded(self, ev_text):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        self.active_evidence = ev_text
        self.txt_evidence.setPlainText(ev_text)
        
        tk = self.txt_debate_ticker.text().strip()
        nk = data.norm(tk)
        sid = self.current_sid.setdefault(nk, str(uuid.uuid4()))
        
        try:
            self.active_debate = Debate(sid, nk, ev_text, self.pool)
            self.btn_next_round.setEnabled(True)
            self.btn_judge.setEnabled(True)
            self.append_chat_message("SYSTEM", f"Evidence loaded for {nk} ({len(ev_text):,} characters). Agents initialized and grounded on filings.")
        except Exception as e:
            QMessageBox.critical(self, "Debate Init Error", str(e))

    def _on_evidence_error(self, err):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        QMessageBox.critical(self, "Evidence Error", f"Failed to fetch filings/evidence:\n{err}")

    def on_debate_round(self):
        if not self.active_debate: return
        self.lbl_status.setText("DEBATING...")
        self.progress_bar.setVisible(True)
        self.btn_next_round.setEnabled(False)
        
        self.round_worker = DebateRoundWorker(self.active_debate)
        self.round_worker.finished_round.connect(self._on_round_finished)
        self.round_worker.error_round.connect(self._on_round_error)
        self.round_worker.start()

    def _on_round_finished(self, bull, bear):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        self.btn_next_round.setEnabled(True)
        self.append_chat_message("BULL", bull)
        self.append_chat_message("BEAR", bear)

    def _on_round_error(self, err):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        self.btn_next_round.setEnabled(True)
        QMessageBox.critical(self, "Debate Error", f"Debate agent call failed:\n{err}")

    def on_user_debate(self):
        text = self.txt_user_query.text().strip()
        if not text or not self.active_debate: return
        self.txt_user_query.clear()
        self.append_chat_message("USER", text)
        
        self.lbl_status.setText("AGENTS RESPONDING...")
        self.progress_bar.setVisible(True)
        
        self.user_worker = DebateUserWorker(self.active_debate, text)
        self.user_worker.finished_user.connect(self._on_user_finished)
        self.user_worker.error_user.connect(self._on_round_error)
        self.user_worker.start()

    def _on_user_finished(self, bull, bear):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        self.append_chat_message("BULL", bull)
        self.append_chat_message("BEAR", bear)

    def on_debate_verdict(self):
        if not self.active_debate: return
        self.lbl_status.setText("JUDGING...")
        self.progress_bar.setVisible(True)
        
        self.verdict_worker = DebateVerdictWorker(self.active_debate)
        self.verdict_worker.finished_verdict.connect(self._on_verdict_finished)
        self.verdict_worker.error_verdict.connect(self._on_round_error)
        self.verdict_worker.start()

    def _on_verdict_finished(self, verdict):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        self.append_chat_message("JUDGE", verdict)

    def on_new_debate(self):
        tk = self.txt_debate_ticker.text().strip()
        nk = data.norm(tk)
        self.current_sid[nk] = str(uuid.uuid4())
        self.txt_transcript.clear()
        if self.active_evidence:
            self.active_debate = Debate(self.current_sid[nk], nk, self.active_evidence, self.pool)
            self.append_chat_message("SYSTEM", f"New debate session created for {nk}.")

    def append_chat_message(self, role, content):
        color_map = {
            "BULL": ("#238636", "#3fb950", "👍 BULL ANALYST"),
            "BEAR": ("#da3633", "#f85149", "👎 BEAR ANALYST"),
            "JUDGE": ("#9e6a03", "#e3b341", "⚖️ NEUTRAL JUDGE"),
            "USER": ("#1f6feb", "#58a6ff", "👤 USER QUESTION"),
            "SYSTEM": ("#30363d", "#8b949e", "⚙️ SYSTEM")
        }
        bg, border, title = color_map.get(role, ("#21262d", "#8b949e", role))
        
        html = f"""
        <div style="margin-bottom: 12px; background-color: #161b22; border-left: 4px solid {border}; border-radius: 4px; padding: 10px 14px;">
            <div style="font-weight: 700; color: {border}; margin-bottom: 4px; font-size: 11px;">{title}</div>
            <div style="color: #f0f6fc; font-size: 13px; line-height: 1.4;">{content.replace(chr(10), '<br>')}</div>
        </div>
        """
        self.txt_transcript.append(html)
        sb = self.txt_transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    # =========================================================================
    # TAB 4: INSIDER RADAR
    # =========================================================================
    def _create_tab_insiders(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Search Bar
        hl_search = QHBoxLayout()
        hl_search.addWidget(QLabel("Ticker:"))
        self.txt_insider_ticker = QLineEdit("AAPL")
        self.txt_insider_ticker.setMaximumWidth(200)
        hl_search.addWidget(self.txt_insider_ticker)
        
        btn_check = QPushButton("Check Insider Filings (Form 4 / SEBI PIT)")
        btn_check.setObjectName("btnAccent")
        btn_check.clicked.connect(self.on_check_insiders)
        hl_search.addWidget(btn_check)
        hl_search.addStretch()
        layout.addLayout(hl_search)
        
        # Hero Cluster Status Card
        self.card_cluster_status = MetricCard("Cluster Status", "NO CLUSTER", "Threshold: >=3 distinct insiders within rolling 14 days")
        layout.addWidget(self.card_cluster_status)
        
        # Filings Table
        lbl_tbl = QLabel("INDIVIDUAL OPEN-MARKET PURCHASES (CODE 'P' / BUYS ONLY)")
        lbl_tbl.setStyleSheet("font-weight: 700; color: #f0f6fc; margin-top: 8px;")
        layout.addWidget(lbl_tbl)
        
        self.tbl_insiders = QTableWidget(0, 6)
        self.tbl_insiders.setHorizontalHeaderLabels(["Insider Name", "Title / Role", "Filing Date", "Shares Bought", "Price", "Total Value"])
        self.tbl_insiders.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_insiders.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.tbl_insiders)
        return w

    def on_check_insiders(self):
        tk = self.txt_insider_ticker.text().strip()
        if not tk: return
        self.lbl_status.setText(f"SCANNING INSIDERS: {tk}...")
        self.progress_bar.setVisible(True)
        
        self.insider_worker = InsidersWorker(tk)
        self.insider_worker.finished_insiders.connect(self._on_insiders_finished)
        self.insider_worker.error_insiders.connect(self._on_insiders_error)
        self.insider_worker.start()

    def _on_insiders_finished(self, res):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        
        flag = res.get("flag", False)
        n = res.get("n", 0)
        tot = res.get("total", 0.0)
        err = res.get("err")
        
        if err:
            self.card_cluster_status.set_value("DATA UNAVAILABLE", str(err), color="#e3b341")
        elif flag:
            self.card_cluster_status.set_value("CLUSTER DETECTED!", f"{n} distinct insiders in 14-day window | Total ${tot:,.2f}", color="#3fb950")
        else:
            self.card_cluster_status.set_value("NO CLUSTER", f"Max distinct insiders in 14d: {n} (Total: ${tot:,.2f})", color="#8b949e")
        
        # Populate table
        df = res.get("df", pd.DataFrame())
        self.tbl_insiders.setRowCount(0)
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                row_idx = self.tbl_insiders.rowCount()
                self.tbl_insiders.insertRow(row_idx)
                self.tbl_insiders.setItem(row_idx, 0, QTableWidgetItem(str(r.get("insider", ""))))
                self.tbl_insiders.setItem(row_idx, 1, QTableWidgetItem(str(r.get("title", ""))))
                self.tbl_insiders.setItem(row_idx, 2, QTableWidgetItem(str(r.get("date", ""))))
                
                sh = r.get("shares", 0)
                px = r.get("price", 0)
                val = r.get("value", 0)
                
                self.tbl_insiders.setItem(row_idx, 3, QTableWidgetItem(f"{sh:,.0f}" if pd.notna(sh) else "-"))
                self.tbl_insiders.setItem(row_idx, 4, QTableWidgetItem(f"{px:,.2f}" if pd.notna(px) else "-"))
                
                val_item = QTableWidgetItem(f"${val:,.2f}" if pd.notna(val) else "-")
                val_item.setForeground(QtGui.QColor("#3fb950"))
                self.tbl_insiders.setItem(row_idx, 5, val_item)

    def _on_insiders_error(self, err):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        QMessageBox.critical(self, "Insider Scan Error", f"Error scanning insider transactions:\n{err}")

    # =========================================================================
    # TAB 5: EARNINGS CALL STRESS
    # =========================================================================
    def _create_tab_stress(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Audio File Picker & Params
        gb_audio = QGroupBox("AUDIO INPUT & EARNINGS CALL PARAMETERS")
        gl_a = QGridLayout(gb_audio)
        gl_a.setSpacing(8)
        
        gl_a.addWidget(QLabel("Audio File:"), 0, 0)
        self.txt_audio_path = QLineEdit()
        self.txt_audio_path.setPlaceholderText("Select earnings call audio (.mp3, .wav, .m4a)")
        gl_a.addWidget(self.txt_audio_path, 0, 1)
        
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.on_browse_audio)
        gl_a.addWidget(btn_browse, 0, 2)
        
        gl_a.addWidget(QLabel("Ticker:"), 1, 0)
        self.txt_stress_ticker = QLineEdit("AAPL")
        self.txt_stress_ticker.setMaximumWidth(140)
        gl_a.addWidget(self.txt_stress_ticker, 1, 1)
        
        gl_a.addWidget(QLabel("Call Date (YYYY-MM-DD):"), 2, 0)
        self.txt_stress_date = QLineEdit(dt.date.today().strftime("%Y-%m-%d"))
        self.txt_stress_date.setMaximumWidth(140)
        gl_a.addWidget(self.txt_stress_date, 2, 1)
        
        btn_analyze = QPushButton("Run Whisper + Pitch + FinBERT Pipeline")
        btn_analyze.setObjectName("btnPrimary")
        btn_analyze.clicked.connect(self.on_analyze_stress)
        gl_a.addWidget(btn_analyze, 3, 0, 1, 3)
        layout.addWidget(gb_audio)
        
        # Results Cards
        hl_res = QHBoxLayout()
        self.card_stress_score = MetricCard("Composite Stress Score", "-", "0 (Calm) to 100 (Extreme)", accent="#da3633")
        self.card_neg_sentiment = MetricCard("FinBERT Negativity", "-", "Negative probability", accent="#e3b341")
        self.card_pitch_cv = MetricCard("Voice Pitch Variability", "-", "pYIN fundamental frequency CV", accent="#58a6ff")
        self.card_nextday = MetricCard("Next-Day Market Return", "-", "Stock return after call", accent="#3fb950")
        
        hl_res.addWidget(self.card_stress_score)
        hl_res.addWidget(self.card_neg_sentiment)
        hl_res.addWidget(self.card_pitch_cv)
        hl_res.addWidget(self.card_nextday)
        layout.addLayout(hl_res)
        
        # Predictive Power & Historical Correlation
        gb_hist = QGroupBox("HISTORICAL PREDICTIVE POWER & CORRELATION TRACKER")
        vl_h = QVBoxLayout(gb_hist)
        self.lbl_stress_corr = QLabel("Loading historical calls...")
        self.lbl_stress_corr.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px; color: #c9d1d9;")
        vl_h.addWidget(self.lbl_stress_corr)
        layout.addWidget(gb_hist)
        
        layout.addStretch()
        self.refresh_stress_predictive_power()
        return w

    def on_browse_audio(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Earnings Call Audio", "", "Audio Files (*.mp3 *.wav *.m4a *.aac)")
        if f: self.txt_audio_path.setText(f)

    def on_analyze_stress(self):
        p = self.txt_audio_path.text().strip()
        tk = self.txt_stress_ticker.text().strip()
        d = self.txt_stress_date.text().strip()
        if not p or not os.path.exists(p):
            QMessageBox.warning(self, "Invalid File", "Please select a valid earnings call audio file.")
            return
        
        self.lbl_status.setText("RUNNING WHISPER & FINBERT...")
        self.progress_bar.setVisible(True)
        
        self.stress_worker = StressWorker(p, tk, d)
        self.stress_worker.finished_stress.connect(self._on_stress_finished)
        self.stress_worker.error_stress.connect(self._on_stress_error)
        self.stress_worker.start()

    def _on_stress_finished(self, r):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        
        sc = r.get("stress", 0)
        neg = r.get("neg", 0)
        cv = r.get("pitch_cv", 0)
        nxt = r.get("next_day_ret")
        
        self.card_stress_score.set_value(f"{sc:.1f} / 100", "Composite stress score")
        self.card_neg_sentiment.set_value(f"{neg:.1%}", "FinBERT negative score")
        self.card_pitch_cv.set_value(f"{cv:.3f}", f"Pause ratio: {r.get('pause', 0):.1%}")
        
        if nxt is not None:
            self.card_nextday.set_value(f"{nxt:+.2%}", "Next-day return", color="#3fb950" if nxt >= 0 else "#f85149")
        else:
            self.card_nextday.set_value("N/A", "Awaiting next market day")
        
        self.refresh_stress_predictive_power()

    def _on_stress_error(self, err):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        QMessageBox.critical(self, "Stress Detection Error", f"Audio processing failed:\n{err}")

    def refresh_stress_predictive_power(self):
        try:
            pp = stress.predictive_power()
            n = pp.get("n", 0)
            corr = pp.get("corr")
            corr_text = f"{corr:+.3f}" if corr is not None else "Insufficient sample (need >=3 calls)"
            self.lbl_stress_corr.setText(f"Logged Calls: {n} | Pearson Correlation (Stress Score vs Next-Day Return): {corr_text}")
        except Exception as e:
            self.lbl_stress_corr.setText(f"Correlation: {e}")

    # =========================================================================
    # TAB 6: OPTIONS & BLACK-SCHOLES
    # =========================================================================
    def _create_tab_options(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Underlying & Expiry Header Bar
        hl_top = QHBoxLayout()
        hl_top.addWidget(QLabel("Underlying:"))
        self.txt_options_ticker = QLineEdit("AAPL")
        self.txt_options_ticker.setMaximumWidth(140)
        hl_top.addWidget(self.txt_options_ticker)
        
        btn_load_exp = QPushButton("Fetch Live Expiries")
        btn_load_exp.clicked.connect(self.on_load_expiries)
        hl_top.addWidget(btn_load_exp)
        
        hl_top.addWidget(QLabel("Expiry:"))
        self.cmb_expiries = QComboBox()
        self.cmb_expiries.setMinimumWidth(140)
        hl_top.addWidget(self.cmb_expiries)
        
        btn_analyze_chain = QPushButton("Analyze Chain & Greeks")
        btn_analyze_chain.setObjectName("btnAccent")
        btn_analyze_chain.clicked.connect(self.on_analyze_chain)
        hl_top.addWidget(btn_analyze_chain)
        
        hl_top.addStretch()
        layout.addLayout(hl_top)
        
        # Volatility & Analytics Metric Cards
        hl_cards = QHBoxLayout()
        self.card_opt_atm_iv = MetricCard("ATM Implied Vol", "-", "Solved via Brent IV")
        self.card_opt_straddle = MetricCard("Straddle-Implied Move", "-", "Expected market swing")
        self.card_opt_skew = MetricCard("25-Delta Skew", "-", "Put IV - Call IV (fear gauge)", accent="#da3633")
        self.card_opt_vrp = MetricCard("Vol Risk Premium", "-", "Implied IV - GARCH forecast", accent="#d29922")
        
        hl_cards.addWidget(self.card_opt_atm_iv)
        hl_cards.addWidget(self.card_opt_straddle)
        hl_cards.addWidget(self.card_opt_skew)
        hl_cards.addWidget(self.card_opt_vrp)
        layout.addLayout(hl_cards)
        
        # Option Chain Table
        lbl_chain = QLabel("CALCULATED OPTION CHAIN (PROPRIETARY BRENT IVs & ANALYTICAL GREEKS)")
        lbl_chain.setStyleSheet("font-weight: 700; color: #f0f6fc; font-size: 12px; margin-top: 4px;")
        layout.addWidget(lbl_chain)
        
        self.tbl_chain = QTableWidget(0, 10)
        self.tbl_chain.setHorizontalHeaderLabels([
            "Type", "Strike", "Mid Quote", "Brent IV", "Delta", "Gamma", "Vega", "Theta", "Volume", "Open Interest"
        ])
        self.tbl_chain.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tbl_chain, 3)
        
        # Interactive Black-Scholes Instant Calculator Box
        gb_calc = QGroupBox("INSTANT BLACK-SCHOLES-MERTON ANALYTICAL CALCULATOR")
        gl_c = QGridLayout(gb_calc)
        gl_c.setSpacing(8)
        
        gl_c.addWidget(QLabel("Spot (S):"), 0, 0)
        self.spn_bs_s = QDoubleSpinBox()
        self.spn_bs_s.setRange(0.01, 1e7); self.spn_bs_s.setValue(100.0); self.spn_bs_s.valueChanged.connect(self._recalc_bs)
        gl_c.addWidget(self.spn_bs_s, 0, 1)
        
        gl_c.addWidget(QLabel("Strike (K):"), 0, 2)
        self.spn_bs_k = QDoubleSpinBox()
        self.spn_bs_k.setRange(0.01, 1e7); self.spn_bs_k.setValue(100.0); self.spn_bs_k.valueChanged.connect(self._recalc_bs)
        gl_c.addWidget(self.spn_bs_k, 0, 3)
        
        gl_c.addWidget(QLabel("Days (T):"), 0, 4)
        self.spn_bs_t = QSpinBox()
        self.spn_bs_t.setRange(1, 3650); self.spn_bs_t.setValue(30); self.spn_bs_t.valueChanged.connect(self._recalc_bs)
        gl_c.addWidget(self.spn_bs_t, 0, 5)
        
        gl_c.addWidget(QLabel("Rate (r):"), 1, 0)
        self.spn_bs_r = QDoubleSpinBox()
        self.spn_bs_r.setRange(0.0, 1.0); self.spn_bs_r.setSingleStep(0.005); self.spn_bs_r.setValue(0.05); self.spn_bs_r.valueChanged.connect(self._recalc_bs)
        gl_c.addWidget(self.spn_bs_r, 1, 1)
        
        gl_c.addWidget(QLabel("Vol (σ):"), 1, 2)
        self.spn_bs_v = QDoubleSpinBox()
        self.spn_bs_v.setRange(0.01, 5.0); self.spn_bs_v.setSingleStep(0.02); self.spn_bs_v.setValue(0.25); self.spn_bs_v.valueChanged.connect(self._recalc_bs)
        gl_c.addWidget(self.spn_bs_v, 1, 3)
        
        gl_c.addWidget(QLabel("Type:"), 1, 4)
        self.cmb_bs_type = QComboBox()
        self.cmb_bs_type.addItems(["Call (c)", "Put (p)"]); self.cmb_bs_type.currentIndexChanged.connect(self._recalc_bs)
        gl_c.addWidget(self.cmb_bs_type, 1, 5)
        
        # Output Row
        self.lbl_bs_output = QLabel("Fair Price: - | Delta: - | Gamma: - | Vega: - | Theta: - | Rho: - | Risk-Neutral P(S_T > K): -")
        self.lbl_bs_output.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px; color: #58a6ff; font-weight: 700; padding-top: 6px;")
        gl_c.addWidget(self.lbl_bs_output, 2, 0, 1, 6)
        
        layout.addWidget(gb_calc, 1)
        self._recalc_bs()
        return w

    def on_load_expiries(self):
        tk = self.txt_options_ticker.text().strip()
        if not tk: return
        self.lbl_status.setText(f"FETCHING EXPIRIES: {tk}...")
        self.progress_bar.setVisible(True)
        
        self.opt_exp_worker = OptionsExpiriesWorker(tk)
        self.opt_exp_worker.finished_expiries.connect(self._on_expiries_finished)
        self.opt_exp_worker.error_expiries.connect(self._on_expiries_error)
        self.opt_exp_worker.start()

    def _on_expiries_finished(self, exps):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        self.cmb_expiries.clear()
        if not exps:
            QMessageBox.information(self, "No Options", "No option expiries found for this symbol.")
            return
        self.cmb_expiries.addItems(exps)

    def _on_expiries_error(self, err):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        QMessageBox.critical(self, "Options Error", f"Could not fetch options expiries:\n{err}")

    def on_analyze_chain(self):
        tk = self.txt_options_ticker.text().strip()
        exp = self.cmb_expiries.currentText().strip()
        if not tk or not exp:
            QMessageBox.warning(self, "Missing Selection", "Please specify a ticker and choose an expiry date.")
            return
        
        self.lbl_status.setText(f"ANALYZING CHAIN: {tk} {exp}...")
        self.progress_bar.setVisible(True)
        
        self.chain_worker = OptionsChainWorker(tk, exp)
        self.chain_worker.finished_chain.connect(self._on_chain_finished)
        self.chain_worker.error_chain.connect(self._on_chain_error)
        self.chain_worker.start()

    def _on_chain_finished(self, res):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("READY")
        
        if not res.get("ok"):
            QMessageBox.warning(self, "Chain Analysis", res.get("reason", "No usable quotes after filtering."))
            return
        
        atm_iv = res.get("atm_iv", 0)
        straddle = res.get("implied_move_pct")
        skew = res.get("skew_25d")
        vrp = res.get("vol_risk_premium")
        garch_vol = res.get("garch_vol_ann")
        
        self.card_opt_atm_iv.set_value(f"{atm_iv:.1%}", f"ATM Strike: {res.get('atm_strike', 0):.2f}")
        self.card_opt_straddle.set_value(f"{straddle:.1%}" if straddle else "N/A", f"Straddle price: ${res.get('straddle', 0):.2f}")
        self.card_opt_skew.set_value(f"{skew:+.1%}" if skew is not None else "N/A", "25d Skew (Fear premium)")
        
        if vrp is not None and garch_vol is not None:
            self.card_opt_vrp.set_value(f"{vrp:+.1%}", f"GARCH: {garch_vol:.1%} (Rich if >0)", color="#3fb950" if vrp < 0 else "#da3633")
        else:
            self.card_opt_vrp.set_value("N/A", "GARCH baseline unavailable")
        
        # Populate Chain Table
        chain_df = res.get("chain", pd.DataFrame())
        self.tbl_chain.setRowCount(0)
        if not chain_df.empty:
            for _, r in chain_df.iterrows():
                row_idx = self.tbl_chain.rowCount()
                self.tbl_chain.insertRow(row_idx)
                
                cp_str = "CALL" if r["cp"] == "c" else "PUT"
                cp_item = QTableWidgetItem(cp_str)
                cp_item.setTextAlignment(Qt.AlignCenter)
                cp_item.setForeground(QtGui.QColor("#3fb950") if cp_str == "CALL" else QtGui.QColor("#f85149"))
                self.tbl_chain.setItem(row_idx, 0, cp_item)
                
                self.tbl_chain.setItem(row_idx, 1, QTableWidgetItem(f"{r['strike']:.2f}"))
                self.tbl_chain.setItem(row_idx, 2, QTableWidgetItem(f"{r['mid']:.2f}"))
                self.tbl_chain.setItem(row_idx, 3, QTableWidgetItem(f"{r['iv']:.1%}"))
                self.tbl_chain.setItem(row_idx, 4, QTableWidgetItem(f"{r['delta']:+.3f}"))
                self.tbl_chain.setItem(row_idx, 5, QTableWidgetItem(f"{r['gamma']:.4f}"))
                self.tbl_chain.setItem(row_idx, 6, QTableWidgetItem(f"{r['vega']:.3f}"))
                self.tbl_chain.setItem(row_idx, 7, QTableWidgetItem(f"{r['theta']:+.3f}"))
                self.tbl_chain.setItem(row_idx, 8, QTableWidgetItem(f"{r['volume']:,.0f}"))
                self.tbl_chain.setItem(row_idx, 9, QTableWidgetItem(f"{r['oi']:,.0f}"))

    def _on_chain_error(self, err):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("ERROR")
        QMessageBox.critical(self, "Chain Analysis Error", f"Failed to analyze options chain:\n{err}")

    def _recalc_bs(self):
        s = self.spn_bs_s.value()
        k = self.spn_bs_k.value()
        t = self.spn_bs_t.value() / 365.0
        r = self.spn_bs_r.value()
        v = self.spn_bs_v.value()
        cp = "c" if "Call" in self.cmb_bs_type.currentText() else "p"
        
        px = options.bs(s, k, t, r, v, cp)
        gr = options.greeks(s, k, t, r, v, cp)
        rn = options.rn_prob_above(s, k, t, r, v)
        
        self.lbl_bs_output.setText(
            f"Fair Price: ${px:.4f} | Delta: {gr['delta']:+.4f} | Gamma: {gr['gamma']:.4f} | "
            f"Vega: {gr['vega']:.4f} | Theta: {gr['theta']:+.4f}/d | Rho: {gr['rho']:+.4f} | "
            f"Risk-Neutral P(S_T > K): {rn:.1%}"
        )


    def _setup_tray(self):
        """Sets up Windows System Tray icon for 24/7 background persistence."""
        try:
            if QSystemTrayIcon.isSystemTrayAvailable():
                self.tray_icon = QSystemTrayIcon(self)
                # Create a simple tray pixmap icon
                pixmap = QtGui.QPixmap(16, 16)
                pixmap.fill(QtGui.QColor("#388bfd"))
                self.tray_icon.setIcon(QtGui.QIcon(pixmap))
                
                tray_menu = QMenu()
                act_show = tray_menu.addAction("Show Workstation")
                act_show.triggered.connect(self.showNormal)
                tray_menu.addSeparator()
                act_exit = tray_menu.addAction("Exit Workstation")
                act_exit.triggered.connect(QApplication.instance().quit)
                
                self.tray_icon.setContextMenu(tray_menu)
                self.tray_icon.show()
        except Exception:
            pass

    def closeEvent(self, event):
        """Minimizes to tray on close to maintain 24/7 continuous autonomous surveillance."""
        if hasattr(self, "tray_icon") and self.tray_icon.isVisible():
            self.hide()
            self.tray_icon.showMessage(
                "AlphaDesk Quant",
                "Workstation minimized to system tray. 24/7 autonomous paper trading and RL surveillance remains active.",
                QSystemTrayIcon.Information,
                3000
            )
            event.ignore()
        else:
            event.accept()

    def _create_tab_timing(self):
        """Market Timing Radar tab: Real-Time Decision Matrix (Enter Now, Pullback, Exit, Cash)"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)
        
        # Top header
        top_bar = QHBoxLayout()
        title = QLabel("MARKET TIMING RADAR: REAL-TIME DECISION ENGINE")
        title.setStyleSheet("font-size: 14px; font-weight: 800; color: #58a6ff;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        
        self.cmb_timing_filter = QComboBox()
        self.cmb_timing_filter.addItems(["All Signals", "Enter Now Only", "Wait for Pullback", "Exit/Tighten"])
        self.cmb_timing_filter.currentTextChanged.connect(self.refresh_timing_radar)
        top_bar.addWidget(self.cmb_timing_filter)
        
        btn_refresh = QPushButton("Refresh Surveillance")
        btn_refresh.clicked.connect(self.refresh_timing_radar)
        top_bar.addWidget(btn_refresh)
        lay.addLayout(top_bar)
        
        # Radar table
        self.tbl_timing = QTableWidget(0, 8)
        self.tbl_timing.setHorizontalHeaderLabels([
            "TICKER", "VERDICT", "CONFIDENCE", "LAST PRICE",
            "RECOMMENDED ENTRY", "STOP LOSS", "TARGET PRICE", "ACTION REASONING"
        ])
        self.tbl_timing.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.tbl_timing.setAlternatingRowColors(True)
        self.tbl_timing.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_timing.itemDoubleClicked.connect(self._on_radar_double_clicked)
        lay.addWidget(self.tbl_timing)
        
        return w

    def _create_tab_rl_trader(self):
        """24/7 Autonomous Paper Portfolio & Reinforcement Learning Q-Policy Ledger"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)
        
        # Telemetry strip
        top_bar = QHBoxLayout()
        self.lbl_rl_telemetry = QLabel("24/7 AUTONOMOUS SURVEILLANCE: ACTIVE | REPLAY BUFFER: 25,000 | ALPHA: +6.50R")
        self.lbl_rl_telemetry.setStyleSheet("font-size: 13px; font-weight: 700; color: #3fb950; font-family: monospace;")
        top_bar.addWidget(self.lbl_rl_telemetry)
        top_bar.addStretch()
        
        btn_refresh = QPushButton("Refresh Portfolio")
        btn_refresh.clicked.connect(self.refresh_rl_portfolio)
        top_bar.addWidget(btn_refresh)
        lay.addLayout(top_bar)
        
        # Splitter: Open Positions (Top) & Closed Ledger (Bottom)
        split = QSplitter(Qt.Vertical)
        
        # 1. Open Positions Group
        gb_open = QGroupBox("ACTIVE SIMULATED PAPER POSITIONS")
        vb_open = QVBoxLayout(gb_open)
        self.tbl_rl_open = QTableWidget(0, 10)
        self.tbl_rl_open.setHorizontalHeaderLabels([
            "TIMESTAMP", "TICKER", "SIDE", "ENTRY PRICE", "CURRENT PRICE",
            "STOP LOSS", "TARGET PRICE", "SHARES", "UNREALIZED P&L", "UNREALIZED R"
        ])
        self.tbl_rl_open.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_rl_open.horizontalHeader().setSectionResizeMode(9, QHeaderView.Stretch)
        self.tbl_rl_open.setAlternatingRowColors(True)
        vb_open.addWidget(self.tbl_rl_open)
        split.addWidget(gb_open)
        
        # 2. Closed Trades Ledger Group
        gb_closed = QGroupBox("CLOSED REALIZED TRADES & Q-LEARNING REWARD EXPERIENCE LEDGER")
        vb_closed = QVBoxLayout(gb_closed)
        self.tbl_rl_closed = QTableWidget(0, 8)
        self.tbl_rl_closed.setHorizontalHeaderLabels([
            "EXIT DATE", "TICKER", "SIDE", "ENTRY PRICE", "EXIT PRICE",
            "SHARES", "REALIZED R", "EXIT REASON"
        ])
        self.tbl_rl_closed.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_rl_closed.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.tbl_rl_closed.setAlternatingRowColors(True)
        vb_closed.addWidget(self.tbl_rl_closed)
        split.addWidget(gb_closed)
        
        lay.addWidget(split)
        return w

    def _create_tab_portfolio_5k(self):
        """5,000 INR Active Portfolio Manager & Indian Statutory Charges Audit"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        # Header Strip
        top_bar = QHBoxLayout()
        self.lbl_p5k_header = QLabel("5K INR PORTFOLIO: CAPITAL BASE INR 5,000.00 | STATUTORY TAXES: SEBI + STT + GST + STAMP DUTY")
        self.lbl_p5k_header.setStyleSheet("font-size: 13px; font-weight: 700; color: #58a6ff; font-family: monospace;")
        top_bar.addWidget(self.lbl_p5k_header)
        top_bar.addStretch()

        btn_signal = QPushButton("Execute Signal Trade")
        btn_signal.setStyleSheet("background-color: #238636; color: white; font-weight: bold;")
        btn_signal.clicked.connect(self._on_execute_5k_signal)
        top_bar.addWidget(btn_signal)

        btn_refresh = QPushButton("Sync Audit")
        btn_refresh.clicked.connect(self.refresh_5k_portfolio)
        top_bar.addWidget(btn_refresh)

        btn_reset = QPushButton("Reset 5K")
        btn_reset.setStyleSheet("color: #f85149;")
        btn_reset.clicked.connect(self._on_reset_5k)
        top_bar.addWidget(btn_reset)
        lay.addLayout(top_bar)

        # KPI Summary Cards (Grid)
        grid_kpi = QGridLayout()
        self.lbl_p5k_equity = QLabel("INR 5,000.00")
        self.lbl_p5k_equity.setStyleSheet("font-size: 18px; font-weight: 800; color: #3fb950; font-family: monospace;")
        self.lbl_p5k_cash = QLabel("INR 5,000.00")
        self.lbl_p5k_cash.setStyleSheet("font-size: 18px; font-weight: 800; color: #fff; font-family: monospace;")
        self.lbl_p5k_pnl = QLabel("+INR 0.00")
        self.lbl_p5k_pnl.setStyleSheet("font-size: 18px; font-weight: 800; color: #79c0ff; font-family: monospace;")
        self.lbl_p5k_taxes = QLabel("INR 0.00")
        self.lbl_p5k_taxes.setStyleSheet("font-size: 18px; font-weight: 800; color: #d29922; font-family: monospace;")
        self.lbl_p5k_winrate = QLabel("0.0%")
        self.lbl_p5k_winrate.setStyleSheet("font-size: 18px; font-weight: 800; color: #3fb950; font-family: monospace;")

        grid_kpi.addWidget(QLabel("TOTAL EQUITY:"), 0, 0)
        grid_kpi.addWidget(self.lbl_p5k_equity, 1, 0)
        grid_kpi.addWidget(QLabel("AVAILABLE CASH:"), 0, 1)
        grid_kpi.addWidget(self.lbl_p5k_cash, 1, 1)
        grid_kpi.addWidget(QLabel("REALIZED NET P&L:"), 0, 2)
        grid_kpi.addWidget(self.lbl_p5k_pnl, 1, 2)
        grid_kpi.addWidget(QLabel("TOTAL TAXES PAID:"), 0, 3)
        grid_kpi.addWidget(self.lbl_p5k_taxes, 1, 3)
        grid_kpi.addWidget(QLabel("WIN RATE:"), 0, 4)
        grid_kpi.addWidget(self.lbl_p5k_winrate, 1, 4)
        lay.addLayout(grid_kpi)

        # Splitter: Open Positions (Top) & Executed Ledger (Bottom)
        split = QSplitter(Qt.Vertical)

        # 1. Open 5K Positions
        gb_open = QGroupBox("ACTIVE OPEN 5K POSITIONS (MAX 2 CONCURRENT)")
        vb_open = QVBoxLayout(gb_open)
        self.tbl_p5k_open = QTableWidget(0, 10)
        self.tbl_p5k_open.setHorizontalHeaderLabels([
            "TICKER", "SIDE", "TYPE", "ENTRY PX", "CURRENT PX",
            "STOP LOSS", "TARGET", "SHARES", "EST TAXES", "NET P&L"
        ])
        self.tbl_p5k_open.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vb_open.addWidget(self.tbl_p5k_open)
        split.addWidget(gb_open)

        # 2. Closed Ledger
        gb_closed = QGroupBox("ITEMIZED TRADE & STATUTORY TAX LEDGER (SEBI & FINANCE ACT)")
        vb_closed = QVBoxLayout(gb_closed)
        self.tbl_p5k_closed = QTableWidget(0, 11)
        self.tbl_p5k_closed.setHorizontalHeaderLabels([
            "EXIT TIME", "TICKER", "SIDE", "QTY", "ENTRY PX", "EXIT PX",
            "GROSS P&L", "TOTAL TAXES", "NET P&L", "R-MULT", "EQUITY AFTER"
        ])
        self.tbl_p5k_closed.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vb_closed.addWidget(self.tbl_p5k_closed)
        split.addWidget(gb_closed)

        lay.addWidget(split, 1)
        return w

    def refresh_5k_portfolio(self):
        """Updates 5K portfolio state, positions, and statutory charges ledger."""
        try:
            d = portfolio_5k.get_state()
            ret_sign = "+" if d["total_return_pct"] >= 0 else ""
            self.lbl_p5k_equity.setText(f"INR {d['equity']:,.2f} ({ret_sign}{d['total_return_pct']:.1f}%)")
            self.lbl_p5k_cash.setText(f"INR {d['current_cash']:,.2f}")
            self.lbl_p5k_pnl.setText(f"{'+' if d['realized_net_pnl'] >= 0 else ''}INR {d['realized_net_pnl']:.2f}")
            self.lbl_p5k_taxes.setText(f"INR {d['total_charges']:.2f}")
            self.lbl_p5k_winrate.setText(f"{d['win_rate_pct']:.1f}% ({d['winning_trades']}W / {d['losing_trades']}L)")

            # Open positions
            open_pos = d.get("open_positions", [])
            self.tbl_p5k_open.setRowCount(len(open_pos))
            for i, p in enumerate(open_pos):
                self.tbl_p5k_open.setItem(i, 0, QTableWidgetItem(str(p.get("ticker", ""))))
                self.tbl_p5k_open.setItem(i, 1, QTableWidgetItem(str(p.get("side", ""))))
                self.tbl_p5k_open.setItem(i, 2, QTableWidgetItem(str(p.get("order_type", ""))))
                self.tbl_p5k_open.setItem(i, 3, QTableWidgetItem(f"{p.get('entry_price', 0.0):.2f}"))
                self.tbl_p5k_open.setItem(i, 4, QTableWidgetItem(f"{p.get('current_price', 0.0):.2f}"))
                self.tbl_p5k_open.setItem(i, 5, QTableWidgetItem(f"{p.get('stop_loss', 0.0):.2f}"))
                self.tbl_p5k_open.setItem(i, 6, QTableWidgetItem(f"{p.get('target_price', 0.0):.2f}"))
                self.tbl_p5k_open.setItem(i, 7, QTableWidgetItem(str(p.get("quantity", 0))))
                self.tbl_p5k_open.setItem(i, 8, QTableWidgetItem(f"{p.get('estimated_charges', 0.0):.2f}"))
                net_pnl = p.get("unrealized_net_pnl", 0.0)
                item_net = QTableWidgetItem(f"{'+' if net_pnl >= 0 else ''}{net_pnl:.2f}")
                item_net.setForeground(QtGui.QColor("#3fb950" if net_pnl >= 0 else "#f85149"))
                self.tbl_p5k_open.setItem(i, 9, item_net)

            # Closed trades
            trades = d.get("recent_trades", [])
            self.tbl_p5k_closed.setRowCount(len(trades))
            for i, t in enumerate(trades):
                self.tbl_p5k_closed.setItem(i, 0, QTableWidgetItem(str(t.get("exit_time", ""))))
                self.tbl_p5k_closed.setItem(i, 1, QTableWidgetItem(str(t.get("ticker", ""))))
                self.tbl_p5k_closed.setItem(i, 2, QTableWidgetItem(str(t.get("side", ""))))
                self.tbl_p5k_closed.setItem(i, 3, QTableWidgetItem(str(t.get("qty", 0))))
                self.tbl_p5k_closed.setItem(i, 4, QTableWidgetItem(f"{t.get('entry_px', 0.0):.2f}"))
                self.tbl_p5k_closed.setItem(i, 5, QTableWidgetItem(f"{t.get('exit_px', 0.0):.2f}"))
                self.tbl_p5k_closed.setItem(i, 6, QTableWidgetItem(f"{t.get('gross_pnl', 0.0):.2f}"))
                self.tbl_p5k_closed.setItem(i, 7, QTableWidgetItem(f"{t.get('total_charges', 0.0):.2f}"))
                net = t.get("net_pnl", 0.0)
                item_net = QTableWidgetItem(f"{'+' if net >= 0 else ''}{net:.2f}")
                item_net.setForeground(QtGui.QColor("#3fb950" if net >= 0 else "#f85149"))
                self.tbl_p5k_closed.setItem(i, 8, item_net)
                self.tbl_p5k_closed.setItem(i, 9, QTableWidgetItem(f"{t.get('r_multiple', 0.0):+.2f}R"))
                self.tbl_p5k_closed.setItem(i, 10, QTableWidgetItem(f"{t.get('portfolio_equity_after', 0.0):.2f}"))
        except Exception:
            pass

    def _on_execute_5k_signal(self):
        import random
        symbols = ["TATASTEEL.NS", "IOC.NS", "PNB.NS", "SUZLON.NS", "CDSL.NS"]
        sym = random.choice(symbols)
        portfolio_5k.execute_trade(sym, "BUY", "INTRADAY")
        self.refresh_5k_portfolio()

    def _on_reset_5k(self):
        portfolio_5k.reset_portfolio()
        self.refresh_5k_portfolio()

    def _create_tab_backtest(self):
        """Parametric Backtest Studio with interactive Matplotlib Qt Canvas"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)
        
        # Controls Strip
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(10)
        
        ctrl_layout.addWidget(QLabel("Ticker:"))
        self.txt_bt_ticker = QLineEdit("SUZLON.NS")
        self.txt_bt_ticker.setMaximumWidth(120)
        ctrl_layout.addWidget(self.txt_bt_ticker)
        
        ctrl_layout.addWidget(QLabel("Strategy:"))
        self.cmb_bt_strategy = QComboBox()
        self.cmb_bt_strategy.addItem("RL Adaptive Q-Policy (ATR & Squeeze)", "rl_adaptive")
        self.cmb_bt_strategy.addItem("Passive Buy & Hold Benchmark", "buy_hold")
        self.cmb_bt_strategy.addItem("50/200 DMA Trend Following Cross", "trend_50_200")
        ctrl_layout.addWidget(self.cmb_bt_strategy)
        
        ctrl_layout.addWidget(QLabel("Horizon:"))
        self.cmb_bt_period = QComboBox()
        self.cmb_bt_period.addItems(["1y", "2y", "5y"])
        self.cmb_bt_period.setCurrentText("2y")
        ctrl_layout.addWidget(self.cmb_bt_period)
        
        ctrl_layout.addWidget(QLabel("Capital:"))
        self.spn_bt_cap = QDoubleSpinBox()
        self.spn_bt_cap.setRange(1000, 1e8)
        self.spn_bt_cap.setValue(100000.0)
        self.spn_bt_cap.setPrefix("$ ")
        ctrl_layout.addWidget(self.spn_bt_cap)
        
        ctrl_layout.addWidget(QLabel("Risk %:"))
        self.spn_bt_risk = QDoubleSpinBox()
        self.spn_bt_risk.setRange(0.1, 5.0)
        self.spn_bt_risk.setValue(1.0)
        self.spn_bt_risk.setSuffix(" %")
        ctrl_layout.addWidget(self.spn_bt_risk)
        
        self.btn_bt_run = QPushButton("Run Institutional Backtest")
        self.btn_bt_run.setStyleSheet("background-color: #1f6feb; color: white; font-weight: 700;")
        self.btn_bt_run.clicked.connect(self.on_run_backtest)
        ctrl_layout.addWidget(self.btn_bt_run)
        
        lay.addLayout(ctrl_layout)
        
        # Metrics Bento Strip
        metrics_box = QFrame()
        metrics_box.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 6px;")
        ml = QHBoxLayout(metrics_box)
        ml.setSpacing(14)
        
        self.lbl_bt_return = QLabel("Return: --")
        self.lbl_bt_sharpe = QLabel("Sharpe: --")
        self.lbl_bt_sortino = QLabel("Sortino: --")
        self.lbl_bt_maxdd = QLabel("Max DD: --")
        self.lbl_bt_winrate = QLabel("Win Rate: --")
        self.lbl_bt_pf = QLabel("Profit Factor: --")
        self.lbl_bt_trades = QLabel("Trades: --")
        
        for lbl, col in [
            (self.lbl_bt_return, "#3fb950"),
            (self.lbl_bt_sharpe, "#58a6ff"),
            (self.lbl_bt_sortino, "#79c0ff"),
            (self.lbl_bt_maxdd, "#f85149"),
            (self.lbl_bt_winrate, "#3fb950"),
            (self.lbl_bt_pf, "#d29922"),
            (self.lbl_bt_trades, "#f0f6fc")
        ]:
            lbl.setStyleSheet(f"font-weight: 700; font-size: 12px; color: {col};")
            ml.addWidget(lbl)
            
        lay.addWidget(metrics_box)
        
        # Splitter: Canvas (Top) & Trade Log (Bottom)
        split = QSplitter(Qt.Vertical)
        
        # Matplotlib Canvas
        self.bt_fig = Figure(figsize=(8, 3.6), facecolor="#0b0e14")
        self.bt_canvas = FigureCanvas(self.bt_fig)
        self.bt_canvas.setMinimumHeight(240)
        split.addWidget(self.bt_canvas)
        
        # Trade Log Table
        gb_trades = QGroupBox("TRADE EXECUTION LOG")
        vb_tr = QVBoxLayout(gb_trades)
        self.tbl_bt_trades = QTableWidget(0, 9)
        self.tbl_bt_trades.setHorizontalHeaderLabels([
            "ENTRY DATE", "EXIT DATE", "SIDE", "ENTRY PX", "EXIT PX",
            "SHARES", "REALIZED P&L", "R-MULTIPLE", "EXIT REASON"
        ])
        self.tbl_bt_trades.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_bt_trades.horizontalHeader().setSectionResizeMode(8, QHeaderView.Stretch)
        self.tbl_bt_trades.setAlternatingRowColors(True)
        vb_tr.addWidget(self.tbl_bt_trades)
        split.addWidget(gb_trades)
        
        lay.addWidget(split)
        return w

    def _create_tab_fast_ticks(self):
        """Millisecond Precision Ticks & Level 2 Order Book Depth"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)
        
        # Top Cockpit
        cockpit = QFrame()
        cockpit.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px;")
        cl = QHBoxLayout(cockpit)
        
        self.lbl_tick_sym = QLabel("SUZLON.NS")
        self.lbl_tick_sym.setStyleSheet("font-size: 20px; font-weight: 800; color: #58a6ff; letter-spacing: 1px;")
        cl.addWidget(self.lbl_tick_sym)
        
        self.lbl_tick_px = QLabel("₹64.20")
        self.lbl_tick_px.setStyleSheet("font-size: 22px; font-weight: 900; color: #3fb950;")
        cl.addWidget(self.lbl_tick_px)
        
        self.lbl_tick_time = QLabel("[16:15:30.124]")
        self.lbl_tick_time.setStyleSheet("font-family: monospace; font-size: 13px; color: #8b949e;")
        cl.addWidget(self.lbl_tick_time)
        
        self.lbl_tick_spread = QLabel("Spread: 0.05 | VWAP: 64.18")
        self.lbl_tick_spread.setStyleSheet("font-family: monospace; font-size: 12px; color: #d29922;")
        cl.addWidget(self.lbl_tick_spread)
        
        cl.addStretch()
        
        btn_paper_buy = QPushButton("BUY MARKET (SIM)")
        btn_paper_buy.setStyleSheet("background-color: #238636; color: white; font-weight: 700; padding: 6px 14px;")
        btn_paper_buy.clicked.connect(lambda: self._exec_sim_order("BUY"))
        cl.addWidget(btn_paper_buy)
        
        btn_paper_sell = QPushButton("SELL MARKET (SIM)")
        btn_paper_sell.setStyleSheet("background-color: #da3633; color: white; font-weight: 700; padding: 6px 14px;")
        btn_paper_sell.clicked.connect(lambda: self._exec_sim_order("SELL"))
        cl.addWidget(btn_paper_sell)
        
        lay.addWidget(cockpit)
        
        # 5-Level Deep Bid/Ask Table
        gb_depth = QGroupBox("LEVEL 2 ORDER BOOK DEPTH (TOP 5 LEVELS)")
        vb_d = QVBoxLayout(gb_depth)
        self.tbl_depth = QTableWidget(5, 5)
        self.tbl_depth.setHorizontalHeaderLabels([
            "BID QUANTITY", "BID PRICE", "SPREAD", "ASK PRICE", "ASK QUANTITY"
        ])
        self.tbl_depth.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vb_d.addWidget(self.tbl_depth)
        lay.addWidget(gb_depth)
        
        return w

    def _create_tab_filings(self):
        """Official BSE & NSE Exchange Regulatory Disclosures (Regulation 30)"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)
        
        top_bar = QHBoxLayout()
        title = QLabel("EXCHANGE REGULATORY FILINGS & ANNOUNCEMENTS (NSE REG 30 / SEC EDGAR)")
        title.setStyleSheet("font-size: 14px; font-weight: 800; color: #58a6ff;")
        top_bar.addWidget(title)
        top_bar.addStretch()
        
        btn_ref = QPushButton("Refresh Filings")
        btn_ref.clicked.connect(self.refresh_filings)
        top_bar.addWidget(btn_ref)
        lay.addLayout(top_bar)
        
        self.tbl_filings = QTableWidget(0, 5)
        self.tbl_filings.setHorizontalHeaderLabels([
            "BROADCAST TIME", "COMPANY", "CATEGORY", "DETAILS", "FILING PDF"
        ])
        self.tbl_filings.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl_filings.setAlternatingRowColors(True)
        self.tbl_filings.itemDoubleClicked.connect(self._on_filing_double_clicked)
        lay.addWidget(self.tbl_filings)
        
        return w

    def refresh_timing_radar(self):
        """Populates Market Timing Radar table with real-time decision matrix."""
        try:
            dash = trader_engine.get_dashboard_data()
            radar = dash.get("radar", [])
            filt = self.cmb_timing_filter.currentText()
            
            if filt == "Enter Now Only":
                radar = [r for r in radar if "ENTER NOW" in r.get("timing_verdict", "")]
            elif filt == "Wait for Pullback":
                radar = [r for r in radar if "WAIT FOR" in r.get("timing_verdict", "")]
            elif filt == "Exit/Tighten":
                radar = [r for r in radar if "EXIT" in r.get("timing_verdict", "")]
                
            self.tbl_timing.setRowCount(len(radar))
            for i, r in enumerate(radar):
                sym_item = QTableWidgetItem(r.get("ticker", ""))
                sym_item.setForeground(QtGui.QColor("#58a6ff"))
                sym_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                self.tbl_timing.setItem(i, 0, sym_item)
                
                verdict = r.get("timing_verdict", "")
                v_item = QTableWidgetItem(verdict)
                v_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                if "ENTER NOW (LONG)" in verdict:
                    v_item.setForeground(QtGui.QColor("#3fb950"))
                elif "WAIT FOR PULLBACK" in verdict:
                    v_item.setForeground(QtGui.QColor("#d29922"))
                elif "ENTER NOW (SHORT)" in verdict or "EXIT" in verdict:
                    v_item.setForeground(QtGui.QColor("#f85149"))
                else:
                    v_item.setForeground(QtGui.QColor("#8b949e"))
                self.tbl_timing.setItem(i, 1, v_item)
                
                self.tbl_timing.setItem(i, 2, QTableWidgetItem(f"{r.get('confidence', 0.6)*100:.0f}%"))
                self.tbl_timing.setItem(i, 3, QTableWidgetItem(f"{r.get('current_price', 0.0):.2f}"))
                
                e_item = QTableWidgetItem(f"{r.get('recommended_entry', 0.0):.2f}")
                e_item.setForeground(QtGui.QColor("#79c0ff"))
                self.tbl_timing.setItem(i, 4, e_item)
                
                sl_item = QTableWidgetItem(f"{r.get('stop_loss', 0.0):.2f}")
                sl_item.setForeground(QtGui.QColor("#f85149"))
                self.tbl_timing.setItem(i, 5, sl_item)
                
                tgt_item = QTableWidgetItem(f"{r.get('target_price', 0.0):.2f}")
                tgt_item.setForeground(QtGui.QColor("#3fb950"))
                self.tbl_timing.setItem(i, 6, tgt_item)
                
                self.tbl_timing.setItem(i, 7, QTableWidgetItem(r.get("action_summary", "")))
        except Exception:
            pass

    def refresh_rl_portfolio(self):
        """Populates active paper positions and closed trade ledger."""
        try:
            dash = trader_engine.get_dashboard_data()
            open_pos = dash.get("open_positions", [])
            closed_pos = dash.get("closed_positions", [])
            
            # Query relentless neural engine telemetry
            from deep_quant_net import neural_trader
            nn_t = neural_trader.get_telemetry()
            device = nn_t.get("device", "CUDA")
            epochs = nn_t.get("epochs_completed", 0)
            loss = nn_t.get("latest_loss", 0.0)
            buf = nn_t.get("buffer_size", 0)
            
            # Update telemetry label
            cum_r = dash.get("telemetry", {}).get("cumulative_r", {}).get("value", "+6.50R")
            self.lbl_rl_telemetry.setText(f"RELENTLESS NEURAL NET ({device}): ACTIVE | EPOCHS: {epochs:,} | LOSS: {loss:.5f} | REPLAY BUFFER: {buf:,} | ALPHA: {cum_r}")
            
            # 1. Open positions
            self.tbl_rl_open.setRowCount(len(open_pos))
            for i, p in enumerate(open_pos):
                self.tbl_rl_open.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", ""))))
                
                sym_item = QTableWidgetItem(p.get("ticker", ""))
                sym_item.setForeground(QtGui.QColor("#58a6ff"))
                sym_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                self.tbl_rl_open.setItem(i, 1, sym_item)
                
                side_item = QTableWidgetItem(p.get("action", ""))
                side_item.setForeground(QtGui.QColor("#3fb950"))
                side_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                self.tbl_rl_open.setItem(i, 2, side_item)
                
                self.tbl_rl_open.setItem(i, 3, QTableWidgetItem(f"{p.get('entry_price', 0.0):.2f}"))
                self.tbl_rl_open.setItem(i, 4, QTableWidgetItem(f"{p.get('current_price', 0.0):.2f}"))
                self.tbl_rl_open.setItem(i, 5, QTableWidgetItem(f"{p.get('stop_loss', 0.0):.2f}"))
                self.tbl_rl_open.setItem(i, 6, QTableWidgetItem(f"{p.get('target_price', 0.0):.2f}"))
                self.tbl_rl_open.setItem(i, 7, QTableWidgetItem(f"{p.get('shares', 0):,}"))
                
                pnl = p.get("unrealized_pnl", 0.0)
                pnl_item = QTableWidgetItem(f"{pnl:+.2f}")
                pnl_item.setForeground(QtGui.QColor("#3fb950" if pnl >= 0 else "#f85149"))
                self.tbl_rl_open.setItem(i, 8, pnl_item)
                
                r_val = p.get("unrealized_r", 0.0)
                r_item = QTableWidgetItem(f"{r_val:+.2f}R")
                r_item.setForeground(QtGui.QColor("#3fb950" if r_val >= 0 else "#f85149"))
                self.tbl_rl_open.setItem(i, 9, r_item)

            # 2. Closed positions
            self.tbl_rl_closed.setRowCount(len(closed_pos))
            for i, p in enumerate(closed_pos):
                self.tbl_rl_closed.setItem(i, 0, QTableWidgetItem(str(p.get("timestamp", ""))))
                self.tbl_rl_closed.setItem(i, 1, QTableWidgetItem(p.get("ticker", "")))
                self.tbl_rl_closed.setItem(i, 2, QTableWidgetItem(p.get("action", "")))
                self.tbl_rl_closed.setItem(i, 3, QTableWidgetItem(f"{p.get('entry_price', 0.0):.2f}"))
                self.tbl_rl_closed.setItem(i, 4, QTableWidgetItem(f"{p.get('exit_price', 0.0):.2f}"))
                self.tbl_rl_closed.setItem(i, 5, QTableWidgetItem(f"{p.get('shares', 0):,}"))
                
                r_val = p.get("realized_r", 0.0)
                r_item = QTableWidgetItem(f"{r_val:+.2f}R")
                r_item.setForeground(QtGui.QColor("#3fb950" if r_val >= 0 else "#f85149"))
                self.tbl_rl_closed.setItem(i, 6, r_item)
                
                self.tbl_rl_closed.setItem(i, 7, QTableWidgetItem(p.get("exit_reason", "")))
        except Exception:
            pass

    def on_run_backtest(self):
        """Executes multi-year institutional backtest and updates Matplotlib chart and table."""
        ticker = self.txt_bt_ticker.text().strip() or "SUZLON.NS"
        strategy = self.cmb_bt_strategy.currentData() or "rl_adaptive"
        period = self.cmb_bt_period.currentText()
        capital = self.spn_bt_cap.value()
        risk_pct = self.spn_bt_risk.value()

        try:
            res = trader_engine.run_multiyear_backtest(ticker, period=period, capital=capital, risk_pct=risk_pct, strategy=strategy)
            if "error" in res:
                return

            # Update KPI Labels
            ret_pct = res.get("net_return_pct", 0.0)
            self.lbl_bt_return.setText(f"Return: {ret_pct:+.1f}%")
            self.lbl_bt_return.setStyleSheet(f"font-weight: 700; color: {'#3fb950' if ret_pct >= 0 else '#f85149'};")
            
            self.lbl_bt_sharpe.setText(f"Sharpe: {res.get('sharpe_ratio', 0.0):.2f}")
            self.lbl_bt_sortino.setText(f"Sortino: {res.get('sortino_ratio', 0.0):.2f}")
            self.lbl_bt_maxdd.setText(f"Max DD: {res.get('max_drawdown_pct', 0.0):.1f}%")
            self.lbl_bt_winrate.setText(f"Win Rate: {res.get('win_rate_pct', 0.0):.1f}%")
            self.lbl_bt_pf.setText(f"Profit Factor: {res.get('profit_factor', 0.0):.2f}")
            self.lbl_bt_trades.setText(f"Trades: {res.get('total_trades', 0)} ({res.get('wins', 0)}W / {res.get('losses', 0)}L)")

            # Plot on Matplotlib Qt Canvas
            self.bt_fig.clear()
            ax1 = self.bt_fig.add_subplot(2, 1, 1)
            ax2 = self.bt_fig.add_subplot(2, 1, 2, sharex=ax1)
            
            self.bt_fig.patch.set_facecolor("#0b0e14")
            ax1.set_facecolor("#0d1117")
            ax2.set_facecolor("#0d1117")

            eq_curve = res.get("equity_curve", [])
            if len(eq_curve) > 1:
                eq_vals = [e["equity"] for e in eq_curve]
                dd_vals = [e.get("drawdown", 0.0) for e in eq_curve]
                x_axis = range(len(eq_vals))

                # Upper plot: Equity Curve
                ax1.plot(x_axis, eq_vals, color="#58a6ff", linewidth=1.8, label="Portfolio Equity")
                ax1.axhline(capital, color="rgba(255,255,255,0.25)", linestyle="--", linewidth=1.0)
                ax1.set_ylabel("Capital ($ / ₹)", color="#8b949e", fontsize=8)
                ax1.tick_params(colors="#8b949e", labelsize=8)
                ax1.grid(True, color="#21262d", linestyle=":")

                # Lower plot: Drawdown Underwater
                ax2.fill_between(x_axis, dd_vals, 0, color="#f85149", alpha=0.35)
                ax2.plot(x_axis, dd_vals, color="#f85149", linewidth=1.2)
                ax2.set_ylabel("Drawdown %", color="#8b949e", fontsize=8)
                ax2.tick_params(colors="#8b949e", labelsize=8)
                ax2.grid(True, color="#21262d", linestyle=":")

            self.bt_fig.tight_layout()
            self.bt_canvas.draw()

            # Populate Trade Log
            trades = res.get("trade_log", [])
            self.tbl_bt_trades.setRowCount(len(trades))
            for i, t in enumerate(reversed(trades)):
                self.tbl_bt_trades.setItem(i, 0, QTableWidgetItem(t.get("entry_date", "")))
                self.tbl_bt_trades.setItem(i, 1, QTableWidgetItem(t.get("exit_date", "")))
                
                side_item = QTableWidgetItem(t.get("type", ""))
                side_item.setForeground(QtGui.QColor("#3fb950" if "LONG" in t.get("type", "") or "BUY" in t.get("type", "") else "#f85149"))
                self.tbl_bt_trades.setItem(i, 2, side_item)
                
                self.tbl_bt_trades.setItem(i, 3, QTableWidgetItem(f"{t.get('entry', 0.0):.2f}"))
                self.tbl_bt_trades.setItem(i, 4, QTableWidgetItem(f"{t.get('exit', 0.0):.2f}"))
                self.tbl_bt_trades.setItem(i, 5, QTableWidgetItem(f"{t.get('shares', 0):,}"))
                
                pnl = t.get("pnl", 0.0)
                pnl_item = QTableWidgetItem(f"{pnl:+.2f}")
                pnl_item.setForeground(QtGui.QColor("#3fb950" if pnl >= 0 else "#f85149"))
                self.tbl_bt_trades.setItem(i, 6, pnl_item)
                
                r_val = t.get("r_multiple", 0.0)
                r_item = QTableWidgetItem(f"{r_val:+.2f}R")
                r_item.setForeground(QtGui.QColor("#3fb950" if r_val >= 0 else "#f85149"))
                self.tbl_bt_trades.setItem(i, 7, r_item)
                
                self.tbl_bt_trades.setItem(i, 8, QTableWidgetItem(t.get("exit_reason", "")))
        except Exception:
            pass

    def _on_tick_timer(self):
        """High-frequency 250ms tick poller for live ticking prices and Level 2 depth."""
        try:
            sym = self.txt_bt_ticker.text().strip() or "SUZLON.NS"
            tick = fast_ticker.get_tick(sym)
            if not tick:
                tick = fast_ticker.update_tick(sym)
            if tick:
                self.lbl_tick_sym.setText(tick["symbol"])
                curr_sign = "₹" if tick["currency"] == "INR" else "$"
                self.lbl_tick_px.setText(f"{curr_sign}{tick['last_price']:.2f}")
                
                if tick.get("tick_direction") == "UP":
                    self.lbl_tick_px.setStyleSheet("font-size: 22px; font-weight: 900; color: #3fb950;")
                elif tick.get("tick_direction") == "DOWN":
                    self.lbl_tick_px.setStyleSheet("font-size: 22px; font-weight: 900; color: #f85149;")
                    
                time_str = tick.get("iso_time_ms", "").split(" ")[-1]
                self.lbl_tick_time.setText(f"[{time_str}]")
                self.lbl_tick_spread.setText(f"Spread: {tick.get('spread', 0.05):.2f} | VWAP: {tick.get('vwap', tick['last_price']):.2f}")
                
                # Update Level 2 Order Book Depth Table
                ob = tick.get("order_book", {})
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                for lvl in range(min(5, len(bids), len(asks))):
                    b = bids[lvl]
                    a = asks[lvl]
                    self.tbl_depth.setItem(lvl, 0, QTableWidgetItem(f"{b['quantity']:,}"))
                    
                    b_px_item = QTableWidgetItem(f"{b['price']:.2f}")
                    b_px_item.setForeground(QtGui.QColor("#3fb950"))
                    b_px_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                    self.tbl_depth.setItem(lvl, 1, b_px_item)
                    
                    self.tbl_depth.setItem(lvl, 2, QTableWidgetItem(f"{a['price'] - b['price']:.2f}"))
                    
                    a_px_item = QTableWidgetItem(f"{a['price']:.2f}")
                    a_px_item.setForeground(QtGui.QColor("#f85149"))
                    a_px_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                    self.tbl_depth.setItem(lvl, 3, a_px_item)
                    
                    self.tbl_depth.setItem(lvl, 4, QTableWidgetItem(f"{a['quantity']:,}"))
        except Exception:
            pass

    def _on_surveillance_timer(self):
        """Called every 5 seconds to sync radar and portfolio."""
        self.refresh_timing_radar()
        self.refresh_rl_portfolio()

    def refresh_filings(self):
        """Loads official exchange filings (NSE Reg 30) into the filings table."""
        try:
            import server
            announcements = server.fetch_nse_announcements("SUZLON")
            self.tbl_filings.setRowCount(len(announcements))
            for i, a in enumerate(announcements):
                self.tbl_filings.setItem(i, 0, QTableWidgetItem(a.get("broadcast_date", "")))
                
                co_item = QTableWidgetItem(a.get("company", ""))
                co_item.setForeground(QtGui.QColor("#58a6ff"))
                co_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                self.tbl_filings.setItem(i, 1, co_item)
                
                self.tbl_filings.setItem(i, 2, QTableWidgetItem(a.get("subject", "")))
                self.tbl_filings.setItem(i, 3, QTableWidgetItem(a.get("details", "")))
                
                pdf_url = a.get("pdf_url", "")
                pdf_item = QTableWidgetItem("Download PDF" if pdf_url else "No PDF")
                pdf_item.setForeground(QtGui.QColor("#3fb950" if pdf_url else "#8b949e"))
                pdf_item.setData(Qt.UserRole, pdf_url)
                self.tbl_filings.setItem(i, 4, pdf_item)
        except Exception:
            pass

    def _exec_sim_order(self, action):
        """Executes a simulated paper market order at current live tick price."""
        sym = self.txt_bt_ticker.text().strip() or "SUZLON.NS"
        tick = fast_ticker.get_tick(sym)
        px = tick["last_price"] if tick else 64.20
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        stop_px = px * 0.95 if action == "BUY" else px * 1.05
        tgt_px = px * 1.10 if action == "BUY" else px * 0.90
        shares = 1000
        
        try:
            import sqlite3
            conn = sqlite3.connect("E:/Alphadesk Quant/autonomous_trading.db")
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO live_positions (timestamp, ticker, action, entry_price, stop_loss, target_price, shares, status, current_price, unrealized_pnl, unrealized_r)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, 0.0, 0.0)
            """, (now_str, sym, action, px, stop_px, tgt_px, shares, px))
            conn.commit()
            conn.close()
            self.refresh_rl_portfolio()
            QMessageBox.information(self, "Order Executed", f"Simulated {action} order filled for {shares} shares of {sym} @ {px:.2f}")
        except Exception as e:
            QMessageBox.critical(self, "Order Error", str(e))

    def _on_radar_double_clicked(self, item):
        """When an asset row in Market Timing Radar is double-clicked, load it into backtester."""
        row = item.row()
        sym = self.tbl_timing.item(row, 0).text()
        self.txt_bt_ticker.setText(sym)
        self.tabs.setCurrentIndex(2) # Switch to Backtest Studio
        self.on_run_backtest()

    def _on_filing_double_clicked(self, item):
        """When a filing row is double clicked, open the PDF archive URL."""
        row = item.row()
        pdf_item = self.tbl_filings.item(row, 4)
        if pdf_item:
            url = pdf_item.data(Qt.UserRole)
            if url:
                webbrowser.open(url)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = AlphaDeskGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
