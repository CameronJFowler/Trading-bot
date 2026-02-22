from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

TRADING_DAYS = 365
MAX_EXPOSURE = 0.30
COST_BPS = 5
VOL_WINDOW = 20
TARGET_ANN_VOL = 0.20
MIN_DAILY_VOL = 1e-6

DEFAULT_FAST = 10
DEFAULT_SLOW = 50
DEFAULT_LOOKBACK = 180
DEFAULT_RIDGE_ALPHA = 10.0


# -----------------------------
# Data
# -----------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column")
    if "Close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated()]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(
            "No valid rows found after cleaning Date/Close columns")

    return df


# -----------------------------
# Feature engineering + ML
# -----------------------------

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    close = df["Close"].astype(float)
    ret_1 = close.pct_change()

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = ret_1
    feat["ret_3"] = close.pct_change(3)
    feat["ret_7"] = close.pct_change(7)
    feat["ret_14"] = close.pct_change(14)

    ma_10 = close.rolling(10).mean()
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    feat["trend_10_20"] = (ma_10 / ma_20) - 1.0
    feat["trend_20_50"] = (ma_20 / ma_50) - 1.0

    vol_10 = ret_1.rolling(10).std()
    vol_30 = ret_1.rolling(30).std()
    feat["vol_10"] = vol_10
    feat["vol_ratio_10_30"] = vol_10 / vol_30

    z20 = (close - ma_20) / close.rolling(20).std()
    feat["zscore_20"] = z20.replace([np.inf, -np.inf], np.nan)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    target = ret_1.shift(-1).reindex(feat.index)

    aligned = feat.join(target.rename("target")).dropna()
    return aligned.drop(columns=["target"]), aligned["target"]


def fit_ridge_and_predict_next(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_next: np.ndarray,
    alpha: float,
) -> float:
    if len(x_train) == 0:
        return 0.0

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    x_scaled = (x_train - x_mean) / x_std

    y_mean = y_train.mean()
    y_centered = y_train - y_mean

    xtx = x_scaled.T @ x_scaled
    ridge = alpha * np.eye(xtx.shape[0])
    beta = np.linalg.solve(xtx + ridge, x_scaled.T @ y_centered)

    x_next_scaled = (x_next - x_mean) / x_std
    y_pred = float(y_mean + (x_next_scaled @ beta))
    return y_pred


def generate_ml_signals(
    df: pd.DataFrame,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> tuple[pd.Series, pd.Series]:
    if lookback < 80:
        raise ValueError("lookback must be at least 80 for stable ML fitting")
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be greater than 0")

    features, target = build_feature_matrix(df)

    if len(features) <= lookback:
        raise ValueError(
            "Not enough data for ML strategy. Provide more history.")

    pred = pd.Series(index=features.index, dtype=float)

    for i in range(lookback, len(features)):
        x_train = features.iloc[i - lookback:i].values
        y_train = target.iloc[i - lookback:i].values
        x_next = features.iloc[i].values
        pred.iloc[i] = fit_ridge_and_predict_next(
            x_train, y_train, x_next, alpha=ridge_alpha)

    pred = pred.reindex(df.index).fillna(0.0)
    confidence = pred.abs().rolling(20, min_periods=1).mean().replace(0.0, np.nan)
    score = (pred / confidence).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    raw_signal = np.tanh(2.2 * score)
    raw_signal = raw_signal.clip(-1.0, 1.0)

    return raw_signal.astype(float), pred.astype(float)


# -----------------------------
# Signals
# -----------------------------

def generate_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype="int8")

    if fast < 1 or slow < 1:
        raise ValueError("SMA windows must be positive integers")
    if fast >= slow:
        raise ValueError("Fast SMA must be less than Slow SMA")

    close = df["Close"].astype(float)
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()

    pos = pd.Series(0.0, index=df.index, dtype=float)
    pos[ma_fast > ma_slow] = 0.8
    pos[1.2 * ma_fast < ma_slow] = -1.0

    return pos.fillna(0.0).clip(-1, 1)


def generate_advanced_hybrid_signals(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    *,
    ml_lookback: int = DEFAULT_LOOKBACK,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> tuple[pd.Series, pd.Series]:
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    trend_signal = generate_signals(df, fast=fast, slow=slow)

    # Mean-reversion sleeve
    ma_20 = close.rolling(20).mean()
    sd_20 = close.rolling(20).std().replace(0.0, np.nan)
    z = ((close - ma_20) /
         sd_20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meanrev_signal = (-z / 2.5).clip(-1.0, 1.0)

    # Breakout sleeve
    high_55 = close.rolling(55).max()
    low_55 = close.rolling(55).min()
    breakout = pd.Series(0.0, index=df.index)
    breakout[close >= high_55] = 1.0
    breakout[close <= low_55] = -1.0
    breakout = breakout.replace(0.0, np.nan).ffill().fillna(0.0)

    # ML sleeve
    ml_signal, ml_pred = generate_ml_signals(
        df, lookback=ml_lookback, ridge_alpha=ridge_alpha)

    # Regime filter: reduce risk in crash/high-volatility states
    vol = ret.rolling(20).std().fillna(ret.std())
    vol_rank = vol.rolling(252, min_periods=30).rank(pct=True)
    crash = close / close.rolling(100).max() - 1.0
    regime_scale = pd.Series(1.0, index=df.index)
    regime_scale[(vol_rank > 0.90) | (crash < -0.18)] = 0.35
    regime_scale[(vol_rank > 0.75) | (crash < -0.10)] = 0.60

    combined = (
        0.35 * trend_signal
        + 0.20 * meanrev_signal
        + 0.20 * breakout
        + 0.25 * ml_signal
    )

    combined = (combined * regime_scale).clip(-1.0, 1.0).fillna(0.0)
    return combined, ml_pred


# -----------------------------
# Strategy pipeline
# -----------------------------

def strategy_pipeline(
    df: pd.DataFrame,
    raw_position: pd.Series,
    *,
    max_exposure: float = MAX_EXPOSURE,
    cost_bps: float = COST_BPS,
    vol_window: int = VOL_WINDOW,
    target_ann_vol: float = TARGET_ANN_VOL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if vol_window < 2:
        raise ValueError("vol_window must be at least 2")
    if max_exposure <= 0:
        raise ValueError("max_exposure must be greater than 0")
    if target_ann_vol <= 0:
        raise ValueError("target_ann_vol must be greater than 0")

    close = df["Close"].astype(float)
    returns = close.pct_change().fillna(0.0)

    vol = returns.rolling(vol_window).std()
    vol = vol.replace(0.0, np.nan).bfill().ffill().clip(lower=MIN_DAILY_VOL)

    target_daily_vol = target_ann_vol / np.sqrt(TRADING_DAYS)
    vol_scale = (target_daily_vol /
                 vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    effective_exposure = raw_position.astype(float) * vol_scale
    effective_exposure = effective_exposure.clip(
        lower=-max_exposure, upper=max_exposure).fillna(0.0)

    turnover = effective_exposure.diff().abs().fillna(0.0)
    cost = turnover * (cost_bps / 10_000.0)

    strat_returns = returns * effective_exposure.shift(1).fillna(0.0) - cost
    equity = (1.0 + strat_returns).cumprod()

    return effective_exposure, strat_returns, equity


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(df: pd.DataFrame, raw_position: pd.Series) -> dict[str, float]:
    effective_exposure, strat_returns, equity = strategy_pipeline(
        df, raw_position)

    total_return = float(equity.iloc[-1] - 1.0)
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_dd = float(drawdown.min())

    mean = float(strat_returns.mean())
    std = float(strat_returns.std())
    sharpe = (mean / std) * np.sqrt(TRADING_DAYS) if std > 0 else 0.0

    periods = len(equity)
    ann_return = float(
        equity.iloc[-1] ** (TRADING_DAYS / periods) - 1.0) if periods > 1 else 0.0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.inf

    downside = float(strat_returns[strat_returns < 0].std())
    sortino = (mean / downside) * \
        np.sqrt(TRADING_DAYS) if downside > 0 else 0.0

    win_rate = float((strat_returns > 0).mean())
    signal_exposure = float((raw_position != 0).mean())
    avg_abs_exposure = float(np.abs(effective_exposure).mean())

    return {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Total Return": total_return,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Win Rate": win_rate,
        "Signal Exposure": signal_exposure,
        "Avg Abs Exposure": avg_abs_exposure,
    }


# -----------------------------
# Trades table
# -----------------------------

def build_trades(df: pd.DataFrame, raw_position: pd.Series) -> pd.DataFrame:
    close = df["Close"].astype(float)
    effective_exposure, strat_returns, _ = strategy_pipeline(df, raw_position)

    trades: list[dict] = []
    in_trade = False
    side = 0.0
    entry_date = None
    entry_price = None

    trade_equity = 1.0
    peak_equity = 1.0
    max_dd_trade = 0.0

    for i in range(len(raw_position)):
        date = raw_position.index[i]
        sig = float(raw_position.iloc[i])
        price = float(close.iloc[i])
        daily_r = float(strat_returns.iloc[i])

        if in_trade:
            trade_equity *= (1.0 + daily_r)
            peak_equity = max(peak_equity, trade_equity)
            dd = (trade_equity / peak_equity) - 1.0
            max_dd_trade = min(max_dd_trade, dd)

        if (not in_trade) and abs(sig) > 1e-9:
            in_trade = True
            side = np.sign(sig)
            entry_date = date
            entry_price = price
            trade_equity = 1.0
            peak_equity = 1.0
            max_dd_trade = 0.0

        elif in_trade and (abs(sig) < 1e-9 or np.sign(sig) != side):
            exit_date = date
            exit_price = price

            trades.append({
                "Side": "LONG" if side > 0 else "SHORT",
                "EntryDate": entry_date,
                "EntryPrice": entry_price,
                "ExitDate": exit_date,
                "ExitPrice": exit_price,
                "PnL": trade_equity - 1.0,
                "Trade Max DD": max_dd_trade,
                "Avg Exposure": float(np.mean(np.abs(effective_exposure.loc[entry_date:exit_date]))),
            })
            in_trade = False

    if in_trade and entry_date is not None and entry_price is not None:
        exit_date = raw_position.index[-1]
        exit_price = float(close.iloc[-1])
        trades.append({
            "Side": "LONG" if side > 0 else "SHORT",
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExitDate": exit_date,
            "ExitPrice": exit_price,
            "PnL": trade_equity - 1.0,
            "Trade Max DD": max_dd_trade,
            "Avg Exposure": float(np.mean(np.abs(effective_exposure.loc[entry_date:exit_date]))),
        })

    return pd.DataFrame(trades)


# -----------------------------
# GUI
# -----------------------------

class TradingBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("⚡ QuantGlow BTC Strategy Studio")
        self.geometry("1260x860")
        self.minsize(1060, 720)
        self.configure(bg="#0b1020")

        self.df: pd.DataFrame | None = None
        self.trades: pd.DataFrame | None = None
        self.metrics: dict[str, float] | None = None
        self.equity_curve: pd.Series | None = None
        self.ml_prediction: pd.Series | None = None

        self.metric_value_labels: dict[str, ttk.Label] = {}
        self.status_var = tk.StringVar(
            value="Ready — load CSV, pick strategy, and run.")

        self._setup_style()
        self.build_ui()

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        bg = "#0b1020"
        panel = "#121a2f"
        card = "#1a2645"
        accent = "#33d1ff"
        fg = "#e6eefc"
        muted = "#9fb0cf"

        style.configure("Root.TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Card.TFrame", background=card, relief="flat")
        style.configure("Title.TLabel", background=bg,
                        foreground=fg, font=("Segoe UI", 20, "bold"))
        style.configure("SubTitle.TLabel", background=bg,
                        foreground=muted, font=("Segoe UI", 10))
        style.configure("Label.TLabel", background=panel,
                        foreground=fg, font=("Segoe UI", 10))
        style.configure("CardName.TLabel", background=card,
                        foreground=muted, font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background=card,
                        foreground=fg, font=("Segoe UI", 14, "bold"))

        style.configure("Glow.TButton", font=("Segoe UI", 10, "bold"),
                        padding=8, background=accent, foreground="#071220", borderwidth=0)
        style.map("Glow.TButton", background=[
                  ("active", "#61ddff"), ("disabled", "#4a5a78")])
        style.configure("Secondary.TButton", font=("Segoe UI", 10),
                        padding=7, background="#2d3b61", foreground=fg)
        style.map("Secondary.TButton", background=[("active", "#3a4a74")])

        style.configure("Dark.TEntry", fieldbackground="#0f172a", foreground=fg, insertcolor=fg,
                        bordercolor="#2b3a61", lightcolor="#2b3a61", darkcolor="#2b3a61", padding=5)
        style.configure("Dark.TCombobox",
                        fieldbackground="#0f172a", foreground=fg)

        style.configure("Dark.TNotebook", background=panel, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background="#26365c",
                        foreground=fg, padding=(15, 8))
        style.map("Dark.TNotebook.Tab", background=[
                  ("selected", accent)], foreground=[("selected", "#071220")])

        style.configure("Dark.Treeview", background="#111a33", fieldbackground="#111a33", foreground=fg,
                        rowheight=28, bordercolor="#253457", borderwidth=0)
        style.configure("Dark.Treeview.Heading", background="#253457",
                        foreground=fg, font=("Segoe UI", 10, "bold"))
        style.map("Dark.Treeview", background=[("selected", "#33507f")])

    def build_ui(self):
        root = ttk.Frame(self, style="Root.TFrame", padding=16)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root, style="Root.TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="⚡ QuantGlow BTC Strategy Studio",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Hybrid alpha engine: trend + breakout + mean-reversion + rolling ML",
                  style="SubTitle.TLabel").pack(anchor="w")

        control_panel = ttk.Frame(root, style="Panel.TFrame", padding=12)
        control_panel.pack(fill=tk.X, pady=(0, 12))

        self.path_var = tk.StringVar()
        ttk.Label(control_panel, text="CSV Path", style="Label.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(control_panel, textvariable=self.path_var, width=60,
                  style="Dark.TEntry").grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(control_panel, text="Browse", command=self.browse,
                   style="Secondary.TButton").grid(row=0, column=2, padx=4)

        ttk.Label(control_panel, text="Strategy", style="Label.TLabel").grid(
            row=0, column=3, padx=(12, 6))
        self.strategy_var = tk.StringVar(value="Advanced ML Hybrid")
        ttk.Combobox(control_panel, textvariable=self.strategy_var, state="readonly", width=22,
                     values=["Advanced ML Hybrid", "SMA Trend Only"], style="Dark.TCombobox").grid(row=0, column=4, sticky="w")

        self.fast_var = tk.StringVar(value=str(DEFAULT_FAST))
        self.slow_var = tk.StringVar(value=str(DEFAULT_SLOW))
        self.lookback_var = tk.StringVar(value=str(DEFAULT_LOOKBACK))
        self.alpha_var = tk.StringVar(value=str(DEFAULT_RIDGE_ALPHA))

        ttk.Label(control_panel, text="Fast SMA", style="Label.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(control_panel, textvariable=self.fast_var, width=10,
                  style="Dark.TEntry").grid(row=1, column=1, sticky="w", pady=(10, 0))
        ttk.Label(control_panel, text="Slow SMA", style="Label.TLabel").grid(
            row=1, column=1, sticky="w", padx=(120, 0), pady=(10, 0))
        ttk.Entry(control_panel, textvariable=self.slow_var, width=10, style="Dark.TEntry").grid(
            row=1, column=1, sticky="w", padx=(200, 0), pady=(10, 0))

        ttk.Label(control_panel, text="ML Lookback", style="Label.TLabel").grid(
            row=1, column=3, sticky="w", pady=(10, 0))
        ttk.Entry(control_panel, textvariable=self.lookback_var, width=12,
                  style="Dark.TEntry").grid(row=1, column=4, sticky="w", pady=(10, 0))
        ttk.Label(control_panel, text="Ridge α", style="Label.TLabel").grid(
            row=1, column=4, sticky="w", padx=(120, 0), pady=(10, 0))
        ttk.Entry(control_panel, textvariable=self.alpha_var, width=12, style="Dark.TEntry").grid(
            row=1, column=4, sticky="w", padx=(180, 0), pady=(10, 0))

        ttk.Button(control_panel, text="Run Strategy", command=self.run,
                   style="Glow.TButton").grid(row=1, column=5, padx=4, pady=(10, 0))
        ttk.Button(control_panel, text="Export Trades", command=self.export_trades,
                   style="Secondary.TButton").grid(row=1, column=6, padx=4, pady=(10, 0))
        control_panel.columnconfigure(1, weight=1)

        cards_container = ttk.Frame(root, style="Root.TFrame")
        cards_container.pack(fill=tk.X, pady=(0, 12))

        self.card_order = [
            "Sharpe Ratio",
            "Sortino Ratio",
            "Total Return",
            "Max Drawdown",
            "Calmar Ratio",
            "Win Rate",
            "Signal Exposure",
            "Avg Abs Exposure",
        ]
        for i, key in enumerate(self.card_order):
            card = ttk.Frame(cards_container, style="Card.TFrame", padding=12)
            card.grid(row=0, column=i, sticky="nsew", padx=4)
            ttk.Label(card, text=key, style="CardName.TLabel").pack(anchor="w")
            value_lbl = ttk.Label(card, text="—", style="CardValue.TLabel")
            value_lbl.pack(anchor="w", pady=(6, 0))
            self.metric_value_labels[key] = value_lbl
            cards_container.columnconfigure(i, weight=1)

        notebook = ttk.Notebook(root, style="Dark.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True)

        trades_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        chart_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        notebook.add(trades_tab, text="Trades")
        notebook.add(chart_tab, text="Equity Curve")

        cols = ("Side", "EntryDate", "EntryPrice", "ExitDate",
                "ExitPrice", "PnL", "Trade Max DD", "Avg Exposure")
        self.tree = ttk.Treeview(
            trades_tab, columns=cols, show="headings", style="Dark.Treeview")
        widths = {"Side": 80, "EntryDate": 140, "EntryPrice": 95, "ExitDate": 140,
                  "ExitPrice": 95, "PnL": 110, "Trade Max DD": 110, "Avg Exposure": 110}
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col], anchor="center")

        self.tree.tag_configure("long", foreground="#7af5a1")
        self.tree.tag_configure("short", foreground="#ff9aa2")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.chart_canvas = tk.Canvas(
            chart_tab, bg="#0f172a", highlightthickness=0)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)
        self.chart_canvas.bind(
            "<Configure>", lambda _event: self.draw_equity_curve())

        status_frame = ttk.Frame(root, style="Panel.TFrame", padding=(12, 6))
        status_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(status_frame, textvariable=self.status_var,
                  style="Label.TLabel").pack(anchor="w")

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.df = load_csv(path)
            self.path_var.set(path)
            self.status_var.set(f"Loaded {len(self.df):,} rows from {path}")
            messagebox.showinfo("CSV Loaded", "Dataset loaded successfully.")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def run(self):
        if self.df is None:
            messagebox.showwarning(
                "Missing data", "Load a CSV file before running the strategy.")
            return

        try:
            fast = int(self.fast_var.get().strip())
            slow = int(self.slow_var.get().strip())
            ml_lookback = int(self.lookback_var.get().strip())
            ridge_alpha = float(self.alpha_var.get().strip())

            if self.strategy_var.get() == "Advanced ML Hybrid":
                raw_position, self.ml_prediction = generate_advanced_hybrid_signals(
                    self.df,
                    fast=fast,
                    slow=slow,
                    ml_lookback=ml_lookback,
                    ridge_alpha=ridge_alpha,
                )
            else:
                raw_position = generate_signals(self.df, fast=fast, slow=slow)
                self.ml_prediction = None

            _, _, self.equity_curve = strategy_pipeline(self.df, raw_position)
            self.metrics = compute_metrics(self.df, raw_position)
            self.trades = build_trades(self.df, raw_position)

            self.show_metrics()
            self.show_trades()
            self.draw_equity_curve()

            mode = self.strategy_var.get()
            self.status_var.set(
                f"{mode} run complete — {len(self.trades) if self.trades is not None else 0} trades across {len(self.df):,} candles."
            )
        except Exception as exc:
            messagebox.showerror("Run Error", str(exc))

    def show_metrics(self):
        if not self.metrics:
            return

        for key, lbl in self.metric_value_labels.items():
            value = self.metrics.get(key)
            if value is None:
                lbl.configure(text="—")
            elif any(tag in key for tag in ["Return", "Drawdown", "Rate", "Exposure"]):
                lbl.configure(text=f"{value:.2%}")
            elif key == "Calmar Ratio" and np.isinf(value):
                lbl.configure(text="∞")
            else:
                lbl.configure(text=f"{value:.3f}")

    def show_trades(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        if self.trades is None or self.trades.empty:
            return

        for _, row in self.trades.iterrows():
            tag = "long" if row["Side"] == "LONG" else "short"
            self.tree.insert("", tk.END, values=(
                row["Side"],
                row["EntryDate"].strftime(
                    "%Y-%m-%d") if hasattr(row["EntryDate"], "strftime") else row["EntryDate"],
                f"{float(row['EntryPrice']):,.2f}",
                row["ExitDate"].strftime(
                    "%Y-%m-%d") if hasattr(row["ExitDate"], "strftime") else row["ExitDate"],
                f"{float(row['ExitPrice']):,.2f}",
                f"{float(row['PnL']):.2%}",
                f"{float(row['Trade Max DD']):.2%}",
                f"{float(row['Avg Exposure']):.2%}",
            ), tags=(tag,))

    def draw_equity_curve(self):
        canvas = self.chart_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 100 or height < 100:
            return

        if self.equity_curve is None or self.equity_curve.empty:
            canvas.create_text(
                width / 2, height / 2, text="Run the strategy to render the equity curve.", fill="#8da2c7", font=("Segoe UI", 12))
            return

        pad = 40
        plot_w = width - 2 * pad
        plot_h = height - 2 * pad

        values = self.equity_curve.values.astype(float)
        x = np.linspace(pad, pad + plot_w, num=len(values))

        vmin, vmax = float(np.min(values)), float(np.max(values))
        if vmax - vmin < 1e-12:
            y = np.full_like(x, pad + plot_h / 2)
        else:
            y = pad + (vmax - values) / (vmax - vmin) * plot_h

        canvas.create_rectangle(pad, pad, pad + plot_w,
                                pad + plot_h, outline="#2c3d63", width=1)
        baseline = y[0]

        points = []
        for xi, yi in zip(x, y):
            points.extend([float(xi), float(yi)])
        canvas.create_line(*points, fill="#33d1ff", width=2, smooth=True)
        canvas.create_line(pad, baseline, pad + plot_w,
                           baseline, fill="#2f7e5f", dash=(4, 3))

        canvas.create_text(pad, pad - 14, anchor="w",
                           text=f"Max: {vmax:.3f}", fill="#9fb0cf", font=("Segoe UI", 9))
        canvas.create_text(pad, pad + plot_h + 14, anchor="w",
                           text=f"Min: {vmin:.3f}", fill="#9fb0cf", font=("Segoe UI", 9))
        canvas.create_text(pad + plot_w, pad - 14, anchor="e",
                           text=f"Final: {values[-1]:.3f}", fill="#9fb0cf", font=("Segoe UI", 9))

    def export_trades(self):
        if self.trades is None or self.trades.empty:
            messagebox.showwarning(
                "No trades", "Run the strategy first to generate trades.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[
                                            ("CSV", "*.csv")], title="Export trades")
        if not path:
            return

        self.trades.to_csv(path, index=False)
        self.status_var.set(f"Trades exported to {path}")
        messagebox.showinfo("Export complete",
                            "Trades were exported successfully.")


if __name__ == "__main__":
    TradingBotGUI().mainloop()
