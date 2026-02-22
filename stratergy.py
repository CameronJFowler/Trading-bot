from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

TRADING_DAYS = 365          # Crypto trades every day
MAX_EXPOSURE = 0.30
COST_BPS = 5
VOL_WINDOW = 20
TARGET_ANN_VOL = 0.20       # More realistic for BTC than 0.15 in many regimes
MIN_DAILY_VOL = 1e-6

DEFAULT_FAST = 10
DEFAULT_SLOW = 50


# -----------------------------
# Data
# -----------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column")

    # Your file contains Close, so we key off that
    if "Close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated()]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Optional: drop rows where Close is missing
    df = df.dropna(subset=["Close"])

    return df


# -----------------------------
# Signals
# -----------------------------

def generate_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype="int8")

    if fast >= slow:
        raise ValueError("Fast SMA must be less than Slow SMA")

    close = df["Close"].astype(float)

    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()

    pos = pd.Series(0, index=df.index, dtype="int8")

    # Long when fast above slow
    pos[ma_fast > ma_slow] = 1

    # Short when slow significantly above fast (asymmetric short rule)
    pos[1.2 * ma_fast < ma_slow] = -1

    return pos.fillna(0).clip(-1, 1).astype("int8")


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

    close = df["Close"].astype(float)
    returns = close.pct_change().fillna(0.0)

    vol = returns.rolling(vol_window).std()
    vol = (
        vol.replace(0.0, np.nan)
           .bfill()
           .ffill()
           .clip(lower=MIN_DAILY_VOL)
    )

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

def compute_metrics(df: pd.DataFrame, raw_position: pd.Series) -> dict:
    _, strat_returns, equity = strategy_pipeline(df, raw_position)

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

    win_rate = float((strat_returns > 0).mean())

    signal_exposure = float((raw_position != 0).mean())
    avg_abs_exposure = float(
        np.abs(strategy_pipeline(df, raw_position)[0]).mean())

    return {
        "Sharpe Ratio": sharpe,
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

    trades = []
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

        if (not in_trade) and sig != 0.0:
            in_trade = True
            side = sig
            entry_date = date
            entry_price = price
            trade_equity = 1.0
            peak_equity = 1.0
            max_dd_trade = 0.0

        elif in_trade and (sig == 0.0 or sig != side):
            exit_date = date
            exit_price = price

            trade_side = "LONG" if side > 0 else "SHORT"
            trade_return = trade_equity - 1.0

            trades.append({
                "Side": trade_side,
                "EntryDate": entry_date,
                "EntryPrice": entry_price,
                "ExitDate": exit_date,
                "ExitPrice": exit_price,
                "PnL": trade_return,
                "Trade Max DD": max_dd_trade,
                "Avg Exposure": float(np.mean(np.abs(effective_exposure.loc[entry_date:exit_date]))),
            })

            in_trade = False

    return pd.DataFrame(trades)


# -----------------------------
# GUI
# -----------------------------

class TradingBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TEntry", padding=4)

        self.title("Crypto Trading Bot (BTC CSV)")
        self.geometry("1100x720")

        self.df: pd.DataFrame | None = None
        self.trades: pd.DataFrame | None = None
        self.metrics: dict | None = None

        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        self.path_var = tk.StringVar()
        ttk.Label(top, text="CSV:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.path_var, width=78).pack(side=tk.LEFT)
        ttk.Button(top, text="Browse", command=self.browse).pack(
            side=tk.LEFT, padx=6)

        controls = ttk.Frame(self, padding=10)
        controls.pack(fill=tk.X)

        self.fast_var = tk.StringVar(value=str(DEFAULT_FAST))
        self.slow_var = tk.StringVar(value=str(DEFAULT_SLOW))

        ttk.Label(controls, text="Fast SMA:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(controls, textvariable=self.fast_var,
                  width=8).pack(side=tk.LEFT)

        ttk.Label(controls, text="Slow SMA:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(controls, textvariable=self.slow_var,
                  width=8).pack(side=tk.LEFT)

        ttk.Button(controls, text="Run", command=self.run).pack(
            side=tk.LEFT, padx=12)

        self.metrics_text = tk.Text(self, height=10)
        self.metrics_text.pack(fill=tk.X, padx=10)

        self.tree = ttk.Treeview(
            self,
            columns=("Side", "EntryDate", "EntryPrice", "ExitDate",
                     "ExitPrice", "PnL", "Trade Max DD", "Avg Exposure"),
            show="headings"
        )

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        try:
            self.path_var.set(path)
            self.df = load_csv(path)
            messagebox.showinfo("Loaded", "CSV loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV first")
            return

        try:
            fast = int(self.fast_var.get())
            slow = int(self.slow_var.get())

            raw_position = generate_signals(self.df, fast=fast, slow=slow)

            self.metrics = compute_metrics(self.df, raw_position)
            self.trades = build_trades(self.df, raw_position)

            self.show_metrics()
            self.show_trades()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_metrics(self):
        self.metrics_text.delete("1.0", tk.END)
        if not self.metrics:
            return

        for k, v in self.metrics.items():
            if any(x in k for x in ["Return", "Drawdown", "Rate", "Exposure"]):
                self.metrics_text.insert(tk.END, f"{k}: {v:.2%}\n")
            else:
                self.metrics_text.insert(tk.END, f"{k}: {v:.4f}\n")

    def show_trades(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        if self.trades is None or self.trades.empty:
            return

        for _, r in self.trades.iterrows():
            self.tree.insert("", tk.END, values=(
                r["Side"],
                r["EntryDate"],
                round(float(r["EntryPrice"]), 2),
                r["ExitDate"],
                round(float(r["ExitPrice"]), 2),
                f"{float(r['PnL']):.2%}",
                f"{float(r['Trade Max DD']):.2%}",
                f"{float(r['Avg Exposure']):.2%}",
            ))


if __name__ == "__main__":
    TradingBotGUI().mainloop()
