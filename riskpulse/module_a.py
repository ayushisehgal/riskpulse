"""Module A - Tactical Index Rebalancing.

Subscribes to the engine's Sentiment Score and dynamically re-weights a mock
index of 15 S&P 100 stocks.

    ema_i   <- (1-a)*ema_i + a * sentiment * (impact/10)        (per signal)
    w_i     <- base_i * exp(tilt * ema_i)  -> clipped to [w_min, w_max], renormalised

Positive sentiment raises a stock's weight, negative lowers it. EMA smoothing
stops one noisy tweet from whipsawing the index; caps keep it diversified.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine import Signal

DEFAULT_INDEX = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM",
                 "BAC", "GS", "XOM", "CVX", "PFE", "KO", "INTC"]


class IndexRebalancer:
    def __init__(self, tickers=None, alpha=0.35, tilt=3.0, w_min=0.02, w_max=0.15):
        self.tickers = list(tickers or DEFAULT_INDEX)
        n = len(self.tickers)
        self.base = np.full(n, 1.0 / n)
        self.ema = np.zeros(n)
        self.alpha, self.tilt, self.w_min, self.w_max = alpha, tilt, w_min, w_max
        self.weights = self.base.copy()
        self.history: list[dict] = []
        self._record("init", self.weights)

    # --------------------------------------------------------------
    def _cap(self, w: np.ndarray) -> np.ndarray:
        w = w / w.sum()
        for _ in range(50):
            w = np.clip(w, self.w_min, self.w_max)
            w = w / w.sum()
            if w.min() >= self.w_min - 1e-9 and w.max() <= self.w_max + 1e-9:
                break
        return w

    def _record(self, ts: str, w: np.ndarray, trigger: str = "") -> None:
        for t, wi, e in zip(self.tickers, w, self.ema):
            self.history.append({"timestamp": ts, "ticker": t, "weight": float(wi),
                                 "ema_sentiment": float(e), "trigger": trigger})

    # engine subscriber ---------------------------------------------
    def on_signal(self, sig: Signal) -> None:
        hit = [t for t in sig.tickers if t in self.tickers]
        if not hit:
            return
        for t in hit:
            i = self.tickers.index(t)
            x = sig.sentiment_score * (sig.impact_score / 10.0)
            self.ema[i] = (1 - self.alpha) * self.ema[i] + self.alpha * x
        raw = self.base * np.exp(self.tilt * self.ema)
        self.weights = self._cap(raw)
        self._record(sig.timestamp, self.weights, trigger=f"{','.join(hit)}: {sig.text[:70]}")

    # views -----------------------------------------------------------
    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def current(self) -> pd.DataFrame:
        df = pd.DataFrame({"ticker": self.tickers, "base": self.base, "weight": self.weights,
                           "ema_sentiment": self.ema})
        df["change_bps"] = (df.weight - df.base) * 1e4
        return df.sort_values("weight", ascending=False).reset_index(drop=True)
