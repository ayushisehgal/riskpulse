"""Price history for the Module A index-value chart. yfinance if installed, else seeded synthetic."""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_prices(tickers: list[str], days: int = 60) -> tuple[pd.DataFrame, str]:
    try:
        import yfinance as yf
        df = yf.download(tickers, period=f"{days}d", progress=False, auto_adjust=True)["Close"].dropna()
        if len(df) > 5:
            return df, "yfinance"
    except Exception:
        pass
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    rets = rng.normal(0.0004, 0.013, (days, len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, 0)), index=idx, columns=tickers), "synthetic"
