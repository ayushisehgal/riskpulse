"""Module B - Strategic Portfolio Stress Testing.

Subscribes to Event Classification + Impact Score. When a high-impact event
(default: impact > 7) of a mapped type is detected, a simplified shock scenario
is applied to a synthetic wholesale-banking portfolio (loans, bonds,
derivatives, equity) built from the sample transaction data.

Shocks are scaled by impact: severity = clip(impact/7, 1.0, 1.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .engine import Signal
from .ingestion import DATA_DIR

# equity shock (%), parallel rate move (bp), credit-spread widening (bp), extra loan loss (% of MV)
SHOCKS = {
    "Geopolitical":  dict(equity=-0.10, rates=+50,  spread=+120, loan_loss=0.010),
    "Macroeconomic": dict(equity=-0.07, rates=+200, spread=+60,  loan_loss=0.005),
    "Credit Event":  dict(equity=-0.12, rates=-30,  spread=+250, loan_loss=0.030),
    "Regulatory":    dict(equity=-0.04, rates=0,    spread=+30,  loan_loss=0.002),
}
TYPE_MAP = {"Term Loan": "Loan", "Revolver": "Loan", "Corporate Bond": "Bond", "Government Bond": "Bond",
            "Interest Rate Swap": "Derivative", "Equity Option": "Derivative", "Equity Holding": "Equity"}
RATING_SPREAD_BETA = {"AAA": 0.3, "AA": 0.5, "A": 0.8, "BBB": 1.0, "BB": 1.6, "B": 2.4}


def build_portfolio(path: str | Path | None = None) -> pd.DataFrame:
    """Turn raw transactions into revaluation-ready positions."""
    tx = pd.read_csv(path or DATA_DIR / "sample_transactions.csv")
    p = pd.DataFrame({
        "asset_id": tx.txn_id, "asset_type": tx["product"].map(TYPE_MAP), "product": tx["product"],
        "sector": tx.sector, "rating": tx.rating, "market_value": tx.amount_usd_m,
    })
    ten = tx.tenor_years.astype(float)
    beta = tx.rating.map(RATING_SPREAD_BETA).fillna(1.0)
    p["duration"] = (ten * 0.9).where(p.asset_type.isin(["Bond", "Loan"]), 0.0)
    p["spread_beta"] = beta.where(p["product"] != "Government Bond", 0.0)
    p["equity_delta"] = p["product"].map({"Equity Holding": 1.0, "Equity Option": 0.6}).fillna(0.0)
    p["rate_dv01"] = p["product"].map({"Interest Rate Swap": 1}).fillna(0) * tx.amount_usd_m * ten * 0.0001 * -0.9
    # floating-rate loans have ~0 rate duration; fixed bonds carry full duration
    p.loc[p.asset_type == "Loan", "duration"] *= 0.15
    return p


def revalue(pf: pd.DataFrame, shock: dict, severity: float = 1.0) -> pd.DataFrame:
    eq, rt, sp = shock["equity"] * severity, shock["rates"] * severity / 1e4, shock["spread"] * severity / 1e4
    out = pf.copy()
    pnl = (-out.duration * rt * out.market_value                      # rate duration
           - out.duration * out.spread_beta * sp * out.market_value   # spread duration
           + out.equity_delta * eq * out.market_value                 # equity delta
           + out.rate_dv01 * (rt * 1e4))                              # swap DV01 ($m per bp)
    pnl = pnl - (out.asset_type == "Loan") * shock["loan_loss"] * severity * out.spread_beta * out.market_value
    out["pnl"] = pnl
    out["stressed_value"] = out.market_value + pnl
    return out


@dataclass
class StressResult:
    timestamp: str
    trigger_text: str
    event_type: str
    impact_score: float
    severity: float
    before: float
    after: float
    by_asset_type: pd.DataFrame = field(repr=False)
    positions: pd.DataFrame = field(repr=False)

    @property
    def pnl(self): return self.after - self.before
    @property
    def pct(self): return self.pnl / self.before * 100


class StressTester:
    def __init__(self, portfolio: pd.DataFrame | None = None, threshold: float = 7.0):
        self.portfolio = portfolio if portfolio is not None else build_portfolio()
        self.threshold = threshold
        self.results: list[StressResult] = []

    def run(self, event_type: str, impact: float, ts: str = "", text: str = "") -> StressResult:
        sev = min(1.5, max(1.0, impact / 7.0))
        pos = revalue(self.portfolio, SHOCKS[event_type], sev)
        agg = pos.groupby("asset_type")[["market_value", "stressed_value", "pnl"]].sum().reset_index()
        res = StressResult(ts, text, event_type, impact, sev, float(pos.market_value.sum()),
                           float(pos.stressed_value.sum()), agg, pos)
        self.results.append(res)
        return res

    def on_signal(self, sig: Signal) -> None:
        if sig.impact_score > self.threshold and sig.event_type in SHOCKS:
            self.run(sig.event_type, sig.impact_score, sig.timestamp, sig.text)
