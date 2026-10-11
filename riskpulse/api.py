"""Simple REST API exposing structured risk signals.   uvicorn riskpulse.api:app --reload"""
from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import BaseModel

from .pipeline import run

app = FastAPI(title="RiskPulse API", version="1.0")
engine, rebalancer, tester, _ = run()


class AnalyzeIn(BaseModel):
    text: str
    source: str = "news"   # news | social


@app.get("/health")
def health():
    return {"status": "ok", "signals": len(engine.signals)}


@app.get("/signals")
def signals(ticker: str | None = None, event_type: str | None = None,
            min_impact: float = Query(0, ge=0, le=10), limit: int = Query(100, le=1000)):
    return [s.to_dict() for s in engine.query(ticker, min_impact, event_type, limit)]


@app.post("/analyze")
def analyze(body: AnalyzeIn):
    """Analyze any text; result also flows to Module A / B subscribers."""
    return engine.analyze_text(body.text, body.source).to_dict()


@app.get("/index/weights")
def weights():
    return rebalancer.current().to_dict(orient="records")


@app.get("/stress/results")
def stress():
    return [{"timestamp": r.timestamp, "event_type": r.event_type, "impact": r.impact_score,
             "before": r.before, "after": r.after, "pnl_pct": round(r.pct, 2), "trigger": r.trigger_text}
            for r in tester.results]
