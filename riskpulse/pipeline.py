"""Wires ingestion -> engine -> Module A + Module B."""
from __future__ import annotations

from .engine import RiskEngine
from .ingestion import fetch_all
from .module_a import IndexRebalancer
from .module_b import StressTester


def build(out_path="output/signals.jsonl", **kw):
    engine = RiskEngine(out_path)
    a, b = IndexRebalancer(**kw.get("a", {})), StressTester(**kw.get("b", {}))
    engine.subscribe(a.on_signal)
    engine.subscribe(b.on_signal)
    return engine, a, b


def run(live=False, **kw):
    engine, a, b = build(**kw)
    docs, warns = fetch_all(live=live, news_provider=kw.get("provider", "gdelt"))
    engine.process_many(docs)
    return engine, a, b, warns


if __name__ == "__main__":
    e, a, b, w = run()
    print(f"{len(e.signals)} signals | {len(b.results)} stress tests triggered | warnings: {w}")
    print(a.current().round(4).to_string(index=False))
    for r in b.results:
        print(f"[{r.event_type} impact {r.impact_score}] portfolio {r.before:,.0f} -> {r.after:,.0f} ({r.pct:+.2f}%)")
