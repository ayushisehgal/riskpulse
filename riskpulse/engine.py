"""RiskEngine: ingest -> NLP -> structured signals -> pub/sub + file/API output."""
from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import nlp
from .ingestion import Doc


@dataclass
class Signal:
    id: str
    timestamp: str          # ISO-8601 UTC
    source: str             # news | social
    origin: str             # gdelt / newsapi / sample / kaggle
    text: str
    tickers: list[str]      # [] => market-wide / macro event
    sentiment_score: float  # -1.0 .. 1.0
    event_type: str         # Geopolitical | Macroeconomic | Credit Event | ...
    impact_score: float     # 1 .. 10
    confidence: float       # classifier confidence 0..1

    def to_dict(self) -> dict:
        return asdict(self)


class RiskEngine:
    """Central processing unit. Downstream modules call ``subscribe``."""

    def __init__(self, out_path: str | Path | None = "output/signals.jsonl"):
        self.signals: list[Signal] = []
        self._subs: list[Callable[[Signal], None]] = []
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self.out_path = Path(out_path) if out_path else None
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

    # pub/sub ---------------------------------------------------------
    def subscribe(self, fn: Callable[[Signal], None]) -> None:
        self._subs.append(fn)

    # processing ------------------------------------------------------
    def process(self, doc: Doc) -> Signal | None:
        with self._lock:
            if doc.id in self._seen or not doc.text.strip():
                return None  # de-duplicate
            self._seen.add(doc.id)
        f = nlp.analyze(doc.text, doc.source)
        sig = Signal(id=doc.id, timestamp=doc.published.isoformat(timespec="seconds") + "Z",
                     source=doc.source, origin=doc.origin, text=doc.text.strip(), **f)
        with self._lock:
            self.signals.append(sig)
            if self.out_path:  # file-based output for downstream consumers
                with self.out_path.open("a") as fh:
                    fh.write(json.dumps(sig.to_dict()) + "\n")
        for fn in self._subs:
            fn(sig)
        return sig

    def process_many(self, docs: Iterable[Doc]) -> list[Signal]:
        return [s for d in docs if (s := self.process(d))]

    def analyze_text(self, text: str, source: str = "news") -> Signal:
        """Ad-hoc analysis that also flows to subscribers (used by API + demo box)."""
        doc = Doc(source, text, dt.datetime.utcnow(), origin="adhoc")
        return self.process(doc) or next(s for s in self.signals if s.id == doc.id)

    def query(self, ticker: str | None = None, min_impact: float = 0, event_type: str | None = None,
              limit: int = 100) -> list[Signal]:
        out = [s for s in self.signals
               if (not ticker or ticker.upper() in s.tickers)
               and s.impact_score >= min_impact
               and (not event_type or s.event_type.lower() == event_type.lower())]
        return out[-limit:]
