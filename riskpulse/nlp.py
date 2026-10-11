"""NLP core: sentiment, event classification, impact scoring, entity linking.

Design: fast, explainable, dependency-light (VADER + finance lexicon + weighted
regex event rules). Set USE_FINBERT=1 (needs `transformers`, `torch`) to swap in
ProsusAI/finbert for the sentiment score without touching anything else.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------------------------------------- sentiment
FIN_LEXICON = {
    # positive
    "beats": 2.2, "beat": 1.8, "surge": 2.5, "surges": 2.5, "soars": 2.7, "rally": 2.2, "rallies": 2.2,
    "record": 1.6, "upgrade": 2.2, "upgraded": 2.2, "bullish": 2.3, "outperform": 2.0, "approval": 1.8,
    "approved": 1.6, "breakthrough": 2.4, "rebound": 1.8, "jumps": 2.0, "profit": 1.5, "growth": 1.4,
    "accretive": 1.8, "buyback": 1.5, "dividend": 1.0, "boom": 2.0,
    # negative
    "plunge": -3.0, "plunges": -3.0, "tumble": -2.6, "tumbles": -2.6, "slump": -2.3, "crash": -3.3,
    "default": -3.2, "defaults": -3.2, "bankruptcy": -3.5, "downgrade": -2.4, "downgraded": -2.4,
    "misses": -2.2, "miss": -1.8, "probe": -1.8, "lawsuit": -1.9, "layoffs": -2.0, "recall": -2.1,
    "fraud": -3.2, "sanctions": -2.2, "war": -2.8, "conflict": -2.2, "invasion": -3.0, "bearish": -2.3,
    "underperform": -2.0, "selloff": -2.6, "sell-off": -2.6, "fears": -1.8, "recession": -2.6,
    "inflation": -1.2, "shortage": -1.8, "outage": -1.8, "breach": -2.4, "fine": -1.4, "fined": -2.0,
    "writedown": -2.4, "bailout": -2.4, "contagion": -2.8, "panic": -3.0, "warns": -1.6, "halts": -1.8,
    "escalating": -2.4, "escalates": -2.4, "spikes": -1.0, "collapse": -3.2, "collapses": -3.2,
}


@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FIN_LEXICON)
    return sia


@lru_cache(maxsize=1)
def _finbert():
    from transformers import pipeline  # optional heavy dependency
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


def sentiment_score(text: str) -> float:
    """Return sentiment in [-1.0, 1.0] (negative .. positive)."""
    if os.getenv("USE_FINBERT") == "1":
        try:
            probs = {d["label"].lower(): d["score"] for d in _finbert()(text[:512])[0]}
            return round(probs.get("positive", 0) - probs.get("negative", 0), 3)
        except Exception:
            pass  # fall through to VADER
    return round(_vader().polarity_scores(text)["compound"], 3)


# ------------------------------------------------------ event classification
EVENT_RULES: dict[str, list[tuple[str, float]]] = {
    "Geopolitical": [(r"\b(war|conflict|invasion|sanction\w*|missile|military|border|ceasefire|"
                      r"tensions?|embargo|strait|terror\w*|coup|nato|geopolitic\w*|blockade|"
                      r"shipping lanes?|escalat\w*)\b", 2.0)],
    "Macroeconomic": [(r"\b(inflation|cpi|gdp|fed|federal reserve|ecb|interest rates?|rate (hike|cut)|"
                       r"unemployment|jobs report|payrolls|recession|central bank|treasury yields?|"
                       r"tariffs?|oil prices?|opec)\b", 2.0)],
    "Credit Event": [(r"\b(default\w*|bankrupt\w*|downgrad\w*|credit rating|insolven\w*|"
                      r"debt restructuring|bailout|liquidity crisis|bond yields? spike|"
                      r"credit spreads?|writedown|write-down|distress\w*|junk)\b", 2.2)],
    "Merger/Acquisition": [(r"\b(acquir\w*|acquisition|merger|merge|takeover|buyout|"
                            r"to buy|bid for|spin-?off|divest\w*|deal valued)\b", 2.0)],
    "Product Launch": [(r"\b(launch\w*|unveil\w*|release[sd]?|introduc\w*|rolls? out|debut\w*|"
                        r"new (iphone|model|chip|product|service|feature))\b", 2.0)],
    "Regulatory": [(r"\b(sec|antitrust|regulat\w*|lawsuit|probe|investigation|fined?|"
                    r"compliance|ban(ned)?|settlement|doj|ftc|eu commission)\b", 1.6)],
    "Earnings": [(r"\b(earnings|eps|revenue|quarterly|guidance|profit|beats?|misses|"
                  r"forecast|outlook|results)\b", 1.4)],
}
_COMPILED = {k: [(re.compile(p, re.I), w) for p, w in v] for k, v in EVENT_RULES.items()}


def classify_event(text: str) -> tuple[str, float]:
    """Return (label, confidence 0..1) using weighted keyword evidence."""
    scores = {}
    for label, rules in _COMPILED.items():
        s = sum(w * len(p.findall(text)) for p, w in rules)
        if s:
            scores[label] = s
    if not scores:
        return "Other", 0.3
    best = max(scores, key=scores.get)
    return best, round(scores[best] / (sum(scores.values()) + 0.5), 2)


# --------------------------------------------------------------- impact
BASE_IMPACT = {"Geopolitical": 6.0, "Credit Event": 6.0, "Macroeconomic": 5.5, "Regulatory": 4.5,
               "Merger/Acquisition": 4.5, "Earnings": 3.5, "Product Launch": 3.0, "Other": 2.0}
INTENSIFIERS = re.compile(r"\b(crisis|collapse\w*|war|default\w*|emergency|unprecedented|plunge\w*|"
                          r"sanction\w*|invasion|bankrupt\w*|crash\w*|panic|escalat\w*|contagion|"
                          r"closes|closure|shutdown|record|massive|global|worst)\b", re.I)
SOURCE_WEIGHT = {"news": 1.0, "social": 0.85}


def impact_score(text: str, event: str, sentiment: float, source: str = "news") -> float:
    """Predicted market-impact severity on a 1..10 scale.

    impact = (event base severity + intensifier keywords (max +3) + 2*|sentiment|) * source weight
    """
    bump = min(3.0, 0.9 * len(INTENSIFIERS.findall(text)))
    raw = BASE_IMPACT.get(event, 2.0) + bump + 2.0 * abs(sentiment)
    return round(max(1.0, min(10.0, raw * SOURCE_WEIGHT.get(source, 0.85))), 1)


# ---------------------------------------------------------- entity linking
COMPANIES = {  # S&P 100 subset used for the mock index
    "AAPL": ["apple", "iphone", "aapl", "tim cook"], "MSFT": ["microsoft", "msft", "azure", "windows"],
    "AMZN": ["amazon", "amzn", "aws"], "GOOGL": ["google", "alphabet", "googl", "youtube"],
    "META": ["meta platforms", "facebook", "instagram", "meta"], "NVDA": ["nvidia", "nvda", "jensen huang"],
    "TSLA": ["tesla", "tsla", "elon musk"], "JPM": ["jpmorgan", "jp morgan", "jpm", "jamie dimon"],
    "BAC": ["bank of america", "bac"], "GS": ["goldman sachs", "goldman"],
    "XOM": ["exxon", "exxonmobil", "xom"], "CVX": ["chevron", "cvx"],
    "PFE": ["pfizer", "pfe"], "JNJ": ["johnson & johnson", "johnson and johnson", "jnj"],
    "KO": ["coca-cola", "coca cola"], "PEP": ["pepsico", "pepsi"], "WMT": ["walmart", "wmt"],
    "DIS": ["disney"], "NFLX": ["netflix", "nflx"], "INTC": ["intel", "intc"],
}
_ENTITY_RX = {t: re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(a) for a in al) + r")(?![A-Za-z])", re.I)
              for t, al in COMPANIES.items()}
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")


def extract_tickers(text: str) -> list[str]:
    found = [t for t, rx in _ENTITY_RX.items() if rx.search(text)]
    found += [t for t in _CASHTAG.findall(text) if t in COMPANIES and t not in found]
    return found


def analyze(text: str, source: str = "news") -> dict:
    """Full NLP pass over one document -> structured signal fields."""
    sent = sentiment_score(text)
    event, conf = classify_event(text)
    return {
        "tickers": extract_tickers(text),
        "sentiment_score": sent,
        "event_type": event,
        "impact_score": impact_score(text, event, sent, source),
        "confidence": conf,
    }
