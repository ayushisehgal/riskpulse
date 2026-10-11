"""Data ingestion from >= 2 heterogeneous text sources.

Sources
-------
* News   : GDELT DOC 2.0 API (no key), NewsAPI.org (free key via NEWSAPI_KEY),
           or the bundled offline sample (data/sample_news.json)
* Social : Twitter/X-style posts from a Kaggle CSV (``text`` column) or the
           bundled offline sample (data/sample_tweets.json)

Every live fetcher fails soft: on any network/API error we fall back to the
bundled sample so the demo never breaks.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Doc:
    source: str          # "news" | "social"
    text: str
    published: dt.datetime
    origin: str = ""     # gdelt / newsapi / sample / kaggle
    url: str = ""
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.source}|{self.text}".encode()).hexdigest()[:12]


# ----------------------------------------------------------------- offline
def _load_sample(fname: str, source: str) -> list[Doc]:
    rows = json.loads((DATA_DIR / fname).read_text())
    now = dt.datetime.utcnow()
    return [Doc(source=source, text=r["text"],
                published=now - dt.timedelta(minutes=int(r["minutes_ago"])), origin="sample")
            for r in rows]


def load_sample_news() -> list[Doc]:
    return _load_sample("sample_news.json", "news")


def load_sample_tweets() -> list[Doc]:
    return _load_sample("sample_tweets.json", "social")


# -------------------------------------------------------------------- live
def fetch_gdelt(query: str = "stock market OR economy OR earnings", max_records: int = 40,
                timeout: int = 15) -> list[Doc]:
    """GDELT DOC 2.0 article list (updated every 15 min, 100+ languages)."""
    params = {"query": f"({query}) sourcelang:english", "mode": "ArtList",
              "format": "json", "maxrecords": max_records, "timespan": "1d", "sort": "DateDesc"}
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=timeout)
    r.raise_for_status()
    out = []
    for a in r.json().get("articles", []):
        try:
            ts = dt.datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ")
        except Exception:
            ts = dt.datetime.utcnow()
        out.append(Doc("news", a.get("title", ""), ts, "gdelt", a.get("url", "")))
    return [d for d in out if d.text]


def fetch_newsapi(query: str = "stocks OR markets OR earnings", page_size: int = 40,
                  api_key: str | None = None, timeout: int = 15) -> list[Doc]:
    key = api_key or os.getenv("NEWSAPI_KEY")
    if not key:
        raise RuntimeError("NEWSAPI_KEY not set")
    r = requests.get("https://newsapi.org/v2/everything",
                     params={"q": query, "language": "en", "sortBy": "publishedAt",
                             "pageSize": page_size, "apiKey": key}, timeout=timeout)
    r.raise_for_status()
    out = []
    for a in r.json().get("articles", []):
        text = " - ".join(x for x in (a.get("title"), a.get("description")) if x)
        ts = dt.datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00")).replace(tzinfo=None)
        out.append(Doc("news", text, ts, "newsapi", a.get("url", "")))
    return out


def load_tweets_csv(path: str, text_col: str = "text", limit: int = 500) -> list[Doc]:
    """Load a Kaggle stock-tweet CSV (e.g. "Tweet Sentiment's Impact on Stock Returns")."""
    now, out = dt.datetime.utcnow(), []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= limit:
                break
            if row.get(text_col):
                out.append(Doc("social", row[text_col], now - dt.timedelta(seconds=30 * i), "kaggle"))
    return out


# ----------------------------------------------------------------- facade
def fetch_all(live: bool = False, news_provider: str = "gdelt", tweets_csv: str | None = None,
              query: str | None = None) -> tuple[list[Doc], list[str]]:
    """Return (docs sorted oldest->newest, warnings). Always returns something."""
    warnings: list[str] = []
    news: list[Doc] = []
    if live:
        try:
            fn = fetch_gdelt if news_provider == "gdelt" else fetch_newsapi
            news = fn(query) if query else fn()
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Live {news_provider} failed ({e.__class__.__name__}); using sample news.")
    if not news:
        news = load_sample_news()
    try:
        social = load_tweets_csv(tweets_csv) if tweets_csv else load_sample_tweets()
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Tweet CSV failed ({e}); using sample tweets.")
        social = load_sample_tweets()
    return sorted(news + social, key=lambda d: d.published), warnings
