# 📡 RiskPulse — AI/NLP Financial Risk Engine

Ingests unstructured text (news + social), turns it into structured, machine-readable risk
signals, and drives **both** downstream modules:

* **Module A** – tactical, sentiment-driven index rebalancer (15 S&P 100 stocks)
* **Module B** – strategic, event-driven stress-testing of a synthetic wholesale-banking portfolio

```
 GDELT / NewsAPI ─┐                                   ┌─► Module A  IndexRebalancer  (Sentiment Score)
                  ├─► Ingestion ─► NLP Risk Engine ───┤
 Tweets (Kaggle)  ┘   (dedupe)    sentiment · event ·  └─► Module B  StressTester     (Event + Impact)
                                   impact · tickers          │
                                        │                    ▼
                                        ├─► signals.jsonl   Streamlit dashboard
                                        └─► REST API (FastAPI)
```

## Engine output (one JSON object per document)
```json
{"id":"47d338f3afae","timestamp":"2026-10-11T12:48:40Z","source":"news","text":"Goldman Sachs beats earnings",
 "tickers":["GS"],"sentiment_score":0.494,"event_type":"Earnings","impact_score":4.5,"confidence":0.85}
```
| Field | Range | How |
|---|---|---|
| `sentiment_score` | −1.0 … 1.0 | VADER + finance lexicon (optional FinBERT via `USE_FINBERT=1`) |
| `event_type` | Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch, Regulatory, Earnings, Other | weighted regex evidence, with confidence |
| `impact_score` | 1 … 10 | (event base severity + intensifier keywords + 2·\|sentiment\|) × source weight |

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q                 # 8 tests
streamlit run app.py                # live dashboard (works offline with bundled sample data)
uvicorn riskpulse.api:app --reload  # REST API → http://127.0.0.1:8000/docs
python -m riskpulse.pipeline        # CLI run, writes output/signals.jsonl
```
**Live data (optional):** toggle *Use live news* (GDELT, no key) or `export NEWSAPI_KEY=...`.
Any network failure automatically falls back to the sample data. Real tweets: `load_tweets_csv("tweets.csv")`.
Live prices for the index chart: `pip install yfinance`.

## API
`GET /signals?ticker=&event_type=&min_impact=` · `POST /analyze {"text","source"}` · `GET /index/weights` · `GET /stress/results`

## Module details
**A – Rebalancer.** Per signal: `ema ← (1−α)·ema + α·sentiment·impact/10`; `w ∝ base·exp(tilt·ema)`, clipped to
[2%, 15%] and renormalised. Positive → weight up, negative → weight down. Dashboard shows weights over time.

**B – Stress tester.** Portfolio built from transaction data (loans, bonds, swaps/options, equity). If
`impact > 7` and the event type has a shock set, apply shocks scaled by `clip(impact/7, 1, 1.5)`:

| Event | Equity | Rates | Spreads | Extra loan loss |
|---|---|---|---|---|
| Geopolitical | −10% | +50bp | +120bp | 1.0% |
| Macroeconomic | −7% | +200bp | +60bp | 0.5% |
| Credit Event | −12% | −30bp | +250bp | 3.0% |
| Regulatory | −4% | 0 | +30bp | 0.2% |

Revaluation: rate + spread duration for bonds/loans, equity delta for options/equity, DV01 for swaps,
loss-given-stress on loans. Dashboard shows before/after, per-asset-type bars, loss contributors, waterfall.

## Known limitations / next steps
Rule-based event classifier (swap for fine-tuned FinBERT / zero-shot LLM), headline-level text only,
sample transactions are synthetic stand-ins for the Kaggle/Salad Money data, shock sizes are illustrative not calibrated,
no transaction costs in the rebalancer.

## 5-minute demo script
1. (0:30) Problem + architecture. 2. (1:00) Run pipeline, show Risk-signals tab (two sources, three outputs).
3. (1:15) Module A: weights shift, NVDA/MSFT up, BAC down. 4. (1:30) Module B: pick the Geopolitical impact-10 event → before/after.
5. (0:45) Tab 4: paste a live headline, show API + `signals.jsonl`.
