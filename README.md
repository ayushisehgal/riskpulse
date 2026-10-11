# RiskPulse: AI/NLP Financial Risk Engine - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Ayushi Sehgal
**College Email ID:** ayushi.23bce10790@vitbhopal.ac.in
**College / Campus:** Vellore Institute of Technology (VIT), Bhopal
**Demo Video Link:** https://www.loom.com/share/0e827ea5db9b451eb580d5cd6a016820
**Live App:** https://riskpulse-as.streamlit.app/
**Slide Deck Link (if hosted externally):** Included in repo at `docs/presentation.pdf`

## 1. Project Overview / Problem Statement & Approach
Markets react to news and social media within minutes, but risk systems consume structured numbers. RiskPulse is a
unified AI/NLP risk engine that ingests unstructured text from two sources (financial news and Twitter/X-style posts)
and converts each document into a structured, machine-readable risk signal: a sentiment score (-1 to 1), an event
classification (Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch, Regulatory, Earnings),
and an impact score (1 to 10).

To demonstrate the signals in practice, **both** downstream modules are implemented. **Module A** is a tactical
index rebalancer that shifts the weights of 15 S&P 100 stocks based on smoothed, impact-weighted sentiment.
**Module B** is a strategic stress-testing tool that applies scaled shock scenarios to a synthetic wholesale-banking
portfolio (loans, bonds, derivatives, equity) whenever a high-impact event (impact > 7) is detected.

## 2. Architecture & Tech Stack
![Architecture](docs/architecture.png)

- **Languages/Frameworks:** Python, Streamlit (dashboard), FastAPI (REST API), Plotly
- **NLP:** VADER with a custom finance lexicon, weighted-regex event classifier, rule-based impact scoring
  (optional FinBERT via `USE_FINBERT=1`)
- **Data/Numerics:** pandas, NumPy; optional yfinance for live prices
- **Design:** the engine publishes signals; Module A and Module B subscribe to them (publish/subscribe). Output is also
  written to `output/signals.jsonl` and served via REST.

## 3. Dataset Used
- `data/sample_news.json` (28 headlines) and `data/sample_tweets.json` (24 posts): **synthetic**, written by the
  candidate to mirror real news/social text. No proprietary or client data is used.
- `data/sample_transactions.csv` (40 transactions): **synthetic** wholesale-banking transactions used to build the
  Module B portfolio. Real datasets (Kaggle stock tweets, financial transactions) can be plugged in via
  `load_tweets_csv()` and `build_portfolio(path)`.
- Optional live news: GDELT DOC 2.0 API (public, no key) and NewsAPI (free key), with automatic fallback to the sample.
- Assumptions: shock sizes in Module B are illustrative, not calibrated; headlines are analysed at sentence level.

## 4. Quickstart & Installation
Runtime: Python 3.11 / 3.12 on Windows 11 (also runs on Linux/macOS)

```bash
git clone https://github.com/ayushisehgal/vitbhopal-ayushi-sehgal-hackathon.git
cd vitbhopal-ayushi-sehgal-hackathon
pip install -r requirements.txt
python -m pytest -q                 # 8 tests
streamlit run app.py                # dashboard (run command)
```
Other entry points: `uvicorn riskpulse.api:app --reload` (API at /docs) and `python -m riskpulse.pipeline` (CLI).

## 5. Key Results & Domain Impact
- 52 signals scored from two sources; sample run: NVDA and MSFT rise to the 15% cap (+833 bps), BAC falls to the 2% floor (-464 bps).
- 9 stress tests triggered; a Geopolitical event with impact 10 reduces the synthetic portfolio from $2,435m to $2,313m (-5.0%).
- Impact: turns unstructured text into actionable, quantified risk that can drive both tactical (index weights) and
  strategic (portfolio stress) decisions faster than manual monitoring.

---
# Details
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

## Extra commands
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
