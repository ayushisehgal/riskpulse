import datetime as dt
from riskpulse import nlp
from riskpulse.engine import RiskEngine
from riskpulse.ingestion import Doc, fetch_all
from riskpulse.module_a import IndexRebalancer
from riskpulse.module_b import StressTester, build_portfolio
from riskpulse.pipeline import build


def test_sentiment_sign():
    assert nlp.sentiment_score("Nvidia soars on record revenue") > 0.4
    assert nlp.sentiment_score("Bank defaults, contagion panic") < -0.4


def test_event_classification():
    assert nlp.classify_event("Rating agency downgrades lender after bond defaults")[0] == "Credit Event"
    assert nlp.classify_event("Apple unveils new iPhone")[0] == "Product Launch"
    assert nlp.classify_event("Exxon to acquire shale producer")[0] == "Merger/Acquisition"


def test_impact_bounds_and_ordering():
    hi = nlp.analyze("Escalating conflict closes strait; global stocks tumble")
    lo = nlp.analyze("Apple unveils new case colour")
    assert 1 <= lo["impact_score"] < hi["impact_score"] <= 10


def test_entities():
    assert nlp.extract_tickers("$TSLA and Goldman Sachs rally") == ["GS", "TSLA"] or set(nlp.extract_tickers("$TSLA and Goldman Sachs rally")) == {"TSLA", "GS"}


def test_signal_schema_and_dedup(tmp_path):
    e = RiskEngine(tmp_path / "s.jsonl")
    d = Doc("news", "Tesla recall hits vehicles", dt.datetime.utcnow())
    assert e.process(d) is not None and e.process(d) is None
    s = e.signals[0].to_dict()
    assert {"sentiment_score", "event_type", "impact_score"} <= s.keys()
    assert (tmp_path / "s.jsonl").read_text().count("\n") == 1


def test_rebalancer_direction_and_constraints():
    a = IndexRebalancer()
    e = RiskEngine(None); e.subscribe(a.on_signal)
    for i in range(6):
        e.process(Doc("news", f"Nvidia soars on record profit {i}", dt.datetime.utcnow()))
        e.process(Doc("news", f"Intel plunges after fraud probe {i}", dt.datetime.utcnow()))
    w = dict(zip(a.tickers, a.weights))
    assert w["NVDA"] > 1 / 15 > w["INTC"]
    assert abs(sum(a.weights) - 1) < 1e-9 and max(a.weights) <= 0.15 + 1e-9 and min(a.weights) >= 0.02 - 1e-9


def test_stress_triggers_only_above_threshold():
    t = StressTester(build_portfolio(), threshold=7.0)
    e = RiskEngine(None); e.subscribe(t.on_signal)
    e.process(Doc("news", "Apple unveils new iPhone", dt.datetime.utcnow()))
    assert not t.results
    e.process(Doc("news", "Escalating conflict closes key shipping strait; global stocks tumble", dt.datetime.utcnow()))
    assert len(t.results) == 1 and t.results[0].after < t.results[0].before


def test_full_pipeline_offline(tmp_path):
    engine, a, b = build(out_path=tmp_path / "o.jsonl")
    docs, w = fetch_all(live=False)
    engine.process_many(docs)
    assert len(engine.signals) >= 40 and b.results and not w
