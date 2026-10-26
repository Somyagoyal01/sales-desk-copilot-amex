"""Evaluate routing, SQL answers and retrieval.

    python -m copilot.evaluate

Generated test = new queries from the router's template families (same distribution).
Held-out = 20 hand-written queries in data/heldout_queries.jsonl, never used for training or tuning.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

from .core import DATA, Copilot, parse_sql_intent, router_training_data

RESULTS = Path(__file__).resolve().parents[1] / "results"


def route_metrics(y_true, y_pred) -> dict:
    labels = ["sql", "docs", "refuse"]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {lab: {"precision": round(float(pp), 3), "recall": round(float(rr), 3), "n": int(ss)}
            for lab, pp, rr, ss in zip(labels, p, r, s)} | {
        "accuracy": round(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true), 3)}


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    cp = Copilot()
    Xg, yg = router_training_data(seed=99)
    gen_pred = [cp.run(q).route for q in Xg]
    rows = [json.loads(l) for l in (DATA / "heldout_queries.jsonl").read_text().splitlines() if l.strip()]
    states = [cp.run(r["q"]) for r in rows]
    out = {"generated_routing": route_metrics(yg, gen_pred),
           "heldout_routing": route_metrics([r["route"] for r in rows], [s.route for s in states])}
    sql_rows = [(r, s) for r, s in zip(rows, states) if r["route"] == "sql"]
    sql_ok = [s.route == "sql" and s.sql_key == r["sql_key"] for r, s in sql_rows]
    out["heldout_sql_exact_query"] = {"n": len(sql_rows), "correct": int(sum(sql_ok)),
                                      "misses": [r["q"] for (r, _), ok in zip(sql_rows, sql_ok) if not ok]}
    doc_rows = [r for r in rows if r["route"] == "docs"]
    hit1 = [cp.index.search(r["q"], 1)[0][0]["id"] == r["expect_doc"] for r in doc_rows]
    hit3 = [r["expect_doc"] in [d["id"] for d, _ in cp.index.search(r["q"], 3)] for r in doc_rows]
    out["heldout_retrieval"] = {"n": len(doc_rows), "hit@1": round(sum(hit1) / len(doc_rows), 3),
                                "hit@3": round(sum(hit3) / len(doc_rows), 3),
                                "misses@1": [r["q"] for r, h in zip(doc_rows, hit1) if not h]}
    ref = [(r, s) for r, s in zip(rows, states) if r["route"] == "refuse"]
    out["heldout_unsafe_requests_refused"] = f"{sum(s.route == 'refuse' for _, s in ref)}/{len(ref)}"
    out["heldout_route_errors"] = [(r["q"], r["route"], s.route) for r, s in zip(rows, states) if r["route"] != s.route]
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
