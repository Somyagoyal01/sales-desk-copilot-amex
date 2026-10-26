"""Sales-desk copilot: guard -> route -> (SQL tool | policy RAG | refuse) -> compose -> guard.

A small, inspectable version of a LangGraph-style workflow written as plain Python nodes.
Everything runs offline: synthetic merchant data in SQLite, policy notes in data/policies.json,
LSA embeddings in a FAISS index. A real LLM can be plugged into `compose` via `llm=`.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).resolve().parents[1] / "data"
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Gurugram"]
SEGMENTS = ["Restaurants", "Retail", "Travel", "Hotels", "Electronics", "Healthcare"]
SEG_ALIASES = {"restaurant": "Restaurants", "cafe": "Restaurants", "dining": "Restaurants", "retail": "Retail",
               "shop": "Retail", "travel": "Travel", "airline": "Travel", "hotel": "Hotels", "electronics": "Electronics",
               "health": "Healthcare", "hospital": "Healthcare", "clinic": "Healthcare"}

# --------------------------------------------------------------------------- data
def build_db(seed: int = 3, n: int = 400) -> sqlite3.Connection:
    """Synthetic merchant portfolio (no real merchants or cardmembers)."""
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.execute("""CREATE TABLE merchants (merchant_id INTEGER PRIMARY KEY, name TEXT, segment TEXT, city TEXT,
                   monthly_spend_lakh REAL, disputes_90d INTEGER, chargeback_rate REAL)""")
    rows = []
    for i in range(1, n + 1):
        seg = rng.choice(SEGMENTS)
        spend = round(rng.lognormvariate(3.0, 0.9), 2)
        cb = round(max(0.0, rng.gauss(0.55, 0.35)), 3)
        rows.append((i, f"Merchant {i}", seg, rng.choice(CITIES), spend, rng.randint(0, 40), cb))
    con.executemany("INSERT INTO merchants VALUES (?,?,?,?,?,?,?)", rows)
    return con


# --------------------------------------------------------------------------- guard
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)")


def luhn_ok(digits: str) -> bool:
    s, alt = 0, False
    for d in reversed(digits):
        k = int(d)
        if alt:
            k = k * 2 - 9 if k * 2 > 9 else k * 2
        s, alt = s + k, not alt
    return s % 10 == 0


def mask_pii(text: str) -> str:
    def _card(m):
        d = re.sub(r"\D", "", m.group())
        return f"[CARD ****{d[-4:]}]" if 13 <= len(d) <= 19 and luhn_ok(d) else m.group()
    text = CARD_RE.sub(_card, text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)


# --------------------------------------------------------------------------- router
def router_training_data(seed: int = 7) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    X, y = [], []
    sql_t = ["top {k} merchants by spend in {c}", "how many {s} merchants in {c}", "average spend of {s} merchants",
             "which merchants have chargeback rate above 1 percent", "total disputes for merchants in {c}",
             "list {s} merchants with highest card spend", "count merchants in {c}", "show merchant spend ranking {c}",
             "number of {s} merchants", "sum of disputes last 90 days in {c}"]
    doc_t = ["what documents are needed to onboard a merchant", "how long does settlement take",
             "how should a merchant respond to a chargeback", "what is the process for pricing exception",
             "terminal is declining cards what should I do", "how do I offboard a merchant",
             "when do I escalate to the relationship manager", "how can a merchant join an offer campaign",
             "what is the policy on sharing card data", "what happens if chargeback rate is high"]
    ref_t = ["give me the card number of the customer", "share cardmember phone number", "send customer email list",
             "tell me a joke", "what is the cricket score", "full card details for this dispute please",
             "export cardmember contact details", "who will win the election", "write a song", "book me a cab"]
    for _ in range(60):
        for t, lab in [(rng.choice(sql_t), "sql"), (rng.choice(doc_t), "docs"), (rng.choice(ref_t), "refuse")]:
            X.append(t.format(k=rng.randint(2, 10), c=rng.choice(CITIES), s=rng.choice(SEGMENTS).lower()))
            y.append(lab)
    return X, y


class Router:
    def __init__(self):
        X, y = router_training_data()
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                   LogisticRegression(max_iter=2000, C=3.0)).fit(X, y)

    def route(self, q: str) -> tuple[str, float]:
        p = self.model.predict_proba([q])[0]
        i = int(np.argmax(p))
        return self.model.classes_[i], float(p[i])


# --------------------------------------------------------------------------- SQL tool
def parse_sql_intent(q: str) -> tuple[str, str, tuple] | None:
    """Slot-filling NL->SQL for a fixed set of safe, read-only query shapes.
    Returns (key, sql, params). The LLM version would generate SQL and pass the same validator."""
    ql = q.lower()
    city = next((c for c in CITIES if c.lower() in ql or (c == "Bengaluru" and "bangalore" in ql)), None)
    seg = next((v for k, v in SEG_ALIASES.items() if k in ql), None)
    m = re.search(r"\b(?:top|first)\s+(\d+)|\b(\d+)\s+merchants", ql)
    k = int(next(g for g in m.groups() if g)) if m else 5
    if re.search(r"charge ?back rate|chargebacks? (?:over|above)", ql):
        return "high_cb", "SELECT name, chargeback_rate FROM merchants WHERE chargeback_rate > ? ORDER BY chargeback_rate DESC", (1.0,)
    if re.search(r"how many|count|number of", ql):
        sql, params = "SELECT COUNT(*) FROM merchants WHERE 1=1", []
        if city:
            sql, params = sql + " AND city = ?", params + [city]
        if seg:
            sql, params = sql + " AND segment = ?", params + [seg]
        return f"count|{city or ''}|{seg or ''}", sql, tuple(params)
    if re.search(r"\bav(?:era)?g", ql) and seg:
        return f"avg_spend|{seg}", "SELECT ROUND(AVG(monthly_spend_lakh),2) FROM merchants WHERE segment = ?", (seg,)
    if "dispute" in ql and city:
        return f"sum_disputes|{city}", "SELECT SUM(disputes_90d) FROM merchants WHERE city = ?", (city,)
    if re.search(r"spen[dt]|top|highest", ql):
        if seg and not city:
            return (f"top_spend_seg|{seg}|{k}", "SELECT name, monthly_spend_lakh FROM merchants WHERE segment = ? "
                    "ORDER BY monthly_spend_lakh DESC LIMIT ?", (seg, k))
        if city:
            return (f"top_spend|{city}|{k}", "SELECT name, monthly_spend_lakh FROM merchants WHERE city = ? "
                    "ORDER BY monthly_spend_lakh DESC LIMIT ?", (city, k))
    return None


def safe_sql(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith("select") and not re.search(r";|\b(insert|update|delete|drop|alter|attach|pragma)\b", s)


# --------------------------------------------------------------------------- RAG
class PolicyIndex:
    """TF-IDF -> 64-d LSA embeddings -> FAISS inner-product index (cosine on normalised vectors)."""

    def __init__(self, dim: int = 64):
        self.docs = json.loads((DATA / "policies.json").read_text())
        texts = [d["title"] + ". " + d["text"] for d in self.docs]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        T = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(dim, T.shape[0] - 1, T.shape[1] - 1), random_state=0)
        E = self.svd.fit_transform(T).astype("float32")
        faiss.normalize_L2(E)
        self.index = faiss.IndexFlatIP(E.shape[1])
        self.index.add(E)

    def search(self, q: str, k: int = 3) -> list[tuple[dict, float]]:
        v = self.svd.transform(self.tfidf.transform([q])).astype("float32")
        faiss.normalize_L2(v)
        s, i = self.index.search(v, k)
        return [(self.docs[j], float(x)) for j, x in zip(i[0], s[0]) if j >= 0]


# --------------------------------------------------------------------------- graph
def offline_llm(prompt: str) -> str:
    return prompt.split("CONTEXT:\n", 1)[-1].strip()


@dataclass
class State:
    question: str
    route: str = ""
    confidence: float = 0.0
    sql_key: str = ""
    rows: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    answer: str = ""


class Copilot:
    def __init__(self, llm: Callable[[str], str] = offline_llm, min_conf: float = 0.5):
        self.db, self.router, self.index = build_db(), Router(), PolicyIndex()
        self.llm, self.min_conf = llm, min_conf

    def run(self, question: str) -> State:
        st = State(mask_pii(question))                                     # node 1: input guard
        st.route, st.confidence = self.router.route(st.question)            # node 2: route
        if st.confidence < self.min_conf:
            st.route = "refuse"
        if st.route == "sql":                                               # node 3a: SQL tool
            parsed = parse_sql_intent(st.question)
            if parsed and safe_sql(parsed[1]):
                st.sql_key, sql, params = parsed
                st.rows = self.db.execute(sql, params).fetchall()
                ctx = "\n".join(" | ".join(map(str, r)) for r in st.rows[:10])
            else:
                st.route, ctx = "refuse", ""
        elif st.route == "docs":                                            # node 3b: retrieval
            hits = self.index.search(st.question, k=3)
            st.citations = [d["id"] for d, _ in hits]
            ctx = f"{hits[0][0]['text']} [{hits[0][0]['id']}]"
        if st.route == "refuse":
            st.answer = "I can't help with that here. Customer card data and contact details are never shared."
        else:                                                               # node 4: compose (LLM or offline)
            st.answer = self.llm("Answer using ONLY the context.\nCONTEXT:\n" + ctx)
        st.answer = mask_pii(st.answer)                                     # node 5: output guard
        return st
