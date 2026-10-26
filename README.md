# Sales-Desk Copilot (route → SQL tool or policy RAG → guarded answer)

> Personal project by Somya Goyal, built while applying to American Express. Not affiliated with or endorsed by American Express; uses public/synthetic data. The merchant data and policy notes are synthetic and fictional.

## Why this matters for American Express

**Problem:** sales and merchant-servicing teams switch between two kinds of questions all day. Some are data questions ("top 5 retail merchants by spend in Pune"); others are process questions ("how long does a merchant have to answer a chargeback?"). Answers live in different systems, and card data must never leak.
**Who feels it:** merchant sales teams and relationship managers.
**What this shows:** a small copilot that routes each question to a **read-only SQL tool** or to **retrieval over policy notes (FAISS)**, refuses requests for customer data, and masks card numbers on the way in and out. It is evaluated with precision and recall, not just demoed.

Public context: the AI Engineer II posting (Amex job 26015436) asks for GenAI copilots and RAG pipelines that "automate manual processes", Python + SQL, evaluating AI outputs with precision/recall, vector databases (FAISS), agentic orchestration (LangGraph), and responsible-AI and data-privacy practice. Each of those appears below.

## Results (real numbers from `python -m copilot.evaluate`)

| Test | Result |
|---|---|
| Routing on generated questions (180, same templates as training) | accuracy 1.00 |
| **Routing on 20 hand-written held-out questions** (written before the code, never tuned on) | **accuracy 0.80**: sql recall 0.86, docs recall 0.63, refuse precision 0.56 / recall 1.00 |
| Unsafe requests refused (card numbers, cardmember contacts, off-topic) | **5 / 5** |
| SQL questions mapped to the exact correct query | 6 / 7 |
| Policy retrieval (FAISS over LSA embeddings) | hit@1 0.75, hit@3 0.875 |

What the gap means: every routing error on the held-out set was an **over-refusal** (for example "POS machine keeps rejecting cards" was refused instead of answered). This is the safe direction for a bank, but it costs usefulness. The fix is more varied training questions or an LLM router, measured again on this same held-out set.

## How it works

`copilot/core.py` is a 5-node graph written as plain Python so every step is testable:
1. **Input guard**: masks Luhn-valid card numbers (keeps the last 4), emails and phone numbers.
2. **Router**: TF-IDF + logistic regression → `sql` / `docs` / `refuse`, with a confidence threshold.
3. **Tools**:
   - **SQL**: slot-filling NL→SQL for a whitelist of query shapes over a synthetic SQLite merchant table. A validator only allows a single `SELECT`.
   - **RAG**: TF-IDF → 64-d LSA embeddings → `faiss.IndexFlatIP`; the top passage is cited.
4. **Compose**: an offline composer by default. Pass `llm=` (Azure OpenAI / Gemini / Claude wrapper) to generate fluent answers from the same context.
5. **Output guard**: PII masking again.

## Run it

```bash
pip install -r requirements.txt
python -m copilot.evaluate      # writes results/metrics.json
pytest -q                       # 6 tests: masking, SQL safety, routing, citations, refusals
```

## Limitations / next steps

- Synthetic data and a small router training set. The held-out score (0.80) is the honest number; the generated score (1.00) is not.
- The NL→SQL step is template-based. An LLM text-to-SQL path would need the same `safe_sql` validator plus a check that executes on a read replica.
- Next: an LLM router with few-shot examples, a rewrite step for paraphrased policy questions ("papers to sign up" → onboarding documents), and tracing and monitoring of refusal rates.
