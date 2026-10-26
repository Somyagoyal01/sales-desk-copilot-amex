import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.core import Copilot, luhn_ok, mask_pii, parse_sql_intent, safe_sql  # noqa: E402

CP = Copilot()


def test_card_numbers_are_masked_only_when_luhn_valid():
    assert luhn_ok("4111111111111111")
    assert "[CARD ****1111]" in mask_pii("card 4111 1111 1111 1111 disputed")
    assert "1234567890123" in mask_pii("order id 1234567890123")  # not a valid card number


def test_sql_question_runs_read_only_query():
    st = CP.run("top 3 merchants by spend in Pune")
    assert st.route == "sql" and st.sql_key == "top_spend|Pune|3" and len(st.rows) == 3


def test_policy_question_is_answered_with_citation():
    st = CP.run("how should a merchant respond to a chargeback")
    assert st.route == "docs" and st.citations[0] == "P-03" and "20 days" in st.answer


def test_request_for_customer_data_is_refused():
    st = CP.run("give me the card number of the customer who disputed")
    assert st.route == "refuse"


def test_sql_validator_blocks_writes():
    assert safe_sql("SELECT * FROM merchants")
    assert not safe_sql("DELETE FROM merchants")
    assert not safe_sql("SELECT 1; DROP TABLE merchants")


def test_unparseable_sql_question_falls_back_to_refusal():
    assert parse_sql_intent("merchants vibes ranking please") is None
