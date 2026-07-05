"""DB-free unit tests for ExtendedPgVector.hybrid_search_with_score_by_vector.

These cover the guard logic that runs BEFORE any database access:
  - an unsafe FTS language identifier is rejected (SQL-literal injection guard)
  - a call with no file filter falls back to standard vector search rather than
    scanning the whole collection

The ranking / fusion behavior against a real pgvector database lives in
tests/integration/test_hybrid_search.py (requires Docker).
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.services.vector_store.extended_pg_vector import ExtendedPgVector


class DummyExtendedPgVector(ExtendedPgVector):
    """Bypass PGVector.__init__ (no DB); enough to exercise the guard logic."""

    def __init__(self):
        self._bind = None


@pytest.mark.parametrize(
    "bad_lang",
    [
        "english'; DROP TABLE langchain_pg_embedding; --",
        "en glish",
        "english; select 1",
        "",
    ],
)
def test_hybrid_rejects_unsafe_language(bad_lang):
    store = DummyExtendedPgVector()
    with pytest.raises(ValueError):
        store.hybrid_search_with_score_by_vector(
            [0.1, 0.2, 0.3],
            "query",
            k=3,
            filter={"file_id": {"$eq": "f1"}},
            lang=bad_lang,
        )


def test_hybrid_without_file_filter_falls_back_to_vector():
    store = DummyExtendedPgVector()
    expected = [(Document(page_content="x", metadata={}), 0.4)]
    with patch.object(
        ExtendedPgVector,
        "similarity_search_with_score_by_vector",
        return_value=expected,
    ) as mock:
        out = store.hybrid_search_with_score_by_vector(
            [0.1, 0.2, 0.3], "query", k=3, filter=None
        )
    mock.assert_called_once_with([0.1, 0.2, 0.3], k=3, filter=None)
    assert out == expected


def test_hybrid_empty_file_in_list_falls_back_to_vector():
    """An $in with an empty list has no file to scope to -> fall back."""
    store = DummyExtendedPgVector()
    expected = [(Document(page_content="x", metadata={}), 0.4)]
    with patch.object(
        ExtendedPgVector,
        "similarity_search_with_score_by_vector",
        return_value=expected,
    ) as mock:
        out = store.hybrid_search_with_score_by_vector(
            [0.1, 0.2, 0.3], "query", k=3, filter={"file_id": {"$in": []}}
        )
    mock.assert_called_once()
    assert out == expected
