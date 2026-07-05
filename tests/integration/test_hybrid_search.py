"""Integration tests for hybrid (vector + full-text) retrieval on real pgvector.

Spins up a real PostgreSQL+pgvector container (see tests/integration/conftest.py)
and exercises ExtendedPgVector.hybrid_search_with_score_by_vector end to end.

Fixture data (one file, four chunks; embeddings are vector(3)):

    A  emb [0.99, 0.01, 0]  very close to query [1,0,0]   no keyword
    D  emb [0.80, 0.20, 0]  fairly close                  no keyword
    C  emb [0.10, 0.90, 0]  far (cosine relevance ~0.11)  no keyword
    B  emb [0.00, 1.00, 0]  orthogonal (relevance ~0)     HAS keyword "zorptastic"

For query vector [1,0,0]:
  - pure-vector top-3 = [A, D, C]  (B is 4th; it is semantically distant)
  - hybrid top-3      = [A, D, B]  (the keyword bonus lifts B above C)

That is the whole point: a chunk the dense vector search ranks too low to return
is surfaced by the lexical signal, without disturbing the vector winners.

Run with:  pytest tests/integration/ -m integration -v
"""

import json
import logging
import uuid

import pytest
from sqlalchemy import text

from app.services.vector_store.extended_pg_vector import ExtendedPgVector

pytestmark = pytest.mark.integration

QUERY_VEC = [1.0, 0.0, 0.0]
QUERY_VEC_SQL = "[1.0,0.0,0.0]"
KEYWORD = "zorptastic"

# (label, document, embedding-literal). Documents are keyword-free EXCEPT B.
CHUNKS = [
    ("A", "alpha bright morning coffee ritual", "[0.99,0.01,0.0]"),
    ("D", "delta gentle river flowing downstream", "[0.80,0.20,0.0]"),
    ("C", "charlie quiet mountain trail at dusk", "[0.10,0.90,0.0]"),
    ("B", f"bravo the {KEYWORD} protocol appears right here", "[0.0,1.0,0.0]"),
]


@pytest.fixture()
def hybrid_store(engine, collection_id):
    """ExtendedPgVector wired to the test engine.

    get_collection is overridden to return the known collection UUID directly,
    so the test exercises OUR hybrid SQL without depending on LangChain's
    collection-lookup internals.
    """

    class HybridTestStore(ExtendedPgVector):
        def __init__(self):
            self._bind = engine
            self.collection_name = "test_collection"
            self.logger = logging.getLogger("test.hybrid")
            self._cid = collection_id

        def get_collection(self, session):
            col = type("Col", (), {})()
            col.uuid = self._cid
            return col

    return HybridTestStore()


def _seed_file(engine, collection_id, file_id):
    """Insert the four fixture chunks under one file_id."""
    with engine.begin() as conn:
        for i, (_label, doc, emb) in enumerate(CHUNKS):
            conn.execute(
                text(
                    "INSERT INTO langchain_pg_embedding "
                    "(collection_id, embedding, document, cmetadata, custom_id) "
                    "VALUES (:cid, CAST(:emb AS vector), :doc, CAST(:meta AS jsonb), :cust)"
                ),
                {
                    "cid": collection_id,
                    "emb": emb,
                    "doc": doc,
                    "meta": json.dumps({"file_id": file_id, "user_id": "u1"}),
                    "cust": f"{file_id}-{i}",
                },
            )


def _vector_topk(engine, collection_id, file_id, k):
    """Pure-vector baseline: the exact semantics of the non-hybrid /query path
    (cosine distance, filtered by file_id, ascending), computed via raw SQL so
    the test does not need a fully constructed PGVector for the baseline."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT document, (embedding <=> CAST(:qvec AS vector)) AS dist "
                "FROM langchain_pg_embedding "
                "WHERE collection_id = :cid AND cmetadata->>'file_id' = :fid "
                "ORDER BY dist ASC LIMIT :k"
            ),
            {"qvec": QUERY_VEC_SQL, "cid": collection_id, "fid": file_id, "k": k},
        ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def test_keyword_only_chunk_surfaced_by_hybrid(engine, collection_id, hybrid_store):
    file_id = f"file-hyb-{uuid.uuid4().hex[:8]}"
    _seed_file(engine, collection_id, file_id)

    # Pure-vector top-3 must NOT include the keyword chunk B.
    vec_docs = [doc for doc, _ in _vector_topk(engine, collection_id, file_id, 3)]
    assert not any(
        KEYWORD in d.lower() for d in vec_docs
    ), f"Precondition failed: keyword chunk unexpectedly in pure-vector top-3: {vec_docs}"

    # Hybrid top-3 MUST surface the keyword chunk B.
    hits = hybrid_store.hybrid_search_with_score_by_vector(
        QUERY_VEC, KEYWORD, k=3, filter={"file_id": {"$eq": file_id}}
    )
    hy_docs = [d.page_content.lower() for d, _ in hits]
    assert any(
        KEYWORD in d for d in hy_docs
    ), f"Hybrid failed to surface the keyword chunk. Got: {hy_docs}"
    # And the vector winners are preserved (A and D still present).
    assert any("alpha" in d for d in hy_docs)
    assert any("delta" in d for d in hy_docs)


def test_non_keyword_query_identical_to_vector(engine, collection_id, hybrid_store):
    file_id = f"file-hyb-{uuid.uuid4().hex[:8]}"
    _seed_file(engine, collection_id, file_id)

    # A query with zero lexical overlap must yield byte-identical results
    # (same docs, same order, same distances) to the pure-vector path.
    hits = hybrid_store.hybrid_search_with_score_by_vector(
        QUERY_VEC, "qwxyz-no-such-token", k=4, filter={"file_id": {"$eq": file_id}}
    )
    hybrid_pairs = [(d.page_content, round(s, 6)) for d, s in hits]
    vector_pairs = [
        (doc, round(dist, 6))
        for doc, dist in _vector_topk(engine, collection_id, file_id, 4)
    ]
    assert hybrid_pairs == vector_pairs


def test_keyword_bonus_only_improves_and_distances_nonnegative(
    engine, collection_id, hybrid_store
):
    file_id = f"file-hyb-{uuid.uuid4().hex[:8]}"
    _seed_file(engine, collection_id, file_id)

    hits = hybrid_store.hybrid_search_with_score_by_vector(
        QUERY_VEC, KEYWORD, k=4, filter={"file_id": {"$eq": file_id}}
    )
    by_doc = {d.page_content: s for d, s in hits}

    # No returned distance is negative (the upper LEAST(1, ..) clamp guarantees it).
    assert all(s >= 0.0 for s in by_doc.values()), by_doc

    # The keyword chunk's fused distance must be strictly SMALLER (better) than
    # its raw cosine distance: the bonus only ever improves a keyword match.
    raw = dict(_vector_topk(engine, collection_id, file_id, 4))
    b_doc = next(d for d in by_doc if KEYWORD in d.lower())
    assert (
        by_doc[b_doc] < raw[b_doc]
    ), f"Keyword bonus did not improve the match: fused={by_doc[b_doc]} raw={raw[b_doc]}"

    # Non-keyword chunks are unchanged (fused distance == raw cosine distance).
    for d, s in by_doc.items():
        if KEYWORD not in d.lower():
            assert round(s, 6) == round(raw[d], 6)


def test_hybrid_supports_in_filter_across_files(engine, collection_id, hybrid_store):
    """/query_multiple uses a $in filter; hybrid must scope to that set."""
    file_a = f"file-hyb-{uuid.uuid4().hex[:8]}"
    file_b = f"file-hyb-{uuid.uuid4().hex[:8]}"
    file_other = f"file-hyb-{uuid.uuid4().hex[:8]}"
    _seed_file(engine, collection_id, file_a)
    _seed_file(engine, collection_id, file_b)
    _seed_file(engine, collection_id, file_other)

    hits = hybrid_store.hybrid_search_with_score_by_vector(
        QUERY_VEC,
        KEYWORD,
        k=100,
        filter={"file_id": {"$in": [file_a, file_b]}},
    )
    # 4 chunks per file, two files in scope -> 8 rows, none from file_other.
    assert len(hits) == 8


def test_fts_index_ddl_is_valid_and_idempotent(engine):
    """The exact CREATE INDEX from ensure_vector_indexes() must run (twice)."""
    ddl = text(
        "CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_document_fts "
        "ON langchain_pg_embedding USING gin (to_tsvector('english', document))"
    )
    with engine.begin() as conn:
        conn.execute(ddl)
    with engine.begin() as conn:
        conn.execute(ddl)  # idempotent
