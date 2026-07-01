"""Unit tests for VoyageContextualizedEmbeddings (mocked Voyage client).

Verifies the load-bearing behaviors:
  - one file's chunks are sent as a single document group (inputs=[chunks])
  - per-chunk vectors come back in order (across token-budget batches too)
  - queries use input_type="query"
  - a count mismatch is rejected rather than silently stored misaligned
"""

import pytest

from app.services.voyage_contextual import VoyageContextualizedEmbeddings


class _FakeResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeResponse:
    def __init__(self, results):
        self.results = results


class _FakeVoyageClient:
    """Records calls and returns one vector per chunk in the (single) input group.
    Each vector's first element is the chunk's char length, so ordering is
    assertable across batches."""

    def __init__(self, force_count=None):
        self.calls = []
        self.force_count = force_count

    def contextualized_embed(
        self, inputs, model, input_type, output_dtype="float", output_dimension=None
    ):
        self.calls.append(
            {
                "inputs": inputs,
                "model": model,
                "input_type": input_type,
                "output_dtype": output_dtype,
                "output_dimension": output_dimension,
            }
        )
        group = inputs[0]
        n = self.force_count if self.force_count is not None else len(group)
        dim = output_dimension or 4
        embeddings = [
            [float(len(group[i]))] + [0.0] * (dim - 1) for i in range(n)
        ]
        return _FakeResponse([_FakeResult(embeddings)])


def _make_embedder(client, **kwargs):
    emb = VoyageContextualizedEmbeddings(api_key="test-key", **kwargs)
    emb._client = client  # bypass lazy SDK import
    return emb


def test_embed_documents_single_group_preserves_order():
    client = _FakeVoyageClient()
    emb = _make_embedder(client, output_dimension=4, max_tokens_per_request=1_000_000)
    texts = ["aaaa", "bb", "cccccc"]

    vectors = emb.embed_documents(texts)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["input_type"] == "document"
    assert call["inputs"] == [texts]  # one document group, all chunks together
    assert call["output_dimension"] == 4
    # First element of each vector encodes the chunk length → order check.
    assert [v[0] for v in vectors] == [4.0, 2.0, 6.0]


def test_embed_documents_splits_on_token_budget_and_keeps_order():
    client = _FakeVoyageClient()
    # ~tokens = chars//4. The embedder clamps the budget to a 1000-token floor, so
    # use ~600/500/700-token chunks against a 1000-token budget to force a split.
    emb = _make_embedder(client, output_dimension=2, max_tokens_per_request=1000)
    texts = ["a" * 2400, "b" * 2000, "c" * 2800]  # ~600, 500, 700 tokens

    vectors = emb.embed_documents(texts)

    assert len(client.calls) >= 2  # split into multiple contextualized requests
    for call in client.calls:
        assert call["input_type"] == "document"
        assert len(call["inputs"]) == 1  # always exactly one document group per call
    assert len(vectors) == 3
    assert [v[0] for v in vectors] == [2400.0, 2000.0, 2800.0]  # order preserved across batches


def test_embed_query_uses_query_input_type():
    client = _FakeVoyageClient()
    emb = _make_embedder(client, output_dimension=4)

    vector = emb.embed_query("what is the refund policy")

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["input_type"] == "query"
    assert call["inputs"] == [["what is the refund policy"]]
    assert vector[0] == float(len("what is the refund policy"))


def test_count_mismatch_raises():
    client = _FakeVoyageClient(force_count=2)  # returns 2 vectors for 3 chunks
    emb = _make_embedder(client, output_dimension=4, max_tokens_per_request=1_000_000)

    with pytest.raises(RuntimeError):
        emb.embed_documents(["a", "b", "c"])


def test_empty_documents_returns_empty():
    client = _FakeVoyageClient()
    emb = _make_embedder(client)
    assert emb.embed_documents([]) == []
    assert client.calls == []


def test_missing_api_key_raises():
    with pytest.raises(ValueError):
        VoyageContextualizedEmbeddings(api_key="")


class _RejectingVoyageClient:
    """Rejects any group larger than ``max_group`` with voyage's real 32K-token
    error, so embed_documents must recursively split. Records every group size it
    is asked to embed (including the rejected attempts)."""

    def __init__(self, max_group):
        self.max_group = max_group
        self.calls = []

    def contextualized_embed(
        self, inputs, model, input_type, output_dtype="float", output_dimension=None
    ):
        group = inputs[0]
        self.calls.append(len(group))
        if len(group) > self.max_group:
            raise Exception(
                "Request to model 'voyage-context-4' failed. The example at index 0 "
                "in your batch has too many tokens and does not fit into the model's "
                "context window of 32000 tokens"
            )
        dim = output_dimension or 4
        embeddings = [[float(len(group[i]))] + [0.0] * (dim - 1) for i in range(len(group))]
        return _FakeResponse([_FakeResult(embeddings)])


def test_over_token_limit_recursively_splits_and_keeps_order():
    # Dense content: a group voyage rejects for exceeding the 32K-token document
    # limit must be halved until each piece fits, without dropping or reordering.
    client = _RejectingVoyageClient(max_group=3)
    emb = _make_embedder(client, output_dimension=4, max_tokens_per_request=10**9)
    texts = [chr(ord("a") + i) * (i + 1) for i in range(10)]  # distinct lengths 1..10

    vectors = emb.embed_documents(texts)

    assert len(vectors) == 10
    assert [v[0] for v in vectors] == [float(i + 1) for i in range(10)]  # order preserved
    # every SUCCESSFUL (non-raising) call embedded a group within the limit
    successful = [n for n in client.calls if n <= 3]
    assert sum(successful) == 10 and all(n <= 3 for n in successful)
