# app/services/voyage_contextual.py
"""
Voyage contextualized-chunk embeddings (voyage-context-4 / voyage-context-3).

This is a native replacement for the two-stage "contextual retrieval" pipeline
(LLM writes a per-chunk context blurb, then a context-agnostic embedder embeds
the prepended text). Voyage's contextualized models process a whole document in
a single encoder pass and emit one vector per chunk that already encodes global
document context. No separate LLM call, one request per document, higher
retrieval accuracy. See:
  https://www.mongodb.com/docs/voyageai/models/contextualized-chunk-embeddings/
  https://blog.voyageai.com/2025/07/23/voyage-context-3/

Integration shape
-----------------
In this service, every `store_data_in_vector_db` call handles exactly one file's
chunks, so when LangChain's vector store calls `embed_documents(texts)`, the
entire `texts` list is one document's ordered chunks. We therefore send them as a
single contextualized group (`inputs=[chunks]`, `input_type="document"`) so the
model sees them together. Queries go through `embed_query`, which uses
`input_type="query"` with a one-element group (context-agnostic, per the API).

The contextualized endpoint caps a single request (pre-chunked: total tokens and
chunk count). A file whose chunks exceed the per-request token budget is split
across multiple requests; context is preserved WITHIN each request but not across
splits (a rare edge for very large docs, logged when it happens).
"""

import logging
from typing import List, Optional

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Conservative per-request token budget for pre-chunked document inputs. MongoDB's
# docs state pre-chunked inputs must stay under 32K tokens per request; we leave
# headroom. Tunable via VOYAGE_MAX_TOKENS_PER_REQUEST. Token count is estimated as
# chars/4 (same convention the LibreChat side uses); chunks are tiny relative to
# this budget (CHUNK_SIZE=1500 chars ~ 375 tokens), so single-chunk overflow is a
# non-issue.
DEFAULT_MAX_TOKENS_PER_REQUEST = 28_000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class VoyageContextualizedEmbeddings(Embeddings):
    """LangChain Embeddings backed by Voyage's contextualized-chunk endpoint.

    `embed_documents` treats its input as ONE document's ordered chunks (true in
    this codebase: one file per store call). `embed_query` embeds a single query
    with `input_type="query"`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-context-4",
        output_dimension: Optional[int] = 1024,
        output_dtype: str = "float",
        max_tokens_per_request: int = DEFAULT_MAX_TOKENS_PER_REQUEST,
        max_retries: int = 3,
        timeout: Optional[float] = None,
    ):
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required for the voyage embeddings provider")
        self.api_key = api_key
        self.model = model
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.max_tokens_per_request = max(1_000, int(max_tokens_per_request))
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            import voyageai

            self._client = voyageai.Client(
                api_key=self.api_key,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._client

    def _kwargs(self, input_type: str) -> dict:
        kwargs = {"model": self.model, "input_type": input_type, "output_dtype": self.output_dtype}
        if self.output_dimension is not None:
            kwargs["output_dimension"] = self.output_dimension
        return kwargs

    def _batch_by_token_budget(self, texts: List[str]) -> List[List[str]]:
        """Split one document's chunks into groups that each fit the per-request
        token budget. Order is preserved across groups."""
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for text in texts:
            t = _estimate_tokens(text)
            if current and current_tokens + t > self.max_tokens_per_request:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(text)
            current_tokens += t
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _extract_group_embeddings(response) -> List[List[float]]:
        """Pull the per-chunk vectors for the single document group out of a
        contextualized_embed response. Documented SDK shape:
        response.results[0].embeddings (List[List[float]])."""
        results = getattr(response, "results", None)
        if results is None:
            # Defensive: some response shapes expose `.data` instead of `.results`.
            results = getattr(response, "data", None)
        if not results:
            raise RuntimeError("Voyage contextualized_embed returned no results")
        first = results[0]
        embeddings = getattr(first, "embeddings", None)
        if embeddings is None and isinstance(first, dict):
            embeddings = first.get("embeddings")
        if embeddings is None:
            raise RuntimeError("Voyage contextualized_embed result missing 'embeddings'")
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        client = self._get_client()
        batches = self._batch_by_token_budget(texts)
        if len(batches) > 1:
            logger.info(
                "Voyage contextual: document has %d chunks exceeding the %d-token "
                "per-request budget; split into %d contextualized groups (context "
                "preserved within each group, not across).",
                len(texts),
                self.max_tokens_per_request,
                len(batches),
            )

        vectors: List[List[float]] = []
        for batch in batches:
            response = client.contextualized_embed(
                inputs=[batch],  # one document group → one inner list of ordered chunks
                **self._kwargs("document"),
            )
            group = self._extract_group_embeddings(response)
            if len(group) != len(batch):
                raise RuntimeError(
                    f"Voyage contextualized_embed returned {len(group)} vectors for "
                    f"{len(batch)} chunks; refusing to store a misaligned set."
                )
            vectors.extend(group)

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Voyage contextual embedding count {len(vectors)} != chunk count "
                f"{len(texts)}; aborting to avoid storing a misaligned set."
            )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        client = self._get_client()
        response = client.contextualized_embed(
            inputs=[[text]],  # single query, embedded context-agnostically
            **self._kwargs("query"),
        )
        group = self._extract_group_embeddings(response)
        if not group:
            raise RuntimeError("Voyage contextualized_embed returned no query vector")
        return group[0]
