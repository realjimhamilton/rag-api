import os
import time
import logging
from typing import Optional, Any, Dict, List, Tuple, Union
from sqlalchemy import event
from sqlalchemy import delete, text
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from langchain_core.documents import Document
from langchain_community.vectorstores.pgvector import (
    PGVector,
    COMPARISONS_TO_NATIVE,
    SUPPORTED_OPERATORS,
)


class ExtendedPgVector(PGVector):
    _query_logging_setup = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_query_logging()

    @staticmethod
    def _sanitize_parameters_for_logging(
        parameters: Union[Dict, List, tuple, Any]
    ) -> Any:
        """Sanitize parameters for logging by truncating embeddings and large values."""
        if parameters is None:
            return parameters

        if isinstance(parameters, dict):
            sanitized = {}
            for key, value in parameters.items():
                # Check if the key contains 'embedding' or if the value looks like an embedding vector
                if "embedding" in str(key).lower() or (
                    isinstance(value, (list, tuple))
                    and len(value) > 10
                    and all(isinstance(x, (int, float)) for x in value[:10])
                ):
                    sanitized[key] = f"<embedding vector of length {len(value)}>"
                elif isinstance(value, str) and len(value) > 500:
                    sanitized[key] = value[:500] + "... (truncated)"
                elif isinstance(value, (dict, list, tuple)):
                    sanitized[key] = ExtendedPgVector._sanitize_parameters_for_logging(
                        value
                    )
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(parameters, (list, tuple)):
            sanitized = []
            # Check if this is a list of embeddings
            if len(parameters) > 0 and all(
                isinstance(item, (list, tuple))
                and len(item) > 10
                and all(isinstance(x, (int, float)) for x in item[: min(10, len(item))])
                for item in parameters
            ):
                return f"<{len(parameters)} embedding vectors>"

            for item in parameters:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) > 10
                    and all(isinstance(x, (int, float)) for x in item[:10])
                ):
                    sanitized.append(f"<embedding vector of length {len(item)}>")
                elif isinstance(item, str) and len(item) > 500:
                    sanitized.append(item[:500] + "... (truncated)")
                elif isinstance(item, (dict, list, tuple)):
                    sanitized.append(
                        ExtendedPgVector._sanitize_parameters_for_logging(item)
                    )
                else:
                    sanitized.append(item)
            return type(parameters)(sanitized)
        else:
            return parameters

    def setup_query_logging(self):
        """Enable query logging for this vector store only if DEBUG_PGVECTOR_QUERIES is set"""
        # Only setup logging if the environment variable is set to a truthy value
        debug_queries = os.getenv("DEBUG_PGVECTOR_QUERIES", "").lower()
        if debug_queries not in ["true", "1", "yes", "on"]:
            return

        # Only setup once per class
        if ExtendedPgVector._query_logging_setup:
            return

        logger = logging.getLogger("pgvector.queries")
        logger.setLevel(logging.INFO)

        # Create handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - PGVECTOR QUERY - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        @event.listens_for(Engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "langchain_pg_embedding" in statement:
                context._query_start_time = time.time()
                logger.info(f"STARTING QUERY: {statement}")
                sanitized_params = ExtendedPgVector._sanitize_parameters_for_logging(
                    parameters
                )
                logger.info(f"PARAMETERS: {sanitized_params}")

        @event.listens_for(Engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "langchain_pg_embedding" in statement:
                total = time.time() - context._query_start_time
                logger.info(f"COMPLETED QUERY in {total:.4f}s")
                logger.info("-" * 50)

        ExtendedPgVector._query_logging_setup = True

    def _handle_field_filter(self, field: str, value: Any) -> Any:
        """Override LangChain's filter to avoid jsonb_path_match() for equality ops.

        LangChain's default _handle_field_filter uses func.jsonb_path_match() for
        $eq/$ne/$lt/$gt etc. That function-call predicate cannot use B-tree expression
        indexes like (cmetadata->>'file_id') or GIN jsonb_path_ops indexes, forcing
        PostgreSQL into sequential scans on large tables.

        This override rewrites $eq and $ne to use the ->>' astext operator instead,
        producing WHERE (cmetadata->>'field') = 'value' which hits expression indexes.
        All other operators ($lt, $gt, $in, $between, etc.) delegate to the parent.
        """
        if not isinstance(field, str):
            raise ValueError(
                f"field should be a string but got: {type(field)} with value: {field}"
            )
        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: {field}"
            )
        if not field.isidentifier():
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

        if isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which "
                    "is a dictionary with a single key that corresponds to an operator "
                    f"but got a dictionary with {len(value)} keys. The first few "
                    f"keys are: {list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. "
                    f"Expected one of {SUPPORTED_OPERATORS}"
                )
        else:
            operator = "$eq"
            filter_value = value

        if operator == "$eq":
            return self.EmbeddingStore.cmetadata[field].astext == str(filter_value)
        elif operator == "$ne":
            return self.EmbeddingStore.cmetadata[field].astext != str(filter_value)

        return super()._handle_field_filter(field, value)

    def get_all_ids(self) -> list[str]:
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_filtered_ids(self, ids: list[str]) -> list[str]:
        with Session(self._bind) as session:
            query = session.query(self.EmbeddingStore.custom_id).filter(
                self.EmbeddingStore.custom_id.in_(ids)
            )
            results = query.all()
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore)
                .filter(self.EmbeddingStore.custom_id.in_(ids))
                .all()
            )
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
                if result.custom_id in ids
            ]

    def hybrid_search_with_score_by_vector(
        self,
        embedding: List[float],
        query_text: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        lang: str = "english",
        keyword_bonus: float = 0.35,
    ) -> List[Tuple[Document, float]]:
        """Hybrid retrieval: fuse pgvector cosine search with Postgres full-text
        (lexical) search for the filtered file(s).

        Returns (Document, distance) tuples where ``distance = 1 - fused_rel`` on
        the SAME cosine scale the pure-vector path returns, so the LibreChat
        fileSearch.js re-rank / pollution-floor logic composes unchanged. Fusion:

            cosine_sim = 1 - (embedding <=> query)     # exactly the vector score
            kw_pos     = rank of the chunk among the keyword matches (1 = best)
            fused_rel  = min(1, cosine_sim + keyword_bonus / kw_pos)   # cap at 1

        A chunk with no lexical match gets no bonus, so ``fused_rel = min(1,
        cosine_sim) = cosine_sim`` (cosine_sim is always <= 1) and ``distance``
        collapses to the raw cosine distance: non-keyword queries return
        byte-for-byte identical results to the pure-vector path. The only cap is
        the upper one (min with 1), which keeps ``distance >= 0`` when a strong
        keyword bonus would otherwise push relevance above 1.

        ``lang`` is interpolated as a SQL literal (a regconfig cannot be a bind
        parameter, and it must match the functional FTS index expression), so it
        MUST be a validated bare identifier. config.HYBRID_FTS_LANGUAGE is
        validated at import; we defensively re-check here.

        Supports the two filter shapes the routes use: ``{"file_id": {"$eq": id}}``
        (or a bare id) and ``{"file_id": {"$in": [ids]}}``. Without a file filter
        it falls back to the standard vector search rather than scanning the whole
        collection.
        """
        if not isinstance(lang, str) or not lang.replace("_", "").isalnum():
            raise ValueError(f"Unsafe FTS language identifier: {lang!r}")

        single_file_id = None
        file_ids = None
        if filter and "file_id" in filter:
            raw = filter["file_id"]
            if isinstance(raw, dict):
                if "$eq" in raw:
                    single_file_id = raw["$eq"]
                elif "$in" in raw:
                    file_ids = list(raw["$in"])
            else:
                single_file_id = raw
        if single_file_id is None and not file_ids:
            # Hybrid is only defined for the per-file query path; without a file
            # filter, fall back to standard vector search to avoid an unbounded
            # cross-collection scan.
            return self.similarity_search_with_score_by_vector(
                embedding, k=k, filter=filter
            )

        params: Dict[str, Any] = {
            "qtext": query_text or "",
            "qvec": "[" + ",".join(str(float(x)) for x in embedding) + "]",
            "keyword_bonus": float(keyword_bonus),
            "k": int(k),
        }
        if file_ids is not None:
            file_predicate = "e.cmetadata->>'file_id' = ANY(:file_ids)"
            params["file_ids"] = [str(f) for f in file_ids]
        else:
            file_predicate = "e.cmetadata->>'file_id' = :file_id"
            params["file_id"] = str(single_file_id)

        sql = text(
            f"""
            WITH q AS (
                SELECT websearch_to_tsquery('{lang}', :qtext) AS tsq
            ),
            scored AS (
                SELECT
                    e.document AS document,
                    e.cmetadata AS cmetadata,
                    (e.embedding <=> CAST(:qvec AS vector)) AS cos_dist,
                    (to_tsvector('{lang}', e.document) @@ q.tsq) AS is_kw,
                    CASE WHEN to_tsvector('{lang}', e.document) @@ q.tsq
                         THEN ts_rank_cd(to_tsvector('{lang}', e.document), q.tsq)
                         ELSE 0 END AS fts_score
                FROM langchain_pg_embedding e, q
                WHERE e.collection_id = :collection_id
                  AND {file_predicate}
            ),
            ranked AS (
                SELECT *,
                    CASE WHEN is_kw THEN
                        row_number() OVER (
                            PARTITION BY is_kw
                            ORDER BY fts_score DESC, cos_dist ASC
                        )
                    ELSE NULL END AS kw_pos
                FROM scored
            )
            SELECT
                document,
                cmetadata,
                (1.0 - LEAST(1.0,
                             (1.0 - cos_dist)
                             + CASE WHEN is_kw THEN (:keyword_bonus / kw_pos)
                                    ELSE 0.0 END
                )) AS distance
            FROM ranked
            ORDER BY distance ASC
            LIMIT :k
            """
        )

        with Session(self._bind) as session:
            collection = self.get_collection(session)
            if collection is None:
                self.logger.warning(
                    "Hybrid search: collection %r not found", self.collection_name
                )
                return []
            params["collection_id"] = collection.uuid
            rows = session.execute(sql, params).fetchall()

        results: List[Tuple[Document, float]] = []
        for document, cmetadata, distance in rows:
            results.append(
                (
                    Document(page_content=document or "", metadata=cmetadata or {}),
                    float(distance),
                )
            )
        return results

    def _delete_multiple(
        self, ids: Optional[list[str]] = None, collection_only: bool = False
    ) -> None:
        with Session(self._bind) as session:
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )
                stmt = delete(self.EmbeddingStore)
                if collection_only:
                    collection = self.get_collection(session)
                    if not collection:
                        self.logger.warning("Collection not found")
                        return
                    stmt = stmt.where(
                        self.EmbeddingStore.collection_id == collection.uuid
                    )
                stmt = stmt.where(self.EmbeddingStore.custom_id.in_(ids))
                session.execute(stmt)
            session.commit()
