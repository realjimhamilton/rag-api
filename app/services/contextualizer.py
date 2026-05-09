# app/services/contextualizer.py
"""
Contextual retrieval: prepends LLM-generated context to each chunk before
embedding, making chunks self-interpretable for better search accuracy.

Supports two providers:
  - Anthropic (Haiku 4.5 + prompt caching) for live uploads
  - OpenAI-compatible (DeepInfra / Kimi K2.6) for bulk backfill

See: https://www.anthropic.com/news/contextual-retrieval
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CONTEXT_PREFIX_START = "[Context: "
CONTEXT_PREFIX_END = "] "

CONTEXTUALIZER_PROMPT = (
    "<document>\n{doc_text}\n</document>\n\n"
    "Here is the chunk we want to situate within the whole document:\n\n"
    "<chunk>\n{chunk_text}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the "
    "overall document for the purposes of improving search retrieval of the "
    "chunk. Answer only with the succinct context and nothing else."
)

MAX_CONTEXT_CHARS = 400_000


class Contextualizer:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        max_concurrency: int = 5,
        max_chunks: int = 200,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.max_chunks = max_chunks
        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._anthropic_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._openai_client = AsyncOpenAI(**kwargs)
        return self._openai_client

    def _truncate_doc(self, full_text: str) -> str:
        if len(full_text) <= MAX_CONTEXT_CHARS:
            return full_text
        truncated = full_text[:MAX_CONTEXT_CHARS]
        word_count = len(truncated.split())
        total_words = len(full_text.split())
        return (
            f"[NOTE: Document truncated to first ~{word_count:,} of "
            f"~{total_words:,} total words.]\n\n" + truncated
        )

    async def _call_anthropic(
        self, full_text: str, chunk_text: str
    ) -> Tuple[str, int, int]:
        client = self._get_anthropic_client()

        response = await client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{full_text}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Here is the chunk we want to situate within "
                                "the whole document:\n\n"
                                f"<chunk>\n{chunk_text}\n</chunk>\n\n"
                                "Please give a short succinct context to situate "
                                "this chunk within the overall document for the "
                                "purposes of improving search retrieval of the "
                                "chunk. Answer only with the succinct context "
                                "and nothing else."
                            ),
                        },
                    ],
                }
            ],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        if cache_read > 0 or cache_create > 0:
            logger.debug(
                "Anthropic cache: read=%d create=%d input=%d output=%d",
                cache_read,
                cache_create,
                input_tokens,
                output_tokens,
            )

        return response.content[0].text.strip(), input_tokens, output_tokens

    async def _call_openai(
        self, full_text: str, chunk_text: str
    ) -> Tuple[str, int, int]:
        client = self._get_openai_client()

        prompt = CONTEXTUALIZER_PROMPT.format(
            doc_text=full_text,
            chunk_text=chunk_text,
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )

        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        return response.choices[0].message.content.strip(), input_tokens, output_tokens

    async def _contextualize_one(
        self,
        full_text: str,
        chunk_text: str,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[str, int, int]:
        async with semaphore:
            if self.provider == "anthropic":
                return await self._call_anthropic(full_text, chunk_text)
            else:
                return await self._call_openai(full_text, chunk_text)

    async def contextualize_documents(
        self,
        full_text: str,
        documents: List[Document],
        file_id: str,
    ) -> List[Document]:
        if not documents:
            return documents

        chunk_count = len(documents)
        if chunk_count > self.max_chunks:
            logger.warning(
                "File %s has %d chunks (max %d). Skipping contextualization.",
                file_id,
                chunk_count,
                self.max_chunks,
            )
            return documents

        full_text = self._truncate_doc(full_text)

        logger.info(
            "Contextualizing %d chunks for file %s via %s/%s",
            chunk_count,
            file_id,
            self.provider,
            self.model,
        )

        # Anthropic: sequential to maximize prompt cache hits (chunk 1 warms
        # cache, chunks 2-N read at 10% cost). OpenAI/DeepInfra: concurrent.
        if self.provider == "anthropic":
            semaphore = asyncio.Semaphore(1)
        else:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        tasks = [
            self._contextualize_one(full_text, doc.page_content, semaphore)
            for doc in documents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contextualized = []
        success_count = 0
        error_count = 0
        total_input = 0
        total_output = 0

        for doc, result in zip(documents, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Contextualization failed for chunk in file %s: %s",
                    file_id,
                    str(result),
                )
                error_count += 1
                contextualized.append(doc)
                continue

            context_text, inp_tokens, out_tokens = result
            success_count += 1
            total_input += inp_tokens
            total_output += out_tokens

            new_content = (
                f"{CONTEXT_PREFIX_START}{context_text}{CONTEXT_PREFIX_END}"
                f"{doc.page_content}"
            )
            new_metadata = {
                **doc.metadata,
                "original_chunk_text": doc.page_content,
                "is_contextualized": True,
                "contextualized_at": datetime.now(timezone.utc).isoformat(),
                "contextualizer_model": f"{self.provider}/{self.model}",
            }
            contextualized.append(
                Document(page_content=new_content, metadata=new_metadata)
            )

        logger.info(
            "Contextualization complete for file %s: %d/%d ok, %d errors, "
            "%d/%d tokens (in/out)",
            file_id,
            success_count,
            chunk_count,
            error_count,
            total_input,
            total_output,
        )

        return contextualized


def strip_context_prefix(text: str) -> str:
    """Remove the [Context: ...] prefix from a contextualized chunk."""
    if text.startswith(CONTEXT_PREFIX_START):
        end_idx = text.find(CONTEXT_PREFIX_END, len(CONTEXT_PREFIX_START))
        if end_idx != -1:
            return text[end_idx + len(CONTEXT_PREFIX_END) :]
    return text
