"""Tests for the markdown-aware chunking branch in _prepare_documents_sync.

Verifies that .md files get header-aware splitting (each ## section becomes
its own chunk, prepended with the header path so it ends up in the
embedding), while non-.md files keep the existing
RecursiveCharacterTextSplitter behavior.
"""
from langchain_core.documents import Document

from app.routes.document_routes import _prepare_documents_sync, _split_markdown_aware


MODULE_LIBRARY_FIXTURE = """# AI Builder Module Library

A curated reference for assembling custom AI bot prompts.

## Client Presentation

Use when the bot is presenting recommendations to a client.
Includes framing for trade-offs, risk language, and a close.
Insert verbatim before the OPERATIONAL CONSTRAINTS block.

### Variant: Executive brief

Compressed form for C-suite. Drop subsection two.

## Security Protocol

Use when the bot will handle PII or financial data.
Includes prompt-leak protection and refusal patterns.

## Diagnostic Framework

A symptom → root cause → prerequisite check pattern.
Includes pitfall interception.
"""


def _by_section(docs):
    """Index chunks by the section_header metadata for assertion convenience."""
    return {d.metadata.get("section_header"): d for d in docs}


class TestSplitMarkdownAware:
    def test_each_h2_section_becomes_its_own_chunk(self):
        chunks = _split_markdown_aware(
            [Document(page_content=MODULE_LIBRARY_FIXTURE)]
        )
        headers = [c.metadata.get("section_header") for c in chunks]
        # Three H2 sections; H3 sub-variant counts as a 4th chunk under the
        # default H2,H3 split config.
        assert "Client Presentation" in headers
        assert "Security Protocol" in headers
        assert "Diagnostic Framework" in headers
        assert any("Variant: Executive brief" in (h or "") for h in headers)

    def test_section_body_carries_the_header_path(self):
        chunks = _split_markdown_aware(
            [Document(page_content=MODULE_LIBRARY_FIXTURE)]
        )
        by_section = _by_section(chunks)
        client_pres = by_section["Client Presentation"]
        # The header path is either embedded in the body (because we strip
        # nothing) or prepended as a bracketed marker — either way the
        # section title must appear in the chunk's page_content so it ends
        # up in the embedding.
        assert "Client Presentation" in client_pres.page_content

    def test_oversized_section_falls_back_to_recursive_split(self):
        # Build a single H2 section larger than CHUNK_SIZE (1500) so the
        # secondary RecursiveCharacterTextSplitter pass has to fire.
        big_body = "lorem ipsum dolor sit amet. " * 200  # ~5400 chars
        oversize_md = f"## Big Section\n\n{big_body}\n"
        chunks = _split_markdown_aware([Document(page_content=oversize_md)])
        # All sub-chunks should still tag the same section_header.
        section_headers = {c.metadata.get("section_header") for c in chunks}
        assert section_headers == {"Big Section"}
        # And there should be more than one sub-chunk because of the size.
        assert len(chunks) > 1

    def test_empty_input_does_not_crash(self):
        assert _split_markdown_aware([]) == []
        assert _split_markdown_aware([Document(page_content="")]) == []

    def test_no_headers_falls_back_to_recursive_split(self):
        # Pure prose with no markdown headers — should still produce chunks
        # via the RecursiveCharacterTextSplitter fallback, with no
        # section_header metadata.
        prose = "Just some plain text. " * 50
        chunks = _split_markdown_aware([Document(page_content=prose)])
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata.get("section_header") is None


class TestPrepareDocumentsSync:
    def test_md_file_uses_header_aware_path(self):
        docs = _prepare_documents_sync(
            data=[Document(page_content=MODULE_LIBRARY_FIXTURE)],
            file_id="f1",
            user_id="u1",
            clean_content=False,
            file_ext="md",
        )
        # Header-aware split should produce one chunk per H2/H3 section
        # (4 sections in the fixture: 3 H2 + 1 H3 variant).
        section_headers = {d.metadata.get("section_header") for d in docs}
        assert "Client Presentation" in section_headers
        assert "Security Protocol" in section_headers
        assert "Diagnostic Framework" in section_headers
        # File/user metadata still injected by the outer prepare step.
        for d in docs:
            assert d.metadata["file_id"] == "f1"
            assert d.metadata["user_id"] == "u1"
            assert d.metadata["digest"]

    def test_non_md_file_uses_recursive_split(self):
        # A .txt file with the same content should NOT carry section_header
        # metadata — it goes through the unchanged
        # RecursiveCharacterTextSplitter path.
        docs = _prepare_documents_sync(
            data=[Document(page_content=MODULE_LIBRARY_FIXTURE)],
            file_id="f2",
            user_id="u2",
            clean_content=False,
            file_ext="txt",
        )
        assert len(docs) >= 1
        for d in docs:
            assert d.metadata.get("section_header") is None
            assert d.metadata["file_id"] == "f2"

    def test_file_ext_none_uses_recursive_split(self):
        # Defensive: when file_ext is not threaded through (legacy callers),
        # behavior matches the pre-change behavior — recursive split,
        # no section_header.
        docs = _prepare_documents_sync(
            data=[Document(page_content=MODULE_LIBRARY_FIXTURE)],
            file_id="f3",
            user_id="u3",
            clean_content=False,
        )
        for d in docs:
            assert d.metadata.get("section_header") is None
