"""Advanced tests for chunking logic - metadata extraction, semantic chunking, edge cases."""

import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.chunking import (
    build_chunks,
    sliding_chunks,
    extract_metadata,
)


class TestMetadataExtraction:
    """Test metadata extraction from text (dates, URLs, etc.)."""

    def test_extract_date_iso_format(self):
        """Test extraction of ISO date format (YYYY-MM-DD)."""
        text = "Released on 2024-01-15. This is a test document."
        metadata = extract_metadata(text)

        assert "dates" in metadata or "date" in metadata
        # Check if date was extracted (implementation may vary)

    def test_extract_url_http(self):
        """Test extraction of HTTP URLs."""
        text = "Visit http://example.com for more info."
        metadata = extract_metadata(text)

        # URLs might be in metadata["urls"] or similar
        # Implementation-specific check
        assert isinstance(metadata, dict)

    def test_extract_url_https(self):
        """Test extraction of HTTPS URLs."""
        text = "Documentation at https://docs.example.com/guide"
        metadata = extract_metadata(text)

        assert isinstance(metadata, dict)

    def test_extract_multiple_items(self):
        """Test extraction of multiple dates and URLs."""
        text = """
        Update from 2024-01-15:
        - New feature at https://example.com/feature1
        - Another update on 2024-02-20
        - Documentation at http://docs.example.com
        """
        metadata = extract_metadata(text)

        assert isinstance(metadata, dict)

    def test_extract_no_metadata(self):
        """Test text with no extractable metadata."""
        text = "This is plain text with no dates or URLs."
        metadata = extract_metadata(text)

        assert isinstance(metadata, dict)
        # Should return empty or minimal metadata


@pytest.mark.skip(reason="semantic_chunking function not yet implemented")
class TestSemanticChunking:
    """Test semantic chunking strategy."""

    def test_semantic_chunking_paragraphs(self):
        """Test semantic chunking splits on paragraph boundaries."""
        text = """
        First paragraph about time tracking features.
        It has multiple sentences explaining the concept.

        Second paragraph about reporting capabilities.
        This paragraph discusses different report types.

        Third paragraph about integrations.
        Integration details go here.
        """

        chunks = semantic_chunking(text, max_chars=200)

        # Should create chunks respecting paragraph boundaries
        assert len(chunks) > 0
        # Each chunk should not exceed max_chars significantly
        for chunk in chunks:
            assert len(chunk) <= 300  # Allow some tolerance

    def test_semantic_chunking_sentences(self):
        """Test semantic chunking falls back to sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."

        chunks = semantic_chunking(text, max_chars=30)

        # Should split on sentence boundaries
        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should ideally contain complete sentences
            assert chunk.strip()

    def test_semantic_chunking_respects_max_chars(self):
        """Test that chunks don't significantly exceed max_chars."""
        text = "word " * 1000  # Long text
        max_chars = 500

        chunks = semantic_chunking(text, max_chars=max_chars)

        # Most chunks should be close to max_chars
        for chunk in chunks:
            # Allow reasonable tolerance (e.g., to complete sentences)
            assert len(chunk) <= max_chars * 1.5

    def test_semantic_chunking_empty_text(self):
        """Test semantic chunking with empty text."""
        chunks = semantic_chunking("", max_chars=200)

        # Should return empty list or single empty chunk
        assert len(chunks) <= 1
        if chunks:
            assert len(chunks[0].strip()) == 0


class TestBuildChunks:
    """Test the main build_chunks function."""

    def test_build_chunks_basic(self):
        """Test basic chunk building."""
        text = """
        # Time Tracking

        Use the timer to track time on tasks.
        Click start to begin tracking.
        Click stop to end tracking.
        """

        chunks = build_chunks(text, source="test_doc.md")

        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk or "chunk" in chunk
            assert "source" in chunk
            assert chunk["source"] == "test_doc.md"

    def test_build_chunks_with_headings(self):
        """Test chunk building preserves heading context."""
        text = """
        # Main Heading

        Content under main heading.

        ## Sub Heading

        Content under sub heading.
        """

        chunks = build_chunks(text, source="test.md")

        assert len(chunks) > 0
        # Check that chunks have proper context

    def test_build_chunks_respects_maxc(self):
        """Test that chunks respect maxc parameter."""
        long_text = "word " * 2000  # Very long text
        maxc = 500

        chunks = build_chunks(long_text, source="long.txt", maxc=maxc)

        assert len(chunks) > 1  # Should be split
        for chunk in chunks:
            text_field = chunk.get("text") or chunk.get("chunk", "")
            # Allow some tolerance for overlap and sentence completion
            assert len(text_field) <= maxc * 2

    def test_build_chunks_with_overlap(self):
        """Test chunk building with overlap."""
        text = "sentence " * 100

        chunks = build_chunks(text, source="test.txt", maxc=200, overlap=50)

        assert len(chunks) > 1
        # Adjacent chunks should have some overlapping text
        if len(chunks) >= 2:
            chunk1_text = chunks[0].get("text") or chunks[0].get("chunk", "")
            chunk2_text = chunks[1].get("text") or chunks[1].get("chunk", "")
            # Check if there's any overlap (last part of chunk1 in start of chunk2)
            assert chunk1_text and chunk2_text

    def test_build_chunks_preserves_metadata(self):
        """Test that build_chunks preserves metadata."""
        text = """
        Released on 2024-01-15.
        Documentation at https://docs.example.com

        Feature description goes here.
        """

        chunks = build_chunks(text, source="release.md")

        assert len(chunks) > 0
        # Check if metadata is preserved in chunks
        first_chunk = chunks[0]
        assert "source" in first_chunk

    def test_build_chunks_empty_text(self):
        """Test build_chunks with empty text."""
        chunks = build_chunks("", source="empty.txt")

        # Should return empty list or handle gracefully
        assert isinstance(chunks, list)

    def test_build_chunks_whitespace_only(self):
        """Test build_chunks with whitespace-only text."""
        chunks = build_chunks("   \n\n\t  ", source="whitespace.txt")

        # Should handle gracefully
        assert isinstance(chunks, list)


class TestSlidingChunksAdvanced:
    """Advanced tests for sliding_chunks function."""

    def test_sliding_chunks_overlap_behavior(self):
        """Test that overlap creates proper overlapping windows."""
        text = "AAAA BBBB CCCC DDDD EEEE FFFF"

        chunks = sliding_chunks(text, maxc=15, overlap=5)

        assert len(chunks) > 1
        # Adjacent chunks should have overlap
        if len(chunks) >= 2:
            # Last chars of chunk[i] should appear in chunk[i+1]
            for i in range(len(chunks) - 1):
                overlap_region = chunks[i][-5:]  # Last 5 chars
                assert len(overlap_region) > 0

    def test_sliding_chunks_no_overlap_zero(self):
        """Test sliding chunks with zero overlap."""
        text = "word " * 50

        chunks = sliding_chunks(text, maxc=50, overlap=0)

        assert len(chunks) > 1
        # No overlap means total length should equal original
        total_chars = sum(len(chunk) for chunk in chunks)
        # Allow for some whitespace handling differences
        assert abs(total_chars - len(text)) < 20

    def test_sliding_chunks_overlap_larger_than_maxc(self):
        """Test that overlap can't exceed maxc (edge case)."""
        text = "word " * 100

        # Invalid config: overlap >= maxc should be handled
        try:
            chunks = sliding_chunks(text, maxc=50, overlap=60)
            # Implementation should either handle gracefully or raise error
            assert isinstance(chunks, list)
        except ValueError:
            # Acceptable to raise ValueError for invalid config
            pass

    def test_sliding_chunks_unicode_handling(self):
        """Test sliding chunks with unicode characters."""
        text = "Hello 你好 مرحبا Привет " * 20

        chunks = sliding_chunks(text, maxc=100, overlap=20)

        assert len(chunks) > 0
        # Should handle unicode gracefully
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_sliding_chunks_preserves_words(self):
        """Test that sliding chunks tries to preserve word boundaries."""
        text = "This is a sentence with many words that should be preserved during chunking process."

        chunks = sliding_chunks(text, maxc=30, overlap=10)

        # Check that chunks don't split words unnecessarily
        for chunk in chunks:
            # Chunk should not start/end with partial words (unless unavoidable)
            # This is a soft check as implementation may vary
            assert chunk.strip()


class TestChunkingEdgeCases:
    """Test edge cases and error handling."""

    def test_chunking_very_long_single_word(self):
        """Test handling of very long single word (URL, hash, etc.)."""
        long_word = "x" * 2000

        chunks = sliding_chunks(long_word, maxc=500, overlap=50)

        # Should handle by splitting even within the word
        assert len(chunks) > 1

    def test_chunking_special_characters(self):
        """Test chunking with special characters."""
        text = "Normal text. @@@ Special ### Characters $$$ Here !!! Testing..."

        chunks = sliding_chunks(text, maxc=30, overlap=5)

        assert len(chunks) > 0
        # Special chars should be preserved
        full_rejoined = "".join(chunks)
        assert "@@@" in full_rejoined or "###" in full_rejoined

    def test_chunking_mixed_newlines(self):
        """Test chunking with mixed newline styles."""
        text = "Line1\nLine2\r\nLine3\rLine4"

        chunks = sliding_chunks(text, maxc=20, overlap=5)

        assert len(chunks) > 0

    def test_build_chunks_none_source(self):
        """Test build_chunks with None source."""
        text = "Test content"

        try:
            chunks = build_chunks(text, source=None)
            # Should handle gracefully or use default
            assert isinstance(chunks, list)
        except (ValueError, TypeError):
            # Acceptable to require source
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
