"""Text parsing and chunking functions for knowledge base processing.

This module provides utilities to parse and chunk documents for the RAG system.
It includes heading-aware splitting, sentence-aware chunking, and overlap management.
"""

import logging
import pathlib
import re
import unicodedata
import uuid
from typing import List, Dict

from .config import CHUNK_CHARS, CHUNK_OVERLAP
from .utils import norm_ws, strip_noise

logger = logging.getLogger(__name__)

# Rank 23: NLTK for sentence-aware chunking
try:
    import nltk

    # Lazy download of punkt tokenizer data (only if needed)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)  # For newer NLTK versions
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# ====== KB PARSING ======
def parse_articles(md_text: str) -> list:
    """Parse articles from markdown. Heuristic: '# [ARTICLE]' + optional URL line."""
    lines = md_text.splitlines()
    articles = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("# [ARTICLE]"):
            title_line = lines[i].replace("# ", "").strip()
            url = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("http"):
                url = lines[i + 1].strip()
                i += 2
            else:
                i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("# [ARTICLE]"):
                buf.append(lines[i])
                i += 1
            body = "\n".join(buf).strip()
            articles.append({"title": title_line, "url": url, "body": body})
        else:
            i += 1
    if not articles:
        articles = [{"title": "KB", "url": "", "body": md_text}]
    return articles


def split_by_headings(body: str) -> list:
    """Split by H2 headers."""
    parts = re.split(r"\n(?=## +)", body)
    return [p.strip() for p in parts if p.strip()]


def sliding_chunks(text: str, maxc: int = None, overlap: int = None) -> list:
    """Advanced overlapping chunks with multiple strategies and semantic awareness.

    Uses hierarchical chunking strategies:
    1. Semantic boundaries (paragraphs, sections)
    2. Sentence-aware splitting with NLTK
    3. Character-based fallback

    Args:
        text: Text to chunk
        maxc: Maximum characters per chunk (defaults to config.CHUNK_CHARS)
        overlap: Overlap in characters (defaults to config.CHUNK_OVERLAP)

    Returns:
        List of text chunks
    """
    if maxc is None:
        maxc = CHUNK_CHARS
    if overlap is None:
        overlap = CHUNK_OVERLAP

    if len(text) <= maxc:
        return [text]

    text = strip_noise(text)
    # Normalize to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Try semantic boundary chunking first (paragraphs, lists, etc.)
    chunks = semantic_boundary_chunking(text, maxc, overlap)

    # If semantic chunking returns chunks that are too small, refine with sentence-aware splitting
    refined_chunks = []
    for chunk in chunks:
        if len(chunk) <= maxc:
            refined_chunks.append(chunk)
        else:
            # For oversized semantic chunks, apply sentence-aware splitting
            sentence_chunks = sentence_aware_chunking(chunk, maxc, overlap)
            refined_chunks.extend(sentence_chunks)

    return refined_chunks


def semantic_boundary_chunking(text: str, maxc: int, overlap: int) -> list:
    """Split text using semantic boundaries like paragraphs, lists, and sections.

    Args:
        text: Text to chunk semantically
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of semantically-boundaried chunks
    """
    # Split on major semantic boundaries (paragraphs, sections, lists)
    # Preserve the separators to maintain context
    parts = re.split(r"(\n\s*\n|\n\s*[-*]\s+|\n\s*\d+\.\s+|\n#{2,}\s+)", text)

    # Merge separators back with the following content
    merged_parts = []
    i = 0
    while i < len(parts):
        current = parts[i]
        if (
            i + 1 < len(parts)
            and parts[i + 1]
            and parts[i + 1].strip() in ["\n\n", r"\n\s*[-*]\s+", r"\n\s*\d+\.\s+", r"\n#{2,}\s+"]
        ):
            # This is a separator that should be merged with the following content
            next_part = parts[i + 1] if i + 1 < len(parts) else ""
            if i + 2 < len(parts):
                current = current + next_part + parts[i + 2]
                i += 3
            else:
                merged_parts.append(current)
                i += 1
        else:
            merged_parts.append(current)
            i += 1

    # Remove empty parts but keep the structure
    merged_parts = [part for part in merged_parts if part.strip()]

    # Group parts into chunks that respect semantic boundaries
    chunks = []
    current_chunk = ""

    for part in merged_parts:
        # If adding this part would exceed the limit
        if len(current_chunk) + len(part) > maxc and current_chunk:
            # Add current chunk to results
            chunks.append(current_chunk.strip())

            # Start a new chunk with overlap if possible
            if len(part) <= maxc:
                # If part fits in a single chunk, start fresh
                current_chunk = part
            else:
                # If part is too large, we'll handle it with sentence splitting later
                current_chunk = part
        else:
            # Add part to current chunk
            current_chunk += part

    # Add the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Now handle any chunks that are still too large with sentence-aware splitting
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= maxc:
            final_chunks.append(chunk)
        else:
            # Use sentence-aware splitting for oversized chunks
            sentence_chunks = sentence_aware_chunking(chunk, maxc, overlap)
            final_chunks.extend(sentence_chunks)

    return final_chunks


def sentence_aware_chunking(text: str, maxc: int, overlap: int) -> list:
    """Overlapping chunks with sentence-aware splitting (Rank 23).

    Uses NLTK sentence tokenization to avoid breaking sentences mid-way.
    Falls back to character-based chunking if NLTK is unavailable.

    Args:
        text: Text to chunk with sentence awareness
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of sentence-aware chunks
    """
    if not text.strip():
        return []

    out = []

    # Rank 23: Use sentence-aware chunking if NLTK is available
    if _NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(text)

            # Build chunks by accumulating sentences
            current_chunk = []
            current_len = 0

            for sent in sentences:
                sent_len = len(sent)

                # If single sentence exceeds maxc, fall back to character splitting
                if sent_len > maxc:
                    # Flush current chunk first
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            out.append(chunk_text)
                        current_chunk = []
                        current_len = 0

                    # Split long sentence by characters with consistent overlap
                    long_chunks = character_chunking(sent, maxc, overlap)
                    for chunk in long_chunks:
                        if chunk.strip():
                            out.append(chunk)
                    continue

                # Check if adding this sentence exceeds maxc
                potential_len = current_len + sent_len + (1 if current_chunk else 0)

                if potential_len > maxc:
                    # Flush current chunk
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            out.append(chunk_text)

                    # Start new chunk with overlap (last N sentences that fit in overlap)
                    overlap_chars = 0
                    overlap_sents = []
                    for prev_sent in reversed(current_chunk):
                        if overlap_chars + len(prev_sent) <= overlap:
                            overlap_sents.insert(0, prev_sent)
                            overlap_chars += len(prev_sent) + 1
                        else:
                            break

                    current_chunk = overlap_sents + [sent]
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sent)
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1

            # Flush final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    out.append(chunk_text)

            return out

        except Exception as e:
            # Fall back to character-based chunking if NLTK fails
            logger.warning(f"NLTK sentence tokenization failed: {e}, falling back to character chunking")

    # Fallback: Character-based chunking (original implementation)
    return character_chunking(text, maxc, overlap)


def character_chunking(text: str, maxc: int, overlap: int) -> list:
    """Basic character-based chunking as fallback.

    Args:
        text: Text to chunk character-wise
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of character-based chunks
    """
    if not text.strip():
        return []

    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + maxc, n)
        chunk = text[i:j].strip()
        if chunk:
            out.append(chunk)
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return out


def yield_sentence_aware_chunk(text: str, maxc: int, overlap: int) -> list:
    """Helper to chunk overly long individual sentences.

    Args:
        text: Long sentence text to chunk
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of chunks from the long sentence
    """
    return character_chunking(text, maxc, overlap)


def build_chunks(md_path: str) -> list:
    """Parse and chunk markdown with enhanced metadata extraction.

    Args:
        md_path: Path to the markdown file to chunk

    Returns:
        List of chunk dictionaries with enhanced metadata
    """
    raw = pathlib.Path(md_path).read_text(encoding="utf-8", errors="ignore")
    chunks = []

    for art in parse_articles(raw):
        sects = split_by_headings(art["body"]) or [art["body"]]

        for sect_idx, sect in enumerate(sects):
            # Extract the section header/title from the content
            head = sect.splitlines()[0] if sect else art["title"]

            # Parse the section for any H3 or H4 headers as subsection indicators
            subsection_headers = extract_subsection_headers(sect)

            # Create chunks for this section
            text_chunks = sliding_chunks(sect)

            for chunk_idx, piece in enumerate(text_chunks):
                # Create a meaningful ID that includes document structure info
                doc_name = pathlib.Path(md_path).stem
                cid = f"{doc_name}_{sect_idx}_{chunk_idx}_{str(uuid.uuid4())[:8]}"

                # Extract additional metadata
                metadata = extract_metadata(piece)

                chunk_obj = {
                    "id": cid,
                    "title": norm_ws(art["title"]),
                    "url": art["url"],
                    "section": norm_ws(head),
                    "subsection": subsection_headers[0] if subsection_headers else "",
                    "text": piece,
                    "doc_path": str(md_path),
                    "doc_name": doc_name,
                    "section_idx": sect_idx,
                    "chunk_idx": chunk_idx,
                    "char_count": len(piece),
                    "word_count": len(piece.split()),
                    "metadata": metadata,
                }

                chunks.append(chunk_obj)

    return chunks


def extract_subsection_headers(section_text: str) -> List[str]:
    """Extract H3 and H4 headers from section text.

    Args:
        section_text: Text of a section to parse for headers

    Returns:
        List of headers found in order of appearance
    """
    headers = []
    lines = section_text.splitlines()

    for line in lines:
        # Match H3 (###) and H4 (####) headers
        h3_match = re.match(r"^###\s+(.+)", line.strip())
        h4_match = re.match(r"^####\s+(.+)", line.strip())

        if h3_match:
            headers.append(h3_match.group(1).strip())
        elif h4_match:
            headers.append(h4_match.group(1).strip())

    return headers


def extract_metadata(text: str) -> Dict[str, str]:
    """Extract basic metadata from text content.

    Args:
        text: Text to extract metadata from

    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}

    # Extract dates (common formats)
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY
        r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",  # MM-DD-YYYY
    ]

    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        if dates:
            metadata["dates"] = dates
            break

    # Extract URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    if urls:
        metadata["urls"] = urls[:5]  # Limit to first 5 URLs

    # Extract email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails = re.findall(email_pattern, text)
    if emails:
        metadata["emails"] = emails[:5]  # Limit to first 5 emails

    return metadata
