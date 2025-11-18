# Chunking Strategy Documentation

## Overview
The RAG system uses a hierarchical chunking strategy designed to preserve semantic boundaries while maintaining coherent context for retrieval.

## Chunking Strategy

### 1. Semantic Boundary Chunking (Primary)
- Splits on major document boundaries: paragraphs, sections, lists
- Preserves logical document structure
- Maintains context within semantic units
- Handles markdown headers, lists, and paragraph breaks

### 2. Sentence-Aware Chunking (Secondary)
- Uses NLTK sentence tokenization
- Avoids breaking sentences mid-way
- Maintains grammatical coherence
- Handles complex sentence structures properly

### 3. Character-Based Chunking (Fallback)
- Pure character-based splitting
- Used when other methods fail
- Ensures consistent max chunk size

## Configuration

The chunking behavior can be configured using these environment variables:

```bash
# Basic chunking parameters
CHUNK_CHARS=1600        # Maximum characters per chunk
CHUNK_OVERLAP=200       # Overlap between chunks in characters

# Advanced options
CHUNK_STRATEGY=hierarchical  # Options: hierarchical, sentence, character
CHUNK_MIN_CHARS=100     # Minimum chunk size (experimental)
CHUNK_SEMANTIC_AWARE=true    # Enable semantic boundary detection
```

## Chunk Metadata

Each chunk now includes enhanced metadata:

- `id`: Unique identifier with document structure info
- `title`: Original document title
- `url`: Document source URL (if any)
- `section`: Section heading
- `subsection`: Subsection headers (H3/H4)
- `doc_path`: Original document path
- `doc_name`: Document name without extension
- `section_idx`: Section index in document
- `chunk_idx`: Chunk index within section
- `char_count`: Character count in chunk
- `word_count`: Word count in chunk
- `metadata`: Additional extracted metadata (dates, URLs, emails)

## Boundary Detection

The system detects and respects these semantic boundaries:

- **Markdown Headers**: H1, H2, H3, H4 sections
- **Paragraph Breaks**: Double newline boundaries
- **List Items**: Bullet points and numbered lists  
- **Code Blocks**: Preserved as single units when possible
- **Tables**: Treated as single semantic units

## Overlap Strategy

The overlap mechanism preserves context across chunk boundaries:

- **Content-Based Overlap**: Includes relevant context from previous chunk
- **Sentence Boundary Respect**: Overlap content maintains sentence integrity
- **Size-Aware**: Calculates overlap based on actual character count

## Performance Considerations

- **NLTK Download**: Sentence tokenization requires NLTK models (downloads automatically)
- **Memory Usage**: Semantic chunking requires more memory for boundary detection
- **Processing Time**: Hierarchical approach takes longer but produces better chunks

## Best Practices

1. **Structure Documents**: Use clear headings and paragraphs for better semantic chunking
2. **Optimal Size**: Balance chunk size between context preservation and retrieval precision
3. **Overlap Settings**: Use 10-15% of chunk size for overlap in most cases
4. **Content Type**: Adjust strategy based on document complexity and structure

## Validation

All chunks are validated for:
- Size constraints (not exceeding maximum)
- Content quality (non-empty with meaningful text)
- Metadata completeness