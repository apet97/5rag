# Document Ingestion Guide

## Overview
The RAG system supports ingestion of various document formats through the ingestion module, which normalizes content into a markdown format expected by the RAG pipeline.

## Supported Formats
- **PDF** - Text extraction with PyPDF2
- **HTML** - Content extraction with BeautifulSoup4
- **Markdown (.md)** - Direct text reading
- **Plain Text (.txt)** - Direct text reading
- **DOCX** - Text extraction with python-docx
- **Directory processing** - Batch processing of mixed file types

## Ingestion Pipeline

### 1. Document Conversion
- Input: Any supported file format
- Output: Normalized text content
- Process: Format-specific extraction and cleaning

### 2. Markdown Formatting
- Input: Extracted text
- Output: RAG-compatible markdown format
- Format: `# [ARTICLE] Title` header + optional URL + content

### 3. Chunking
- Input: Formatted markdown
- Output: Chunked data structure with metadata
- Process: Heading-aware splitting with overlap

## Usage Examples

### Single Document Ingestion

```python
from clockify_rag.ingestion import ingest_document

# Convert single document to markdown string
markdown_content = ingest_document("/path/to/document.pdf")

# Convert and save to file
output_path = ingest_document("/path/to/document.pdf", output_path="output.md")
```

### Directory Ingestion

```python
from clockify_rag.ingestion import ingest_directory

# Process all documents in a directory
combined_kb = ingest_directory("/path/to/documents/", output_path="knowledge_base.md")
```

### Custom File Type Processing

```python
from clockify_rag.ingestion import ingest_document

# Process with custom validation
content = ingest_document("/path/to/document.txt")
is_valid, issues = validate_ingestion_output(content)

if is_valid:
    print("Document ready for RAG pipeline")
else:
    print(f"Validation issues: {issues}")
```

## Configuration

Document processing can be configured through environment variables:

```bash
# Chunk size and overlap
CHUNK_CHARS=1600
CHUNK_OVERLAP=200

# PDF processing options
PDF_MAX_PAGES=100  # Maximum pages to process
PDF_PAGE_LIMIT=50  # Limit pages per document (optional)

# HTML processing options
HTML_STRIP_TAGS=true  # Whether to strip HTML tags
HTML_PRESERVE_FORMAT=false  # Preserve formatting in extracted text
```

## Metadata Preservation

During ingestion, the system preserves and captures:

- **Document source**: File path and original format
- **Structure**: Headings and section hierarchy
- **Content metadata**: Dates, URLs, email addresses
- **Processing info**: Chunk indices and content statistics

## Validation

All ingestion output is validated to ensure it conforms to the RAG system expectations:

- Proper ARTICLE header format
- Sufficient content length
- Valid markdown structure

## Best Practices

1. **File Naming**: Use descriptive names that will be helpful when reviewing chunks
2. **Document Structure**: Organize documents with clear headings for better chunking
3. **Content Quality**: Ensure documents are well-formatted and contain searchable text
4. **Batch Processing**: Use directory ingestion for large collections
5. **Validation**: Always validate ingestion output before building the knowledge base