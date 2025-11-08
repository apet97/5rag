# Clockify RAG Plugin System - Developer Guide

**Version**: 5.1
**Date**: 2025-11-08
**Audience**: Developers extending the Clockify RAG system

---

## Table of Contents

1. [Overview](#overview)
2. [Plugin Architecture](#plugin-architecture)
3. [Available Plugin Types](#available-plugin-types)
4. [Creating Custom Plugins](#creating-custom-plugins)
5. [Complete Examples](#complete-examples)
6. [Testing Your Plugins](#testing-your-plugins)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Clockify RAG system includes a flexible plugin architecture that allows you to:

- **Customize retrieval strategies** - Implement alternative search algorithms
- **Add reranking methods** - Improve result quality with custom scoring
- **Use different embedding models** - Integrate new embedding providers
- **Implement custom indexes** - Use specialized index structures

### When to Use Plugins

✅ **Use plugins when you need to**:
- Experiment with different retrieval algorithms
- Integrate domain-specific ranking models
- Use proprietary embedding models
- Implement specialized index structures (e.g., graph-based)

❌ **Don't use plugins for**:
- Simple configuration changes (use environment variables)
- One-off experiments (use the existing codebase directly)
- Performance tuning (adjust existing parameters first)

---

## Plugin Architecture

### Core Components

```
clockify_rag/plugins/
├── __init__.py          # Public API
├── interfaces.py        # Abstract base classes
├── registry.py          # Plugin registration system
└── examples.py          # Example implementations
```

### Plugin Lifecycle

1. **Define**: Implement a plugin interface
2. **Register**: Add plugin to registry
3. **Activate**: Configure system to use plugin
4. **Execute**: Plugin runs during query processing

### Plugin Interface Hierarchy

```
Plugin (ABC)
├── RetrieverPlugin      # Custom retrieval strategies
├── RerankPlugin         # Result reranking
├── EmbeddingPlugin      # Custom embeddings
└── IndexPlugin          # Custom index structures
```

---

## Available Plugin Types

### 1. RetrieverPlugin

**Purpose**: Implement custom retrieval strategies

**Required Methods**:
- `retrieve(question: str, top_k: int) -> List[dict]` - Retrieve relevant chunks
- `get_name() -> str` - Return plugin name

**Use Cases**:
- TF-IDF only retrieval
- Keyword-based search
- Neural retrieval models
- Hybrid custom algorithms

### 2. RerankPlugin

**Purpose**: Rerank initial retrieval results

**Required Methods**:
- `rerank(question: str, chunks: List[dict], scores: List[float]) -> Tuple[List[dict], List[float]]`
- `get_name() -> str`

**Use Cases**:
- Cross-encoder reranking
- Domain-specific scoring
- Query-dependent reordering
- Diversity-based reranking

### 3. EmbeddingPlugin

**Purpose**: Use custom embedding models

**Required Methods**:
- `embed(texts: List[str]) -> np.ndarray` - Generate embeddings
- `get_dimension() -> int` - Return embedding dimension
- `get_name() -> str`

**Use Cases**:
- Proprietary embedding models
- Domain-adapted embeddings
- Multilingual embeddings
- API-based embedding services

### 4. IndexPlugin

**Purpose**: Implement custom index structures

**Required Methods**:
- `build(vectors, metadata)` - Build index
- `search(query_vector, top_k)` - Search index
- `save(path)` - Persist index
- `load(path)` - Load index
- `get_name() -> str`

**Use Cases**:
- Graph-based indexes
- LSH (Locality-Sensitive Hashing)
- Custom ANN algorithms
- Hybrid index structures

---

## Creating Custom Plugins

### Step 1: Choose Plugin Type

Determine which plugin interface best fits your use case.

### Step 2: Implement Interface

Create a new class that inherits from the appropriate plugin interface.

### Step 3: Implement Required Methods

All abstract methods must be implemented.

### Step 4: Register Plugin

Use the plugin registry to make your plugin available.

### Step 5: Test Plugin

Write tests to verify plugin behavior.

---

## Complete Examples

### Example 1: TF-IDF Only Retriever

**Use Case**: Simple keyword-based retrieval without dense embeddings.

```python
from clockify_rag.plugins import RetrieverPlugin, register_plugin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TFIDFRetriever(RetrieverPlugin):
    """Pure TF-IDF retrieval without dense embeddings."""

    def __init__(self, chunks):
        """Initialize with document corpus.

        Args:
            chunks: List of chunk dicts with 'id' and 'text' fields
        """
        self.chunks = chunks
        self.chunk_texts = [c["text"] for c in chunks]

        # Build TF-IDF index
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def retrieve(self, question: str, top_k: int = 12) -> list:
        """Retrieve chunks using TF-IDF cosine similarity.

        Args:
            question: User query
            top_k: Number of chunks to retrieve

        Returns:
            List of dicts with 'id', 'text', and 'score'
        """
        # Transform query
        query_vec = self.vectorizer.transform([question])

        # Compute similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            results.append({
                "id": self.chunks[idx]["id"],
                "text": self.chunks[idx]["text"],
                "score": float(similarities[idx])
            })

        return results

    def get_name(self) -> str:
        return "tfidf_retriever"

    def validate(self) -> bool:
        """Validate that vectorizer is trained."""
        return self.vectorizer is not None and self.tfidf_matrix is not None


# Register plugin
register_plugin(TFIDFRetriever)
```

**Usage**:
```python
from clockify_rag import load_index
from your_module import TFIDFRetriever

# Load index
idx = load_index()

# Create retriever plugin
retriever = TFIDFRetriever(idx["chunks"])

# Use for retrieval
results = retriever.retrieve("How do I track time?", top_k=10)
```

---

### Example 2: Keyword-Based Reranker

**Use Case**: Boost chunks containing specific domain keywords.

```python
from clockify_rag.plugins import RerankPlugin, register_plugin
from typing import List, Tuple
import re


class KeywordReranker(RerankPlugin):
    """Rerank results by boosting chunks with important keywords."""

    def __init__(self, keyword_weights: dict = None):
        """Initialize with keyword weights.

        Args:
            keyword_weights: Dict mapping keywords to boost multipliers
                            Example: {"pricing": 1.5, "integration": 1.3}
        """
        self.keyword_weights = keyword_weights or {
            "pricing": 1.5,
            "plan": 1.3,
            "integration": 1.3,
            "sso": 1.4,
            "enterprise": 1.3,
            "api": 1.2
        }

        # Compile regex patterns for efficiency
        self.patterns = {
            keyword: re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for keyword in self.keyword_weights
        }

    def rerank(
        self,
        question: str,
        chunks: List[dict],
        scores: List[float]
    ) -> Tuple[List[dict], List[float]]:
        """Rerank chunks by boosting keyword matches.

        Args:
            question: User query
            chunks: Retrieved chunks
            scores: Initial scores

        Returns:
            (reranked_chunks, reranked_scores)
        """
        # Calculate keyword boosts
        boosted_scores = []
        for chunk, score in zip(chunks, scores):
            boost = 1.0
            text = chunk["text"].lower()

            # Apply keyword boosts
            for keyword, weight in self.keyword_weights.items():
                if self.patterns[keyword].search(text):
                    boost *= weight

            boosted_scores.append(score * boost)

        # Sort by boosted scores
        sorted_indices = sorted(
            range(len(chunks)),
            key=lambda i: boosted_scores[i],
            reverse=True
        )

        # Reorder chunks and scores
        reranked_chunks = [chunks[i] for i in sorted_indices]
        reranked_scores = [boosted_scores[i] for i in sorted_indices]

        return reranked_chunks, reranked_scores

    def get_name(self) -> str:
        return "keyword_reranker"

    def validate(self) -> bool:
        """Validate keyword configuration."""
        return (
            self.keyword_weights is not None
            and len(self.keyword_weights) > 0
            and all(w > 0 for w in self.keyword_weights.values())
        )


# Register plugin
register_plugin(KeywordReranker)
```

**Usage**:
```python
from clockify_rag import retrieve
from your_module import KeywordReranker

# Create reranker
reranker = KeywordReranker(keyword_weights={
    "pricing": 1.8,
    "enterprise": 1.5
})

# Use in retrieval pipeline
chunks, scores = retrieve(question, top_k=12)
reranked_chunks, reranked_scores = reranker.rerank(question, chunks, scores)
```

---

### Example 3: OpenAI Embeddings Plugin

**Use Case**: Use OpenAI's embedding API instead of local models.

```python
from clockify_rag.plugins import EmbeddingPlugin, register_plugin
import numpy as np
import openai


class OpenAIEmbeddings(EmbeddingPlugin):
    """Use OpenAI's embedding API."""

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """Initialize with OpenAI API credentials.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

        # Embedding dimensions for OpenAI models
        self.dim_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            NumPy array of shape (len(texts), dimension)
        """
        # Batch API call
        response = openai.Embedding.create(
            input=texts,
            model=self.model
        )

        # Extract embeddings
        embeddings = [item["embedding"] for item in response["data"]]

        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dim_map.get(self.model, 1536)

    def get_name(self) -> str:
        return f"openai_{self.model}"

    def validate(self) -> bool:
        """Validate API key and model."""
        return (
            self.api_key is not None
            and len(self.api_key) > 0
            and self.model in self.dim_map
        )


# Register plugin
register_plugin(OpenAIEmbeddings)
```

**Usage**:
```python
from your_module import OpenAIEmbeddings

# Create embedding plugin
embedder = OpenAIEmbeddings(api_key="sk-...")

# Generate embeddings
texts = ["How do I track time?", "What is the pricing?"]
embeddings = embedder.embed(texts)

print(f"Generated {len(embeddings)} embeddings of dimension {embedder.get_dimension()}")
```

---

## Testing Your Plugins

### Unit Tests

Create unit tests for each plugin method:

```python
import pytest
from your_module import TFIDFRetriever


def test_tfidf_retriever():
    """Test TF-IDF retriever basic functionality."""
    # Sample chunks
    chunks = [
        {"id": "1", "text": "Track time with the timer button"},
        {"id": "2", "text": "Generate reports for your team"},
        {"id": "3", "text": "Time tracking features"}
    ]

    # Create retriever
    retriever = TFIDFRetriever(chunks)

    # Test retrieval
    results = retriever.retrieve("time tracking", top_k=2)

    # Assertions
    assert len(results) == 2
    assert all("id" in r for r in results)
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)
    assert results[0]["score"] >= results[1]["score"]  # Sorted by score


def test_tfidf_retriever_validation():
    """Test retriever validation."""
    chunks = [{"id": "1", "text": "test"}]
    retriever = TFIDFRetriever(chunks)

    assert retriever.validate() is True
    assert retriever.get_name() == "tfidf_retriever"
```

### Integration Tests

Test plugins in the full RAG pipeline:

```python
def test_plugin_in_pipeline():
    """Test plugin integration with RAG system."""
    from clockify_rag import build, load_index
    from your_module import TFIDFRetriever

    # Build test index
    build("test_kb.md")
    idx = load_index()

    # Use plugin
    retriever = TFIDFRetriever(idx["chunks"])
    results = retriever.retrieve("test query", top_k=5)

    assert len(results) > 0
```

---

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
def retrieve(self, question: str, top_k: int = 12) -> list:
    try:
        # Plugin logic
        return results
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []  # Return empty results rather than crashing
```

### 2. Input Validation

Validate inputs in plugin methods:

```python
def rerank(self, question: str, chunks: List[dict], scores: List[float]):
    # Validate inputs
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if len(chunks) != len(scores):
        raise ValueError("Chunks and scores must have same length")

    # Plugin logic...
```

### 3. Performance Optimization

- **Cache expensive operations** (e.g., model loading)
- **Batch API calls** when possible
- **Use vectorized operations** (NumPy) over loops
- **Profile your code** to identify bottlenecks

### 4. Documentation

Document your plugin thoroughly:

```python
class MyPlugin(RetrieverPlugin):
    """One-line summary of what this plugin does.

    Detailed description explaining:
    - What problem it solves
    - When to use it
    - Performance characteristics
    - Dependencies required

    Example:
        >>> plugin = MyPlugin(config)
        >>> results = plugin.retrieve("query", top_k=10)

    Attributes:
        config: Plugin configuration
        model: Underlying model instance
    """
```

### 5. Configuration Management

Use a configuration dict for flexibility:

```python
class MyPlugin(RetrieverPlugin):
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()

    def _default_config(self) -> dict:
        return {
            "param1": 10,
            "param2": 0.5,
            "param3": "value"
        }
```

---

## Troubleshooting

### Common Issues

#### 1. Plugin Not Found

**Error**: `Plugin 'my_plugin' not registered`

**Solution**:
```python
# Ensure plugin is registered
from clockify_rag.plugins import register_plugin
register_plugin(MyPlugin)

# Or import the module that registers it
import my_plugin_module
```

#### 2. Dimension Mismatch

**Error**: `Embedding dimension mismatch: expected 384, got 768`

**Solution**:
- Ensure `get_dimension()` returns correct value
- Rebuild index if switching embedding models
- Check that all embeddings use same dimension

#### 3. Validation Failure

**Error**: `Plugin validation failed`

**Solution**:
- Check `validate()` method implementation
- Verify all required resources are loaded
- Ensure configuration is valid

#### 4. Performance Issues

**Symptoms**: Slow retrieval or high memory usage

**Solutions**:
- Profile your plugin code
- Cache expensive operations
- Use batch processing
- Consider approximate methods for large datasets

### Debug Mode

Enable debug logging to troubleshoot plugins:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now plugin errors will be logged
```

### Getting Help

1. Check plugin interface documentation: `clockify_rag/plugins/interfaces.py`
2. Review example implementations: `clockify_rag/plugins/examples.py`
3. Run unit tests to verify plugin behavior
4. Check system logs for error messages

---

## Advanced Topics

### Plugin Composition

Combine multiple plugins for complex workflows:

```python
# Use custom retriever + custom reranker
retriever = MyCustomRetriever(chunks)
reranker = MyCustomReranker()

# Two-stage retrieval
initial_results = retriever.retrieve(question, top_k=20)
final_results, scores = reranker.rerank(question, initial_results, scores)
```

### Plugin Configuration via Environment

```python
import os


class ConfigurablePlugin(RetrieverPlugin):
    def __init__(self):
        self.top_k = int(os.getenv("PLUGIN_TOP_K", "12"))
        self.threshold = float(os.getenv("PLUGIN_THRESHOLD", "0.3"))
```

### Async Plugin Support

For I/O-bound plugins (API calls, database queries):

```python
import asyncio


class AsyncEmbedder(EmbeddingPlugin):
    async def embed_async(self, texts: List[str]) -> np.ndarray:
        """Async embedding generation."""
        # Async API calls
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        """Sync wrapper for async method."""
        return asyncio.run(self.embed_async(texts))
```

---

## Migration Guide

### From v1.0 to v5.0

If you have custom retrieval code from v1.0, migrate to plugin system:

**Old (v1.0)**:
```python
# Custom retrieval in main code
def my_custom_retrieval(question, chunks):
    # Custom logic
    return results
```

**New (v5.0)**:
```python
# Plugin-based retrieval
class MyRetriever(RetrieverPlugin):
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question: str, top_k: int) -> list:
        # Same custom logic
        return results

    def get_name(self) -> str:
        return "my_retriever"

register_plugin(MyRetriever)
```

---

## Appendix

### Plugin Interface Reference

See `clockify_rag/plugins/interfaces.py` for complete interface definitions.

### Example Plugins

See `clockify_rag/plugins/examples.py` for working examples.

### System Requirements

- Python 3.7+
- NumPy
- Dependencies vary by plugin (e.g., sklearn for TF-IDF, openai for OpenAI embeddings)

---

**Last Updated**: 2025-11-08
**Version**: 5.1
**Maintainer**: Clockify RAG Team
