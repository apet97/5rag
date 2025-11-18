# Error Handling and Resilience

## Overview
The RAG system implements comprehensive error handling with graceful degradation, clear error messages, and actionable hints for troubleshooting.

## Error Classification

### Configuration Errors
- **CONFIG_ERROR**: Invalid environment variables or configuration
- **Validation**: Checked at startup and provides detailed hints

### Connection Errors
- **CONNECTION_ERROR**: Network connectivity issues to Ollama endpoint
- **TIMEOUT_ERROR**: Request timeouts during API calls
- **RESOLVABLE**: Provides hints for network configuration

### Model Errors
- **LLM_ERROR**: Issues with language model requests
- **EMBEDDING_ERROR**: Issues with embedding generation
- **FALLBACK**: Includes retry mechanisms and alternative strategies

### Index Errors
- **INDEX_ERROR**: Issues with index loading or validation
- **BUILD_ERROR**: Issues during index construction
- **RECOVERY**: Automatic rebuild attempts when possible

## Error Message Format

All error messages follow a consistent format:
```
[ERROR_TYPE] Description message [hint: resolution hint]
```

### Examples
- `[LLM_ERROR] Connection timeout after 120s [hint: check OLLAMA_URL or increase CHAT timeouts]`
- `[CONFIG_ERROR] Invalid float for CTX_BUDGET='abc': could not convert string to float [hint: use numeric value like 12000]`
- `[INDEX_ERROR] Embedding dimension mismatch: stored=768, expected=384 [hint: python3 clockify_support_cli.py build knowledge_full.md]`

## Graceful Degradation

### Fallback Mechanisms
1. **FAISS Fallback**: If FAISS fails, falls back to full-scan retrieval
2. **Embedding Backend**: Switch between local and Ollama embeddings
3. **Rerank Fallback**: If LLM reranking fails, returns original order
4. **Connection Fallback**: Multiple retry attempts with exponential backoff

### Safe Defaults
- Conservative timeout values that can be overridden
- Reasonable chunk sizes and retrieval parameters
- Graceful handling of missing optional dependencies

## Troubleshooting Common Issues

### Ollama Connection Issues
```
# Check Ollama status
curl http://127.0.0.1:11434/api/tags

# Verify model availability
ollama list

# Start Ollama server
ollama serve
```

### Index Building Issues
```
# Check file permissions
ls -la knowledge_full.md

# Verify file content
head -20 knowledge_full.md

# Rebuild index
python clockify_support_cli.py build knowledge_full.md
```

### Configuration Validation
```
# Run health check
python -c "from clockify_rag.error_handlers import print_system_health; print_system_health()"

# Check specific environment variables
echo $OLLAMA_URL
echo $GEN_MODEL
echo $EMB_MODEL
```

## Error Recovery Strategies

### Automatic Recovery
- **Model endpoint retries**: Configurable retry attempts (default: 2)
- **Index validation**: Detects and rebuilds broken indexes automatically
- **Cache recovery**: Recovers query cache from disk on startup

### Manual Recovery
- **Complete rebuild**: Delete index files and rebuild from source
- **Configuration reset**: Use default values when environment variables are invalid
- **Component isolation**: Continue operation when non-critical components fail

## Error Logging

### Log Levels
- **DEBUG**: Detailed diagnostic information for troubleshooting
- **INFO**: High-level operational information
- **WARNING**: Potential issues that don't stop execution
- **ERROR**: Errors that affect functionality
- **CRITICAL**: System-wide errors requiring immediate attention

### Log Structure
```json
{
  "timestamp": "2025-11-11T10:30:45.123Z",
  "level": "ERROR",
  "module": "retrieval",
  "function": "ask_llm",
  "message": "[LLM_ERROR] Connection timeout after 120s [hint: check OLLAMA_URL]",
  "context": {
    "model": "qwen2.5:32b",
    "timeout": 120.0
  }
}
```

## Custom Exception Types

### Primary Exceptions
```python
from clockify_rag.exceptions import (
    LLMError,           # Language model related errors
    EmbeddingError,     # Embedding generation errors
    IndexLoadError,     # Index loading errors
    BuildError,         # Index building errors
    ValidationError     # Input validation errors
)
```

### Usage Pattern
Each exception includes:
- Clear error message
- Contextual information
- Actionable resolution hints

## Monitoring and Health Checks

### System Health Command
```bash
# Check complete system health
python -c "from clockify_rag.error_handlers import print_system_health; print_system_health()"
```

### Self-Test Integration
```bash
# Run built-in self-test
python clockify_support_cli.py --selftest
```

## Best Practices

### For Developers
1. Always use typed exceptions from `clockify_rag.exceptions`
2. Include actionable hints in error messages
3. Log both at error and debug levels when appropriate
4. Handle expected error cases gracefully

### For Operators
1. Monitor error logs regularly
2. Check system health before troubleshooting
3. Use environment variables for configuration adjustments
4. Maintain backup knowledge bases for recovery

## Recovery Commands

### Common Recovery Actions
```bash
# Rebuild index
python clockify_support_cli.py build knowledge_full.md

# Test connectivity
curl http://127.0.0.1:11434/api/tags

# Run self-test
python clockify_support_cli.py --selftest

# Clear cache files
rm -f emb_cache.jsonl query_cache.json
```

This comprehensive error handling system ensures the RAG system remains operational under various failure conditions and provides clear guidance for users to resolve issues.