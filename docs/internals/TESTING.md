# Testing Documentation

## Overview
The RAG system includes a comprehensive test suite to ensure reliability and correctness across all components.

## Test Structure

### Unit Tests
Located in `tests/` directory, these test individual components:

- `test_chunker.py` - Chunking and document parsing
- `test_embedding.py` - Embedding generation and caching
- `test_retrieval.py` - Retrieval algorithms and hybrid search
- `test_answer.py` - Answer generation pipeline
- `test_packer.py` - Context packing with token budgets
- `test_query_cache.py` - Query caching and rate limiting
- And many more...

### Integration Tests
- `test_integration.py` - Complete end-to-end workflow
- `test_end_to_end.py` - Full pipeline integration test

### Performance Tests
- `test_thread_safety.py` - Thread safety under concurrent load
- `test_metrics.py` - Performance metrics and timing

## Running Tests

### All Tests
```bash
# Run all tests with coverage
make test

# Or directly with pytest
pytest tests/ --cov=clockify_rag --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Specific Test Categories
```bash
# Run only unit tests
pytest tests/test_*.py -k "not integration"

# Run only integration tests
pytest tests/test_integration.py

# Run with specific markers
pytest -m "integration" tests/
pytest -m "slow" tests/  # Only slow tests
```

### Individual Test Files
```bash
# Run a specific test file
pytest tests/test_chunker.py

# Run with verbose output
pytest -v tests/test_chunker.py

# Run with detailed output and capture
pytest -v -s tests/test_chunker.py
```

## Test Commands

### Development Workflow
```bash
# Quick syntax check
python -m py_compile clockify_support_cli_final.py

# Run a quick smoke test
python -m pytest tests/test_chunker.py -v

# Run self-test command (built into the CLI)
python clockify_support_cli_final.py --selftest
```

### Full Test Suite
```bash
# All tests with coverage
pytest tests/ --cov=clockify_rag --cov-report=term-missing

# All tests with parallel execution (faster)
pytest tests/ -n auto --dist=loadfile

# All tests with specific configuration
OLLAMA_URL=http://127.0.0.1:11434 pytest tests/
```

### Continuous Integration
```bash
# CI command (as used in workflows)
pytest tests/ --cov=clockify_rag --cov-report=xml --junitxml=test-results.xml
```

## Test Configuration

Tests can be customized with environment variables:

```bash
# Test-specific environment variables
export RAG_LLM_CLIENT="mock"                # Force deterministic mock client (default for pytest)
export RAG_OLLAMA_URL="http://localhost:11434"  # Real Ollama endpoint (only needed for manual integration tests)
export TEST_EMBEDDING_BACKEND="local"       # Use local embeddings for tests
export TEST_TIMEOUT_FACTOR=2                # Multiply timeouts by factor
```

## Writing Tests

### Best Practices
1. **Isolation**: Each test should be independent
2. **Reproducibility**: Tests should have deterministic outputs
3. **Speed**: Keep tests fast; use mocking for external dependencies
4. **Coverage**: Aim for 80%+ coverage of critical paths
5. **Documentation**: Include docstrings explaining test purpose

### Test Structure
```python
def test_feature_behavior():
    """Test that feature X behaves correctly under condition Y."""
    # Arrange: Set up test conditions
    # Act: Execute the functionality
    # Assert: Verify expected outcomes
    # Cleanup: Remove temporary resources
```

### Mocking External Dependencies
```python
from unittest.mock import patch
import pytest

@patch('clockify_rag.embedding.embed_texts')
def test_build_with_mock_embeddings(mock_embed):
    mock_embed.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
    # Test build functionality
```

## Test Coverage

### Critical Areas to Test
1. **Configuration loading** - Environment variables and defaults
2. **Document ingestion** - All supported file formats
3. **Chunking** - Boundary detection and overlap handling
4. **Indexing** - Embedding consistency and storage
5. **Retrieval** - Accuracy and performance of search
6. **Answer generation** - Quality and safety of responses
7. **Error handling** - Graceful degradation on failures
8. **Concurrency** - Thread safety under load

### Current Coverage
The test suite aims to maintain >80% code coverage across:

- Core functionality: 90%+
- CLI commands: 85%+
- Utility functions: 80%+
- Error handling: 75%+

## Test Data

### Fixtures
Test fixtures are located in `tests/fixtures/`:
- Sample documents for ingestion testing
- Mock embeddings for isolation
- Configuration files for validation

### Sample Data
For integration tests, the system uses:
- `sample_kb.md` - Small knowledge base for quick tests
- `large_kb.md` - Larger knowledge base for performance tests
- Various document formats for ingestion validation

## Troubleshooting Tests

### Common Issues
```bash
# If tests fail due to missing dependencies
pip install -r requirements.txt

# If tests fail due to missing NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# If FAISS tests fail on M1 Macs
conda install -c conda-forge faiss-cpu
```

### Debugging Tips
```bash
# Run tests in verbose mode
pytest -v -s tests/test_specific.py

# Run with Python debugger on failures
pytest --pdb tests/

# Run with logging enabled
pytest --log-cli-level=DEBUG tests/
```

## Performance Testing

### Benchmark Tests
```bash
# Run performance benchmarks
python benchmark.py

# Run with specific parameters
BENCHMARK_QUERIES=100 BENCHMARK_CONCURRENCY=10 python benchmark.py
```

### Load Testing
```bash
# Test under concurrent load
pytest tests/test_thread_safety.py -n 4
```

## Quality Gates

### Pre-merge Checks
Before submitting changes, ensure:
- [ ] All tests pass: `pytest tests/`
- [ ] New functionality is tested: `pytest tests/test_new_feature.py`
- [ ] Coverage hasn't decreased significantly
- [ ] Self-test passes: `python clockify_support_cli_final.py --selftest`
- [ ] CLI commands work: `python clockify_support_cli_final.py build --help`
