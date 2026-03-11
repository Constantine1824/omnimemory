# Integration Tests

This directory contains integration tests for OmniMemory that validate system behavior with real external dependencies (LLM providers, vector databases).

## Test Organization

```
tests/integration/
├── __init__.py              # Package initialization
├── conftest.py              # Shared pytest fixtures
├── helpers/                 # Test utilities
│   ├── __init__.py
│   ├── environment.py       # Docker detection and environment validation
│   ├── config.py            # Configuration management
│   └── fixtures.py          # Test data generators and helpers
├── llm/                     # Phase 1: LLM integration tests
│   ├── __init__.py
│   ├── test_llm_connection.py
│   ├── test_conflict_agent.py
│   └── test_synthesis_agent.py
└── vectordb/                # Phase 2: Vector DB tests (future)
    ├── __init__.py
    └── ...
```

## Running Tests

### Prerequisites

- Python 3.10+
- pytest and pytest-asyncio installed
- LLM API keys configured (see Configuration section)

### Basic Commands

```bash
# Run only unit tests (default - no integration tests)
pytest

# Run all integration tests
pytest -m integration

# Run only LLM integration tests
pytest -m llm

# Run with verbose output
pytest -m llm -v

# Run specific test file
pytest tests/integration/llm/test_llm_connection.py -m llm -v

# Run with coverage
pytest -m llm --cov=src/omnimemory --cov-report=term-missing
```

### Pytest Markers

| Marker | Description |
|--------|-------------|
| `integration` | All integration tests |
| `llm` | LLM integration tests (embeddings, agents) |
| `vectordb` | Vector database tests (Phase 2) |
| `docker` | Tests requiring Docker (Phase 2) |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` or `OPENAI_API_KEY` | API key for LLM provider | - |
| `LLM_PROVIDER` | LLM provider name | `openai` |
| `LLM_MODEL` | LLM model to use | `gpt-4o-mini` |
| `EMBEDDING_API_KEY` | API key for embedding provider | Same as LLM_API_KEY |
| `EMBEDDING_PROVIDER` | Embedding provider name | Same as LLM_PROVIDER |
| `EMBEDDING_MODEL` | Embedding model to use | `text-embedding-3-small` |
| `EMBEDDING_DIMENSIONS` | Embedding vector dimensions | `1536` |
| `USE_DOCKER` | Whether to use Docker | `true` |
| `INTEGRATION_TEST_TIMEOUT` | Test timeout in seconds | `30` |
| `PERFORMANCE_THRESHOLD_MS` | Performance threshold in ms | `5000` |
| `MIN_SIMILARITY_THRESHOLD` | Minimum similarity for embeddings | `0.99` |

### Example Configuration

```bash
# Set required API keys
export OPENAI_API_KEY="sk-..."

# Optional: Override defaults
export LLM_MODEL="gpt-4o-mini"
export EMBEDDING_MODEL="text-embedding-3-small"
export EMBEDDING_DIMENSIONS=1536

# Disable Docker (for local testing)
export USE_DOCKER=false

# Run tests
pytest -m llm -v
```

## Test Types

### LLM Integration Tests (Phase 1)

Tests in `tests/integration/llm/` validate LLM operations with real API calls:

- **test_llm_connection.py**: Embedding generation, dimensions, determinism
- **test_conflict_agent.py**: Conflict resolution agent decisions
- **test_synthesis_agent.py**: Synthesis agent consolidation

### Property-Based Tests

The integration tests use [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing:

- **Embedding Dimension Consistency**: Embeddings always have correct dimensions
- **Embedding Determinism**: Same text produces similar embeddings
- **Conflict Agent Response Structure**: Decisions have required fields
- **Conflict Agent Operation Validity**: Operations are UPDATE/DELETE/SKIP
- **Conflict Agent Confidence Bounds**: Scores are between 0.0 and 1.0
- **Synthesis Agent Response Structure**: Consolidated memory has required fields

### Example Tests

- **Missing API Key Skip Behavior**: Tests are skipped when API keys are missing
- **Docker Detection**: Tests detect Docker availability and fallback to local environment
- **Performance Tracking**: Tests measure and log operation latency

## Docker Fallback

The test infrastructure automatically detects Docker availability:

1. If Docker is installed and running, use Docker containers
2. If Docker is not available, fall back to local environment
3. If local services are unavailable, skip tests with clear messages

To explicitly disable Docker:

```bash
export USE_DOCKER=false
pytest -m integration
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on:
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run LLM integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest -m llm -v --log-cli-level=INFO
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results/
```

## Development

### Adding New Tests

1. Create test file in appropriate subdirectory (`tests/integration/llm/`)
2. Add `@pytest.mark.integration` and `@pytest.mark.llm` decorators
3. Use fixtures from `conftest.py`
4. Import helper utilities from `helpers/`
5. Add property-based tests using Hypothesis where appropriate

### Test Quality Guidelines

- Use property-based tests for universal properties
- Use example tests for specific scenarios
- Measure and log performance metrics
- Skip tests gracefully when dependencies are missing
- Provide clear error messages for failures

## Troubleshooting

### Tests are skipped

- Check that API keys are configured
- Verify LLM provider is accessible
- Check logs for skip messages

### Docker errors

- Verify Docker is installed: `docker --version`
- Verify Docker daemon is running: `docker ps`
- Or disable Docker: `export USE_DOCKER=false`

### Performance issues

- Check network connectivity to LLM provider
- Consider using faster models for testing
- Increase performance threshold if needed

## Future Work (Phase 2)

- Vector database integration tests (ChromaDB, PostgreSQL, Qdrant, MongoDB)
- Memory manager integration tests
- End-to-end workflow tests
- Performance benchmarking suite
