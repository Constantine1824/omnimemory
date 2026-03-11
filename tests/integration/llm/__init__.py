"""
LLM integration tests.

Tests in this module validate LLM operations with real API calls:
- Embedding generation (dimensions, determinism, performance)
- Conflict resolution agent (decision structure, operations, JSON parsing)
- Synthesis agent (consolidation, content quality, metadata)

All tests require LLM API keys to be configured via environment variables.
Tests will be skipped if API keys are not available.
"""
