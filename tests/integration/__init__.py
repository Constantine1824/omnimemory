"""
Integration tests for OmniMemory.

These tests validate system behavior with real external dependencies (LLM providers, vector databases).
They complement unit tests by testing end-to-end functionality in production-like environments.

Test Organization:
- llm/: LLM integration tests (embeddings, agents)
- vectordb/: Vector database integration tests (Phase 2)
- helpers/: Test utilities and fixtures

Running Tests:
- All integration tests: pytest -m integration
- LLM tests only: pytest -m llm
- Vector DB tests only: pytest -m vectordb (Phase 2)
"""
