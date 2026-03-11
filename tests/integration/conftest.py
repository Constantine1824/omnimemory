"""
Pytest configuration and shared fixtures for OmniMemory integration tests.

This module provides fixtures for integration tests that validate system behavior
with real external dependencies (LLM providers, vector databases).
"""

import logging
import os
import pytest
from typing import Any, Dict

from omnimemory.core.llm import LLMConnection

from .helpers.config import IntegrationTestConfig
from .helpers.environment import EnvironmentDetector

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest with custom markers and options.
    
    This function is called by pytest during initialization and sets up
    custom markers for integration tests.
    """
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line("markers", "llm: mark test as requiring LLM access")
    config.addinivalue_line("markers", "vectordb: mark test as requiring vector DB")
    config.addinivalue_line("markers", "docker: mark test as requiring Docker")
    config.addinivalue_line("markers", "timeout: set a timeout for long-running tests")


@pytest.fixture(scope="session")
def integration_config() -> IntegrationTestConfig:
    """
    Load and validate integration test configuration.
    
    This fixture loads configuration from environment variables and validates
    that required settings are present. It is session-scoped so configuration
    is loaded once per test session.
    
    Returns:
        IntegrationTestConfig with values from environment variables.
    """
    config = IntegrationTestConfig.from_environment()
    errors = config.validate()
    
    if errors:
        logger.warning("Configuration validation warnings:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return config


@pytest.fixture(scope="session")
def environment_detector() -> EnvironmentDetector:
    """
    Provide environment detection utilities.
    
    This fixture provides an EnvironmentDetector instance that can be used
    to check Docker availability and validate service configurations.
    
    Returns:
        EnvironmentDetector instance.
    """
    return EnvironmentDetector()


@pytest.fixture(scope="session")
def environment_info(
    environment_detector: EnvironmentDetector, integration_config: IntegrationTestConfig
) -> Dict[str, Any]:
    """
    Get environment information and log summary.
    
    This fixture runs environment detection at session start and logs a
    summary of the detected environment configuration.
    
    Args:
        environment_detector: EnvironmentDetector fixture.
        integration_config: IntegrationTestConfig fixture.
    
    Returns:
        Dictionary with environment detection results.
    """
    # Update docker_available in config
    integration_config.docker_available = environment_detector.should_use_docker()
    
    # Get environment summary and log it
    env_info = environment_detector.get_environment_summary()
    
    # Log the summary
    logger.info(env_info["summary"])
    
    return env_info


@pytest.fixture(scope="session")
def real_llm_connection(
    integration_config: IntegrationTestConfig, environment_info: Dict[str, Any]
) -> LLMConnection:
    """
    Provide real LLM connection for integration tests.
    
    This fixture creates an LLMConnection instance configured with the
    environment's API keys and settings. If LLM is not configured, the
    fixture will skip all tests that depend on it.
    
    LLMConnection reads all configuration from environment variables, so we
    set the required env vars before construction.
    
    Args:
        integration_config: IntegrationTestConfig fixture.
        environment_info: Environment info from environment_info fixture.
    
    Returns:
        LLMConnection instance configured for real API calls.
    
    Raises:
        pytest.skip: If LLM is not configured.
    """
    if not environment_info["llm_configured"]:
        pytest.skip(
            f"LLM not configured: {environment_info.get('llm_error', 'API key missing')}"
        )
    
    # LLMConnection reads from environment variables, so set them first
    llm_config = integration_config.get_llm_config()
    embedding_config = integration_config.get_embedding_config()
    
    if llm_config.get("api_key"):
        os.environ["LLM_API_KEY"] = llm_config["api_key"]
    if llm_config.get("provider"):
        os.environ["LLM_PROVIDER"] = llm_config["provider"]
    if llm_config.get("model"):
        os.environ["LLM_MODEL"] = llm_config["model"]
    
    if embedding_config.get("api_key"):
        os.environ["EMBEDDING_API_KEY"] = embedding_config["api_key"]
    if embedding_config.get("provider"):
        os.environ["EMBEDDING_PROVIDER"] = embedding_config["provider"]
    if embedding_config.get("model"):
        os.environ["EMBEDDING_MODEL"] = embedding_config["model"]
    if embedding_config.get("dimensions"):
        os.environ["EMBEDDING_DIMENSIONS"] = str(embedding_config["dimensions"])
    
    llm = LLMConnection()
    
    logger.debug(f"Created LLMConnection: {llm_config['provider']}/{llm_config['model']}")
    
    return llm


@pytest.fixture
def sample_embedding_text() -> str:
    """
    Provide sample text for embedding tests.
    
    Returns:
        Sample text string for embedding generation tests.
    """
    return "This is a sample text for testing embedding functionality."


@pytest.fixture
def sample_conflict_scenario() -> Dict[str, Any]:
    """
    Provide sample conflict scenario for agent tests.
    
    Returns:
        Dictionary with conflict scenario data including new memory and linked memories.
    """
    return {
        "new_memory": {
            "natural_memory_note": "User asked about the weather in London",
            "embedding": [0.1] * 1536,
        },
        "linked_memories": [
            {
                "memory_id": "mem-1",
                "document": "User previously asked about weather in London",
                "composite_score": 0.85,
            },
            {
                "memory_id": "mem-2",
                "document": "User asked about weather in Paris",
                "composite_score": 0.3,
            },
        ],
    }


@pytest.fixture
def sample_conflict_scenario_high_similarity() -> Dict[str, Any]:
    """
    Provide conflict scenario where all linked memories are highly similar.
    
    Returns:
        Dictionary with conflict scenario where all memories have high scores.
    """
    return {
        "new_memory": {
            "natural_memory_note": "User's favorite color is blue",
            "embedding": [0.1] * 1536,
        },
        "linked_memories": [
            {
                "memory_id": "mem-1",
                "document": "User said their favorite color is blue",
                "composite_score": 0.95,
            },
            {
                "memory_id": "mem-2",
                "document": "User mentioned they love the color blue",
                "composite_score": 0.90,
            },
            {
                "memory_id": "mem-3",
                "document": "User prefers blue over other colors",
                "composite_score": 0.88,
            },
        ],
    }


@pytest.fixture
def sample_conflict_scenario_borderline() -> Dict[str, Any]:
    """
    Provide conflict scenario with borderline similarity scores.
    
    Returns:
        Dictionary with conflict scenario where scores are near decision boundaries.
    """
    return {
        "new_memory": {
            "natural_memory_note": "User works from home on Fridays",
            "embedding": [0.1] * 1536,
        },
        "linked_memories": [
            {
                "memory_id": "mem-1",
                "document": "User sometimes works remotely",
                "composite_score": 0.55,
            },
            {
                "memory_id": "mem-2",
                "document": "User has a flexible work schedule",
                "composite_score": 0.50,
            },
        ],
    }


@pytest.fixture
def sample_synthesis_scenario() -> Dict[str, Any]:
    """
    Provide sample synthesis scenario for agent tests.
    
    Returns:
        Dictionary with synthesis scenario data including new memory and existing memories.
    """
    return {
        "new_memory": {
            "natural_memory_note": "User is interested in machine learning",
            "embedding": [0.1] * 1536,
        },
        "existing_memories": [
            {
                "document": "User works as a software engineer",
                "composite_score": 0.7,
            },
            {
                "document": "User has experience with Python",
                "composite_score": 0.6,
            },
        ],
    }


@pytest.fixture
def performance_tracker():
    """
    Track and report test performance metrics.
    
    This fixture provides a context manager for measuring operation latency
    and tracking performance metrics during tests.
    
    Yields:
        Context manager function for tracking performance.
    """
    import time
    from contextlib import contextmanager
    from dataclasses import dataclass
    from typing import Optional, Callable, Any
    
    @dataclass
    class PerformanceMetric:
        operation: str
        latency_ms: float
        tokens_used: Optional[int]
        model: str
        success: bool
        error: Optional[str]
    
    metrics: list[PerformanceMetric] = []
    
    @contextmanager
    def track(operation: str, model: str = "unknown"):
        """
        Track performance of an operation.
        
        Args:
            operation: Name of the operation being tracked.
            model: Model name being used.
        
        Yields:
            Function to record token usage.
        """
        start_time = time.time()
        tokens_used: Optional[int] = None
        error: Optional[str] = None
        
        def record_tokens(tokens: int):
            nonlocal tokens_used
            tokens_used = tokens
        
        try:
            yield record_tokens
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            metrics.append(
                PerformanceMetric(
                    operation=operation,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                    model=model,
                    success=error is None,
                    error=error,
                )
            )
    
    yield track
    
    # Log performance summary at end of test
    if metrics:
        logger.debug("Performance metrics summary:")
        for m in metrics:
            status = "✓" if m.success else "✗"
            logger.debug(
                f"  {status} {m.operation}: {m.latency_ms:.2f}ms, "
                f"tokens={m.tokens_used}, model={m.model}"
            )
