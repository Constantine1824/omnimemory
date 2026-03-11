"""
Configuration management for integration tests.

This module provides configuration loading and validation for integration tests,
reading from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    
    # LLM Configuration
    llm_api_key: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    
    # Embedding Configuration
    embedding_api_key: Optional[str] = None
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Docker Configuration
    use_docker: bool = True
    docker_available: bool = False
    
    # Test Configuration
    test_timeout: int = 30
    performance_threshold_ms: int = 5000
    min_similarity_threshold: float = 0.99
    
    # Validation errors
    _validation_errors: List[str] = field(default_factory=list, init=False, repr=False)
    
    @classmethod
    def from_environment(cls) -> "IntegrationTestConfig":
        """
        Load configuration from environment variables.
        
        Environment Variables:
            LLM_API_KEY or OPENAI_API_KEY: API key for LLM provider
            LLM_PROVIDER: LLM provider name (default: openai)
            LLM_MODEL: LLM model to use (default: gpt-4o-mini)
            
            EMBEDDING_API_KEY: API key for embedding provider (falls back to LLM_API_KEY)
            EMBEDDING_PROVIDER: Embedding provider name (default: openai)
            EMBEDDING_MODEL: Embedding model to use (default: text-embedding-3-small)
            EMBEDDING_DIMENSIONS: Embedding vector dimensions (default: 1536)
            
            USE_DOCKER: Whether to use Docker (default: true)
            
            INTEGRATION_TEST_TIMEOUT: Test timeout in seconds (default: 30)
            PERFORMANCE_THRESHOLD_MS: Performance threshold in milliseconds (default: 5000)
            MIN_SIMILARITY_THRESHOLD: Minimum similarity for embedding tests (default: 0.99)
        
        Returns:
            IntegrationTestConfig instance with values from environment.
        """
        # LLM configuration
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Embedding configuration (falls back to LLM config)
        embedding_api_key = os.getenv("EMBEDDING_API_KEY") or llm_api_key
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", llm_provider)
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        try:
            embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        except ValueError:
            embedding_dimensions = 1536
        
        # Docker configuration
        use_docker_env = os.getenv("USE_DOCKER", "true").lower()
        use_docker = use_docker_env not in ("false", "0", "no")
        
        # Test configuration
        try:
            test_timeout = int(os.getenv("INTEGRATION_TEST_TIMEOUT", "30"))
        except ValueError:
            test_timeout = 30
        
        try:
            performance_threshold_ms = int(os.getenv("PERFORMANCE_THRESHOLD_MS", "5000"))
        except ValueError:
            performance_threshold_ms = 5000
        
        try:
            min_similarity_threshold = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.99"))
        except ValueError:
            min_similarity_threshold = 0.99
        
        return cls(
            llm_api_key=llm_api_key,
            llm_provider=llm_provider,
            llm_model=llm_model,
            embedding_api_key=embedding_api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            use_docker=use_docker,
            docker_available=False,  # Will be set by environment detector
            test_timeout=test_timeout,
            performance_threshold_ms=performance_threshold_ms,
            min_similarity_threshold=min_similarity_threshold,
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []
        
        # Validate LLM configuration
        if not self.llm_api_key:
            errors.append(
                "LLM API key not configured. Set LLM_API_KEY or OPENAI_API_KEY environment variable."
            )
        
        if not self.llm_provider:
            errors.append("LLM provider not specified")
        
        if not self.llm_model:
            errors.append("LLM model not specified")
        
        # Validate embedding configuration
        if not self.embedding_api_key:
            errors.append(
                "Embedding API key not configured. Set EMBEDDING_API_KEY or use LLM_API_KEY."
            )
        
        if not self.embedding_provider:
            errors.append("Embedding provider not specified")
        
        if not self.embedding_model:
            errors.append("Embedding model not specified")
        
        if self.embedding_dimensions <= 0:
            errors.append(f"Invalid embedding dimensions: {self.embedding_dimensions}")
        
        # Validate test configuration
        if self.test_timeout <= 0:
            errors.append(f"Invalid test timeout: {self.test_timeout}")
        
        if self.performance_threshold_ms <= 0:
            errors.append(f"Invalid performance threshold: {self.performance_threshold_ms}")
        
        if not (0.0 <= self.min_similarity_threshold <= 1.0):
            errors.append(
                f"Invalid similarity threshold: {self.min_similarity_threshold} "
                "(must be between 0.0 and 1.0)"
            )
        
        self._validation_errors = errors
        return errors
    
    def is_llm_configured(self) -> bool:
        """
        Check if LLM configuration is complete.
        
        Returns:
            True if LLM is configured with API key, False otherwise.
        """
        return bool(self.llm_api_key and self.llm_provider and self.llm_model)
    
    def is_embedding_configured(self) -> bool:
        """
        Check if embedding configuration is complete.
        
        Returns:
            True if embedding is configured with API key, False otherwise.
        """
        return bool(
            self.embedding_api_key
            and self.embedding_provider
            and self.embedding_model
            and self.embedding_dimensions > 0
        )
    
    def get_llm_config(self) -> dict:
        """
        Get LLM configuration as a dictionary.
        
        Returns:
            Dictionary with LLM configuration.
        """
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "api_key": self.llm_api_key,
        }
    
    def get_embedding_config(self) -> dict:
        """
        Get embedding configuration as a dictionary.
        
        Returns:
            Dictionary with embedding configuration.
        """
        return {
            "provider": self.embedding_provider,
            "model": self.embedding_model,
            "dimensions": self.embedding_dimensions,
            "api_key": self.embedding_api_key,
        }
