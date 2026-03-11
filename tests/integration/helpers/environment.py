"""
Environment detection utilities for integration tests.

This module provides functionality to detect Docker availability, validate
service configurations, and determine the appropriate test environment.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Information about the detected test environment."""
    
    docker_installed: bool
    docker_daemon_running: bool
    use_docker: bool
    llm_configured: bool
    embedding_configured: bool
    warnings: List[str]
    summary: str


class EnvironmentDetector:
    """Detects Docker availability and validates test environment configuration."""
    
    def __init__(self):
        """Initialize the environment detector."""
        self._docker_available_cache: Optional[bool] = None
        self._docker_daemon_cache: Optional[bool] = None
    
    def is_docker_available(self) -> bool:
        """
        Check if Docker is installed on the system.
        
        Returns:
            True if Docker is installed, False otherwise.
        """
        if self._docker_available_cache is not None:
            return self._docker_available_cache
        
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self._docker_available_cache = result.returncode == 0
            
            if self._docker_available_cache:
                version = result.stdout.strip()
                logger.debug(f"Docker detected: {version}")
            else:
                logger.debug("Docker command failed")
                
            return self._docker_available_cache
            
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"Docker not available: {e}")
            self._docker_available_cache = False
            return False
    
    def is_docker_daemon_running(self) -> bool:
        """
        Check if Docker daemon is running and accessible.
        
        Returns:
            True if Docker daemon is running, False otherwise.
        """
        if self._docker_daemon_cache is not None:
            return self._docker_daemon_cache
        
        if not self.is_docker_available():
            self._docker_daemon_cache = False
            return False
        
        try:
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self._docker_daemon_cache = result.returncode == 0
            
            if self._docker_daemon_cache:
                logger.debug("Docker daemon is running")
            else:
                logger.debug(f"Docker daemon not accessible: {result.stderr}")
                
            return self._docker_daemon_cache
            
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug(f"Docker daemon check failed: {e}")
            self._docker_daemon_cache = False
            return False
    
    def should_use_docker(self) -> bool:
        """
        Determine if Docker should be used for tests.
        
        Respects the USE_DOCKER environment variable override.
        If USE_DOCKER is set to "false" or "0", Docker will not be used
        regardless of availability.
        
        Returns:
            True if Docker should be used, False otherwise.
        """
        # Check for explicit override
        use_docker_env = os.getenv("USE_DOCKER", "").lower()
        if use_docker_env in ("false", "0", "no"):
            logger.debug("USE_DOCKER environment variable set to false")
            return False
        
        # Check Docker availability
        return self.is_docker_available() and self.is_docker_daemon_running()
    
    def validate_llm_config(self) -> Tuple[bool, str]:
        """
        Validate LLM API configuration.
        
        Checks for required environment variables for LLM providers.
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        # Check for OpenAI API key (primary provider)
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        
        if openai_key:
            logger.debug("LLM API key found")
            return True, ""
        
        error_msg = (
            "LLM API key not configured. Set OPENAI_API_KEY or LLM_API_KEY "
            "environment variable to run LLM integration tests."
        )
        logger.debug(error_msg)
        return False, error_msg
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the detected environment configuration.
        
        Returns:
            Dictionary containing environment detection results.
        """
        docker_installed = self.is_docker_available()
        docker_daemon = self.is_docker_daemon_running()
        use_docker = self.should_use_docker()
        llm_valid, llm_error = self.validate_llm_config()
        
        warnings = []
        
        # Docker warnings
        if not docker_installed:
            warnings.append("Docker is not installed")
        elif not docker_daemon:
            warnings.append("Docker daemon is not running")
        elif not use_docker:
            warnings.append("Docker usage disabled via USE_DOCKER environment variable")
        
        # LLM warnings
        if not llm_valid:
            warnings.append(llm_error)
        
        # Build summary text
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("Integration Test Environment Detection")
        summary_lines.append("=" * 60)
        
        # Docker status
        if docker_installed and docker_daemon:
            summary_lines.append("✓ Docker: Available and running")
        elif docker_installed:
            summary_lines.append("✗ Docker: Installed but daemon not running")
        else:
            summary_lines.append("✗ Docker: Not installed")
        
        if not use_docker and docker_installed:
            summary_lines.append("  Note: Docker usage disabled via USE_DOCKER=false")
        
        # LLM status
        if llm_valid:
            provider = os.getenv("LLM_PROVIDER", "openai")
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            summary_lines.append(f"✓ LLM: Configured ({provider}/{model})")
        else:
            summary_lines.append("✗ LLM: Not configured (tests will be skipped)")
        
        # Warnings
        if warnings:
            summary_lines.append("")
            summary_lines.append("Warnings:")
            for warning in warnings:
                summary_lines.append(f"  - {warning}")
        
        summary_lines.append("=" * 60)
        summary = "\n".join(summary_lines)
        
        return {
            "docker_installed": docker_installed,
            "docker_daemon_running": docker_daemon,
            "use_docker": use_docker,
            "llm_configured": llm_valid,
            "llm_error": llm_error if not llm_valid else None,
            "warnings": warnings,
            "summary": summary,
        }
    
    def get_environment_info(self) -> EnvironmentInfo:
        """
        Get structured environment information.
        
        Returns:
            EnvironmentInfo dataclass with detection results.
        """
        summary_dict = self.get_environment_summary()
        
        return EnvironmentInfo(
            docker_installed=summary_dict["docker_installed"],
            docker_daemon_running=summary_dict["docker_daemon_running"],
            use_docker=summary_dict["use_docker"],
            llm_configured=summary_dict["llm_configured"],
            embedding_configured=bool(
                os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            ),
            warnings=summary_dict["warnings"],
            summary=summary_dict["summary"],
        )
