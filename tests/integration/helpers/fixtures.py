"""
Test helper utilities for integration tests.

This module provides utility functions for test assertions, data generation,
and performance tracking.
"""

import json
import re
import time
from contextlib import contextmanager
from typing import List, Tuple, Dict, Any, Optional, Callable


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
    
    Returns:
        Cosine similarity value between -1 and 1.
    
    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same length, got {len(vec1)} and {len(vec2)}")
    
    if len(vec1) == 0:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Calculate magnitudes
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def validate_embedding_dimensions(embedding: List[float], expected_dim: int) -> bool:
    """
    Validate embedding has correct dimensions.
    
    Args:
        embedding: The embedding vector to validate.
        expected_dim: Expected number of dimensions.
    
    Returns:
        True if embedding has correct dimensions, False otherwise.
    """
    return len(embedding) == expected_dim


def validate_agent_decision(decision: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate agent decision structure and fields.
    
    Args:
        decision: The decision dictionary to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    required_fields = ["memory_id", "operation", "confidence_score", "reasoning"]
    
    for field in required_fields:
        if field not in decision:
            return False, f"Missing required field: {field}"
    
    # Validate operation is valid
    valid_operations = ["UPDATE", "DELETE", "SKIP"]
    if decision["operation"] not in valid_operations:
        return False, f"Invalid operation: {decision['operation']}. Must be one of {valid_operations}"
    
    # Validate confidence score is in range
    confidence = decision["confidence_score"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        return False, f"Invalid confidence_score: {confidence}. Must be float between 0.0 and 1.0"
    
    # Validate reasoning is a string
    if not isinstance(decision["reasoning"], str):
        return False, f"reasoning must be a string, got {type(decision['reasoning'])}"
    
    return True, ""


def extract_json_from_response(content: str) -> Dict[str, Any]:
    """
    Extract JSON from various response formats.
    
    Args:
        content: The response content to parse.
    
    Returns:
        Parsed JSON dictionary.
    
    Raises:
        ValueError: If JSON cannot be extracted or parsed.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code block
    markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(markdown_pattern, content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text (non-greedy to avoid over-matching)
    json_pattern = r"\{[\s\S]*?\}"
    matches = re.findall(json_pattern, content)
    for match in reversed(matches):  # Try last matches first (usually more complete)
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    raise ValueError(f"Could not extract JSON from response: {content[:200]}...")


def measure_latency(func: Callable) -> Tuple[Any, float]:
    """
    Measure function execution latency in milliseconds.
    
    Args:
        func: The function to measure.
    
    Returns:
        Tuple of (result, latency_ms).
    """
    start_time = time.time()
    result = func()
    latency_ms = (time.time() - start_time) * 1000
    return result, latency_ms


@contextmanager
def measure_latency_context(operation: str = "operation"):
    """
    Context manager for measuring operation latency.
    
    Args:
        operation: Name of the operation being measured.
    
    Yields:
        Function to record token usage.
    
    Example:
        with measure_latency_context("embedding_generation") as record_tokens:
            result = generate_embedding(text)
            record_tokens(100)  # Optional: record token usage
    """
    start_time = time.time()
    tokens_used: Optional[int] = None
    
    def record_tokens(tokens: int):
        nonlocal tokens_used
        tokens_used = tokens
    
    try:
        yield record_tokens
    finally:
        latency_ms = (time.time() - start_time) * 1000
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"{operation} latency: {latency_ms:.2f}ms, tokens: {tokens_used}")


def generate_random_text(num_words: int = 20) -> str:
    """
    Generate random text for testing.
    
    Args:
        num_words: Number of words to generate.
    
    Returns:
        Random text string.
    """
    import random
    
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "artificial", "intelligence", "data",
        "algorithm", "neural", "network", "deep", "learning", "model",
    ]
    
    return " ".join(random.choice(words) for _ in range(num_words))


def validate_consolidated_memory(consolidated: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate consolidated memory structure from synthesis agent.
    
    Args:
        consolidated: The consolidated memory dictionary.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(consolidated, dict):
        return False, "Consolidated memory must be a dictionary"
    
    if "natural_memory_note" not in consolidated:
        return False, "Missing natural_memory_note field"
    
    if not isinstance(consolidated["natural_memory_note"], str):
        return False, "natural_memory_note must be a string"
    
    if len(consolidated["natural_memory_note"].strip()) == 0:
        return False, "natural_memory_note cannot be empty"
    
    return True, ""
