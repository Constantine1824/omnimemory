"""
Synthesis Agent Integration Tests.

These tests validate SynthesisAgent with real LLM API calls to verify:
- Consolidated memory structure with required fields
- Content quality and information retention
- JSON parsing from various response formats
- Error handling for edge cases

All tests require LLM API keys to be configured via environment variables.
Tests will be skipped if API keys are not available.

Usage:
    pytest tests/integration/llm/test_synthesis_agent.py -m llm -v
"""

import os
import pytest
import logging
from typing import Dict, Any

from hypothesis import given, settings, strategies as st

from omnimemory.core.agents import SynthesisAgent

from ..helpers.fixtures import validate_consolidated_memory

logger = logging.getLogger(__name__)


# Property-based testing configuration
# Keep max_examples low for integration tests to avoid excessive API costs.
settings.register_profile("ci", max_examples=5, deadline=30000)
settings.register_profile("dev", max_examples=3, deadline=10000)
settings.load_profile("ci" if "CI" in os.environ else "dev")


@pytest.mark.integration
@pytest.mark.llm
class TestSynthesisAgentStructure:
    """
    Feature: integration-test-infrastructure, Property 6: Synthesis Agent Response Structure
    
    For any synthesis scenario (new memory and existing memories), the SynthesisAgent
    SHALL return a dictionary containing consolidated_memory with a natural_memory_note
    field and a synthesis_summary string.
    """
    
    @pytest.mark.asyncio
    async def test_synthesis_agent_consolidation(
        self, real_llm_connection, sample_synthesis_scenario: Dict[str, Any]
    ):
        """
        Test synthesis agent produces valid consolidated memory.
        
        Validates that agent.consolidate_memories() returns valid structure.
        """
        agent = SynthesisAgent(llm_connection=real_llm_connection)
        
        new_memory = sample_synthesis_scenario["new_memory"]
        existing_memories = sample_synthesis_scenario["existing_memories"]
        
        result = await agent.consolidate_memories(new_memory, existing_memories)
        
        assert isinstance(result, dict), "Result should be a dictionary"
        
        is_valid, error = validate_consolidated_memory(result.get("consolidated_memory", {}))
        assert is_valid, f"Invalid consolidated memory: {error}"
        
        # Validate synthesis_summary exists
        assert "synthesis_summary" in result, "Missing synthesis_summary field"
        assert isinstance(result["synthesis_summary"], str), (
            "synthesis_summary must be a string"
        )
        assert len(result["synthesis_summary"].strip()) > 0, (
            "synthesis_summary cannot be empty"
        )
    
    @pytest.mark.asyncio
    @settings(max_examples=5, deadline=30000)
    @given(
        new_memory_note=st.text(min_size=10, max_size=100),
        existing_count=st.integers(min_value=1, max_value=3),
    )
    async def test_synthesis_agent_consolidation_property(
        self,
        real_llm_connection,
        new_memory_note: str,
        existing_count: int,
    ):
        """
        Feature: integration-test-infrastructure, Property 6: Synthesis Agent Response Structure
        
        For any synthesis scenario, the agent SHALL return valid consolidated memory.
        
        This property test validates that regardless of input scenario, the agent
        always returns properly structured consolidated memory.
        """
        agent = SynthesisAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": new_memory_note,
            "embedding": [0.1] * 1536,
        }
        
        existing_memories = [
            {
                "document": f"Existing memory {i}",
                "composite_score": 0.5 + (i * 0.1),
            }
            for i in range(existing_count)
        ]
        
        result = await agent.consolidate_memories(new_memory, existing_memories)
        
        assert isinstance(result, dict), "Result should be a dictionary"
        
        is_valid, error = validate_consolidated_memory(result.get("consolidated_memory", {}))
        assert is_valid, f"Invalid consolidated memory: {error}"
        
        # Validate synthesis_summary exists
        assert "synthesis_summary" in result, "Missing synthesis_summary field"
        assert isinstance(result["synthesis_summary"], str), (
            "synthesis_summary must be a string"
        )
        assert len(result["synthesis_summary"].strip()) > 0, (
            "synthesis_summary cannot be empty"
        )


@pytest.mark.integration
@pytest.mark.llm
class TestSynthesisAgentContent:
    """Test content quality for synthesis agent."""
    
    @pytest.mark.asyncio
    async def test_synthesis_agent_content_quality(
        self, real_llm_connection, sample_synthesis_scenario: Dict[str, Any]
    ):
        """
        Test synthesis agent produces quality consolidated memory.
        
        Validates that consolidated memory contains meaningful content.
        """
        agent = SynthesisAgent(llm_connection=real_llm_connection)
        
        new_memory = sample_synthesis_scenario["new_memory"]
        existing_memories = sample_synthesis_scenario["existing_memories"]
        
        result = await agent.consolidate_memories(new_memory, existing_memories)
        
        consolidated = result.get("consolidated_memory", {})
        natural_note = consolidated.get("natural_memory_note", "")
        
        # Validate content is not empty
        assert len(natural_note.strip()) > 10, (
            "Consolidated memory should have meaningful content"
        )
        
        # Validate synthesis summary is descriptive
        summary = result.get("synthesis_summary", "")
        assert len(summary.strip()) > 10, (
            "Synthesis summary should be descriptive"
        )
    
    @pytest.mark.asyncio
    async def test_synthesis_agent_metadata(
        self, real_llm_connection, sample_synthesis_scenario: Dict[str, Any]
    ):
        """
        Test synthesis agent includes metadata in response.
        
        Validates that response includes synthesis_metadata field.
        """
        agent = SynthesisAgent(llm_connection=real_llm_connection)
        
        new_memory = sample_synthesis_scenario["new_memory"]
        existing_memories = sample_synthesis_scenario["existing_memories"]
        
        result = await agent.consolidate_memories(new_memory, existing_memories)
        
        # Validate synthesis_metadata exists
        assert "synthesis_metadata" in result, "Missing synthesis_metadata field"
        assert isinstance(result["synthesis_metadata"], dict), (
            "synthesis_metadata must be a dictionary"
        )


@pytest.mark.integration
@pytest.mark.llm
class TestSynthesisAgentEdgeCases:
    """Test edge cases for synthesis agent."""
    
    @pytest.mark.asyncio
    async def test_synthesis_agent_empty_existing_memories(
        self, real_llm_connection
    ):
        """
        Test synthesis agent handles empty existing memories.
        
        Validates that agent.consolidate_memories() returns valid structure
        even with no existing memories.
        """
        agent = SynthesisAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": "Test memory",
            "embedding": [0.1] * 1536,
        }
        existing_memories = []
        
        result = await agent.consolidate_memories(new_memory, existing_memories)
        
        assert isinstance(result, dict), "Result should be a dictionary"
        
        is_valid, error = validate_consolidated_memory(result.get("consolidated_memory", {}))
        assert is_valid, f"Invalid consolidated memory: {error}"
    
    @pytest.mark.asyncio
    async def test_synthesis_agent_single_existing_memory(
        self, real_llm_connection
    ):
        """
        Test synthesis agent handles single existing memory.
        
        Validates that agent.consolidate_memories() returns valid structure
        with single existing memory.
        """
        agent = SynthesisAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": "Test memory",
            "embedding": [0.1] * 1536,
        }
        existing_memories = [
            {
                "document": "Existing memory",
                "composite_score": 0.8,
            }
        ]
        
        result = await agent.consolidate_memories(new_memory, existing_memories)
        
        assert isinstance(result, dict), "Result should be a dictionary"
        
        is_valid, error = validate_consolidated_memory(result.get("consolidated_memory", {}))
        assert is_valid, f"Invalid consolidated memory: {error}"


@pytest.mark.integration
@pytest.mark.llm
class TestSynthesisAgentPerformance:
    """Test performance for synthesis agent."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_synthesis_agent_performance(
        self, real_llm_connection, sample_synthesis_scenario: Dict[str, Any]
    ):
        """
        Test synthesis agent performance.
        
        Measures and validates synthesis operation latency.
        """
        import time
        
        agent = SynthesisAgent(llm_connection=real_llm_connection)
        
        new_memory = sample_synthesis_scenario["new_memory"]
        existing_memories = sample_synthesis_scenario["existing_memories"]
        
        start = time.time()
        result = await agent.consolidate_memories(new_memory, existing_memories)
        latency_ms = (time.time() - start) * 1000
        
        assert isinstance(result, dict), "Result should be a dictionary"
        
        threshold_ms = 10000  # 10s threshold for LLM synthesis calls
        
        logger.info(f"Synthesis latency: {latency_ms:.2f}ms (threshold: {threshold_ms}ms)")
        
        assert latency_ms < threshold_ms, (
            f"Synthesis latency {latency_ms:.0f}ms exceeded threshold {threshold_ms}ms"
        )
