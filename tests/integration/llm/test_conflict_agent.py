"""
Conflict Resolution Agent Integration Tests.

These tests validate ConflictResolutionAgent with real LLM API calls to verify:
- Decision structure with required fields
- Valid operations (UPDATE, DELETE, SKIP)
- Confidence scores in valid range
- JSON parsing from various response formats
- Error handling for edge cases

All tests require LLM API keys to be configured via environment variables.
Tests will be skipped if API keys are not available.
"""

import os
import pytest
import logging
from typing import List, Dict, Any

from hypothesis import given, settings, strategies as st

from omnimemory.core.agents import ConflictResolutionAgent

from ..helpers.fixtures import validate_agent_decision

logger = logging.getLogger(__name__)


# Property-based testing configuration
# Keep max_examples low for integration tests to avoid excessive API costs.
settings.register_profile("ci", max_examples=5, deadline=30000)
settings.register_profile("dev", max_examples=3, deadline=10000)
settings.load_profile("ci" if "CI" in os.environ else "dev")


@pytest.mark.integration
@pytest.mark.llm
class TestConflictAgentStructure:
    """
    Feature: integration-test-infrastructure, Property 3: Conflict Agent Response Structure
    
    For any conflict scenario (new memory and linked memories), the ConflictResolutionAgent
    SHALL return a list of decisions where each decision contains the required fields:
    memory_id, operation, confidence_score, and reasoning.
    """
    
    @pytest.mark.asyncio
    async def test_conflict_agent_decision_structure(
        self, real_llm_connection, sample_conflict_scenario: Dict[str, Any]
    ):
        """
        Test conflict agent returns valid decision structure.
        
        Validates that agent.decide() returns decisions with all required fields.
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = sample_conflict_scenario["new_memory"]
        linked_memories = sample_conflict_scenario["linked_memories"]
        
        result = await agent.decide(new_memory, linked_memories)
        
        assert isinstance(result, list), "Result should be a list of decisions"
        assert len(result) > 0, "Result should not be empty"
        
        for decision in result:
            is_valid, error = validate_agent_decision(decision)
            assert is_valid, f"Invalid decision: {error}"
    
    @pytest.mark.asyncio
    @settings(max_examples=5, deadline=30000)
    @given(
        new_memory_note=st.text(min_size=10, max_size=100),
        linked_count=st.integers(min_value=1, max_value=5),
    )
    async def test_conflict_agent_decision_structure_property(
        self,
        real_llm_connection,
        new_memory_note: str,
        linked_count: int,
    ):
        """
        Feature: integration-test-infrastructure, Property 3: Conflict Agent Response Structure
        
        For any conflict scenario, the agent SHALL return decisions with required fields.
        
        This property test validates that regardless of input scenario, the agent
        always returns properly structured decisions.
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": new_memory_note,
            "embedding": [0.1] * 1536,
        }
        
        linked_memories = [
            {
                "memory_id": f"mem-{i}",
                "document": f"Linked memory {i}",
                "composite_score": 0.5 + (i * 0.1),
            }
            for i in range(linked_count)
        ]
        
        result = await agent.decide(new_memory, linked_memories)
        
        assert isinstance(result, list), "Result should be a list of decisions"
        assert len(result) == len(linked_memories), (
            f"Should have {len(linked_memories)} decisions, got {len(result)}"
        )
        
        for decision in result:
            is_valid, error = validate_agent_decision(decision)
            assert is_valid, f"Invalid decision: {error}"


@pytest.mark.integration
@pytest.mark.llm
class TestConflictAgentOperations:
    """
    Feature: integration-test-infrastructure, Property 4: Conflict Agent Operation Validity
    
    For any decision returned by ConflictResolutionAgent, the operation field
    SHALL be one of the valid values: UPDATE, DELETE, or SKIP.
    """
    
    @pytest.mark.asyncio
    async def test_conflict_agent_operations(
        self, real_llm_connection, sample_conflict_scenario: Dict[str, Any]
    ):
        """
        Test conflict agent returns valid operations.
        
        Validates that agent.decide() returns decisions with valid operations.
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = sample_conflict_scenario["new_memory"]
        linked_memories = sample_conflict_scenario["linked_memories"]
        
        result = await agent.decide(new_memory, linked_memories)
        
        valid_operations = {"UPDATE", "DELETE", "SKIP"}
        
        for decision in result:
            operation = decision.get("operation")
            assert operation in valid_operations, (
                f"Invalid operation: {operation}. Must be one of {valid_operations}"
            )
    
    @pytest.mark.asyncio
    @settings(max_examples=5, deadline=30000)
    @given(
        new_memory_note=st.text(min_size=10, max_size=100),
        linked_count=st.integers(min_value=1, max_value=3),
    )
    async def test_conflict_agent_operations_property(
        self,
        real_llm_connection,
        new_memory_note: str,
        linked_count: int,
    ):
        """
        Feature: integration-test-infrastructure, Property 4: Conflict Agent Operation Validity
        
        For any decision, the operation field SHALL be UPDATE, DELETE, or SKIP.
        
        This property test validates that all operations returned by the agent
        are from the valid set of operations.
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": new_memory_note,
            "embedding": [0.1] * 1536,
        }
        
        linked_memories = [
            {
                "memory_id": f"mem-{i}",
                "document": f"Linked memory {i}",
                "composite_score": 0.5 + (i * 0.1),
            }
            for i in range(linked_count)
        ]
        
        result = await agent.decide(new_memory, linked_memories)
        
        valid_operations = {"UPDATE", "DELETE", "SKIP"}
        
        for decision in result:
            operation = decision.get("operation")
            assert operation in valid_operations, (
                f"Invalid operation: {operation}. Must be one of {valid_operations}"
            )


@pytest.mark.integration
@pytest.mark.llm
class TestConflictAgentConfidence:
    """
    Feature: integration-test-infrastructure, Property 5: Conflict Agent Confidence Bounds
    
    For any decision returned by ConflictResolutionAgent, the confidence_score
    SHALL be a float between 0.0 and 1.0 inclusive.
    """
    
    @pytest.mark.asyncio
    async def test_conflict_agent_confidence_scores(
        self, real_llm_connection, sample_conflict_scenario: Dict[str, Any]
    ):
        """
        Test conflict agent returns valid confidence scores.
        
        Validates that agent.decide() returns decisions with confidence scores
        in the valid range [0.0, 1.0].
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = sample_conflict_scenario["new_memory"]
        linked_memories = sample_conflict_scenario["linked_memories"]
        
        result = await agent.decide(new_memory, linked_memories)
        
        for decision in result:
            confidence = decision.get("confidence_score")
            assert isinstance(confidence, (int, float)), (
                f"confidence_score must be numeric, got {type(confidence)}"
            )
            assert 0.0 <= confidence <= 1.0, (
                f"confidence_score must be between 0.0 and 1.0, got {confidence}"
            )
    
    @pytest.mark.asyncio
    @settings(max_examples=5, deadline=30000)
    @given(
        new_memory_note=st.text(min_size=10, max_size=100),
        linked_count=st.integers(min_value=1, max_value=3),
    )
    async def test_conflict_agent_confidence_property(
        self,
        real_llm_connection,
        new_memory_note: str,
        linked_count: int,
    ):
        """
        Feature: integration-test-infrastructure, Property 5: Conflict Agent Confidence Bounds
        
        For any decision, the confidence_score SHALL be between 0.0 and 1.0.
        
        This property test validates that all confidence scores returned by the
        agent are within the valid range.
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": new_memory_note,
            "embedding": [0.1] * 1536,
        }
        
        linked_memories = [
            {
                "memory_id": f"mem-{i}",
                "document": f"Linked memory {i}",
                "composite_score": 0.5 + (i * 0.1),
            }
            for i in range(linked_count)
        ]
        
        result = await agent.decide(new_memory, linked_memories)
        
        for decision in result:
            confidence = decision.get("confidence_score")
            assert isinstance(confidence, (int, float)), (
                f"confidence_score must be numeric, got {type(confidence)}"
            )
            assert 0.0 <= confidence <= 1.0, (
                f"confidence_score must be between 0.0 and 1.0, got {confidence}"
            )


@pytest.mark.integration
@pytest.mark.llm
class TestConflictAgentEdgeCases:
    """Test edge cases for conflict resolution agent."""
    
    @pytest.mark.asyncio
    async def test_conflict_agent_empty_linked_memories(
        self, real_llm_connection
    ):
        """
        Test conflict agent handles empty linked memories.
        
        Validates that agent.decide() returns empty list when no linked memories.
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": "Test memory",
            "embedding": [0.1] * 1536,
        }
        linked_memories = []
        
        result = await agent.decide(new_memory, linked_memories)
        
        assert result == [], "Empty linked memories should return empty list"
    
    @pytest.mark.asyncio
    async def test_conflict_agent_single_linked_memory(
        self, real_llm_connection
    ):
        """
        Test conflict agent handles single linked memory.
        
        Validates that agent.decide() returns single decision for single linked memory.
        """
        agent = ConflictResolutionAgent(llm_connection=real_llm_connection)
        
        new_memory = {
            "natural_memory_note": "Test memory",
            "embedding": [0.1] * 1536,
        }
        linked_memories = [
            {
                "memory_id": "mem-1",
                "document": "Linked memory",
                "composite_score": 0.8,
            }
        ]
        
        result = await agent.decide(new_memory, linked_memories)
        
        assert len(result) == 1, "Single linked memory should return single decision"
        is_valid, error = validate_agent_decision(result[0])
        assert is_valid, f"Invalid decision: {error}"
