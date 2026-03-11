"""
Unit tests for the Episodic Agent evaluation gate.
"""

import json

import pytest

from omnimemory.core.evals.episodic_eval import EvalResult, validate_episodic_output


# ── Fixtures ────────────────────────────────────────────────────────────────


def _valid_episodic_output() -> dict:
    """Return a fully valid episodic agent output dict."""
    return {
        "context": {
            "available_data": "Full conversation with both user and assistant messages",
            "user_intent": "User wanted to debug a Python asyncio issue",
            "analysis_limitation": "N/A",
        },
        "what_worked": {
            "strategies": [
                "Providing step-by-step debugging instructions",
                "Asking clarifying questions about the error traceback",
            ],
            "pattern": "Structured troubleshooting with concrete examples worked well",
        },
        "what_failed": {
            "strategies": ["Initial generic suggestion was too vague"],
            "pattern": "Generic advice without context fell flat",
        },
        "behavioral_profile": {
            "communication": "Direct and technical. Prefers concise answers with code snippets.",
            "learning": "Hands-on learner who prefers working examples over theory.",
            "problem_solving": "Systematic approach — isolates variables one at a time.",
            "decision_making": "Data-driven; wants to see evidence before changing approach.",
        },
        "interaction_insights": {
            "engagement_triggers": "Gets energized when shown working code examples",
            "friction_points": "Frustrated by vague or overly theoretical explanations",
            "optimal_approach": "Lead with a concrete example, then explain the why",
        },
        "future_guidance": {
            "recommended_approaches": [
                "Start with a minimal reproducible example",
                "Include type annotations in code snippets",
            ],
            "avoid_approaches": ["Don't provide long theoretical preambles"],
            "adaptation_note": "This user values efficiency — get to the point quickly.",
        },
    }


def _minimal_valid_output() -> dict:
    """Return minimum viable valid output (required keys only, N/A everywhere)."""
    return {
        "context": {
            "available_data": "Partial transcript only",
            "user_intent": "Unknown",
            "analysis_limitation": "Only one side of conversation available",
        },
        "behavioral_profile": {
            "communication": "N/A",
            "learning": "N/A",
            "problem_solving": "N/A",
            "decision_making": "N/A",
        },
        "future_guidance": {
            "recommended_approaches": ["N/A"],
            "avoid_approaches": ["N/A"],
            "adaptation_note": "N/A",
        },
    }


# ── Happy path tests ────────────────────────────────────────────────────────


class TestValidOutputs:
    """Tests that valid outputs pass evaluation."""

    def test_valid_complete_output_passes(self):
        """Full valid output with all fields → passes with no failures."""
        raw = json.dumps(_valid_episodic_output())
        result = validate_episodic_output(raw)

        assert result.passed is True
        assert result.hard_failures == []

    def test_valid_minimal_with_na_passes(self):
        """Minimal output with N/A for optional content → passes."""
        raw = json.dumps(_minimal_valid_output())
        result = validate_episodic_output(raw)

        assert result.passed is True
        assert result.hard_failures == []

    def test_valid_output_with_extra_keys_passes(self):
        """Extra keys beyond the schema → ignored, still passes."""
        data = _valid_episodic_output()
        data["extra_field"] = "bonus data"
        raw = json.dumps(data)
        result = validate_episodic_output(raw)

        assert result.passed is True

    def test_valid_output_with_markdown_wrapper(self):
        """JSON wrapped in markdown code block → passes (clean_and_parse_json handles it)."""
        inner = json.dumps(_valid_episodic_output())
        raw = f"```json\n{inner}\n```"
        result = validate_episodic_output(raw)

        assert result.passed is True


# ── Hard failure tests ───────────────────────────────────────────────────────


class TestHardFailures:
    """Tests that invalid outputs fail with hard failures."""

    def test_empty_string_fails(self):
        """Empty string → hard failure."""
        result = validate_episodic_output("")
        assert result.passed is False
        assert any("empty" in f.lower() for f in result.hard_failures)

    def test_whitespace_only_fails(self):
        """Whitespace-only string → hard failure."""
        result = validate_episodic_output("   \n\t  ")
        assert result.passed is False
        assert any("empty" in f.lower() for f in result.hard_failures)

    def test_invalid_json_fails(self):
        """Broken JSON → hard failure."""
        result = validate_episodic_output("{broken json!!!")
        assert result.passed is False
        assert any("json" in f.lower() for f in result.hard_failures)

    def test_list_instead_of_dict_fails(self):
        """Top-level array → hard failure."""
        result = validate_episodic_output('[{"context": {}}]')
        assert result.passed is False
        assert any("dict" in f.lower() for f in result.hard_failures)

    def test_plain_text_fails(self):
        """Plain text (no JSON) → hard failure."""
        result = validate_episodic_output(
            "Here is the behavioral analysis of the user."
        )
        assert result.passed is False

    def test_missing_context_key_fails(self):
        """Missing 'context' top-level key → hard failure."""
        data = _valid_episodic_output()
        del data["context"]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("context" in f for f in result.hard_failures)

    def test_missing_behavioral_profile_key_fails(self):
        """Missing 'behavioral_profile' → hard failure."""
        data = _valid_episodic_output()
        del data["behavioral_profile"]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("behavioral_profile" in f for f in result.hard_failures)

    def test_missing_future_guidance_key_fails(self):
        """Missing 'future_guidance' → hard failure."""
        data = _valid_episodic_output()
        del data["future_guidance"]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("future_guidance" in f for f in result.hard_failures)

    def test_missing_all_required_keys_fails(self):
        """Missing all 3 required top-level keys → hard failure."""
        data = {"what_worked": {"strategies": [], "pattern": "N/A"}}
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert len(result.hard_failures) >= 1

    def test_context_wrong_type_fails(self):
        """'context' is a string instead of dict → hard failure."""
        data = _valid_episodic_output()
        data["context"] = "just a string"
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("context" in f and "dict" in f for f in result.hard_failures)

    def test_behavioral_profile_wrong_type_fails(self):
        """'behavioral_profile' is a list → hard failure."""
        data = _valid_episodic_output()
        data["behavioral_profile"] = ["item1", "item2"]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any(
            "behavioral_profile" in f and "dict" in f
            for f in result.hard_failures
        )

    def test_missing_context_sub_keys_fails(self):
        """'context' present but missing required sub-keys → hard failure."""
        data = _valid_episodic_output()
        data["context"] = {"analysis_limitation": "N/A"}  # missing available_data, user_intent
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("available_data" in f or "user_intent" in f for f in result.hard_failures)

    def test_missing_behavioral_profile_sub_keys_fails(self):
        """'behavioral_profile' present but missing 'communication' and 'learning' → hard failure."""
        data = _valid_episodic_output()
        data["behavioral_profile"] = {"problem_solving": "Systematic"}
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("communication" in f or "learning" in f for f in result.hard_failures)

    def test_missing_recommended_approaches_sub_key_fails(self):
        """'future_guidance' missing 'recommended_approaches' → hard failure."""
        data = _valid_episodic_output()
        data["future_guidance"] = {
            "avoid_approaches": ["Don't"],
            "adaptation_note": "Note",
        }
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("recommended_approaches" in f for f in result.hard_failures)

    def test_recommended_approaches_wrong_type_fails(self):
        """'recommended_approaches' is a string instead of list → hard failure."""
        data = _valid_episodic_output()
        data["future_guidance"]["recommended_approaches"] = "just a string"
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("recommended_approaches" in f and "list" in f for f in result.hard_failures)

    def test_what_worked_strategies_wrong_type_fails(self):
        """'what_worked.strategies' is a string instead of list → hard failure."""
        data = _valid_episodic_output()
        data["what_worked"]["strategies"] = "not a list"
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is False
        assert any("strategies" in f and "list" in f for f in result.hard_failures)


# ── Soft failure tests ───────────────────────────────────────────────────────


class TestSoftFailures:
    """Tests for soft failures that pass but log warnings."""

    def test_missing_optional_keys_soft_warning(self):
        """Missing 'what_worked', 'what_failed', 'interaction_insights' → soft warnings."""
        data = _minimal_valid_output()
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is True
        # Should have soft warnings about missing optional keys
        optional_mentioned = [
            f for f in result.soft_failures if "Optional key" in f
        ]
        assert len(optional_mentioned) >= 1

    def test_empty_string_field_soft_warning(self):
        """Empty string in behavioral_profile field → soft warning about N/A."""
        data = _valid_episodic_output()
        data["behavioral_profile"]["communication"] = ""  # should be "N/A"
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is True
        assert any("N/A" in f for f in result.soft_failures)

    def test_too_many_strategies_soft_warning(self):
        """what_worked.strategies > 3 items → soft warning."""
        data = _valid_episodic_output()
        data["what_worked"]["strategies"] = [
            "Strategy 1",
            "Strategy 2",
            "Strategy 3",
            "Strategy 4",  # exceeds max 3
        ]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is True
        assert any(
            "what_worked.strategies" in f and "max 3" in f
            for f in result.soft_failures
        )

    def test_too_many_what_failed_strategies_soft_warning(self):
        """what_failed.strategies > 2 items → soft warning."""
        data = _valid_episodic_output()
        data["what_failed"]["strategies"] = [
            "Failure 1",
            "Failure 2",
            "Failure 3",  # exceeds max 2
        ]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is True
        assert any(
            "what_failed.strategies" in f and "max 2" in f
            for f in result.soft_failures
        )

    def test_too_many_recommended_approaches_soft_warning(self):
        """recommended_approaches > 3 items → soft warning."""
        data = _valid_episodic_output()
        data["future_guidance"]["recommended_approaches"] = [
            "Approach 1",
            "Approach 2",
            "Approach 3",
            "Approach 4",  # exceeds max 3
        ]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is True
        assert any(
            "recommended_approaches" in f and "max 3" in f
            for f in result.soft_failures
        )

    def test_too_many_avoid_approaches_soft_warning(self):
        """avoid_approaches > 2 items → soft warning."""
        data = _valid_episodic_output()
        data["future_guidance"]["avoid_approaches"] = [
            "Avoid 1",
            "Avoid 2",
            "Avoid 3",  # exceeds max 2
        ]
        result = validate_episodic_output(json.dumps(data))

        assert result.passed is True
        assert any(
            "avoid_approaches" in f and "max 2" in f for f in result.soft_failures
        )


# ── EvalResult dataclass tests ──────────────────────────────────────────────


class TestEvalResult:
    """Tests for the EvalResult dataclass."""

    def test_eval_result_defaults(self):
        """EvalResult with defaults has empty lists."""
        result = EvalResult(passed=True)
        assert result.passed is True
        assert result.hard_failures == []
        assert result.soft_failures == []

    def test_eval_result_with_failures(self):
        """EvalResult stores failure lists correctly."""
        result = EvalResult(
            passed=False,
            hard_failures=["Missing key"],
            soft_failures=["Too many items"],
        )
        assert result.passed is False
        assert len(result.hard_failures) == 1
        assert len(result.soft_failures) == 1


# ── Integration with memory_manager retry logic ─────────────────────────────


class TestMemoryManagerEvalGateIntegration:
    """Tests that create_episodic_memory retries on eval failure."""

    @pytest.mark.asyncio
    async def test_passes_on_first_attempt(self, monkeypatch, mock_llm_connection):
        """Valid output on first try → returns immediately, 1 LLM call."""
        from omnimemory.memory_management.memory_manager import MemoryManager

        self._stub_pool(monkeypatch)
        manager = MemoryManager(mock_llm_connection)

        valid_json = json.dumps(_valid_episodic_output())
        from unittest.mock import AsyncMock, Mock

        mock_llm_connection.llm_call = AsyncMock(
            return_value=Mock(choices=[Mock(message=Mock(content=valid_json))])
        )

        result = await manager.create_episodic_memory("hello", mock_llm_connection)

        assert result == valid_json
        assert mock_llm_connection.llm_call.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_hard_failure_then_succeeds(
        self, monkeypatch, mock_llm_connection
    ):
        """Invalid output first, valid second → retries once, returns valid."""
        from omnimemory.memory_management.memory_manager import MemoryManager

        self._stub_pool(monkeypatch)
        manager = MemoryManager(mock_llm_connection)

        invalid_json = '{"only": "partial"}'  # missing required keys
        valid_json = json.dumps(_valid_episodic_output())
        from unittest.mock import AsyncMock, Mock

        mock_llm_connection.llm_call = AsyncMock(
            side_effect=[
                Mock(choices=[Mock(message=Mock(content=invalid_json))]),
                Mock(choices=[Mock(message=Mock(content=valid_json))]),
            ]
        )

        result = await manager.create_episodic_memory(
            "hello", mock_llm_connection, max_retries=2
        )

        assert result == valid_json
        assert mock_llm_connection.llm_call.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_after_all_retries_exhausted(
        self, monkeypatch, mock_llm_connection
    ):
        """All attempts return invalid → returns None after max_retries."""
        from omnimemory.memory_management.memory_manager import MemoryManager

        self._stub_pool(monkeypatch)
        manager = MemoryManager(mock_llm_connection)

        invalid_json = '{"broken": true}'
        from unittest.mock import AsyncMock, Mock

        mock_llm_connection.llm_call = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content=invalid_json))]
            )
        )

        result = await manager.create_episodic_memory(
            "hello", mock_llm_connection, max_retries=1
        )

        assert result is None
        # 1 initial + 1 retry = 2 calls
        assert mock_llm_connection.llm_call.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_on_no_choices(
        self, monkeypatch, mock_llm_connection
    ):
        """LLM returns no choices → returns None immediately (no retry)."""
        from omnimemory.memory_management.memory_manager import MemoryManager

        self._stub_pool(monkeypatch)
        manager = MemoryManager(mock_llm_connection)

        from unittest.mock import AsyncMock, Mock

        mock_llm_connection.llm_call = AsyncMock(
            return_value=Mock(choices=[])
        )

        result = await manager.create_episodic_memory("hello", mock_llm_connection)

        assert result is None
        assert mock_llm_connection.llm_call.await_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(
        self, monkeypatch, mock_llm_connection
    ):
        """LLM call throws exception → returns None."""
        from omnimemory.memory_management.memory_manager import MemoryManager

        self._stub_pool(monkeypatch)
        manager = MemoryManager(mock_llm_connection)

        from unittest.mock import AsyncMock

        mock_llm_connection.llm_call = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )

        result = await manager.create_episodic_memory("hello", mock_llm_connection)

        assert result is None

    @pytest.mark.asyncio
    async def test_zero_retries_only_one_attempt(
        self, monkeypatch, mock_llm_connection
    ):
        """max_retries=0 → exactly 1 attempt, no retries."""
        from omnimemory.memory_management.memory_manager import MemoryManager

        self._stub_pool(monkeypatch)
        manager = MemoryManager(mock_llm_connection)

        invalid_json = '{"broken": true}'
        from unittest.mock import AsyncMock, Mock

        mock_llm_connection.llm_call = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content=invalid_json))]
            )
        )

        result = await manager.create_episodic_memory(
            "hello", mock_llm_connection, max_retries=0
        )

        assert result is None
        assert mock_llm_connection.llm_call.await_count == 1

    @staticmethod
    def _stub_pool(monkeypatch):
        """Patch the handler pool + agents so MemoryManager can be constructed."""
        from unittest.mock import AsyncMock, Mock

        fake_ctx = AsyncMock()
        fake_ctx.__aenter__ = AsyncMock(return_value=Mock(enabled=True))
        fake_ctx.__aexit__ = AsyncMock(return_value=False)

        fake_pool = Mock()
        fake_pool.get_handler.return_value = fake_ctx

        monkeypatch.setattr(
            "omnimemory.memory_management.memory_manager.VectorDBHandlerPool.get_instance",
            lambda max_connections=None: fake_pool,
        )
        monkeypatch.setattr(
            "omnimemory.memory_management.memory_manager.ConflictResolutionAgent",
            lambda llm: Mock(),
        )
        monkeypatch.setattr(
            "omnimemory.memory_management.memory_manager.SynthesisAgent",
            lambda llm: Mock(),
        )
