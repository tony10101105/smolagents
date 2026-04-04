# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for prompt caching support (TokenUsage cache fields, cache_control
injection, cache usage extraction in LiteLLMModel and OpenAIModel)."""

from unittest.mock import MagicMock, patch

import pytest

from smolagents import CodeAgent
from smolagents.memory import ActionStep, PlanningStep
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    LiteLLMModel,
    MessageRole,
    Model,
    TokenUsage,
    _extract_cache_usage_from_openai,
    agglomerate_stream_deltas,
)
from smolagents.monitoring import Monitor, AgentLogger, LogLevel


# ---------------------------------------------------------------------------
# TokenUsage tests
# ---------------------------------------------------------------------------

class TestTokenUsageCacheFields:
    def test_default_cache_fields_are_zero(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.cache_read_input_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.total_tokens == 150

    def test_explicit_cache_fields(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=20,
        )
        assert usage.cache_read_input_tokens == 30
        assert usage.cache_creation_input_tokens == 20
        # total_tokens is input + output (cache fields are informational)
        assert usage.total_tokens == 150

    def test_dict_includes_cache_fields(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=20,
        )
        d = usage.dict()
        assert d["cache_read_input_tokens"] == 30
        assert d["cache_creation_input_tokens"] == 20
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["total_tokens"] == 150

    def test_dict_includes_cache_fields_when_zero(self):
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        d = usage.dict()
        assert "cache_read_input_tokens" in d
        assert "cache_creation_input_tokens" in d
        assert d["cache_read_input_tokens"] == 0
        assert d["cache_creation_input_tokens"] == 0


# ---------------------------------------------------------------------------
# Monitor tests
# ---------------------------------------------------------------------------

class TestMonitorCacheTracking:
    def _make_monitor(self):
        logger = AgentLogger(level=LogLevel.OFF)
        model = MagicMock()
        return Monitor(tracked_model=model, logger=logger)

    def test_initial_cache_counts_are_zero(self):
        monitor = self._make_monitor()
        assert monitor.total_cache_read_input_token_count == 0
        assert monitor.total_cache_creation_input_token_count == 0

    def test_update_metrics_accumulates_cache_tokens(self):
        monitor = self._make_monitor()

        step1 = MagicMock()
        step1.timing.duration = 1.0
        step1.token_usage = TokenUsage(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=30, cache_creation_input_tokens=20,
        )
        monitor.update_metrics(step1)

        assert monitor.total_cache_read_input_token_count == 30
        assert monitor.total_cache_creation_input_token_count == 20

        step2 = MagicMock()
        step2.timing.duration = 2.0
        step2.token_usage = TokenUsage(
            input_tokens=200, output_tokens=80,
            cache_read_input_tokens=50, cache_creation_input_tokens=0,
        )
        monitor.update_metrics(step2)

        assert monitor.total_cache_read_input_token_count == 80
        assert monitor.total_cache_creation_input_token_count == 20

    def test_get_total_token_counts_includes_cache(self):
        monitor = self._make_monitor()

        step = MagicMock()
        step.timing.duration = 1.0
        step.token_usage = TokenUsage(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=30, cache_creation_input_tokens=20,
        )
        monitor.update_metrics(step)

        total = monitor.get_total_token_counts()
        assert total.cache_read_input_tokens == 30
        assert total.cache_creation_input_tokens == 20

    def test_reset_clears_cache_counts(self):
        monitor = self._make_monitor()

        step = MagicMock()
        step.timing.duration = 1.0
        step.token_usage = TokenUsage(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=30, cache_creation_input_tokens=20,
        )
        monitor.update_metrics(step)
        monitor.reset()

        assert monitor.total_cache_read_input_token_count == 0
        assert monitor.total_cache_creation_input_token_count == 0

    def test_update_metrics_no_cache_fields(self):
        """Steps without cache info should still work (default 0)."""
        monitor = self._make_monitor()

        step = MagicMock()
        step.timing.duration = 1.0
        step.token_usage = TokenUsage(input_tokens=100, output_tokens=50)
        monitor.update_metrics(step)

        assert monitor.total_cache_read_input_token_count == 0
        assert monitor.total_cache_creation_input_token_count == 0
        assert monitor.total_input_token_count == 100
        assert monitor.total_output_token_count == 50


# ---------------------------------------------------------------------------
# LiteLLMModel._add_cache_control tests
# ---------------------------------------------------------------------------

class TestAddCacheControl:
    def test_marks_system_message_string_content(self):
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = LiteLLMModel._add_cache_control(messages)
        # System message should be converted to list with cache_control
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"] == {"type": "ephemeral"}
        assert sys_content[0]["text"] == "You are an assistant."

    def test_marks_system_message_list_content(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an assistant."}]},
            {"role": "user", "content": "Hello"},
        ]
        result = LiteLLMModel._add_cache_control(messages)
        assert result[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_marks_last_user_message(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second user message"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Current query"},
        ]
        result = LiteLLMModel._add_cache_control(messages)
        # Last user message ("Current query") should be marked
        target = result[5]
        assert isinstance(target["content"], list)
        assert target["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert target["content"][0]["text"] == "Current query"

        # Earlier user messages should NOT be marked
        first_user = result[1]
        assert isinstance(first_user["content"], str)

    def test_single_user_message_gets_cached(self):
        """With only one user message, it should be cache-marked (it's the last)."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Only user message"},
        ]
        result = LiteLLMModel._add_cache_control(messages)
        # System should be marked
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Single user message should be marked (it is the last user message)
        user_msg = result[1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_message(self):
        """Should not fail when there is no system message."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        result = LiteLLMModel._add_cache_control(messages)
        # Last user message ("Second") should be marked
        assert isinstance(result[2]["content"], list)
        assert result[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # First user message should NOT be marked
        assert isinstance(result[0]["content"], str)


# ---------------------------------------------------------------------------
# LiteLLMModel._extract_cache_usage tests
# ---------------------------------------------------------------------------

class TestExtractCacheUsage:
    def test_anthropic_style(self):
        usage = MagicMock()
        usage.cache_read_input_tokens = 500
        usage.cache_creation_input_tokens = 200
        result = LiteLLMModel._extract_cache_usage(usage)
        assert result["cache_read_input_tokens"] == 500
        assert result["cache_creation_input_tokens"] == 200

    def test_openai_style(self):
        usage = MagicMock()
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0
        details = MagicMock()
        details.cached_tokens = 300
        usage.prompt_tokens_details = details
        result = LiteLLMModel._extract_cache_usage(usage)
        assert result["cache_read_input_tokens"] == 300
        assert result["cache_creation_input_tokens"] == 0

    def test_no_cache_info(self):
        usage = MagicMock(spec=[])  # No attributes at all
        result = LiteLLMModel._extract_cache_usage(usage)
        assert result["cache_read_input_tokens"] == 0
        assert result["cache_creation_input_tokens"] == 0

    def test_none_values_treated_as_zero(self):
        usage = MagicMock()
        usage.cache_read_input_tokens = None
        usage.cache_creation_input_tokens = None
        usage.prompt_tokens_details = None
        result = LiteLLMModel._extract_cache_usage(usage)
        assert result["cache_read_input_tokens"] == 0
        assert result["cache_creation_input_tokens"] == 0


# ---------------------------------------------------------------------------
# _extract_cache_usage_from_openai tests
# ---------------------------------------------------------------------------

class TestExtractCacheUsageFromOpenAI:
    def test_with_cached_tokens(self):
        usage = MagicMock()
        details = MagicMock()
        details.cached_tokens = 1000
        usage.prompt_tokens_details = details
        result = _extract_cache_usage_from_openai(usage)
        assert result["cache_read_input_tokens"] == 1000
        assert result["cache_creation_input_tokens"] == 0

    def test_without_details(self):
        usage = MagicMock(spec=[])
        result = _extract_cache_usage_from_openai(usage)
        assert result["cache_read_input_tokens"] == 0

    def test_none_cached_tokens(self):
        usage = MagicMock()
        details = MagicMock()
        details.cached_tokens = None
        usage.prompt_tokens_details = details
        result = _extract_cache_usage_from_openai(usage)
        assert result["cache_read_input_tokens"] == 0


# ---------------------------------------------------------------------------
# agglomerate_stream_deltas with cache fields
# ---------------------------------------------------------------------------

class TestAgglomerateStreamDeltasCacheFields:
    def test_accumulates_cache_fields(self):
        deltas = [
            ChatMessageStreamDelta(
                content="Hi",
                token_usage=None,
            ),
            ChatMessageStreamDelta(
                content="",
                token_usage=TokenUsage(
                    input_tokens=100, output_tokens=20,
                    cache_read_input_tokens=30, cache_creation_input_tokens=10,
                ),
            ),
        ]
        result = agglomerate_stream_deltas(deltas)
        assert result.token_usage.cache_read_input_tokens == 30
        assert result.token_usage.cache_creation_input_tokens == 10
        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 20

    def test_default_zero_when_no_cache(self):
        deltas = [
            ChatMessageStreamDelta(
                content="Hello",
                token_usage=TokenUsage(input_tokens=50, output_tokens=10),
            ),
        ]
        result = agglomerate_stream_deltas(deltas)
        assert result.token_usage.cache_read_input_tokens == 0
        assert result.token_usage.cache_creation_input_tokens == 0


# ---------------------------------------------------------------------------
# LiteLLMModel.generate with prompt_cache
# ---------------------------------------------------------------------------

class TestLiteLLMModelPromptCache:
    def test_generate_with_prompt_cache_adds_cache_control(self):
        """When prompt_cache=True, cache_control should be injected into messages."""
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.cache_read_input_tokens = 50
        mock_response.usage.cache_creation_input_tokens = 30
        mock_response.usage.prompt_tokens_details = None
        mock_litellm.completion.return_value = mock_response

        with patch("smolagents.models.LiteLLMModel.create_client", return_value=mock_litellm):
            model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-20250514", prompt_cache=True)
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": "System prompt"}]),
                ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}]),
            ]
            result = model.generate(messages)

        # Verify cache_control was injected in the messages passed to litellm
        call_kwargs = mock_litellm.completion.call_args
        sent_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        sys_msg = sent_messages[0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}

        # Verify cache usage is extracted
        assert result.token_usage.cache_read_input_tokens == 50
        assert result.token_usage.cache_creation_input_tokens == 30

    def test_generate_without_prompt_cache_no_cache_control(self):
        """When prompt_cache=False (default), no cache_control should be injected."""
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.prompt_tokens_details = None
        mock_litellm.completion.return_value = mock_response

        with patch("smolagents.models.LiteLLMModel.create_client", return_value=mock_litellm):
            model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-20250514", prompt_cache=False)
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": "System prompt"}]),
                ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}]),
            ]
            result = model.generate(messages)

        call_kwargs = mock_litellm.completion.call_args
        sent_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        sys_msg = sent_messages[0]
        # System message should NOT have cache_control (content is a list from get_clean_message_list)
        if isinstance(sys_msg["content"], list):
            for block in sys_msg["content"]:
                assert "cache_control" not in block

    def test_generate_extracts_cache_usage_even_without_prompt_cache(self):
        """Cache usage should always be extracted from response, regardless of prompt_cache flag."""
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        # OpenAI auto-cached tokens come back even without explicit cache_control
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        details = MagicMock()
        details.cached_tokens = 40
        mock_response.usage.prompt_tokens_details = details
        mock_litellm.completion.return_value = mock_response

        with patch("smolagents.models.LiteLLMModel.create_client", return_value=mock_litellm):
            model = LiteLLMModel(model_id="openai/gpt-4o", prompt_cache=False)
            messages = [
                ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}]),
            ]
            result = model.generate(messages)

        assert result.token_usage.cache_read_input_tokens == 40
        assert result.token_usage.cache_creation_input_tokens == 0


# ---------------------------------------------------------------------------
# RunResult aggregation with cache tokens (CodeAgent)
# ---------------------------------------------------------------------------

class TestRunResultCacheAggregation:
    def test_run_result_aggregates_cache_tokens(self):
        """When return_full_result=True, RunResult should include aggregated cache token counts."""

        class FakeLLMModelWithCache(Model):
            def __init__(self):
                super().__init__()
                self._call_count = 0

            def generate(self, prompt, **kwargs):
                self._call_count += 1
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="""<code>\nfinal_answer('answer')\n</code>""",
                    token_usage=TokenUsage(
                        input_tokens=100,
                        output_tokens=20,
                        cache_read_input_tokens=30,
                        cache_creation_input_tokens=10,
                    ),
                )

        agent = CodeAgent(
            tools=[],
            model=FakeLLMModelWithCache(),
            max_steps=1,
        )
        result = agent.run("Fake task", return_full_result=True)

        assert result.token_usage is not None
        assert result.token_usage.cache_read_input_tokens == 30
        assert result.token_usage.cache_creation_input_tokens == 10
        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 20
