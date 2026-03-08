"""Tests for configuration."""

import pytest
from agentdbg.config import DebugConfig, get_model_cost, MODEL_COSTS


class TestDebugConfig:
    """Tests for DebugConfig."""

    def test_default_values(self):
        config = DebugConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8765
        assert config.ui_port == 8766
        assert config.auto_pause_on_error is True
        assert config.capture_inputs is True
        assert config.capture_outputs is True

    def test_custom_values(self):
        config = DebugConfig(
            host="0.0.0.0",
            port=9000,
            auto_pause_on_cost=1.0,
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.auto_pause_on_cost == 1.0

    def test_to_dict(self):
        config = DebugConfig()
        d = config.to_dict()

        assert "host" in d
        assert "port" in d
        assert "auto_pause_on_error" in d
        assert d["host"] == "127.0.0.1"


class TestModelCosts:
    """Tests for model cost calculation."""

    def test_known_model_costs(self):
        # Test GPT-4o
        input_cost, output_cost = get_model_cost("gpt-4o", 1000, 500)
        assert input_cost == 0.0025  # $2.50 per 1M input
        assert output_cost == 0.005  # $10 per 1M output

    def test_claude_model_costs(self):
        # Test Claude 3.5 Sonnet
        input_cost, output_cost = get_model_cost("claude-3-5-sonnet", 1000, 500)
        assert input_cost == 0.003  # $3 per 1M input
        assert output_cost == 0.0075  # $15 per 1M output

    def test_unknown_model_defaults_to_gpt4o(self):
        # Unknown model should use GPT-4o pricing
        input_cost, output_cost = get_model_cost("unknown-model-xyz", 1000, 500)
        assert input_cost == 0.0025
        assert output_cost == 0.005

    def test_model_name_matching(self):
        # Test that partial model names match
        input_cost1, _ = get_model_cost("gpt-4o-2024-08-06", 1000, 0)
        input_cost2, _ = get_model_cost("gpt-4o", 1000, 0)
        assert input_cost1 == input_cost2

    def test_zero_tokens(self):
        input_cost, output_cost = get_model_cost("gpt-4o", 0, 0)
        assert input_cost == 0.0
        assert output_cost == 0.0

    def test_large_token_counts(self):
        # 1 million tokens
        input_cost, output_cost = get_model_cost("gpt-4o", 1_000_000, 500_000)
        assert input_cost == 2.5  # $2.50 for 1M input tokens
        assert output_cost == 5.0  # $10 for 1M * 0.5 output tokens
