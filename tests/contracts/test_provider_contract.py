# tests/contracts/test_provider_contract.py
"""Provider 合约测试：所有 Provider 必须满足的行为规范"""

from agent_framework.llm.base import BaseLLMProvider


class ProviderContractTests:
    """Provider 必须满足的合约，子类化并实现 get_provider()"""

    def get_provider(self) -> BaseLLMProvider:
        raise NotImplementedError

    def test_format_tool_result_returns_dict(self):
        p = self.get_provider()
        result = p.format_tool_result("id_123", "output")
        assert isinstance(result, dict)
        assert "id_123" in str(result)  # 必须包含 tool_call_id

    def test_tool_results_as_message_returns_list(self):
        p = self.get_provider()
        results = [p.format_tool_result("id_1", "a")]
        msgs = p.tool_results_as_message(results)
        assert isinstance(msgs, list)
        assert len(msgs) >= 1
        for m in msgs:
            assert isinstance(m, dict)

    def test_format_tool_result_contains_content(self):
        p = self.get_provider()
        result = p.format_tool_result("id_456", "test_content")
        assert "test_content" in str(result)

    def test_tool_results_as_message_with_multiple_results(self):
        p = self.get_provider()
        results = [
            p.format_tool_result("id_1", "result_1"),
            p.format_tool_result("id_2", "result_2"),
        ]
        msgs = p.tool_results_as_message(results)
        assert isinstance(msgs, list)
        assert len(msgs) >= 1
        # 验证所有结果都被包含（通过字符串检查）
        combined = str(msgs)
        assert "result_1" in combined
        assert "result_2" in combined


class TestAnthropicProviderContract(ProviderContractTests):
    def get_provider(self):
        from agent_framework.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider.__new__(AnthropicProvider)


class TestOpenAIProviderContract(ProviderContractTests):
    def get_provider(self):
        from agent_framework.llm.openai_provider import OpenAIProvider

        return OpenAIProvider.__new__(OpenAIProvider)
