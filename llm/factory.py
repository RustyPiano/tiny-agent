# llm/factory.py
from llm.base import BaseLLMProvider


def create_provider(cfg: dict) -> BaseLLMProvider:
    """
    cfg 示例：
      {"type": "anthropic", "model": "claude-opus-4-6"}
      {"type": "openai",    "model": "gpt-4o"}
      {"type": "openai",    "model": "qwen2.5:14b",
       "base_url": "http://localhost:11434/v1", "api_key": "ollama"}
    """
    t = cfg.get("type", "anthropic")

    if t == "anthropic":
        from llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=cfg["model"], api_key=cfg.get("api_key"))

    if t == "openai":
        from llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=cfg["model"],
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
        )

    raise ValueError(f"未知 provider 类型: {t!r}，支持: anthropic, openai")
