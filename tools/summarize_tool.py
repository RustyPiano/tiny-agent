import textwrap

from tools.registry import register


def summarize(text: str, max_chars: int = 300) -> str:
    cleaned = text.strip()
    if max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    suffix = "..."
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return textwrap.shorten(cleaned, width=max_chars, placeholder=suffix)


def register_summarize_tool() -> None:
    register(
        name="summarize",
        description="压缩长文本为短摘要，便于快速预览。",
        parameters={
            "text": {"type": "string", "description": "待摘要文本"},
            "max_chars": {"type": "integer", "description": "摘要最大字符数，默认 300"},
        },
        required=["text"],
        handler=summarize,
    )
