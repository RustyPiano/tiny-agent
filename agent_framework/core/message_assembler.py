from __future__ import annotations

from agent_framework._config import SYSTEM_PROMPT_DYNAMIC_BOUNDARY


def assemble_messages(
    static_system_prompt: str,
    memory_text: str,
    compacted_history: str,
    last_observation: str,
    current_task: str,
) -> str:
    return "\n\n".join(
        [
            static_system_prompt.strip(),
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
            "## Current MEMORY\n" + (memory_text.strip() or "[none]"),
            "## Compacted History\n"
            + ("[sent as compacted provider message]" if compacted_history.strip() else "[none]"),
            "## Current Task + Last Observation\n"
            + "Current Task:\n"
            + (current_task.strip() or "[none]")
            + "\n\n"
            + "Last Observation:\n"
            + (last_observation.strip() or "[none]"),
        ]
    )


def assemble_compacted_history_message(compacted_history: str) -> dict | None:
    compacted = compacted_history.strip()
    if not compacted:
        return None
    return {
        "role": "assistant",
        "content": "Earlier conversation summary:\n" + compacted,
    }
