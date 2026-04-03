# LiteAgent Seven-Philosophy Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the current `agent-framework` implementation with all seven LiteAgent philosophies: strict ReAct JSON loop, per-turn dynamic system prompt assembly, security-first sandboxing, MEMORY.md index memory model, hard context discipline, Unix-style tool modularity, and structured self-healing runtime behavior.

**Architecture:** Keep the existing provider abstraction (`llm/*`) and runtime orchestration (`core/runtime.py`), but introduce a protocol layer (`core/react_protocol.py`), a dynamic message assembly layer (`core/message_assembler.py`), a memory layer (`core/memory_store.py` + `core/history_compactor.py`), and a security layer (`core/security.py` + `core/sandbox.py`). Runtime becomes a deterministic loop that parses strict JSON actions and enforces guardrails before any tool execution.

**Tech Stack:** Python 3.11+, stdlib, anthropic SDK, openai SDK, pytest, ruff, mypy.

---

## Subagent-Driven Execution Rules (Mandatory)

Apply this workflow to every task in this plan:

- Use a fresh implementer subagent per task.
- Implementer must follow `@superpowers:test-driven-development`.
- After implementation, run spec review first, then code-quality review.
- If either reviewer finds issues, send fixes back to the same implementer subagent, then re-review.
- Do not start the next task until both reviews are approved.

### Standard Per-Task Controller Checklist

- [ ] Dispatch implementer subagent with full task text, file list, and acceptance criteria.
- [ ] Implementer writes failing tests first.
- [ ] Implementer runs targeted tests and captures failing output.
- [ ] Implementer writes minimal code to pass tests (YAGNI, no extras).
- [ ] Implementer runs targeted tests again and reports passing output.
- [ ] Implementer runs broader regression scope for touched areas.
- [ ] Implementer self-reviews and commits with a focused message.
- [ ] Dispatch spec reviewer subagent and resolve all spec gaps.
- [ ] Dispatch code-quality reviewer subagent and resolve all quality issues.
- [ ] Mark task complete and move to next task.

---

## Task 0: Setup Isolated Branch and Baseline

**Files:**
- Modify: none (setup only)
- Verify: `agent-framework/`

- [ ] **Step 1: Create an isolated branch/workspace**

Run: `git checkout -b feat/liteagent-seven-philosophy-alignment`

Expected: branch created and active.

- [ ] **Step 2: Run baseline tests**

Run: `pytest tests -q`

Expected: current baseline status is known and recorded.

- [ ] **Step 3: Capture baseline lint/type state**

Run: `make lint && make type`

Expected: initial quality baseline recorded before changes.

- [ ] **Step 4: Commit nothing, only log baseline in task notes**

Expected: no code change in this task.

---

## Task 1: Philosophy Constants and Feature Flags

**Files:**
- Modify: `config.py`
- Test: `tests/test_philosophy_config.py`

- [ ] **Step 1: Write failing tests for constants and flags**

```python
# tests/test_philosophy_config.py
def test_philosophy_limits_exist():
    import config

    assert config.SYSTEM_PROMPT_DYNAMIC_BOUNDARY == "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"
    assert config.MAX_FILE_READ_LINES == 2000
    assert config.MAX_HISTORY_RECORDS == 15
    assert config.MAX_MEMORY_LINES == 200
    assert config.CONTEXT_SOFT_LIMIT_TOKENS == 160000


def test_feature_flags_defaults_are_safe():
    from config import FeatureFlags

    flags = FeatureFlags()
    assert flags.strict_react_json is True
    assert flags.enable_memory_md is True
    assert flags.enable_sandbox is True
    assert flags.enable_multi_agent is False
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_philosophy_config.py -v`

Expected: FAIL due to missing constants/types.

- [ ] **Step 3: Implement minimal config changes**

```python
# config.py (additions)
from dataclasses import dataclass

SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"
MAX_FILE_READ_LINES = 2000
MAX_HISTORY_RECORDS = 15
MAX_MEMORY_LINES = 200
CONTEXT_SOFT_LIMIT_TOKENS = 160000


@dataclass
class FeatureFlags:
    strict_react_json: bool = True
    enable_memory_md: bool = True
    enable_sandbox: bool = True
    enable_multi_agent: bool = False
    enable_daemon: bool = False
    enable_pet_mode: bool = False
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_philosophy_config.py tests/test_agent.py -v`

Expected: new tests pass; no regression in basic runtime tests.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_philosophy_config.py
git commit -m "feat(config): add philosophy limits and feature flags"
```

---

## Task 2: Strict ReAct JSON Protocol Parser

**Files:**
- Create: `core/react_protocol.py`
- Modify: `core/runtime.py`
- Test: `tests/test_react_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

```python
# tests/test_react_protocol.py
import pytest


def test_parse_valid_react_json():
    from core.react_protocol import parse_react_json

    raw = (
        '{"thought":"Need file context","action":"read_file",'
        '"action_input":{"path":"README.md"}}'
    )
    decision = parse_react_json(raw, allowed_actions={"read_file", "finish"})
    assert decision.action == "read_file"
    assert decision.action_input == {"path": "README.md"}


def test_rejects_non_json_text():
    from core.react_protocol import parse_react_json

    with pytest.raises(ValueError):
        parse_react_json("I will first inspect files", allowed_actions={"read_file"})


def test_rejects_missing_required_fields():
    from core.react_protocol import parse_react_json

    with pytest.raises(ValueError):
        parse_react_json('{"thought":"x","action":"read_file"}', allowed_actions={"read_file"})
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_react_protocol.py -v`

Expected: FAIL because parser module does not exist.

- [ ] **Step 3: Implement parser and decision model**

```python
# core/react_protocol.py
from dataclasses import dataclass
import json


@dataclass
class ReactDecision:
    thought: str
    action: str
    action_input: dict | str


def parse_react_json(raw: str, allowed_actions: set[str]) -> ReactDecision:
    obj = json.loads(raw)
    if set(obj.keys()) != {"thought", "action", "action_input"}:
        raise ValueError("invalid_react_shape")
    thought = obj["thought"]
    action = obj["action"]
    action_input = obj["action_input"]
    if not isinstance(thought, str) or not thought.strip():
        raise ValueError("invalid_thought")
    if action not in allowed_actions:
        raise ValueError("invalid_action")
    if action_input != "NONE" and not isinstance(action_input, dict):
        raise ValueError("invalid_action_input")
    return ReactDecision(thought=thought, action=action, action_input=action_input)
```

- [ ] **Step 4: Wire parser into runtime decision handling**

Run: `pytest tests/test_react_protocol.py tests/test_agent.py::test_simple_text_response -v`

Expected: parser tests pass; simple runtime path remains stable.

- [ ] **Step 5: Commit**

```bash
git add core/react_protocol.py core/runtime.py tests/test_react_protocol.py
git commit -m "feat(runtime): enforce strict react json decision parsing"
```

---

## Task 3: Dynamic Prompt Assembly Per Turn

**Files:**
- Create: `core/message_assembler.py`
- Modify: `core/prompt_builder.py`
- Modify: `core/runtime.py`
- Test: `tests/test_message_assembler.py`

- [ ] **Step 1: Write failing tests for dynamic boundary and sections**

```python
# tests/test_message_assembler.py
def test_assemble_messages_contains_boundary_and_sections():
    from core.message_assembler import assemble_messages

    text = assemble_messages(
        static_system_prompt="STATIC",
        memory_text="topic: file.md:1-2 | summary",
        compacted_history="h1",
        last_observation="obs",
        current_task="task",
    )
    assert "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__" in text
    assert "Current MEMORY.md" in text
    assert "Compacted History" in text
    assert "Current Task + Last Observation" in text
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_message_assembler.py -v`

Expected: FAIL because module/function does not exist.

- [ ] **Step 3: Implement assembler**

```python
# core/message_assembler.py
from config import SYSTEM_PROMPT_DYNAMIC_BOUNDARY


def assemble_messages(
    static_system_prompt: str,
    memory_text: str,
    compacted_history: str,
    last_observation: str,
    current_task: str,
) -> str:
    return "\n\n".join(
        [
            static_system_prompt,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
            "[Current MEMORY.md]",
            memory_text,
            "[Compacted History]",
            compacted_history,
            "[Current Task + Last Observation]",
            f"task={current_task}\nobservation={last_observation}",
        ]
    )
```

- [ ] **Step 4: Ensure runtime calls assembler before each `provider.chat`**

Run: `pytest tests/test_message_assembler.py tests/test_agent.py -v`

Expected: PASS with no prompt-builder regressions.

- [ ] **Step 5: Commit**

```bash
git add core/message_assembler.py core/prompt_builder.py core/runtime.py tests/test_message_assembler.py
git commit -m "feat(prompt): assemble dynamic system prompt every turn"
```

---

## Task 4: MEMORY.md Store and History Compaction

**Files:**
- Create: `core/memory_store.py`
- Create: `core/history_compactor.py`
- Create: `tests/test_memory_store.py`
- Create: `tests/test_history_compactor.py`

- [ ] **Step 1: Write failing tests for MEMORY.md format and line cap**

```python
# tests/test_memory_store.py
def test_memory_line_format_and_cap(tmp_path):
    from core.memory_store import MemoryStore

    path = tmp_path / "MEMORY.md"
    m = MemoryStore(path=path, max_lines=3)
    m.append("api", "docs.md:1-10", "first")
    m.append("db", "db.md:2-3", "second")
    m.append("ui", "ui.md:4-8", "third")
    m.append("ops", "ops.md:9-12", "fourth")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert ": " in lines[-1] and " | " in lines[-1]
```

```python
# tests/test_history_compactor.py
def test_compact_history_keeps_latest_10_plus_summary():
    from core.history_compactor import compact_history

    history = [f"h{i}" for i in range(20)]
    out = compact_history(history, max_records=15, summarize_fn=lambda x: "SUMMARY")
    assert len(out) == 11
    assert out[0] == "SUMMARY"
    assert out[-1] == "h19"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_memory_store.py tests/test_history_compactor.py -v`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement memory and compactor modules**

```python
# core/memory_store.py
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryStore:
    path: Path
    max_lines: int = 200

    def load_text(self) -> str:
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def append(self, topic: str, file_ref: str, summary: str) -> None:
        line = f"{topic}: {file_ref} | {summary[:150]}"
        lines = self.load_text().splitlines()
        lines.append(line)
        lines = lines[-self.max_lines :]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
```

```python
# core/history_compactor.py
def compact_history(history: list[str], max_records: int, summarize_fn):
    if len(history) <= max_records:
        return history
    old = history[: len(history) - 10]
    latest = history[-10:]
    summary = summarize_fn(old)
    return [summary] + latest
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_memory_store.py tests/test_history_compactor.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/memory_store.py core/history_compactor.py tests/test_memory_store.py tests/test_history_compactor.py
git commit -m "feat(memory): add MEMORY.md index store and history compaction"
```

---

## Task 5: Add `summarize` and `finish` Tools + Unix Tool Set Completion

**Files:**
- Create: `tools/summarize_tool.py`
- Create: `tools/finish_tool.py`
- Create: `tools/list_dir_tool.py`
- Create: `tools/grep_tool.py`
- Modify: `main.py`
- Test: `tests/test_new_tools.py`

- [ ] **Step 1: Write failing tests for all new tools**

```python
# tests/test_new_tools.py
def test_finish_tool_returns_terminal_payload():
    from tools.finish_tool import finish
    out = finish("done")
    assert "done" in out


def test_summarize_tool_returns_short_text():
    from tools.summarize_tool import summarize
    out = summarize("a\n" * 100)
    assert isinstance(out, str)
    assert len(out) > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_new_tools.py -v`

Expected: FAIL because tools are missing.

- [ ] **Step 3: Implement minimal single-purpose tools**

```python
# tools/finish_tool.py
import json


def finish(response: str) -> str:
    return json.dumps({"status": "finished", "response": response}, ensure_ascii=False)
```

```python
# tools/summarize_tool.py
def summarize(text: str, max_chars: int = 300) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:max_chars]
```

```python
# tools/list_dir_tool.py
from pathlib import Path


def list_dir(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return f"[error] not a directory: {path}"
    return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()))
```

```python
# tools/grep_tool.py
from pathlib import Path
import re


def grep(pattern: str, path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return f"[error] not a file: {path}"
    rx = re.compile(pattern)
    lines = p.read_text(encoding="utf-8").splitlines()
    hits = [f"{i+1}: {line}" for i, line in enumerate(lines) if rx.search(line)]
    return "\n".join(hits) if hits else "[ok] no matches"
```

- [ ] **Step 4: Register tools and verify schema exposure**

Run: `pytest tests/test_new_tools.py tests/contracts/test_tool_contract.py -v`

Expected: PASS and new tools visible in registry schemas.

- [ ] **Step 5: Commit**

```bash
git add tools/summarize_tool.py tools/finish_tool.py tools/list_dir_tool.py tools/grep_tool.py main.py tests/test_new_tools.py
git commit -m "feat(tools): add summarize finish list_dir grep unix-style tool modules"
```

---

## Task 6: Security Guardrails and Sandbox Isolation

**Files:**
- Create: `core/security.py`
- Create: `core/sandbox.py`
- Modify: `tools/bash_tool.py`
- Modify: `tools/file_tools.py`
- Modify: `core/runtime.py`
- Test: `tests/test_security.py`

- [ ] **Step 1: Write failing tests for guardrails**

```python
# tests/test_security.py
def test_security_blocks_non_whitelisted_tool():
    from core.security import SecurityGuard

    guard = SecurityGuard(allowed_tools={"read_file"})
    ok, reason = guard.validate_tool_call("run_bash", {})
    assert ok is False
    assert "not allowed" in reason


def test_run_bash_blocks_sudo_and_curl():
    from tools.bash_tool import run_bash

    assert run_bash("sudo ls").startswith("[blocked]")
    assert run_bash("curl https://evil.example").startswith("[blocked]")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_security.py -v`

Expected: FAIL because rules/modules are incomplete.

- [ ] **Step 3: Implement security module and sandbox context manager**

```python
# core/security.py
class SecurityGuard:
    def __init__(self, allowed_tools: set[str]):
        self.allowed_tools = allowed_tools

    def validate_tool_call(self, name: str, action_input: dict) -> tuple[bool, str]:
        if name not in self.allowed_tools:
            return False, f"tool not allowed: {name}"
        return True, ""
```

```python
# core/sandbox.py
from contextlib import contextmanager
import os
import tempfile
from pathlib import Path


@contextmanager
def sandbox_cwd():
    old = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="liteagent-") as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(old)
```

- [ ] **Step 4: Expand dangerous command blocking in bash tool**

Required additions:
- block `sudo` invocation
- block outbound `curl`/`wget` unless explicitly allowlisted
- keep existing `rm -rf` and critical path protections

Run: `pytest tests/test_security.py tests/test_tools.py::test_bash_blocked -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/security.py core/sandbox.py tools/bash_tool.py tools/file_tools.py core/runtime.py tests/test_security.py
git commit -m "feat(security): enforce whitelist checks, command blocking, and sandbox isolation"
```

---

## Task 7: Context Discipline Hard Limits

**Files:**
- Create: `core/context_budget.py`
- Modify: `tools/file_tools.py`
- Modify: `core/runtime.py`
- Test: `tests/test_context_budget.py`

- [ ] **Step 1: Write failing tests for file-read and history budgets**

```python
# tests/test_context_budget.py
def test_read_file_caps_to_2000_lines(tmp_path):
    from tools.file_tools import read_file

    p = tmp_path / "big.txt"
    p.write_text("\n".join(str(i) for i in range(3000)), encoding="utf-8")
    out = read_file(str(p))
    assert len(out.splitlines()) <= 2000


def test_context_budget_estimate_tokens():
    from core.context_budget import estimate_tokens

    assert estimate_tokens("abcd" * 1000) > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_context_budget.py -v`

Expected: FAIL because cap/estimator are missing.

- [ ] **Step 3: Implement budget module and integrate line cap**

```python
# core/context_budget.py
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def should_compact(estimated_tokens: int, soft_limit: int) -> bool:
    return estimated_tokens >= soft_limit
```

Also update `read_file` default behavior to cap output to `MAX_FILE_READ_LINES`.

- [ ] **Step 4: Add runtime compact trigger near 160k token soft limit**

Run: `pytest tests/test_context_budget.py tests/test_agent.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/context_budget.py tools/file_tools.py core/runtime.py tests/test_context_budget.py
git commit -m "feat(context): enforce read/history/token discipline limits"
```

---

## Task 8: Runtime Self-Healing and Format Retry Loop

**Files:**
- Modify: `core/runtime.py`
- Test: `tests/test_self_healing.py`

- [ ] **Step 1: Write failing tests for invalid JSON recovery (max 3 retries)**

```python
# tests/test_self_healing.py
def test_runtime_retries_invalid_format_up_to_three_then_fails():
    # provider returns invalid outputs repeatedly; runtime should stop with structured error
    ...


def test_runtime_recovers_when_second_response_is_valid_json():
    # first invalid, second valid tool call, third finish
    ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_self_healing.py -v`

Expected: FAIL until recovery loop is implemented.

- [ ] **Step 3: Implement runtime repair protocol**

Required logic:
- On parse error, append a structured observation requesting strict JSON fix.
- Retry up to 3 times.
- If still invalid, return deterministic JSON error payload.
- Do not crash runtime on tool errors; keep observation-driven recovery.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_self_healing.py tests/test_agent.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/runtime.py tests/test_self_healing.py
git commit -m "feat(runtime): add strict json self-healing retry loop"
```

---

## Task 9: Deterministic Provider Behavior (temperature=0.0)

**Files:**
- Modify: `llm/anthropic_provider.py`
- Modify: `llm/openai_provider.py`
- Modify: `tests/test_openai_provider.py`
- Create: `tests/test_anthropic_provider.py`

- [ ] **Step 1: Write failing tests for deterministic request arguments**

```python
# tests/test_anthropic_provider.py
def test_anthropic_chat_sets_temperature_zero(monkeypatch):
    ...

# tests/test_openai_provider.py
def test_openai_chat_sets_temperature_zero():
    ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_openai_provider.py tests/test_anthropic_provider.py -v`

Expected: FAIL because temperature is not forced.

- [ ] **Step 3: Implement provider argument updates**

Required updates:
- Add `temperature=0.0` to Anthropic messages call.
- Add `temperature=0.0` to OpenAI chat completion call.
- Keep existing max token compatibility logic unchanged.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_openai_provider.py tests/test_anthropic_provider.py tests/test_providers.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm/anthropic_provider.py llm/openai_provider.py tests/test_openai_provider.py tests/test_anthropic_provider.py
git commit -m "fix(provider): enforce deterministic temperature zero for react protocol"
```

---

## Task 10: Static Prompt Rewrite and Documentation Alignment

**Files:**
- Modify: `config.py`
- Modify: `README.md`
- Create: `docs/liteagent-philosophy-alignment.md`

- [ ] **Step 1: Write failing doc-level assertions (lightweight)**

```python
# tests/test_prompt_contract.py
def test_base_system_prompt_contains_json_contract():
    import config
    prompt = config.BASE_SYSTEM_PROMPT
    assert '"thought"' in prompt
    assert '"action"' in prompt
    assert '"action_input"' in prompt
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_prompt_contract.py -v`

Expected: FAIL with current prompt content.

- [ ] **Step 3: Rewrite static prompt contract**

Prompt must include:
- strict ReAct JSON output schema
- tool whitelist names
- security rules
- context discipline limits
- explicit `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`

- [ ] **Step 4: Update README and philosophy mapping doc**

Run: `pytest tests/test_prompt_contract.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.py README.md docs/liteagent-philosophy-alignment.md tests/test_prompt_contract.py
git commit -m "docs(prompt): align static contract and docs with seven philosophies"
```

---

## Final Integration Task: End-to-End Verification and Branch Readiness

**Files:**
- Modify: none unless fixes required

- [ ] **Step 1: Run full test suite**

Run: `pytest tests -q`

Expected: all tests pass.

- [ ] **Step 2: Run lint and type checks**

Run: `make lint && make type`

Expected: no blocking issues.

- [ ] **Step 3: Execute final spec compliance review subagent**

Expected: confirms all seven philosophies are implemented as specified.

- [ ] **Step 4: Execute final code quality review subagent**

Expected: approves maintainability, no major correctness/safety risks.

- [ ] **Step 5: Prepare integration options**

Use `@superpowers:finishing-a-development-branch` to decide merge/PR/cleanup path.

---

## Acceptance Matrix (Must All Be Green)

- [ ] Philosophy 1 (ReAct heart): strict JSON parser + runtime action loop verified.
- [ ] Philosophy 2 (dynamic system prompt): per-turn assembly with boundary verified.
- [ ] Philosophy 3 (security first): whitelist + parameter checks + dangerous command blocking + sandbox verified.
- [ ] Philosophy 4 (minimal memory): MEMORY.md line format + max 200 + indexed summaries verified.
- [ ] Philosophy 5 (resource discipline): read <= 2000 lines, history compaction >15, token soft-limit compaction verified.
- [ ] Philosophy 6 (Unix tools): single-purpose tools (`read_file`, `write_file`, `run_bash`, `list_dir`, `grep`, `summarize`, `finish`) verified.
- [ ] Philosophy 7 (structured output + self-healing): deterministic JSON output path and 3-retry recovery verified.

---

## Notes for Implementers

- Keep each commit tightly scoped to one task.
- Do not introduce non-stdlib dependencies.
- Do not add multi-agent orchestration now; only preserve flag stubs.
- Prefer additive changes over broad rewrites in one commit.
- Every task must finish with targeted tests and at least one reviewer approval cycle.
