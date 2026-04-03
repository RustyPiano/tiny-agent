# Agent 流程与工具重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Agent 在长参数工具调用时工具不执行、用户无中间反馈、`write_file` 易截断等核心体验问题，补齐 `edit_file` 局部编辑工具，增强 `end_turn` 容错，增加逐轮事件可见性。

**Architecture:** 保持现有 provider/runtime 分层，新增 `edit_file` 工具、`write_file` payload 上限、`end_turn` 解析失败自愈回路、CLI 逐轮事件输出、`beforeToolCall` 钩子。

**Tech Stack:** Python 3.11+, stdlib, anthropic/openai SDK, pytest, ruff, mypy。

---

## Task 1: 新增 `edit_file` 局部编辑工具

**Files:**
- Create: `tools/edit_file_tool.py`
- Modify: `main.py`
- Create: `tests/test_edit_file_tool.py`

- [ ] **Step 1: 写失败测试**
```python
def test_edit_file_replaces_text(tmp_path):
    from tools.edit_file_tool import edit_file
    p = tmp_path / "test.txt"
    p.write_text("hello world\nfoo bar", encoding="utf-8")
    result = edit_file(str(p), "hello", "hi")
    assert "[ok]" in result
    assert p.read_text(encoding="utf-8") == "hi world\nfoo bar"
```

- [ ] **Step 2: 运行测试，确认失败**
Run: `pytest tests/test_edit_file_tool.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 edit_file 工具**
```python
# tools/edit_file_tool.py
import pathlib
from tools.registry import register
import config

def edit_file(path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
    p = pathlib.Path(path)
    # 复用 _validate_path 或自行校验
    if not p.exists():
        return f"[error] 文件不存在: {path}"
    content = p.read_text(encoding="utf-8")
    if old_str not in content:
        return f"[error] 未找到匹配文本: {old_str!r}"
    if not replace_all and content.count(old_str) > 1:
        return f"[error] 找到多处匹配，请提供更长上下文以唯一定位；或使用 replace_all=true"
    new_content = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
    p.write_text(new_content, encoding="utf-8")
    return f"[ok] 已替换 {path}"
```

- [ ] **Step 4: 注册工具**
在 `main.py` 的 `bootstrap()` 里调用 `register_edit_file_tool()`。

- [ ] **Step 5: 运行测试**
Run: `pytest tests/test_edit_file_tool.py -v`
Expected: PASS

- [ ] **Step 6: 提交**
```bash
git add tools/edit_file_tool.py main.py tests/test_edit_file_tool.py
git commit -m "feat(tools): add edit_file for safe local file modifications"
```

---

## Task 2: `write_file` payload 上限与软拒绝

**Files:**
- Modify: `tools/file_tools.py`
- Modify: `config.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: 写失败测试**
```python
def test_write_file_rejects_oversized_content(tmp_path, monkeypatch):
    from tools.file_tools import write_file
    monkeypatch.setattr("config.MAX_WRITE_FILE_CHARS", 64000)
    result = write_file(str(tmp_path / "big.txt"), "x" * 100000)
    assert "[error]" in result
    assert "超长" in result or "MAX_WRITE_FILE_CHARS" in result
```

- [ ] **Step 2: 运行测试，确认失败**
Run: `pytest tests/test_tools.py::test_write_file_rejects_oversized_content -v`
Expected: FAIL

- [ ] **Step 3: 实现上限检查**
在 `config.py` 新增 `MAX_WRITE_FILE_CHARS = 64000`。
在 `write_file()` 开头加：
```python
if len(content) > config.MAX_WRITE_FILE_CHARS:
    return (
        f"[error] content 超长（{len(content)} 字符，上限 {config.MAX_WRITE_FILE_CHARS}）。"
        "请改用 edit_file 做局部修改，或分块写入。"
    )
```

- [ ] **Step 4: 运行测试**
Run: `pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: 提交**
```bash
git add config.py tools/file_tools.py tests/test_tools.py
git commit -m "feat(tools): add write_file payload limit with soft rejection"
```

---

## Task 3: `end_turn` + ReAct 解析失败自愈回路

**Files:**
- Modify: `core/runtime.py`
- Modify: `core/react_protocol.py`
- Create: `tests/test_end_turn_recovery.py`

- [ ] **Step 1: 写失败测试**
```python
def test_end_turn_with_malformed_react_json_retries_then_recovers():
    # 1st: end_turn + malformed JSON, 2nd: valid tool call, 3rd: end_turn + text
    ...
```

- [ ] **Step 2: 运行测试，确认失败**
Run: `pytest tests/test_end_turn_recovery.py -v`
Expected: FAIL

- [ ] **Step 3: 实现自愈回路**
在 `core/runtime.py` 的 `Step.DONE` 分支：
- 如果文本像 JSON（以 `{` 开头或包含 `"action"`）但解析失败：
  - 注入 observation："你的输出像 ReAct JSON 但解析失败，错误：{error}。请仅输出严格 JSON。"
  - 重试最多 2 次
  - 超限后返回确定性失败信息

在 `core/react_protocol.py` 增加 `parse_react_json_with_error(raw, allowed_actions) -> (ReactDecision | None, error_msg | None)`。

- [ ] **Step 4: 运行测试**
Run: `pytest tests/test_end_turn_recovery.py tests/test_react_protocol.py -v`
Expected: PASS

- [ ] **Step 5: 提交**
```bash
git add core/runtime.py core/react_protocol.py tests/test_end_turn_recovery.py
git commit -m "feat(runtime): add end_turn + malformed react json self-healing loop"
```

---

## Task 4: CLI 逐轮事件可见性

**Files:**
- Modify: `core/runtime.py`
- Modify: `main.py`
- Modify: `core/logging.py`
- Create: `tests/test_cli_events.py`

- [ ] **Step 1: 写失败测试**
```python
def test_runtime_emits_turn_events():
    # 验证 run() 过程中产生 turn_start/assistant_decision/tool_call/tool_result 事件
    ...
```

- [ ] **Step 2: 运行测试，确认失败**
Run: `pytest tests/test_cli_events.py -v`
Expected: FAIL

- [ ] **Step 3: 实现事件输出**
在 `core/runtime.py` 的 `run()` 循环中：
- `call_llm` 后：`log_event("assistant_decision", ...)` 输出 `stop_reason` + `tool_calls_count`
- `execute_tools` 前/后：已有 `tool_call`/`tool_result` 日志

在 `main.py`：
- 新增 `--show-turns` 参数
- 当开启时，stderr 日志（已有）之外，stdout 打印每轮摘要：
  ```
  [t1] read_file(snake-game.html) → ok
  [t2] write_file(snake-game.html) → [error] content 超长
  ```

- [ ] **Step 4: 运行测试**
Run: `pytest tests/test_cli_events.py tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: 提交**
```bash
git add core/runtime.py main.py core/logging.py tests/test_cli_events.py
git commit -m "feat(cli): add per-turn event visibility with --show-turns flag"
```

---

## Task 5: 提高 `MAX_TOKENS`

**Files:**
- Modify: `config.py`
- Modify: `tests/test_philosophy_config.py`（如有相关断言）

- [ ] **Step 1: 修改常量**
```python
# config.py
MAX_TOKENS = 16000  # 从 4096 提升到 16000，支持代码密集型任务
```

- [ ] **Step 2: 运行测试**
Run: `pytest tests/ -v`
Expected: PASS（无回归）

- [ ] **Step 3: 提交**
```bash
git add config.py
git commit -m "config: increase MAX_TOKENS to 16000 for code-heavy tasks"
```

---

## Task 6: `beforeToolCall` 钩子（参数修复）

**Files:**
- Modify: `tools/registry.py`
- Modify: `core/runtime.py`
- Create: `tests/test_tool_hooks.py`

- [ ] **Step 1: 写失败测试**
```python
def test_before_tool_call_can_repair_arguments():
    repairs = []
    def repair(name, inputs):
        if name == "write_file" and "content" in inputs:
            inputs["content"] = inputs["content"][:100]
            repairs.append(name)
        return inputs
    # 注册钩子后调用 execute，验证修复生效
    ...
```

- [ ] **Step 2: 运行测试，确认失败**
Run: `pytest tests/test_tool_hooks.py -v`
Expected: FAIL

- [ ] **Step 3: 实现钩子机制**
在 `tools/registry.py` 新增：
```python
_BEFORE_TOOL_CALL_HOOKS: list[callable] = []

def register_before_tool_call(hook):
    _BEFORE_TOOL_CALL_HOOKS.append(hook)
```

在 `execute()` 中，参数校验前遍历 hooks 并调用：
```python
for hook in _BEFORE_TOOL_CALL_HOOKS:
    inputs = hook(name, inputs)
```

在 `core/runtime.py` 注册默认修复钩子（如截断超长 content）。

- [ ] **Step 4: 运行测试**
Run: `pytest tests/test_tool_hooks.py tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: 提交**
```bash
git add tools/registry.py core/runtime.py tests/test_tool_hooks.py
git commit -m "feat(tools): add beforeToolCall hook for argument repair"
```

---

## 最终集成验证

- [ ] **Step 1: 全量测试**
Run: `pytest tests -q`
Expected: 全绿

- [ ] **Step 2: 质量门禁**
Run: `make lint && make type`
Expected: 全绿

- [ ] **Step 3: 手动验证**
```bash
python3 main.py --session verify_edit --show-turns
# 输入：用 edit_file 修改 snake-game.html 的主题按钮区域
```
Expected: 工具执行成功，CLI 显示逐轮摘要。

---

## 验收矩阵

- [ ] `edit_file` 工具可用，支持唯一定位与 replace_all
- [ ] `write_file` 超长 content 被拒绝并提示改用 edit_file
- [ ] `end_turn` + 破损 JSON 进入修复回路，不直接结束
- [ ] CLI 显示每轮 tool_call/tool_result 摘要
- [ ] `MAX_TOKENS` 提升到 16000
- [ ] `beforeToolCall` 钩子可注册并生效
- [ ] 全量测试 + lint + type 全绿
