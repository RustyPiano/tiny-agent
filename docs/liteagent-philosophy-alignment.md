# LiteAgent Philosophy Alignment

本文档说明 Task 10 对静态提示词契约（`BASE_SYSTEM_PROMPT`）的对齐目标与边界。

## 目标

- 把 Agent 的核心行为约束前置到静态 prompt，降低 provider 差异导致的行为漂移。
- 让 ReAct 输出协议、工具使用范围、安全约束、上下文纪律可测试、可审计。
- 明确静态契约与动态上下文的分界，避免模型混淆不可变规则与运行时信息。

## 静态契约更新点

`config.py` 中的 `BASE_SYSTEM_PROMPT` 已覆盖以下条目：

1. 严格 ReAct JSON 输出协议
   - 每轮只能输出一个 JSON object。
   - 必须且仅允许包含 `"thought"`、`"action"`、`"action_input"` 三个键。
   - 明确 `"NONE"` 分支的约束。

2. 工具白名单
   - 明确可调用工具名：`read_file`、`write_file`、`run_bash`、`grep`、`list_dir`、`use_skill`、`summarize`、`finish`、`NONE`。
   - 强调只能调用白名单中的 action。

3. 安全规则
   - 禁止调用未授权工具。
   - 禁止越权路径访问和高风险不可逆命令。
   - 对有副作用操作要求先做最小化探查。

4. 上下文纪律
   - 明确轮次、压缩历史、记忆行数上限。
   - 强调按观察驱动下一步，避免连续盲调工具。

5. 动态边界
   - 静态 prompt 显式引用 `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`。
   - 约定该标记之后为运行时动态上下文区域。

## 测试保障

- 新增 `tests/test_prompt_contract.py`，按章节对关键契约片段做存在性断言。
- 该测试作为轻量 anti-regression 护栏，确保后续改动不会意外移除关键治理语句。
- 这类断言不覆盖完整语义正确性（例如模型是否始终按契约执行），完整行为仍需结合端到端与运行时验证。

## 兼容性说明

- 本次调整只修改提示词文本和文档，不改变工具实现、运行时逻辑或 provider 接口。
- `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` 常量保持不变，继续由消息组装层注入动态区域。
