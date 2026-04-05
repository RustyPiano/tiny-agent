# Agent Framework

轻量 Python Agent，不依赖第三方框架（仅 anthropic/openai SDK）。

## 前置条件

- Python 3.11+
- Anthropic 或 OpenAI API Key

## 快速开始

```bash
# 安装依赖（推荐）
pip install -e ".[dev]"

# 或仅安装运行时依赖
pip install -e .

# 配置 API Key
export ANTHROPIC_API_KEY=xxx

# 运行（推荐）
agent

# 或模块方式
python -m agent_framework
```

运行后应看到类似输出：

```
Agent 已启动，输入任务（Ctrl+C 退出）：

你:
```

## 用法

```bash
# 基本用法
agent

# 使用 OpenAI provider
agent --provider openai --model gpt-4o

# 使用本地模型 (Ollama)
agent --provider openai --model qwen2.5:14b --base-url http://localhost:11434/v1

# 预加载 skills（可选；skills 名称来自已发现目录）
agent --skills code-review,safe-ops

# 持久化会话
agent --session my_project

# 界面模式
agent --ui concise
agent --ui detailed

# JSON 日志格式
agent --log-format json --log-level DEBUG
```

## 架构

```
agent-framework/
├── agent_framework/
│   ├── __main__.py         # python -m agent_framework 入口
│   ├── main.py             # CLI 入口（project.scripts: agent）
│   ├── _config.py          # 配置集中化
│   ├── core/               # 核心组件
│   ├── llm/                # Provider 抽象层
│   ├── tools/              # 工具系统
│   ├── skills/             # Skill 系统
│   ├── sessions/           # 会话持久化
│   └── extensions/         # 扩展加载器与内置扩展示例
├── tests/
└── pyproject.toml
```

### 核心概念

- **ReAct 循环**: `agent_framework/core/agent.py` 实现 Reasoning-Acting 循环
- **Provider 抽象**: `agent_framework/llm/base.py` 定义统一接口
- **工具注册**: `agent_framework/tools/registry.py` 提供声明式工具注册
- **Skill 注入**: `agent_framework/skills/` 动态注入 prompt 片段

## LiteAgent 哲学对齐

- 静态 system prompt 现在声明了严格 ReAct JSON 契约（`"thought"` / `"action"` / `"action_input"`）。
- prompt 明确了运行时注册工具集合、安全约束和上下文纪律（轮次/历史/记忆边界）。
- 静态区域显式引用 `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`，用于区分静态契约与运行时动态上下文。
- 相关测试是轻量 anti-regression 护栏（检查关键条款是否存在），不等价于对模型行为做完整语义验证。
- `workspace_root` 和 `sessions_dir` 由 `AgentSettings` 注入到 runtime 和工具层，不再只依赖全局配置。
- 详细说明见 `docs/liteagent-philosophy-alignment.md`。

## 扩展指南

- [如何新增 Tool](docs/how-to-add-tool.md)
- [如何新增 Provider](docs/how-to-add-provider.md)
- [如何新增 Skill](docs/how-to-add-skill.md)

## 扩展机制（PoC）

框架会在启动时按约定目录尝试加载扩展模块：

- `agent_framework/extensions/tools/*.py`
- `agent_framework/extensions/providers/*.py`

扩展模块需要提供模块级 `register()` 合约函数，加载器会导入模块后调用该函数完成注册。

启动时会把当前已注册工具集合动态写入 prompt 的 allowed-tools 段，因此扩展工具一旦注册，就会进入模型可见的工具列表。

安全说明（重要）：扩展本质上是 Python 代码，加载时会执行任意模块代码与 `register()` 逻辑。请仅安装和启用你信任来源的扩展。

## 开发

```bash
# 运行测试
make test

# 代码检查
make lint

# 类型检查
make type

# 全部检查
make check

# 清理缓存
make clean
```

## 配置

环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API Key | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `AGENT_PROVIDER` | Provider 类型 | `anthropic` |
| `AGENT_MODEL` | 模型名称 | `claude-opus-4-6` |
| `AGENT_BASE_URL` | 自定义 API 地址 | - |
| `AGENT_WORKSPACE` | 工作空间根目录 | 当前目录 |
| `AGENT_PROJECT_SKILLS_DIR` | 项目级 skills 目录 | `<workspace>/.agents/skills` |
| `AGENT_GLOBAL_SKILLS_DIR` | 全局 skills 目录 | `~/.agents/skills` |

## Skills 目录规范

Skill 采用主流目录布局：

`<skills_root>/<skill_name>/SKILL.md`

`SKILL.md` 顶部使用 frontmatter：

```md
---
name: coding
description: 代码实现与重构最佳实践
---
<skill 的完整指令正文>
```

当前解析器是轻量实现，仅支持简单的 `key: value` 行（例如 `name`、`description`），不支持嵌套 YAML 结构或复杂语法。

启动时会自动发现两级 skills 并注入 metadata 到 system prompt 的 `Available Skills` 段落：

1. 全局级：`~/.agents/skills`
2. 项目级：`<project>/.agents/skills`

同名 skill 以项目级覆盖全局级（project > global）。

## Skills 加载流程

1. 启动时扫描全局目录与项目目录，读取 `SKILL.md` frontmatter 的 `name`/`description`。
2. 自动把技能元数据注入 system prompt 的 `Available Skills`。
3. 当模型需要技能完整内容时，调用 `use_skill(name)` 按需加载正文。
4. 同名技能按项目级覆盖全局级。

## 会话与工作区

- 文件工具、`run_bash`、`grep`、`list_dir`、`start_job`、`edit_file` 都会使用 `AgentSettings.workspace_root` 作为运行时工作区边界。
- 会话历史由 `AgentSettings.sessions_dir` 决定，嵌入式调用时可以通过 settings 覆盖默认目录。
- `main.py` 的 CLI 仍然保留当前默认行为，只是底层不再依赖单一全局配置真源。
