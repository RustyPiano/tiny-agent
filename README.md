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

# 运行
python main.py
```

运行后应看到类似输出：

```
Agent 已启动，输入任务（Ctrl+C 退出）：

你:
```

## 用法

```bash
# 基本用法
python main.py

# 使用 OpenAI provider
python main.py --provider openai --model gpt-4o

# 使用本地模型 (Ollama)
python main.py --provider openai --model qwen2.5:14b --base-url http://localhost:11434/v1

# 启用 skills
python main.py --skills coding,project_explorer

# 持久化会话
python main.py --session my_project

# JSON 日志格式
python main.py --log-format json --log-level DEBUG
```

## 架构

```
agent-framework/
├── core/                   # 核心组件
│   ├── agent.py            # ReAct 主循环
│   ├── context.py          # 对话上下文管理
│   ├── logging.py          # 结构化日志
│   └── prompt_builder.py   # System prompt 构建
├── llm/                    # Provider 抽象层
│   ├── base.py             # BaseLLMProvider 抽象类
│   ├── factory.py          # Provider 工厂
│   ├── anthropic_provider.py
│   └── openai_provider.py
├── tools/                  # 工具系统
│   ├── registry.py         # 工具注册中心
│   ├── bash_tool.py        # Shell 命令执行
│   └── file_tools.py       # 文件读写
├── skills/                 # Skill 系统（动态 prompt 注入）
│   ├── registry.py         # Skill 注册中心
│   └── builtin/            # 内置 skills
├── sessions/               # 会话持久化
│   ├── store.py            # 会话存储
│   └── migrations.py       # 版本迁移
├── config.py               # 配置集中化
└── main.py                 # 入口
```

### 核心概念

- **ReAct 循环**: `core/agent.py` 实现 Reasoning-Acting 循环，LLM 决定调用工具或直接回答
- **Provider 抽象**: `llm/base.py` 定义统一接口，屏蔽 Anthropic/OpenAI API 差异
- **工具注册**: `tools/registry.py` 提供声明式工具注册，自动转换为 LLM schema
- **Skill 注入**: `skills/` 动态注入 prompt 片段，扩展 Agent 能力

## 扩展指南

- [如何新增 Tool](docs/how-to-add-tool.md)
- [如何新增 Provider](docs/how-to-add-provider.md)

## 扩展机制（PoC）

框架会在启动时按约定目录尝试加载扩展模块：

- `extensions/tools/*.py`
- `extensions/skills/*.py`
- `extensions/providers/*.py`

扩展模块需要提供模块级 `register()` 合约函数，加载器会导入模块后调用该函数完成注册。

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
