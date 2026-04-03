# 如何新增 Tool

本文档说明如何为 Agent Framework 添加自定义工具。

## 快速开始

### 1. 创建工具模块

在 `tools/` 目录下创建新文件，例如 `tools/my_tool.py`:

```python
# tools/my_tool.py
from tools.registry import register


def my_function(param1: str, param2: int = 10) -> str:
    """工具的实际逻辑"""
    # 你的实现
    return f"结果: {param1}, {param2}"


def register_my_tool() -> None:
    register(
        name="my_tool",
        description="工具的描述，LLM 会根据这个决定是否调用",
        parameters={
            "param1": {"type": "string", "description": "参数1的说明"},
            "param2": {"type": "integer", "description": "参数2的说明，默认10"},
        },
        required=["param1"],  # 必填参数列表
        handler=my_function,
    )
```

### 2. 注册工具

推荐方式（PoC）：使用扩展目录自动发现，不修改核心文件。

在 `extensions/tools/` 下放置模块，并提供模块级 `register()` 合约函数：

```python
# extensions/tools/my_tool.py
from tools.registry import register as register_tool


def my_function(param1: str, param2: int = 10) -> str:
    return f"结果: {param1}, {param2}"


def register() -> None:
    register_tool(
        name="my_tool",
        description="工具的描述，LLM 会根据这个决定是否调用",
        parameters={
            "param1": {"type": "string", "description": "参数1的说明"},
            "param2": {"type": "integer", "description": "参数2的说明，默认10"},
        },
        required=["param1"],
        handler=my_function,
    )
```

兼容方式：你仍可在 `main.py` 中手动注册，但不推荐。

## 详细说明

### register() 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 工具名称，LLM 调用时使用 |
| `description` | `str` | 工具描述，LLM 根据此决定是否调用 |
| `parameters` | `dict` | 参数 schema，遵循 JSON Schema 格式 |
| `handler` | `Callable` | 实际执行的函数 |
| `required` | `list[str] \| None` | 必填参数列表，默认全部必填 |

### 参数 Schema 格式

```python
parameters={
    "param_name": {
        "type": "string",        # string | integer | number | boolean
        "description": "参数说明"
    }
}
```

### Handler 函数签名

Handler 接收与 `parameters` 对应的关键字参数，返回字符串:

```python
def handler(**kwargs) -> str:
    # 处理逻辑
    return "结果字符串"
```

### 返回值格式

建议使用前缀标识结果类型:

```python
return "[ok] 操作成功"
return "[error] 错误信息"
return "[warn] 警告信息"
return "[blocked] 操作被拒绝"
```

## 完整示例

### 示例 1: 简单计算工具

```python
# tools/calc_tool.py
from tools.registry import register


def calculate(expression: str) -> str:
    """安全的计算器"""
    try:
        # 注意：生产环境应使用更安全的求值方式
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "[error] 包含不允许的字符"
        result = eval(expression)
        return f"[ok] {expression} = {result}"
    except Exception as e:
        return f"[error] 计算失败: {e}"


def register_calc_tool() -> None:
    register(
        name="calculate",
        description="计算数学表达式，支持加减乘除和括号",
        parameters={
            "expression": {"type": "string", "description": "数学表达式，如 '2 + 3 * 4'"},
        },
        required=["expression"],
        handler=calculate,
    )
```

### 示例 2: 带可选参数的工具

```python
# tools/search_tool.py
from tools.registry import register


def search(query: str, limit: int = 5, case_sensitive: bool = False) -> str:
    """搜索工具"""
    # 实现搜索逻辑
    return f"搜索 '{query}' 的前 {limit} 个结果"


def register_search_tool() -> None:
    register(
        name="search",
        description="在文件中搜索内容",
        parameters={
            "query": {"type": "string", "description": "搜索关键词"},
            "limit": {"type": "integer", "description": "返回结果数量，默认5"},
            "case_sensitive": {"type": "boolean", "description": "是否区分大小写，默认否"},
        },
        required=["query"],  # 只有 query 是必填
        handler=search,
    )
```

## 最佳实践

1. **描述要清晰**: LLM 根据 description 决定是否调用，写清楚适用场景
2. **参数验证**: 在 handler 中验证参数，返回明确的错误信息
3. **输出截断**: 长输出应截断，避免超出 token 限制
4. **安全性**: 危险操作要有确认机制或使用 `[blocked]` 拒绝
5. **错误处理**: 捕获异常，返回 `[error]` 前缀的错误信息

## 信任边界（重要）

扩展模块会被 Python 直接导入并执行模块顶层代码，再调用 `register()`。这意味着扩展具备执行任意代码的能力。请仅使用可信来源的扩展文件。

## 测试工具

创建测试文件 `tests/test_my_tool.py`:

```python
from tools.my_tool import my_function


def test_my_function():
    result = my_function("test", 5)
    assert "结果" in result
```

运行测试:

```bash
make test
```
