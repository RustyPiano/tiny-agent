# tests/contracts/test_tool_contract.py
"""Tool 合约测试：所有 Tool 必须满足的行为规范"""

import pytest

from agent_framework.tools import registry


class ToolContractTests:
    """Tool 注册与执行的合约测试基类，子类化并实现 get_tool_info()"""

    def get_tool_name(self) -> str:
        raise NotImplementedError

    def get_tool_schema(self) -> dict:
        raise NotImplementedError

    def get_tool_handler(self):
        raise NotImplementedError

    def test_schema_has_required_fields(self):
        schema = self.get_tool_schema()
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["name"] == self.get_tool_name()

    def test_schema_input_schema_has_type(self):
        schema = self.get_tool_schema()
        input_schema = schema["input_schema"]
        assert input_schema.get("type") == "object"

    def test_schema_input_schema_has_properties(self):
        schema = self.get_tool_schema()
        input_schema = schema["input_schema"]
        assert "properties" in input_schema
        assert isinstance(input_schema["properties"], dict)

    def test_handler_returns_str(self):
        handler = self.get_tool_handler()
        result = handler()
        assert isinstance(str(result), str)


class TestBashToolContract(ToolContractTests):
    def get_tool_name(self):
        return "run_bash"

    def get_tool_schema(self):
        from agent_framework.tools.bash_tool import register_bash_tool

        # 确保工具已注册
        if "run_bash" not in registry.list_tools():
            register_bash_tool()
        schemas = registry.get_schemas()
        for s in schemas:
            if s["name"] == "run_bash":
                return s
        pytest.fail("run_bash tool not found in registry")

    def get_tool_handler(self):
        from agent_framework.tools.bash_tool import run_bash

        return lambda: run_bash("echo test")

    def test_handler_returns_str(self):
        handler = self.get_tool_handler()
        result = handler()
        assert isinstance(result, str)
        assert "test" in result


class TestReadFileToolContract(ToolContractTests):
    def get_tool_name(self):
        return "read_file"

    def get_tool_schema(self):
        from agent_framework.tools.file_tools import register_file_tools

        if "read_file" not in registry.list_tools():
            register_file_tools()
        schemas = registry.get_schemas()
        for s in schemas:
            if s["name"] == "read_file":
                return s
        pytest.fail("read_file tool not found in registry")

    def get_tool_handler(self):
        from agent_framework.tools.file_tools import read_file

        return lambda: read_file("/tmp/nonexistent_test_file.txt")

    def test_handler_returns_str(self):
        handler = self.get_tool_handler()
        result = handler()
        assert isinstance(result, str)


class TestWriteFileToolContract(ToolContractTests):
    def get_tool_name(self):
        return "write_file"

    def get_tool_schema(self):
        from agent_framework.tools.file_tools import register_file_tools

        if "write_file" not in registry.list_tools():
            register_file_tools()
        schemas = registry.get_schemas()
        for s in schemas:
            if s["name"] == "write_file":
                return s
        pytest.fail("write_file tool not found in registry")

    def get_tool_handler(self):
        from agent_framework.tools.file_tools import write_file

        return lambda: write_file("/tmp/test_contract.txt", "test content")

    def test_handler_returns_str(self):
        handler = self.get_tool_handler()
        result = handler()
        assert isinstance(result, str)


class TestToolRegistryContract:
    """工具注册表的合约测试"""

    def test_execute_unknown_tool_returns_error(self):
        result = registry.execute("nonexistent_tool_xyz", {})
        assert isinstance(result, str)
        assert "[error]" in result

    def test_get_schemas_returns_list(self):
        schemas = registry.get_schemas()
        assert isinstance(schemas, list)

    def test_list_tools_returns_list(self):
        tools = registry.list_tools()
        assert isinstance(tools, list)

    def test_execute_validates_required_inputs(self):
        def _handler(path: str, content: str) -> str:
            return f"[ok] {path}:{len(content)}"

        registry.register(
            name="contract_required_tool",
            description="contract test required inputs",
            parameters={
                "path": {"type": "string", "description": "path"},
                "content": {"type": "string", "description": "content"},
            },
            required=["path", "content"],
            handler=_handler,
        )

        result = registry.execute("contract_required_tool", {})

        assert isinstance(result, str)
        assert result.startswith("[error]")
        assert "缺少必填参数" in result
        assert "path" in result
        assert "content" in result

    def test_execute_surfaces_parse_error_hint(self):
        def _handler(path: str, content: str) -> str:
            return f"[ok] {path}:{len(content)}"

        registry.register(
            name="contract_parse_error_tool",
            description="contract test parse error",
            parameters={
                "path": {"type": "string", "description": "path"},
                "content": {"type": "string", "description": "content"},
            },
            required=["path", "content"],
            handler=_handler,
        )

        result = registry.execute(
            "contract_parse_error_tool",
            {
                "_tool_parse_error": "arguments JSON 解析失败: Unterminated string",
                "_tool_raw_arguments": '{"path":"x", "content":"oops"',
            },
        )

        assert isinstance(result, str)
        assert result.startswith("[error]")
        assert "工具参数解析失败" in result
        assert "raw_arguments" in result

    def test_register_preserves_explicit_empty_required_list(self):
        def _handler(name: str = "world") -> str:
            return f"hello, {name}"

        registry.register(
            name="contract_optional_tool",
            description="contract test optional parameter",
            parameters={
                "name": {"type": "string", "description": "name"},
            },
            required=[],
            handler=_handler,
        )

        result = registry.execute("contract_optional_tool", {})

        assert isinstance(result, str)
        assert result == "hello, world"

    def test_execute_rejects_non_object_inputs(self):
        def _handler() -> str:
            return "[ok]"

        registry.register(
            name="contract_noop_tool",
            description="contract test non-object inputs",
            parameters={},
            required=[],
            handler=_handler,
        )

        result = registry.execute("contract_noop_tool", None)

        assert isinstance(result, str)
        assert result.startswith("[error]")
        assert "工具参数必须是 object" in result

    def test_execute_keeps_unknown_parameter_error_visible(self):
        def _handler(name: str = "world") -> str:
            return f"hello, {name}"

        registry.register(
            name="contract_unknown_param_tool",
            description="contract test unknown param visibility",
            parameters={
                "name": {"type": "string", "description": "name"},
            },
            required=[],
            handler=_handler,
        )

        result = registry.execute("contract_unknown_param_tool", {"mod": "append"})

        assert isinstance(result, str)
        assert result.startswith("[error]")
        assert "unexpected keyword" in result
