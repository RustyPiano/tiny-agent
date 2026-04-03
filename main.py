# main.py
"""
Agent Framework 入口。
用法：
    python main.py
    python main.py --session my_project --skills coding,project_explorer
    python main.py --provider openai --model gpt-4o
    python main.py --provider openai --model qwen2.5:14b --base-url http://localhost:11434/v1
    python main.py --log-format json
"""

import argparse
import sys

from config import AgentSettings
from core.agent import run
from core.logging import RunContext, log_event, setup_logging
from extensions.loader import load_extensions
from llm.factory import create_provider
from skills import discover_skills
from tools.bash_tool import register_bash_tool
from tools.file_tools import register_file_tools
from tools.skill_tool import register_skill_tool


def bootstrap(settings: AgentSettings) -> None:
    try:
        summary = discover_skills(
            project_dir=settings.project_skills_dir,
            global_dir=settings.global_skills_dir,
        )
    except Exception as e:
        summary = {
            "discovered": 0,
            "loaded": 0,
            "overridden": 0,
            "failed": 1,
            "total": 0,
            "failure_details": [
                {
                    "path": f"{settings.global_skills_dir} | {settings.project_skills_dir}",
                    "reason": "discover_exception",
                }
            ],
        }
        log_event(
            "skills_discovery_error",
            RunContext(),
            project_dir=str(settings.project_skills_dir),
            global_dir=str(settings.global_skills_dir),
            error=str(e),
        )

    event_fields = {
        "project_dir": str(settings.project_skills_dir),
        "global_dir": str(settings.global_skills_dir),
        "discovered": summary["discovered"],
        "loaded": summary["loaded"],
        "overridden": summary["overridden"],
        "failed": summary["failed"],
        "total": summary["total"],
    }
    if summary.get("failure_details"):
        event_fields["failure_details"] = summary["failure_details"]

    log_event(
        "skills_discovered",
        RunContext(),
        **event_fields,
    )
    register_file_tools()
    register_bash_tool()
    register_skill_tool()
    try:
        result = load_extensions()
        for ext in result["loaded"]:
            log_event("extension_loaded", RunContext(), extension=ext)
        for failed in result["failed"]:
            log_event(
                "extension_load_failed",
                RunContext(),
                extension=failed["id"],
                category=failed["category"],
                path=failed["path"],
                error=failed["error"],
            )
    except Exception as e:
        log_event("extension_loader_error", RunContext(), error=str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Agent Framework")
    parser.add_argument("--session", default=None, help="会话 ID，用于持久化历史")
    parser.add_argument("--skills", default="", help="逗号分隔的 skill 名称")
    parser.add_argument("--verbose", action="store_true", help="打印工具调用详情")
    parser.add_argument("--provider", default=None, help="llm provider 类型: anthropic | openai")
    parser.add_argument("--model", default=None, help="模型名称")
    parser.add_argument("--base-url", default=None, dest="base_url", help="本地模型 base_url")
    parser.add_argument(
        "--log-format",
        default="text",
        choices=["json", "text"],
        help="日志格式: json | text",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别: DEBUG | INFO | WARNING | ERROR",
    )
    args = parser.parse_args()

    # 配置日志
    setup_logging(level=args.log_level, fmt=args.log_format)

    # 从环境变量构建，再用命令行参数覆盖
    settings = AgentSettings.from_env()
    if args.provider:
        settings.provider_type = args.provider
    if args.model:
        settings.model = args.model
    if args.base_url:
        settings.base_url = args.base_url

    # 校验配置
    errors = settings.validate()
    if errors:
        for e in errors:
            log_event("config_error", RunContext(), error=e)
        sys.exit(1)

    bootstrap(settings)

    provider = create_provider(settings.to_provider_config())

    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    log_event("agent_start", RunContext(), provider=settings.provider_type, model=settings.model)
    print("Agent 已启动，输入任务（Ctrl+C 退出）：\n")

    while True:
        try:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            run_ctx = RunContext(session_id=args.session)
            log_event("session_start", run_ctx, input=user_input[:100])
            result = run(
                user_input,
                settings=settings,
                provider=provider,
                session_id=args.session,
                skills=skills,
                verbose=args.verbose,
                run_ctx=run_ctx,
            )
            print(f"\nAgent: {result}\n")
        except KeyboardInterrupt:
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
