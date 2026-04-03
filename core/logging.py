# core/logging.py
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class RunContext:
    """一次运行的追踪上下文"""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str | None = None
    turn: int = 0


class JsonFormatter(logging.Formatter):
    """输出 JSON 格式的日志"""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, str | int] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # 附加运行上下文
        ctx = getattr(record, "run_ctx", None)
        if ctx is not None:
            log_data["run_id"] = ctx.run_id
            if ctx.session_id:
                log_data["session_id"] = ctx.session_id
            if ctx.turn:
                log_data["turn"] = ctx.turn
        # 附加额外字段
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            log_data.update(extra_fields)
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """输出人类可读格式的日志，包含追踪字段"""

    def format(self, record: logging.LogRecord) -> str:
        prefix = f"[{record.levelname}]"
        ctx = getattr(record, "run_ctx", None)
        if ctx is not None:
            prefix += f"[{ctx.run_id}"
            if ctx.session_id:
                prefix += f":{ctx.session_id}"
            if ctx.turn:
                prefix += f":t{ctx.turn}"
            prefix += "]"
        msg = record.getMessage()
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            extras = " ".join(f"{k}={v}" for k, v in extra_fields.items())
            msg = f"{msg} {extras}"
        return f"{prefix} {msg}"


def setup_logging(level: str = "INFO", fmt: str = "json") -> logging.Logger:
    """配置全局日志"""
    logger = logging.getLogger("agent")
    # 校验 log level
    level_no = getattr(logging, level.upper(), None)
    if level_no is None:
        raise ValueError(f"无效的日志级别: {level}")
    logger.setLevel(level_no)

    # 显式清理之前的 handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())

    logger.addHandler(handler)
    return logger


# 全局 logger
logger = logging.getLogger("agent")


def log_event(event: str, ctx: RunContext, **kwargs) -> None:
    """记录结构化事件"""
    # 使用标准 logging API，通过 extra 传递上下文和额外字段
    extra: dict[str, object] = {"run_ctx": ctx}
    if kwargs:
        extra["extra_fields"] = kwargs
    logger.info(event, extra=extra)
