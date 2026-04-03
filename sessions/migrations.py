# sessions/migrations.py
"""
Session 数据迁移系统。

如何添加新版本迁移：
1. 在 MIGRATIONS 字典中添加新版本号和对应的迁移函数
2. 迁移函数接收 dict，返回迁移后的 dict
3. 更新 LATEST_VERSION 常量

示例：
    # 版本 2: 添加新字段
    MIGRATIONS[2] = lambda d: {**d, "new_field": "default"} if "new_field" not in d else d
"""

MIGRATIONS = {
    # 0 -> 1: 添加 schema_version
    1: lambda d: {**d, "schema_version": 1} if d.get("schema_version", 0) < 1 else d,
    # 未来：2: lambda d: ...,
}

LATEST_VERSION = max(MIGRATIONS.keys())


def migrate(data: dict) -> dict:
    """将 data 从当前版本迁移到最新版本"""
    current = data.get("schema_version", 0)
    for version in range(current + 1, LATEST_VERSION + 1):
        if version in MIGRATIONS:
            data = MIGRATIONS[version](data)
    return data
