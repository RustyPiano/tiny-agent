# 如何新增 Skill

本文档说明如何按当前技能规范为 Agent Framework 添加 Skill。

## 目录结构

Skill 使用目录式布局：

`<skills_root>/<skill_name>/SKILL.md`

其中 `<skills_root>` 有两级：

1. 全局级：`~/.agents/skills`
2. 项目级：`<project>/.agents/skills`

同名 skill 以项目级覆盖全局级。

## 最小可用示例

在项目内创建 `./.agents/skills/code-review/SKILL.md`：

```md
---
name: code-review
description: 审查代码改动并给出可执行改进建议。
---

# Code Review

当用户要求审核代码时，先总结改动，再按严重级别给出问题与修复建议。
```

然后直接运行：

```bash
agent
```

启动后框架会自动发现该 skill，并把 `name + description` 注入 `Available Skills`。

## Frontmatter 规则

当前解析器是轻量实现，建议仅使用简单 `key: value`：

- `name`
- `description`

例如：

```md
---
name: safe-ops
description: 执行高风险操作前进行检查与确认。
---
```

不建议使用复杂 YAML（嵌套对象、数组、多行块标量等）。

## 加载流程

1. 框架启动时发现技能目录并读取 metadata。
2. 自动注入 `Available Skills` 到 system prompt。
3. 需要完整技能正文时，模型调用 `use_skill(name)`。
4. `use_skill` 返回对应 `SKILL.md` 正文；未知名称返回错误。

## 调试建议

- 若 skill 没生效，先检查目录与文件名是否为 `SKILL.md`。
- 检查 frontmatter 是否包含 `name` 和 `description`。
- 查看启动日志中的 `skills_discovered` 事件：
  - `discovered`
  - `loaded`
  - `overridden`
  - `failed`
  - `failure_details`

## 常见问题

### Q: 为什么同名 skill 没使用全局版本？

A: 当前规则是项目级优先，项目目录中的同名 skill 会覆盖全局 skill。

### Q: 是否必须用 `--skills`？

A: 不是必须。默认流程是自动注入 metadata，按需通过 `use_skill(name)` 拉取正文。`--skills` 只是可选预加载。
