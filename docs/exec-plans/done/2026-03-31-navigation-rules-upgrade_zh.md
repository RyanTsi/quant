# 执行计划：Navigation 规则升级

- 日期：2026-03-31
- 任务 UUID：1bb80e51-c329-42c8-9c12-bccb536118ae

## 目标
建立统一的 Navigation 规则体系，具备明确图结构与可执行规则。

## 范围
- 范围内：
  - 新增 `docs/NAVIGATION.md` 作为顶层 Navigation 契约。
  - 新增 `docs/navigation-docs/*` 作为图结构与规则细则文档。
  - 补齐中文对应文档（`docs/` 下 `_zh` 后缀）。
  - 更新 `AGENTS.md`，明确要求读取 Navigation 文档。
- 范围外：
  - 代码运行逻辑变更。
  - `server`/`news_module` 变更。

## 假设
- Navigation 文档将成为文档路由与执行规则的权威来源。

## 步骤
1. 创建顶层 Navigation 文档（中英文），写清图结构与执行流程。
2. 在 `docs/navigation-docs/` 下创建细分导航文档（中英文）。
3. 更新 `AGENTS.md` 的必读顺序与 Navigation 强制规则。
4. 更新 docs 索引链接。
5. 输出任务日志并将计划归档到 `done/`。

## 验收标准
- `docs/NAVIGATION.md` 与 `docs/navigation-docs/*` 已存在且图结构明确。
- `AGENTS.md` 已明确要求遵循 Navigation。
- NAVIGATION 文档明确写出该规则集的执行步骤。
- 中英文版本齐全且内容一致。

## 回滚说明
- 回滚以下文件：
  - `AGENTS.md`
  - `docs/NAVIGATION.md`、`docs/NAVIGATION_zh.md`
  - `docs/navigation-docs/*`
  - 可选的 docs 索引改动
