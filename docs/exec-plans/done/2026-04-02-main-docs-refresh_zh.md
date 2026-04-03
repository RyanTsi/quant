# 执行计划：主文档刷新

- 日期：2026-04-02
- 任务 UUID：`9db9b42a-0c47-4c94-815b-180cec6fc376`

## 目标
刷新项目的中英文主文档，使其与兼容层清理后的当前 runtime-first 代码结构保持一致。

## 范围
- 范围内：
  - `README.md`
  - `docs/README_zh.md`
  - `ARCHITECTURE.md`
  - `docs/ARCHITECTURE_zh.md`
  - `docs/index.md`
  - `docs/index_zh.md`
  - `docs/navigation-docs/*`
  - `docs/python-runtime-guide*.md`
  - 当活跃契约文案与代码漂移时，补修 `docs/product-specs/python-runtime-v2*.md`
  - 本次文档任务的追踪产物
- 范围外：
  - Python runtime、训练流程或网关代码的行为改动
  - 作为历史记录保留的旧计划/旧日志

## 假设
- 当前代码与测试是现行行为事实来源。
- 历史文档可以保留历史语境；只更新当前仍作为主入口的文档。
- 当前运行时已不再依赖 `scheduler/`、`quantcore.settings`、`quantcore.history`、`quantcore.registry`、`data_pipeline/ingest.py` 与 `news_module/`。

## 步骤
1. 读取导航文档与当前 runtime 入口，识别事实漂移点。
2. 更新中英文主文档，反映当前文件布局、runtime 归属、CLI 入口与配置面。
3. 刷新导航内容层，使模块路由与现有仓库树一致。
4. 若 critic 指出活跃运行时/产品文档仍有漂移，则先补修再收口。
5. 对文档涉及的主要 runtime 表面执行聚焦验证和一致性检查。
6. 记录中英文结构化日志，附上 critic review，并将本计划移入 `docs/exec-plans/done/`。

## 验收标准
- 主文档能准确描述当前 runtime-first 架构，且中英文一致。
- Navigation 文档不再把任务路由到已删除模块。
- 配置与用法章节只引用仍然存在的代码路径和支持的环境变量。
- 验证覆盖文档涉及的主要 runtime 入口。
- 结构化日志包含 UUID 追踪和 critic 反馈。

## 回滚说明
- 若文档刷新被证明不准确，则连同本次追踪产物一起回滚。
