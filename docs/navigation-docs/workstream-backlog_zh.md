# Navigation 内容层：工作流待办

本文件用于维护当前代码库的状态与待办摘要。

## 1. 活跃工作流

| 工作流 | 范围 | 当前状态 |
|---|---|---|
| Python 运行时架构 | `main.py`, `runtime`, `scripts`, `data_pipeline`, `alpha_models`, `utils` | runtime-first 归属已完成；剩余兼容壳已于 2026-04-02 删除，当前活跃工作主要是行为打磨以及文档/测试对齐 |
| 数据管线可靠性 | `runtime/adapters/fetching.py`, `runtime/adapters/ingest.py`, `runtime/adapters/exporting.py`, `data_pipeline/*` | 活跃；重点仍是抓取窗口正确性、ingest 安全性和网关失败报告 |
| 模型管线可用性 | `model_function`, `alpha_models`, `runtime/adapters/modeling.py`, `scripts/filter.py`, `scripts/predict.py`, `scripts/build_portfolio.py`, `scripts/view.py`, `scripts/eval_test.py` | 活跃；确定性股票池构建已沉到 `model_function/`，当前重点是让训练/预测/组合行为和操作体验持续围绕这份共享契约收敛 |
| 测试完备性 | `test/*` | 活跃；当前已覆盖 bootstrap、registry、runlog、adapters、CLI wrapper 与 pipeline semantics |
| 文档导航体系 | `docs/NAVIGATION*`, `docs/navigation-docs/*`, `docs/python-runtime-guide*.md` 与主文档 | 已于 2026-04-02 刷新，并持续按代码事实做对齐；runtime 细节继续收敛时需保持中英文同步 |

## 2. 延后 / 范围外

| 工作流 | 原因 |
|---|---|
| `server/*` 深度重构 | 不在当前 Python 文档/运行时任务范围 |
| RL 组合生产化 | 目前仍只是占位阶段 |

## 3. 更新规则

状态发生变化时：
1. 更新本文件。
2. 若范围或路由变化，同步更新 `module-index`。
3. 若系统拓扑变化，同步更新 `system-map`。
