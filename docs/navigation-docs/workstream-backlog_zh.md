# Navigation 内容层：工作流待办

本文件用于维护当前状态与待办摘要。

## 1. 活跃工作流

| 工作流 | 范围 | 当前状态 |
|---|---|---|
| Python 运行时架构 | `quantcore`, `scheduler`, `scripts`, `config`, `utils` | 已重构并生效 |
| 数据管线可靠性 | `data_pipeline/*` | 持续增强中 |
| 模型管线可用性 | `alpha_models`, `scripts/predict.py`, `scripts/view.py` | 活跃，训练后自动可视化已接入 |
| 测试完备性 | `test/*` | 活跃，单元/整体覆盖较完整 |
| 文档导航体系 | `docs/NAVIGATION*`, `docs/navigation-docs/*` | 活跃且权威 |

## 2. 延后 / 范围外

| 工作流 | 原因 |
|---|---|
| `server/*` 深度重构 | 不在当前 Python 重构范围 |
| `news_module/*` 启用 | 模块为弃用/WIP 且隔离 |
| RL 组合生产化 | 仍处占位阶段 |

## 3. 更新规则

状态发生变化时：
1. 更新本文件。
2. 若模块路由变化，同步更新 `module-index`。
3. 若系统拓扑变化，同步更新 `system-map`。
