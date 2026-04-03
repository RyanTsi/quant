# 执行计划：详细工作流 Notebook 阅读总结

## 目标

仔细阅读 `docs/detailed_workflow.ipynb`，提取数据集、训练流程和模型使用方式等具体信息，并整理成面向用户的说明。

## 范围

- `docs/detailed_workflow.ipynb`
- 为区分教程内容与当前项目运行主线而需要交叉核对的仓库文件：
  - `alpha_models/qlib_workflow.py`
  - `alpha_models/workflow_config_transformer_Alpha158.yaml`
  - `runtime/adapters/modeling.py`
  - `scripts/predict.py`

## 假设

- 该 notebook 主要是 Qlib 教程材料，不是本仓库当前生产训练流程的唯一权威入口。
- 用户需要的是细读后的信息总结，而不是代码修改。

## 步骤

1. 按 `AGENTS.md` 要求阅读导航和核心文档。
2. 逐个单元检查 notebook，提取数据集、预处理、训练、推理和评估细节。
3. 交叉核对仓库当前模型入口，说明 notebook 与当前运行主线的一致处和差异处。
4. 在仓库中记录双语追踪材料，并向用户输出精炼说明。

## 验收标准

- 说明中清楚区分 notebook 教程行为与仓库当前偏生产化的运行流程。
- 对 notebook 的数据集、训练、推理和评估信息总结准确。
- 双语追踪材料已写入 `docs/exec-plans/*` 和 `docs/logs/*`。

## 回滚说明

- 本任务仅新增文档；若无需保留，可删除新增的追踪文件。
