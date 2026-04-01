# 执行计划：Workflow Runner 重构（可脱离 YAML）

- 日期：2026-03-31
- 任务 UUID：3e83f7da-4b24-41d8-887e-d768bbaba772

## 目标
重构 `alpha_models/workflow/runner.py`，使训练流程支持代码配置控制，不再完全依赖 YAML，同时保持向后兼容。

## 范围
- 范围内：
  - 增加 Python dict/配置对象加载路径。
  - 保留现有 YAML 路径可用。
  - 增加 runtime override 合并能力。
  - 增加针对新能力和兼容性的测试。
  - 更新最小必要中英文文档。
- 范围外：
  - 模型结构或训练算法重构。
  - 调度流程重设计。

## 假设
- 现有 `run_from_yaml` 行为必须保持可用。
- `docs/detailed_workflow.ipynb` 体现了期望的“代码分步可控”风格。

## 步骤
1. 扩展 runner 的配置模型，增加通用配置组装入口。
2. 新增 `run_from_config`，并让 `run_from_yaml` 复用同一运行主路径。
3. 保持 `alpha_models/qlib_workflow.py` 调用兼容（不改或最小改）。
4. 增加单元测试，覆盖配置合并与兼容行为。
5. 在 `conda quant` 运行最小相关测试。
6. 更新中英文 changelog。

## 验收标准
- runner 支持无 YAML 的 dict/config 训练运行。
- 现有 YAML 训练路径不受影响。
- runtime override 合并行为可预测。
- 新增/更新测试通过。

## 回滚说明
- 回滚以下文件：
  - `alpha_models/workflow/runner.py`
  - `test/test_workflow_runner.py`（新增）
  - 可选文档更新（`README.md`, `docs/README_zh.md`）
- 如取消实施，删除该计划文档。
