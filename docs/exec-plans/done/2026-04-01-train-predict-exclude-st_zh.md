# 执行计划：训练与预测排除 ST 股票

## 目标

确保项目运行流程中，模型训练与预测默认都排除 ST 股票。

## 范围

- `scripts/filter.py`（训练股票池生成路径）
- `scripts/predict.py`（预测候选池构建路径）
- 相关单元测试：`test/test_filter_stocks.py`、`test/test_predict_pool.py`
- 轻量文档更新

## 假设

- 原始行情行数据包含 `isST`（DB 侧摄入），Qlib 特征中可通过 `$isst` 访问。
- 训练股票池通过 `scripts/filter.py` 生成，并由 workflow 配置中的 `my_800_stocks` 使用。
- ST 排除采用保守策略：相关周期/快照为 ST 的标的应被排除。

## 步骤

1. 在 `scripts/filter.py` 中为按流动性筛选增加 ST 季度剔除。
2. 在 `scripts/predict.py` 中为当日候选与前一日补池都增加 ST 标的剔除。
3. 保持输出路径在运行和测试场景下稳定（确保目录存在）。
4. 补充/更新测试：
   - filter 在季度数据含 ST 时的行为
   - prediction pool 对 `$isst` 的剔除行为
5. 运行聚焦测试并修复回归。
6. 更新文档与日志，并将本计划移至 `docs/exec-plans/done/`。

## 验收标准

- 训练股票池输出中，相关 ST 季度/标的不参与入选。
- 预测股票池在当日流动性候选与前一日补池两处都排除 ST 标的。
- 相关测试通过。
- 文档与日志同步更新该行为变更。

## 回滚说明

- 若新排除逻辑引入下游异常，可回滚 `scripts/filter.py`、`scripts/predict.py` 及相应测试改动。

