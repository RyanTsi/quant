# 执行计划：导出代码列表回退

- Goal: 保证 `main.py --run export` 在网关 `GET /api/v1/symbols` 超时或断连时仍可继续执行。
- Scope: Python 导出 adapter、数据 service 接线、聚焦测试，以及 runtime guide 文档。
- Assumptions:
  - 相比全量 symbol 列表查询，按单个 symbol 的网关查询更稳定。
  - `.data/` 下的本地代码清单足够新，可以作为导出时的优先 symbol 来源。
- Steps:
  1. 确认失败发生在哪个运行时步骤，并定位 adapter / service 边界。
  2. 让导出时的 symbol 解析优先使用本地工件，并保留网关列表作为次级回退。
  3. 更新 adapter 行为与 service 接线测试。
  4. 补充回退行为文档，并记录追踪产物。
  5. 运行最小验证，并做一次基于回退的真实导出探测。
- Acceptance Criteria:
  - 当网关 symbol 列表查询失败时，导出不再直接失败。
  - 现有导出结果元数据保持不变。
  - 聚焦测试覆盖回退加载与 service 参数传递。
- Rollback Notes:
  - 若回退逻辑导致 symbol 覆盖错误，则回滚导出 adapter 回退 helper 与 service 接线改动。
