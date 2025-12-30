我正在训练A股量化模型，我的样本是近15年深沪主板的daily数据。

我们将系统拆分为三个串联的模块，每个模块解决一个特定的量化难题。

1. Model A：全能分析师 (The Eye)
   1. 模型架构：Multi-task LSTM/GRU。
   2. 输入：个股过去 30 天的量价/因子序列（归一化数据）。
   3. 核心任务 (Multi-task)：同时预测三个指标，形成“立体画像”：1天涨幅（短期择时信号）。5天累计涨幅（中期趋势信号，核心权重）。5天波动率（风险信号）。
   4. 输出：
      1. 显式输出：预测值 $[r_{1d}, r_{5d}, \sigma_{vol}]$，供 Model B 排序用。
      2. 隐式输出：Context Vector (Embedding)，供 Model C 决策用。
   5. 状态：先进行监督学习预训练，然后在 RL 阶段冻结 (Frozen)。
2. Model B：选股基金经理 (The Filter)
   1. 模型架构：规则排序 (Heuristic Ranking)，非神经网络。
   2. 核心任务：从 3000+ 只股票中筛选出约 300 只放入候选池。
   3. 筛选逻辑：
      1. 夏普评分：$Score = (w_1 \cdot r_{1d} + w_2 \cdot r_{5d}) / \sigma_{vol}$。优先选“涨得稳”的，剔除“涨得猛但风险大”的。
      2. 缓冲区 (Hysteresis)：Top 200 买入，跌出 Top 500 才卖出。防止候选池剧烈变动，保护 Model C。
3. Model C：王牌交易员 (The Hand)
   1. 模型架构：SAC (Soft Actor-Critic)。
   2. 输入：
      1. 市场状态：来自 Model A 的 Context Vector (高质量特征)。
      2. 账户状态：当前持仓、资金、盈亏情况。
   3. 核心任务：决定买卖力度 $[-1, 1]$。


原始数据格式
```
```

数据预处理阶段
```python
```

下面是我的训练环境的代码
```python
```

下面是我的训练的代码
```python
```