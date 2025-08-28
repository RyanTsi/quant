# Quantitative trading

## Plan

### 金融市场基础
- 金融知识：股票、期货、期权、外汇的基础知识；交易规则（T+1、涨跌停、保证金、手续费等）；基本概念（如开盘价、收盘价、K线、成交量、市盈率、资金流）。
- 投资理论：现代投资组合理论、CAPM模型、有效市场假说等（虽然不一定直接用于策略，但是理论基础）。

### 数据处理

Python核心库：

- NumPy, Pandas： 数据处理和分析的基石，必须极其熟练。
- Scikit-learn： 传统机器学习模型库（回归、分类、聚类）。
- TensorFlow / PyTorch： 深度学习框架。
- Statsmodels： 统计模型，用于时间序列分析。
- Matplotlib, Seaborn： 数据可视化，用于分析结果和策略表现。

### 回测工具

- AkShare 数据
- Qlib 框架

### 实盘部署

## Record

### 8.27

#### Conda Usage

`conda init`

**error:**

> . : 无法加载文件 D:\Documents\WindowsPowerShell\profile.ps1，因为在此系统上禁止运行脚本。有关详细信息，请参阅 https:/go.microsoft.com/fwlink/?LinkID=135170 中的 
about_Execution_Policies。
所在位置 行:1 字符: 3
+ . 'D:\Documents\WindowsPowerShell\profile.ps1'
+   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) []，PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
PS D:\Desktop\quant> 

**solve:**

`Set-ExecutionPolicy RemoteSigned` in sudo

`conda env list` list all environments

`conda create -n <env_name> python=3.13.5` create a new environment

`conda activate <env_name>` activate the environment

#### install packages

`pip install numpy pandas matplotlib`
`pip install akshare mplfinance`

#### akshare usage

使用东财接口

##### history data

```python
stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20240528', adjust="")
```

IN

| 名称       | 类型   | 描述                                                                 |
|------------|--------|----------------------------------------------------------------------|
| symbol     | str    | 股票代码（示例：`'603777'`），可通过 `ak.stock_zh_a_spot_em()` 获取    |
| period     | str    | 周期选择（`'daily'`/`'weekly'`/`'monthly'`）                           |
| start_date | str    | 开始日期（格式：`YYYYMMDD`，示例：`'20210301'`）                      |
| end_date   | str    | 结束日期（格式：`YYYYMMDD`，示例：`'20210616'`）                      |
| adjust     | str    | 复权方式（默认不复权；`'qfq'`前复权；`'hfq'`后复权）                   |
| timeout    | float  | 超时设置（默认不设超时）                                             |

OUT

| 名称       | 类型     | 描述                                       |
|------------|----------|--------------------------------------------|
| 日期       | object   | 交易日                                     |
| 股票代码   | object   | 不带市场标识的股票代码                     |
| 开盘       | float64  | 开盘价                                     |
| 收盘       | float64  | 收盘价                                     |
| 最高       | float64  | 最高价                                     |
| 最低       | float64  | 最低价                                     |
| 成交量     | int64    | 单位：手                                   |
| 成交额     | float64  | 单位：元                                   |
| 振幅       | float64  | 单位：百分比（%）                          |
| 涨跌幅     | float64  | 单位：百分比（%）                          |
| 涨跌额     | float64  | 单位：元                                   |
| 换手率     | float64  | 单位：百分比（%）                          |


1. 为何要复权：由于股票存在配股、分拆、合并和发放股息等事件，会导致股价出现较大的缺口。 若使用不复权的价格处理数据、计算各种指标，将会导致它们失去连续性，且使用不复权价格计算收益也会出现错误。 为了保证数据连贯性，常通过前复权和后复权对价格序列进行调整。

2. 前复权：保持当前价格不变，将历史价格进行增减，从而使股价连续。 前复权用来看盘非常方便，能一眼看出股价的历史走势，叠加各种技术指标也比较顺畅，是各种行情软件默认的复权方式。 这种方法虽然很常见，但也有两个缺陷需要注意。
    2.1 为了保证当前价格不变，每次股票除权除息，均需要重新调整历史价格，因此其历史价格是时变的。 这会导致在不同时点看到的历史前复权价可能出现差异。
    2.2 对于有持续分红的公司来说，前复权价可能出现负值。

3. 后复权：保证历史价格不变，在每次股票权益事件发生后，调整当前的股票价格。 后复权价格和真实股票价格可能差别较大，不适合用来看盘。 其优点在于，可以被看作投资者的长期财富增长曲线，反映投资者的真实收益率情况。

4. 在量化投资研究中普遍采用后复权数据

##### real time data

```python
stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
```

OUT
| 名称             | 类型     | 描述                         |
|------------------|----------|------------------------------|
| 序号             | int64    | -                            |
| 代码             | object   | -                            |
| 名称             | object   | -                            |
| 最新价           | float64  | -                            |
| 涨跌幅           | float64  | 注意单位: %                  |
| 涨跌额           | float64  | -                            |
| 成交量           | float64  | 注意单位: 手                 |
| 成交额           | float64  | 注意单位: 元                 |
| 振幅             | float64  | 注意单位: %                  |
| 最高             | float64  | -                            |
| 最低             | float64  | -                            |
| 今开             | float64  | -                            |
| 昨收             | float64  | -                            |
| 量比             | float64  | -                            |
| 换手率           | float64  | 注意单位: %                  |
| 市盈率-动态      | float64  | -                            |
| 市净率           | float64  | -                            |
| 总市值           | float64  | 注意单位: 元                 |
| 流通市值         | float64  | 注意单位: 元                 |
| 涨速             | float64  | -                            |
| 5分钟涨跌        | float64  | 注意单位: %                  |
| 60日涨跌幅       | float64  | 注意单位: %                  |
| 年初至今涨跌幅   | float64  | 注意单位: %                  |


### 8.28

#### Data of Core

1. 收盘价，成交量
2. 开盘价，最高，最低
3. 振幅，涨跌幅，涨跌额，换手率

#### 时序数据分析 Model

1. RNN
2. LSTM
3. GRU
   
时序大模型
1. VLDB 2025
2. ICLR 2025
3. NIPS 2024
4. ICML 2024

#### PyTorch

##### install 

- CUDA 12.9
`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129`

- CPU
`pip3 install torch torchvision`