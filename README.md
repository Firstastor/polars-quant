# polars-quant 🧮📊

> 基于 Rust + Polars 的高性能量化分析与回测工具集，提供丰富的技术指标计算和独立资金池回测引擎。

[![PyPI version](https://img.shields.io/pypi/v/polars-quant.svg)](https://pypi.org/project/polars-quant/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.90+-orange.svg)](https://www.rust-lang.org/)

## ✨ 特性

- 🚀 **高性能**：基于 Rust 实现，底层使用 Polars 数据处理，速度快、内存占用低
- 📊 **丰富指标**：提供 50+ 常用技术指标（移动平均、动量、震荡、成交量等）
- 🎯 **交易策略**：15+ 内置策略（MA、MACD、RSI、布林带、KDJ、突破、反转等）
- 🔬 **因子挖掘**：15+ 技术因子计算 + 8种专业评估指标（IC、IR、Rank IC等）
- 🎯 **股票筛选**：链式调用的选择器，支持 30+ 筛选条件组合，批量加载多种文件格式
- 💰 **独立资金池**：每只股票使用独立资金池回测，智能并行处理
- 📈 **杠杆交易**：支持融资融券回测，可设置杠杆倍数、保证金阈值、利息计算
- 🎯 **真实模拟**：支持佣金（含最低佣金）、滑点、整百股交易等实盘规则
- 📊 **详细统计**：提供夏普比率、索提诺比率、卡尔马比率等 12 类详细指标
- 🔍 **灵活分析**：支持全局汇总和单股票深度分析
- 📉 **基准对比**：支持与基准指数对比，计算Alpha、Beta和相对收益
- 💹 **每日绩效**：详细记录每日盈亏、累计收益等绩效指标

## 📦 安装

```bash
pip install polars-quant
```

或从源码安装：

```bash
git clone https://github.com/Firstastor/polars-quant.git
cd polars-quant
pip install maturin
maturin develop --release
```

## 📚 API 参考

### 一、数据处理函数

#### 1. 收益率计算

##### `returns(df, price_col, period, method, return_col)`

计算价格收益率。

**参数**：
- `df` (DataFrame): 包含价格数据的 DataFrame
- `price_col` (str): 价格列名，默认 `"close"`
- `period` (int): 收益率周期，默认 1
- `method` (str): 计算方法，默认 `"simple"`
  - `"simple"`: 简单收益率 = (price[t] - price[t-period]) / price[t-period]
  - `"log"`: 对数收益率 = ln(price[t] / price[t-period])
- `return_col` (str): 收益率列名，默认 `"return"`

**返回**：DataFrame（在原 DataFrame 基础上添加收益率列）

**示例**：
```python
import polars as pl
from polars_quant import returns

# 准备数据
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    "close": [100.0, 102.0, 101.0, 105.0]
})

# 1日简单收益率
df = returns(df, price_col="close", period=1, method="simple")
print(df)
# 输出包含 "return" 列：[None, 0.02, -0.0098, 0.0396]

# 5日对数收益率
df = returns(df, price_col="close", period=5, method="log", return_col="log_return_5d")

# 计算多个周期的收益率
df = returns(df, period=1, return_col="return_1d")
df = returns(df, period=5, return_col="return_5d")
df = returns(df, period=20, return_col="return_20d")
```

---

#### 2. 批量数据加载

##### `load(folder, file_type, prefix, suffix, has_header)`

从文件夹批量加载股票数据，支持多种文件格式。

**参数**：
- `folder` (str): 数据文件夹路径
- `file_type` (List[str], 可选): 文件类型列表，默认加载所有支持的格式
  - 支持：`"parquet"`, `"csv"`, `"json"`, `"feather"`, `"ipc"`, `"xlsx"`, `"xls"`
- `prefix` (str, 可选): 文件名前缀过滤
- `suffix` (str, 可选): 文件名后缀过滤（扩展名之前的部分）
- `has_header` (bool): 是否包含表头，默认 True

**返回**：DataFrame，格式为：
- 第一列：`date`（日期）
- 其余列：`{symbol}_{column}`（如 `AAPL_open`, `AAPL_close` 等）

**数据要求**：
- 每个文件代表一只股票
- 文件名即为股票代码（不含扩展名）
- 必须包含 `date` 列
- 自动按日期进行 Full Join 合并

**示例**：
```python
from polars_quant import load

# 1. 加载所有 parquet 文件
data = load("data/stocks", file_type=["parquet"])

# 2. 加载 CSV 和 Excel 文件
data = load("data/stocks", file_type=["csv", "xlsx"])

# 3. 使用前缀过滤（只加载上海股票）
data = load("data/stocks", prefix="SH", file_type=["parquet"])

# 4. 使用后缀过滤
data = load("data/stocks", suffix="_daily", file_type=["csv"])

# 5. 加载所有支持格式（默认）
data = load("data/stocks")

# 6. 查看结果
print(data.columns)
# ['date', 'AAPL_open', 'AAPL_high', 'AAPL_low', 'AAPL_close', 'AAPL_volume', 
#  'GOOGL_open', 'GOOGL_high', ...]

# 7. 配合 Selector 使用
from polars_quant import Selector
selector = Selector(data)
selected = selector.filter(price_min=10, volume_min=1000000).result()
```

**文件夹结构示例**：
```
data/stocks/
├── AAPL.parquet
├── GOOGL.parquet
├── MSFT.csv
├── TSLA.xlsx
└── ...
```

**返回的 DataFrame 结构**：
```python
shape: (252, 11)
┌────────────┬────────────┬────────────┬─────────────┬───┐
│ date       │ AAPL_open  │ AAPL_high  │ AAPL_close  │ … │
├────────────┼────────────┼────────────┼─────────────┼───┤
│ 2024-01-01 │ 150.0      │ 155.0      │ 153.0       │ … │
│ 2024-01-02 │ 152.0      │ 157.0      │ 154.0       │ … │
└────────────┴────────────┴────────────┴─────────────┴───┘
```

---

#### 3. 线性回归

##### `linear(df, x_cols, y_col, pred_col, resid_col, return_stats)`

对数据进行线性回归分析，支持一元和多元线性回归。

**参数**：
- `df` (DataFrame): 输入数据
- `x_cols` (List[str]): 自变量列名列表（支持多个特征）
- `y_col` (str): 因变量列名
- `pred_col` (str, 可选): 预测值列名，默认 `"pred"`
- `resid_col` (str, 可选): 残差列名，默认 `"resid"`
- `return_stats` (bool, 可选): 是否返回回归统计量，默认 False

**返回**：
- 如果 `return_stats=False`: 返回 DataFrame（包含预测值和残差列）
- 如果 `return_stats=True`: 返回 `(DataFrame, (coefficients, r_squared))`
  - `coefficients`: 回归系数列表 `[b0, b1, b2, ..., bn]`，其中 b0 为截距
  - `r_squared`: R² 决定系数

**示例**：
```python
import polars as pl
from polars_quant import linear

# 准备数据
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    "market_cap": [1000.0, 1100.0, 1050.0, 1200.0],
    "pe_ratio": [15.0, 16.0, 14.5, 17.0],
    "pb_ratio": [2.0, 2.1, 1.95, 2.2],
    "return": [0.02, 0.01, -0.005, 0.03]
})

# 1. 一元线性回归
df_result, (coeffs, r2) = linear(
    df, 
    x_cols=["market_cap"], 
    y_col="return",
    return_stats=True
)
print(f"截距: {coeffs[0]:.4f}, 斜率: {coeffs[1]:.4f}, R²: {r2:.4f}")
# 输出：截距: 0.0123, 斜率: 0.0001, R²: 0.8234

# 2. 多元线性回归（同时考虑多个因素）
df_result, (coeffs, r2) = linear(
    df,
    x_cols=["market_cap", "pe_ratio", "pb_ratio"],
    y_col="return",
    pred_col="predicted_return",
    resid_col="residual",
    return_stats=True
)
print(f"截距: {coeffs[0]:.4f}")
print(f"市值系数: {coeffs[1]:.4f}, PE系数: {coeffs[2]:.4f}, PB系数: {coeffs[3]:.4f}")
print(f"R²: {r2:.4f}")

# 3. 因子中性化（去除市值影响）
df_neutral = linear(
    df,
    x_cols=["market_cap"],
    y_col="return",
    resid_col="market_neutral_return"
)
# 使用残差作为市值中性化后的收益

# 4. 计算股票相对市场的 Beta
df_beta, (coeffs, r2) = linear(
    df,
    x_cols=["market_return"],
    y_col="stock_return",
    return_stats=True
)
beta = coeffs[1]  # 斜率即为 Beta 值
alpha = coeffs[0]  # 截距即为 Alpha 值
```

---

#### 4. 因子清洗

##### `clean(df, col, winsorize, winsorize_n, neutralize_market_cap, cap_col, neutralize_industry, industry_col, standardize)`

对因子数据进行清洗，包括去极值、中性化和标准化处理。

**参数**：
- `df` (DataFrame): 输入数据
- `col` (str): 要清洗的因子列名
- `winsorize` (str, 可选): 去极值方法，默认不去极值
  - `"mad"`: MAD（中位数绝对偏差）法
  - `"sigma"`: 标准差法，配合 `winsorize_n` 参数
  - `"percentile"`: 百分位法，配合 `winsorize_n` 参数
- `winsorize_n` (float, 可选): 去极值参数
  - sigma 方法：保留均值 ± n 倍标准差内的数据，默认 3.0
  - percentile 方法：保留 n% 到 (100-n)% 分位数内的数据，默认 1.0
- `neutralize_market_cap` (bool, 可选): 是否进行市值中性化，默认 False
- `cap_col` (str, 可选): 市值列名，当 `neutralize_market_cap=True` 时必须提供
- `neutralize_industry` (bool, 可选): 是否进行行业中性化，默认 False
- `industry_col` (str, 可选): 行业列名，当 `neutralize_industry=True` 时必须提供
- `standardize` (bool, 可选): 是否标准化，默认 False

**返回**：DataFrame（在原 DataFrame 基础上添加清洗后的因子列，列名为 `{col}_cleaned`）

**处理顺序**：去极值 → 市值中性化 → 行业中性化 → 标准化

**示例**：
```python
import polars as pl
from polars_quant import clean

# 准备数据
df = pl.DataFrame({
    "date": ["2024-01-01"] * 5,
    "stock_code": ["000001", "000002", "000003", "000004", "000005"],
    "factor": [1.5, 2.3, 10.0, 1.8, 2.1],  # 存在极值 10.0
    "market_cap": [100.0, 200.0, 150.0, 300.0, 250.0],
    "industry": ["金融", "科技", "金融", "科技", "消费"]
})

# 1. 仅标准化
df_result = clean(df, col="factor", standardize=True)

# 2. MAD 去极值 + 标准化
df_result = clean(
    df, 
    col="factor",
    winsorize="mad",
    standardize=True
)

# 3. 3 倍标准差去极值 + 标准化
df_result = clean(
    df,
    col="factor",
    winsorize="sigma",
    winsorize_n=3.0,
    standardize=True
)

# 4. 百分位法去极值（保留 1%-99%）+ 标准化
df_result = clean(
    df,
    col="factor",
    winsorize="percentile",
    winsorize_n=1.0,
    standardize=True
)

# 5. 完整清洗流程：去极值 + 市值中性化 + 行业中性化 + 标准化
df_result = clean(
    df,
    col="factor",
    winsorize="mad",
    neutralize_market_cap=True,
    cap_col="market_cap",
    neutralize_industry=True,
    industry_col="industry",
    standardize=True
)
print(df_result["factor_cleaned"])

# 6. 仅市值中性化（适合单因子测试）
df_result = clean(
    df,
    col="factor",
    neutralize_market_cap=True,
    cap_col="market_cap"
)

# 7. 仅行业中性化
df_result = clean(
    df,
    col="factor",
    neutralize_industry=True,
    industry_col="industry",
    standardize=True
)
```

---

### 二、回测类 (Backtest)

#### 1. 构造函数

##### `Backtest(prices, buy_signals, sell_signals, initial_capital, position_size, leverage, margin_call_threshold, interest_rate, commission_rate, min_commission, slippage, benchmark)`

创建回测实例。

**参数**：
- `prices` (DataFrame): 价格数据，第一列为日期，其余列为各股票价格
- `buy_signals` (DataFrame): 买入信号，第一列为日期，其余列为布尔值（True 表示买入，False 表示不买入）
- `sell_signals` (DataFrame): 卖出信号，第一列为日期，其余列为布尔值（True 表示卖出，False 表示不卖出）
- `initial_capital` (float): 初始资金，默认 100000.0
- `position_size` (float): 每次交易仓位大小（0.0-1.0），默认 1.0
- `leverage` (float): 杠杆倍数，默认 1.0（无杠杆）
- `margin_call_threshold` (float): 保证金预警阈值，默认 0.3
- `interest_rate` (float): 融资年化利率，默认 0.06
- `commission_rate` (float): 佣金费率，默认 0.0003（万三）
- `min_commission` (float): 最低佣金，默认 5.0 元
- `slippage` (float): 滑点，默认 0.0（0.001 表示 0.1%）
- `benchmark` (DataFrame, 可选): 基准指数数据，第一列为日期，第二列为基准价格

**示例**：
```python
from polars_quant import Backtest

# 准备基准数据（如上证指数）
benchmark_df = pl.DataFrame({
    "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    "SH000001": [3100.0, 3120.0, 3095.0]
})

bt = Backtest(
    prices=prices_df,
    buy_signals=buy_df,
    sell_signals=sell_df,
    initial_capital=100000.0,
    leverage=2.0,             # 2倍杠杆
    commission_rate=0.0003,   # 万三
    min_commission=5.0,       # 最低5元
    slippage=0.001,           # 0.1%滑点
    benchmark=benchmark_df    # 基准对比
)
```

---

#### 2. 回测执行

##### `run()`

执行回测。

**返回**：None

**特性**：
- 自动识别股票数量，智能选择串行/并行策略
- <4 只股票：串行执行（避免线程开销）
- ≥4 只股票：并行执行（动态线程池）
- 每只股票独立资金池，互不影响

**示例**：
```python
bt.run()
```

---

#### 3. 结果查询 - 全局数据

##### `get_daily_records()`

获取所有股票的每日资金记录。

**返回**：DataFrame，包含以下列：
- `symbol` (str): 股票代码
- `date` (str): 日期
- `cash` (float): 现金
- `stock_value` (float): 持仓市值
- `total_value` (float): 总资产

**示例**：
```python
daily = bt.get_daily_records()
print(daily)
```

---

##### `get_position_records()`

获取所有股票的交易记录。

**返回**：DataFrame，包含以下列：
- `symbol` (str): 股票代码
- `entry_date` (str): 开仓日期
- `entry_price` (float): 开仓价格
- `quantity` (float): 持仓数量（整百股）
- `exit_date` (str): 平仓日期
- `exit_price` (float): 平仓价格
- `pnl` (float): 盈亏金额
- `pnl_pct` (float): 盈亏百分比
- `holding_days` (int): 持仓天数

**示例**：
```python
positions = bt.get_position_records()
print(positions)
```

---

##### `get_performance_metrics()`

获取每日绩效指标（包括每日盈亏、累计收益、与基准对比）。

**返回**：DataFrame，包含以下列：
- `date` (str): 日期
- `portfolio_value` (float): 组合总市值
- `daily_pnl` (float): 当日盈亏金额
- `daily_return_pct` (float): 当日收益率（%）
- `cumulative_pnl` (float): 累计盈亏金额
- `cumulative_return_pct` (float): 累计收益率（%）

**如果提供了基准数据，还包括**：
- `benchmark_return_pct` (float): 基准当日收益率（%）
- `alpha_pct` (float): 超额收益率（策略收益 - 基准收益）（%）
- `relative_return_pct` (float): 相对基准的累计收益率（%）
- `beta` (float): Beta系数（策略相对基准的系统性风险敞口）
  - Beta < 1: 策略波动小于基准，防御性
  - Beta = 1: 策略波动与基准一致
  - Beta > 1: 策略波动大于基准，进攻性

**示例**：
```python
# 获取绩效指标
metrics = bt.get_performance_metrics()
print(metrics)

# 查看Beta值
if "beta" in metrics.columns:
    beta = metrics["beta"][0]
    print(f"策略Beta系数: {beta:.4f}")

# 分析每日盈亏
daily_stats = metrics.select([
    pl.col("daily_pnl").mean().alias("平均日盈亏"),
    pl.col("daily_pnl").max().alias("最大日盈利"),
    pl.col("daily_pnl").min().alias("最大日亏损"),
])
print(daily_stats)

# 如果有基准数据，对比分析
if "benchmark_return_pct" in metrics.columns:
    comparison = metrics.select([
        pl.col("date"),
        pl.col("cumulative_return_pct").alias("策略累计收益(%)"),
        pl.col("benchmark_return_pct").cum_sum().alias("基准累计收益(%)"),
        pl.col("relative_return_pct").alias("相对收益(%)"),
    ])
    print(comparison)
```

---

##### `summary()`

打印所有股票的综合统计摘要。

**返回**：None（直接打印输出）

**包含统计项**：
1. **基本信息**：回测期间、初始资金、最终资金、总盈亏、仓位大小、执行时间
2. **收益指标**：总收益率、年化收益率、日均收益率
3. **风险指标**：最大回撤、最大回撤持续、日波动率、年化波动率
4. **风险调整收益**：夏普比率、索提诺比率、卡尔马比率
5. **交易统计**：总交易次数、盈利/亏损交易、胜率、盈亏比
6. **盈利分析**：总盈利、平均盈利、最大单笔盈利、平均盈利持仓
7. **亏损分析**：总亏损、平均亏损、最大单笔亏损、平均亏损持仓
8. **持仓分析**：平均持仓周期、总持仓天数、最大连续盈利/亏损
9. **交易成本**：总交易额、总手续费、手续费占比
10. **资金使用**：平均单笔交易额、资金使用率
11. **日收益分析**：正收益天数、负收益天数、日胜率
12. **股票维度**：交易股票数量、表现最好/最差的股票
13. **基准对比**（仅当提供了 `benchmark` 参数时）：
    - **收益对比**：策略累计收益率 vs 基准累计收益率、超额收益
    - **风险分析**：Alpha（超额收益的平均值）、Beta（市场敏感度）、IR（信息比率）
    - **相对表现**：跑赢基准天数、跑赢基准比例、综合评价

**示例**：
```python
# 无基准对比
bt.summary()

# 有基准对比（需在创建 Backtest 时提供 benchmark 参数）
benchmark_df = pl.DataFrame({
    "date": ["2023-01-01", "2023-01-02"],
    "SH000001": [3100.0, 3120.0]
})
bt = Backtest(prices, buy_signals, sell_signals, benchmark=benchmark_df)
bt.run()
bt.summary()  # 将额外显示基准对比分析
```

---

#### 4. 结果查询 - 单只股票

##### `get_stock_performance(symbol)`

获取指定股票的每日绩效指标。

**参数**：
- `symbol` (str): 股票代码

**返回**：DataFrame，包含以下列：
- `symbol` (str): 股票代码
- `date` (str): 日期
- `stock_value` (float): 股票总资产
- `daily_pnl` (float): 当日盈亏金额
- `daily_return_pct` (float): 当日收益率（%）
- `cumulative_pnl` (float): 累计盈亏金额
- `cumulative_return_pct` (float): 累计收益率（%）

**如果提供了基准数据，还包括**：
- `benchmark_return_pct` (float): 基准当日收益率（%）
- `alpha_pct` (float): 超额收益率（%）
- `relative_return_pct` (float): 相对收益率（%）

**示例**：
```python
# 获取AAPL的每日绩效
aapl_perf = bt.get_stock_performance("AAPL")
print(aapl_perf)

# 查看Beta值（从get_stock_summary中获取）
print(bt.get_stock_summary("AAPL"))  # 包含Beta信息

# 分析统计
stats = aapl_perf.select([
    pl.col("daily_pnl").mean().alias("平均日盈亏"),
    pl.col("cumulative_return_pct").tail(1).alias("总收益率")
])
```

---

##### `get_stock_daily(symbol)`

获取指定股票的每日资金记录。

**参数**：
- `symbol` (str): 股票代码

**返回**：DataFrame，列同 `get_daily_records()`

**示例**：
```python
stock_a_daily = bt.get_stock_daily("AAPL")
print(stock_a_daily)
```

---

##### `get_stock_positions(symbol)`

获取指定股票的交易记录。

**参数**：
- `symbol` (str): 股票代码

**返回**：DataFrame，列同 `get_position_records()`

**示例**：
```python
stock_a_positions = bt.get_stock_positions("AAPL")
print(stock_a_positions)
```

---

##### `get_stock_summary(symbol)`

打印指定股票的统计摘要。

**参数**：
- `symbol` (str): 股票代码

**返回**：str（统计摘要字符串）

**示例**：
```python
print(bt.get_stock_summary("AAPL"))
```

---

### 三、股票选择器 (Selector)

股票选择器提供链式调用的股票筛选功能，支持从文件夹批量加载数据，并使用 30+ 筛选参数进行多条件组合筛选。

#### 1. 创建选择器

##### `Selector(ohlcv_data)`

从 DataFrame 创建选择器。

**参数**：
- `ohlcv_data` (DataFrame): OHLCV 数据，要求包含列：
  - `date`: 日期列
  - `{symbol}_open`: 各股票的开盘价
  - `{symbol}_high`: 各股票的最高价
  - `{symbol}_low`: 各股票的最低价
  - `{symbol}_close`: 各股票的收盘价
  - `{symbol}_volume`: 各股票的成交量

**示例**：
```python
from polars_quant import Selector
import polars as pl

df = pl.DataFrame({
    "date": ["2023-01-01", "2023-01-02"],
    "AAPL_open": [150.0, 152.0],
    "AAPL_high": [155.0, 157.0],
    "AAPL_low": [149.0, 151.0],
    "AAPL_close": [153.0, 154.0],
    "AAPL_volume": [1000000.0, 1200000.0]
})

selector = Selector(df)
```

---

#### 2. 筛选方法

##### `filter(...)`

筛选股票，支持链式调用。包含 30+ 筛选参数，所有参数均为可选。

**价格筛选**：
- `price_min` (float): 最低价格
- `price_max` (float): 最高价格

**成交量筛选**：
- `volume_min` (float): 最小成交量
- `volume_avg_days` (int): 平均成交量天数

**收益率筛选**：
- `return_min` (float): 最小收益率（百分比）
- `return_max` (float): 最大收益率（百分比）
- `return_period` (int): 收益率周期，默认 1

**波动率筛选**：
- `volatility_min` (float): 最小年化波动率（百分比）
- `volatility_max` (float): 最大年化波动率（百分比）
- `volatility_period` (int): 波动率计算周期，默认 20

**均线筛选**：
- `ma_above` (int): 价格在 N 日均线之上
- `ma_below` (int): 价格在 N 日均线之下

**RSI 筛选**：
- `rsi_min` (float): 最小 RSI 值
- `rsi_max` (float): 最大 RSI 值
- `rsi_period` (int): RSI 周期，默认 14

**MACD 筛选**：
- `macd` (str): MACD 条件 - `"golden_cross"` (金叉), `"death_cross"` (死叉), `"above_zero"` (零轴上方), `"below_zero"` (零轴下方)
- `macd_fast` (int): 快线周期，默认 12
- `macd_slow` (int): 慢线周期，默认 26
- `macd_signal` (int): 信号线周期，默认 9

**KDJ 筛选**：
- `kdj` (str): KDJ 条件 - `"golden_cross"` (金叉), `"death_cross"` (死叉), `"oversold"` (超卖), `"overbought"` (超买)
- `kdj_period` (int): KDJ 周期，默认 9

**涨跌停筛选**：
- `limit_type` (str): 涨跌停类型 - `"limit_up"` (涨停), `"limit_down"` (跌停), `"near_limit_up"` (接近涨停), `"near_limit_down"` (接近跌停)
- `limit_threshold` (float): 涨跌停阈值，默认 9.9

**成交量变化筛选**：
- `volume_change` (str): 成交量变化类型 - `"volume_surge"` (放量), `"volume_shrink"` (缩量)
- `volume_multiplier` (float): 成交量倍数，默认 2.0
- `volume_change_days` (int): 平均成交量天数，默认 5

**连续上涨/下跌筛选**：
- `consecutive` (str): 连续类型 - `"consecutive_up"` (连续上涨), `"consecutive_down"` (连续下跌)
- `consecutive_days` (int): 连续天数，默认 3

**突破筛选**：
- `breakout` (str): 突破类型 - `"breakout_high"` (突破新高), `"breakdown_low"` (跌破新低)
- `breakout_period` (int): 突破周期，默认 20

**返回**：self (支持链式调用)

**示例**：
```python
# 单条件筛选
result = selector.filter(price_min=10, price_max=100).result()

# 多条件筛选
result = selector.filter(
    price_min=10,
    volume_min=1000000,
    ma_above=20,
    rsi_min=30, rsi_max=70
).result()

# 链式调用
result = (
    selector
    .filter(price_min=10, price_max=100)
    .filter(volume_min=1000000)
    .filter(ma_above=20)
    .filter(volatility_min=15, volatility_max=40)
    .sort(by="return_5d", ascending=False, top_n=10)
    .result()
)

# 技术指标筛选
macd_golden = selector.filter(macd="golden_cross").result()
kdj_oversold = selector.filter(kdj="oversold").result()
limit_up = selector.filter(limit_type="limit_up").result()

# 放量突破
breakout = selector.filter(
    volume_change="volume_surge",
    volume_multiplier=2.0,
    breakout="breakout_high"
).result()
```

---

##### `sort(by, ascending, top_n)`

对股票进行排序并可选择取 TopN。

**参数**：
- `by` (str): 排序指标 - `"price"`, `"return_1d"`, `"return_5d"`, `"return_20d"`, `"volume"`, `"volatility"`
- `ascending` (bool): 是否升序，默认 False（降序）
- `top_n` (int, 可选): 取前 N 个股票

**返回**：self (支持链式调用)

**示例**：
```python
# 按5日收益率降序排序，取前10
selector.sort(by="return_5d", ascending=False, top_n=10)

# 按波动率升序排序
selector.sort(by="volatility", ascending=True)
```

---

##### `result()`

获取筛选结果（股票代码列表）。

**返回**：List[str]

**示例**：
```python
symbols = selector.filter(price_min=10).result()
print(symbols)  # ['SH600000', 'SH600001', ...]
```

---

##### `reset()`

重置筛选条件。

**返回**：self (支持链式调用)

**示例**：
```python
# 重置后重新筛选
selector.filter(price_min=10).reset().filter(volume_min=1000000)
```

---

##### `info()`

获取股票详细信息。

**返回**：DataFrame，包含以下 15 列：
- `symbol`: 股票代码
- `price`: 最新收盘价
- `open`: 最新开盘价
- `high`: 最新最高价
- `low`: 最新最低价
- `volume`: 最新成交量
- `return_1d`: 1日收益率（%）
- `return_5d`: 5日收益率（%）
- `return_20d`: 20日收益率（%）
- `volatility`: 年化波动率（%，基于20日数据）
- `ma_5`: 5日均线
- `ma_10`: 10日均线
- `ma_20`: 20日均线
- `volume_ratio`: 量比（当前成交量 / 5日平均成交量）
- `amplitude`: 振幅（%，当日最高最低价差 / 收盘价）

**示例**：
```python
df = selector.filter(price_min=10).info()
print(df)
# 输出包含15列详细信息的DataFrame
```

---

### 四、交易策略 (Strategy)

交易策略模块提供 15 种常用交易策略，每个策略返回包含 `buy_signal` 和 `sell_signal` 列的 DataFrame。

#### 1. 创建策略实例

```python
from polars_quant import Strategy

strategy = Strategy()
```

#### 2. 策略方法

##### MA 均线策略

**`ma(df, price_col, fast_period, slow_period, ma_type, trend_period, trend_filter, slope_filter, distance_pct)`**

MA均线策略，支持多种过滤条件。

**参数**：
- `df` (DataFrame): 包含价格数据的DataFrame
- `price_col` (str): 价格列名，默认 `"close"`
- `fast_period` (int): 快线周期，默认 10
- `slow_period` (int): 慢线周期，默认 20
- `ma_type` (str): 均线类型，默认 `"sma"`，可选 `"sma"`, `"ema"`, `"wma"`, `"dema"`, `"tema"`
- `trend_period` (int): 趋势过滤均线周期，默认 0（不使用）
- `trend_filter` (bool): 是否启用趋势过滤，默认 False
- `slope_filter` (bool): 是否启用斜率过滤，默认 False
- `distance_pct` (float): 价格与均线最小距离百分比，默认 0.0（不使用）

**返回**：包含 `buy_signal` 和 `sell_signal` 列的DataFrame

**示例**：
```python
# 简单MA策略
signals = strategy.ma(df, fast_period=5, slow_period=10)

# 带趋势过滤的MA策略
signals = strategy.ma(df, fast_period=10, slow_period=20,
                      trend_period=60, trend_filter=True)

# EMA策略
signals = strategy.ma(df, ma_type="ema", fast_period=12, slow_period=26)
```

---

##### MACD 策略

**`macd(df, price_col, fast_period, slow_period, signal_period)`**

**参数**：
- `fast_period` (int): 快线周期，默认 12
- `slow_period` (int): 慢线周期，默认 26
- `signal_period` (int): 信号线周期，默认 9

**示例**：
```python
signals = strategy.macd(df)
```

---

##### RSI 策略

**`rsi(df, price_col, period, oversold, overbought)`**

**参数**：
- `period` (int): RSI周期，默认 14
- `oversold` (float): 超卖阈值，默认 30.0
- `overbought` (float): 超买阈值，默认 70.0

**示例**：
```python
signals = strategy.rsi(df, period=14, oversold=30, overbought=70)
```

---

##### 其他策略方法

- **`bband(df, ...)`** - 布林带策略
- **`stoch(df, ...)`** - KDJ/随机指标策略
- **`cci(df, ...)`** - CCI顺势指标策略
- **`adx(df, ...)`** - ADX趋势强度策略
- **`breakout(df, ...)`** - 突破策略（Donchian Channel）
- **`reversion(df, ...)`** - 均值回归策略
- **`volume(df, ...)`** - 成交量突破策略
- **`grid(df, ...)`** - 网格交易策略
- **`gap(df, ...)`** - 跳空缺口策略
- **`pattern(df, ...)`** - K线形态策略
- **`trend(df, ...)`** - 多均线趋势策略

详细参数请参考 API 文档（`polars_quant.pyi`）。

#### 3. 策略组合示例

```python
import polars as pl
from polars_quant import Strategy

# 准备数据
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02"],
    "open": [100.0, 102.0],
    "high": [105.0, 106.0],
    "low": [99.0, 101.0],
    "close": [103.0, 104.0],
    "volume": [1000000, 1200000]
})

strategy = Strategy()

# 1. 单一策略
ma_signals = strategy.ma(df, fast_period=5, slow_period=10)

# 2. 组合策略（同时满足多个条件）
ma_sig = strategy.ma(df, fast_period=5, slow_period=10)
rsi_sig = strategy.rsi(df, period=14, oversold=30, overbought=70)

# 买入信号：MA金叉且RSI不超买
combined = ma_sig.with_columns([
    (pl.col("buy_signal") & rsi_sig["buy_signal"]).alias("buy_signal"),
    (pl.col("sell_signal") | rsi_sig["sell_signal"]).alias("sell_signal")
])

# 3. 趋势过滤策略
# 只在长期趋势向上时交易
trend_ma = strategy.ma(df, fast_period=10, slow_period=20,
                       trend_period=60, trend_filter=True,
                       slope_filter=True)
```

---

### 五、技术指标函数

所有指标函数接受 Polars Series 作为输入，返回 Polars Series 或元组。

#### 1. 趋势指标 (Overlap Studies)

##### 移动平均类

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `sma(series, period)` | 简单移动平均 | series: 价格序列<br>period: 周期 | Series |
| `ema(series, period)` | 指数移动平均 | series: 价格序列<br>period: 周期 | Series |
| `wma(series, period)` | 加权移动平均 | series: 价格序列<br>period: 周期 | Series |
| `dema(series, period)` | 双重指数移动平均 | series: 价格序列<br>period: 周期 | Series |
| `tema(series, period)` | 三重指数移动平均 | series: 价格序列<br>period: 周期 | Series |
| `trima(series, period)` | 三角移动平均 | series: 价格序列<br>period: 周期 | Series |
| `kama(series, period)` | 考夫曼自适应移动平均 | series: 价格序列<br>period: 周期 | Series |
| `ma(series, period, ma_type)` | 通用移动平均 | series: 价格序列<br>period: 周期（默认20）<br>ma_type: 类型（默认"SMA"，支持"SMA"/"EMA"/"WMA"） | Series |
| `t3(series, period, vfactor)` | T3 移动平均 | series: 价格序列<br>period: 周期<br>vfactor: 体积因子 | Series |

**ma 函数说明**：
- **SMA**: 简单移动平均，所有值权重相等
- **EMA**: 指数移动平均，对近期数据赋予更高权重
- **WMA**: 加权移动平均，线性加权（权重为 1, 2, 3, ..., period）

**示例**：
```python
import polars as pl
from polars_quant import sma, ema, wma, ma

df = pl.DataFrame({"close": [100, 102, 101, 105, 107, 110]})
df = df.with_columns([
    sma(pl.col("close"), 3).alias("sma_3"),
    ema(pl.col("close"), 3).alias("ema_3"),
    wma(pl.col("close"), 3).alias("wma_3"),
    # 使用 ma 通用函数
    ma(pl.col("close"), 5).alias("ma_sma_5"),  # 默认 SMA
    ma(pl.col("close"), 5, "EMA").alias("ma_ema_5"),
    ma(pl.col("close"), 5, "WMA").alias("ma_wma_5"),
])
```

---

##### 布林带与通道

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `bband(series, period, std_dev)` | 布林带 | series: 价格序列<br>period: 周期<br>std_dev: 标准差倍数 | (upper, middle, lower) |

**示例**：
```python
from polars_quant import bband

upper, middle, lower = bband(pl.col("close"), 20, 2.0)
df = df.with_columns([
    upper.alias("bb_upper"),
    middle.alias("bb_middle"),
    lower.alias("bb_lower"),
])
```

---

##### MAMA 自适应均线

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `mama(series, fast_limit, slow_limit)` | MESA 自适应移动平均 | series: 价格序列<br>fast_limit: 快速限制<br>slow_limit: 慢速限制 | (mama, fama) |

**示例**：
```python
from polars_quant import mama

mama_line, fama_line = mama(pl.col("close"), 0.5, 0.05)
```

---

##### 可变周期均线

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `mavp(series, periods, min_period, max_period)` | 可变周期移动平均 | series: 价格序列<br>periods: 周期序列<br>min_period: 最小周期<br>max_period: 最大周期 | Series |

---

##### 价格位置指标

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `midpoint(series, period)` | 中点价格 | series: 价格序列<br>period: 周期 | Series |
| `midprice_hl(high, low, period)` | 最高最低中点 | high: 最高价<br>low: 最低价<br>period: 周期 | Series |

---

##### 抛物线指标

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `sar(high, low, acceleration, maximum)` | 抛物线转向指标 | high: 最高价<br>low: 最低价<br>acceleration: 加速因子<br>maximum: 最大值 | Series |
| `sarext(high, low, startvalue, offsetonreverse, accelerationinitlong, accelerationlong, accelerationmaxlong, accelerationinitshort, accelerationshort, accelerationmaxshort)` | 扩展抛物线指标 | 多个参数配置 | Series |

**示例**：
```python
from polars_quant import sar

sar_values = sar(pl.col("high"), pl.col("low"), 0.02, 0.2)
```

---

#### 2. 动量指标 (Momentum Indicators)

##### ADX 系列（平均趋向指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `adx(high, low, close, period)` | 平均趋向指标 | high/low/close: 价格<br>period: 周期 | Series |
| `adxr(high, low, close, period)` | 平均趋向评估 | 同上 | Series |
| `plus_di(high, low, close, period)` | 正向指标 | 同上 | Series |
| `minus_di(high, low, close, period)` | 负向指标 | 同上 | Series |
| `plus_dm(high, low, period)` | 正向动向 | high/low: 价格<br>period: 周期 | Series |
| `minus_dm(high, low, period)` | 负向动向 | 同上 | Series |
| `dx(high, low, close, period)` | 方向性指标 | high/low/close: 价格<br>period: 周期 | Series |

**示例**：
```python
from polars_quant import adx, plus_di, minus_di

df = df.with_columns([
    adx(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("adx"),
    plus_di(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("plus_di"),
    minus_di(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("minus_di"),
])
```

---

##### APO/PPO（价格震荡器）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `apo(series, fast_period, slow_period)` | 绝对价格震荡器 | series: 价格序列<br>fast_period: 快周期<br>slow_period: 慢周期 | Series |
| `ppo(series, fast_period, slow_period)` | 百分比价格震荡器 | 同上 | Series |

---

##### Aroon 系列

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `aroon(high, low, period)` | Aroon 指标 | high/low: 价格<br>period: 周期 | (aroon_up, aroon_down) |
| `aroonosc(high, low, period)` | Aroon 震荡器 | 同上 | Series |

---

##### BOP（均势指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `bop(open, high, low, close)` | 均势指标 | OHLC 价格 | Series |

---

##### CCI（商品通道指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `cci(high, low, close, period)` | 商品通道指标 | high/low/close: 价格<br>period: 周期 | Series |

**示例**：
```python
from polars_quant import cci

cci_values = cci(pl.col("high"), pl.col("low"), pl.col("close"), 14)
```

---

##### CMO（钱德动量摆动指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `cmo(series, period)` | 钱德动量摆动指标 | series: 价格序列<br>period: 周期 | Series |

---

##### MACD 系列

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `macd(series, fast, slow, signal)` | MACD 指标 | series: 价格序列<br>fast/slow/signal: 周期 | (macd, signal, hist) |
| `macdext(series, fast, slow, signal, fast_matype, slow_matype, signal_matype)` | 扩展 MACD | 同上 + MA 类型 | (macd, signal, hist) |
| `macdfix(series, signal)` | 固定参数 MACD | series: 价格序列<br>signal: 信号周期 | (macd, signal, hist) |

**示例**：
```python
from polars_quant import macd

macd_line, signal_line, hist = macd(pl.col("close"), 12, 26, 9)
df = df.with_columns([
    macd_line.alias("macd"),
    signal_line.alias("signal"),
    hist.alias("hist"),
])
```

---

##### MFI（资金流量指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `mfi(high, low, close, volume, period)` | 资金流量指标 | HLCV 数据<br>period: 周期 | Series |

---

##### MOM（动量指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `mom(series, period)` | 动量指标 | series: 价格序列<br>period: 周期 | Series |

---

##### ROC 系列（变化率）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `roc(series, period)` | 变化率 | series: 价格序列<br>period: 周期 | Series |
| `rocp(series, period)` | 百分比变化率 | 同上 | Series |
| `rocr(series, period)` | 比率变化率 | 同上 | Series |
| `rocr100(series, period)` | 百倍比率变化率 | 同上 | Series |

---

##### RSI（相对强弱指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `rsi(series, period)` | 相对强弱指标 | series: 价格序列<br>period: 周期（通常14） | Series |

**示例**：
```python
from polars_quant import rsi

rsi_values = rsi(pl.col("close"), 14)
df = df.with_columns(rsi_values.alias("rsi"))
```

---

##### Stochastic 系列（随机指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `stoch(high, low, close, k_period, d_period)` | 慢速随机指标 | HLC 价格<br>k/d 周期 | (k, d) |
| `stochf(high, low, close, k_period, d_period)` | 快速随机指标 | 同上 | (k, d) |
| `stochrsi(series, period, k_period, d_period)` | RSI 随机指标 | series: 价格序列<br>period/k/d: 周期 | (k, d) |

**示例**：
```python
from polars_quant import stoch

k, d = stoch(pl.col("high"), pl.col("low"), pl.col("close"), 14, 3)
df = df.with_columns([k.alias("stoch_k"), d.alias("stoch_d")])
```

---

##### TRIX（三重指数平滑移动平均）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `trix(series, period)` | TRIX 指标 | series: 价格序列<br>period: 周期 | Series |

---

##### ULTOSC（终极震荡器）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `ultosc(high, low, close, period1, period2, period3)` | 终极震荡器 | HLC 价格<br>三个周期 | Series |

---

##### WILLR（威廉指标）

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `willr(high, low, close, period)` | 威廉 %R | HLC 价格<br>period: 周期 | Series |

**示例**：
```python
from polars_quant import willr

willr_values = willr(pl.col("high"), pl.col("low"), pl.col("close"), 14)
```

---

#### 3. 成交量指标 (Volume Indicators)

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `ad(high, low, close, volume)` | 累积/派发线 | HLCV 数据 | Series |
| `adosc(high, low, close, volume, fast, slow)` | 累积/派发震荡器 | HLCV 数据<br>fast/slow: 周期 | Series |
| `obv(close, volume)` | 能量潮指标 | close/volume: 价格和成交量 | Series |

**示例**：
```python
from polars_quant import ad, obv

df = df.with_columns([
    ad(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("ad"),
    obv(pl.col("close"), pl.col("volume")).alias("obv"),
])
```

---

#### 4. 波动率指标 (Volatility Indicators)

| 函数 | 说明 | 参数 | 返回 |
|------|------|------|------|
| `trange(high, low, close)` | 真实范围 | HLC 价格 | Series |

**示例**：
```python
from polars_quant import trange

tr = trange(pl.col("high"), pl.col("low"), pl.col("close"))
```

---

## 🚀 快速开始

```python
import polars as pl
from polars_quant import Backtest, Selector, Strategy, sma, rsi, load

# 1. 股票筛选
# 使用 load 函数加载数据
data = load("data/stocks", file_type=["parquet"])
selector = Selector(data)
selected = selector.filter(
    price_min=10.0,
    price_max=100.0,
    volume_min=1000000,
    return_min=0.02,  # 收益率 > 2%
    return_period=5
).sort(by="return_5d", ascending=False, top_n=10).result()

# 2. 使用交易策略生成信号
strategy = Strategy()
signals = strategy.ma(df, fast_period=5, slow_period=20)
# 或使用组合策略
ma_signals = strategy.ma(df, fast_period=10, slow_period=20)
rsi_signals = strategy.rsi(df, period=14, oversold=30, overbought=70)

# 3. 或手动计算技术指标
df = pl.read_parquet("stock_data.parquet")
df = df.with_columns([
    sma(pl.col("close"), 20).alias("ma20"),
    rsi(pl.col("close"), 14).alias("rsi")
])

# 4. 生成买卖信号
buy_signals = df.select([
    pl.col("date"),
    ((pl.col("close") > pl.col("ma20")) & (pl.col("rsi") < 30)).alias("AAPL")
])

sell_signals = df.select([
    pl.col("date"),
    ((pl.col("close") < pl.col("ma20")) | (pl.col("rsi") > 70)).alias("AAPL")
])

# 5. 回测
bt = Backtest(
    prices=df.select(["date", "AAPL"]),
    buy_signals=buy_signals,
    sell_signals=sell_signals,
    initial_capital=100000.0,
    leverage=2.0  # 杠杆倍数
)
bt.run()
bt.summary()  # 查看详细统计
```

---

## 🔬 因子挖掘与评估

polars-quant 提供了强大的因子计算和评估工具，支持通用因子计算方法和专业的因子评估指标，适用于基本面、宏观、技术等各类因子研究。

### 因子分析完整流程

```python
import polars as pl
from polars_quant import Factor

# 1. 准备数据
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "symbol": ["AAPL", "AAPL", "AAPL"],
    "close": [150.0, 152.0, 148.0],
    "volume": [1000000, 1200000, 900000],
    "market_cap": [2.5e12, 2.52e12, 2.46e12],
    "net_profit": [100e9, 98e9, 102e9],
    "pe_ratio": [25.5, 26.0, 24.8]
})

# 2. 创建Factor实例
factor = Factor()

# 3. 计算通用因子
df = factor.ratio(df, "market_cap", "net_profit", "pe_ratio")  # 市盈率
df = factor.diff(df, "pe_ratio", "pe_prev", "pe_growth", normalize=True)  # PE增长率
df = factor.weighted(df, "pe_ratio", "market_cap", "weighted_pe")  # 市值加权PE
df = factor.normalize(df, "pe_ratio", "zscore", "pe_zscore")  # 标准化
df = factor.rank(df, "market_cap", "cap_rank", False, True)  # 市值排名

# 4. 计算技术因子
df = factor.moving_average(df, "close", 20, "ma20")  # 移动平均
df = factor.momentum(df, "close", 20, "mom20")  # 动量
df = factor.volatility(df, "close", 20, "vol20")  # 波动率

# 5. 评估因子（需要先添加收益率列 "next_return"）
ic_result = factor.ic(df, "pe_ratio", "next_return")  # IC值
rank_ic_result = factor.rank_ic(df, "pe_ratio", "next_return")  # Rank IC
quantile_result = factor.quantile(df, "pe_ratio", "next_return", 5)  # 分层分析
ls_result = factor.long_short(df, "pe_ratio", "next_return", 0.2, 0.2)  # 多空收益

print(f"IC结果: {ic_result}")
print(f"Rank IC结果: {rank_ic_result}")
```

### 通用因子计算示例

```python
# 1. 比值因子（适用于基本面分析）
df = factor.ratio(df, "market_cap", "book_value", "pb_ratio")  # 市净率
df = factor.ratio(df, "current_assets", "current_liabilities", "current_ratio")  # 流动比率

# 2. 差值因子（适用于增长分析）
df = factor.diff(df, "revenue", "revenue_last_year", "revenue_growth", normalize=True)
df = factor.diff(df, "net_profit", "operating_profit", "profit_diff")

# 3. 加权因子（适用于市值加权）
df = factor.weighted(df, "pe_ratio", "market_cap", "weighted_pe")
df = factor.weighted(df, "roe", "market_cap", "weighted_roe", ["industry"])

# 4. 标准化因子
df = factor.normalize(df, "pb_ratio", "zscore", "pb_zscore")  # Z-Score
df = factor.normalize(df, "market_cap", "minmax", "cap_scaled")  # MinMax
df = factor.normalize(df, "pe_ratio", "quantile", "pe_quantile")  # 分位数

# 5. 排名因子
df = factor.rank(df, "pe_ratio", "pe_rank", True, False)  # 排名（1-N）
df = factor.rank(df, "market_cap", "cap_pct", False, True)  # 百分比排名（0-1）
```

### 可用的因子方法

**通用因子计算（5种）**：
- `ratio()` - 比值计算（PE、PB、财务比率等）
- `diff()` - 差值计算（增长率、利润差等）
- `weighted()` - 加权计算（市值加权、成交量加权等）
- `normalize()` - 标准化（Z-Score、MinMax、分位数等）
- `rank()` - 排名（升序/降序、数值/百分比）

**常用技术因子（5种）**：
- `moving_average()` - 移动平均
- `momentum()` - 动量因子
- `volatility()` - 波动率因子
- `skewness()` - 偏度因子
- `relative_strength()` - 相对强弱

**评估指标（8种）**：
- `ic()` - IC值（信息系数，Pearson相关）
- `ir()` - IR值（信息比率）
- `rank_ic()` - Rank IC（Spearman秩相关）
- `quantile()` - 分层分析
- `coverage()` - 因子覆盖率
- `ic_win_rate()` - IC胜率
- `long_short()` - 多空收益
- `turnover()` - 因子换手率

---

### 因子检验三步流程

polars-quant 提供完整的因子检验框架，包括单因子检验、多因子回归分析和稳健性检验。

#### 前置处理：因子预处理

在进行因子检验前，通常需要对多因子进行预处理，消除因子间的线性相关性：

```python
from polars_quant import Factor

factor = Factor()

# 正交化处理（直接修改因子值）
df = factor.clean(
    df,
    factor_cols=["size", "value", "momentum"],
    method="orthogonalize"
)
# 结果: size不变，value和momentum被正交化

# 中性化处理（保留原始值，添加残差列）
df = factor.clean(
    df,
    factor_cols=["size", "value", "momentum"],
    method="neutralize"
)
# 结果: 新增 value_residual 和 momentum_residual 列
```

#### 第一步：单因子检验

评估单个因子的预测能力和收益特征：

```python
# 1. IC检验（信息系数）
ic_result = factor.ic_test(
    df,
    factor_col="value",
    return_col="next_return",
    time_col="date",
    method="pearson"  # 或 "spearman"
)
# 返回: time, ic, t_stat, p_value

# 2. 投资组合排序
sorts_result = factor.portfolio_sorts(
    df,
    factor_col="value",
    return_col="next_return",
    time_col="date",
    n_quantiles=5
)
# 返回: quantile, mean_return, std_return, sharpe

# 3. 因子收益率（截面回归）
fr_result = factor.factor_return(
    df,
    factor_col="value",
    return_col="next_return",
    time_col="date"
)
# 返回: time, factor_return, t_stat, p_value

# 4. IC衰减分析
decay_result = factor.ic_decay(
    df,
    factor_col="value",
    return_col="next_return",
    time_col="date",
    id_col="stock_code",
    max_lag=10
)
# 返回: lag, ic, t_stat, p_value
```

#### 第二步：多因子回归分析

分析多个因子的联合解释能力和相对重要性：

```python
# 1. Fama-MacBeth回归
fm_result = factor.fama_macbeth(
    df,
    factor_cols=["size", "value", "momentum"],
    return_col="next_return",
    time_col="date"
)
# 返回: factor, mean_coef, t_stat, p_value

# 2. 时间序列回归
ts_result = factor.time_series_regression(
    df,
    factor_cols=["size", "value", "momentum"],
    return_col="next_return",
    id_col="stock_code"
)
# 返回: id, factor, coefficient, t_stat, p_value, r_squared

# 3. 因子模拟组合
fmp_result = factor.factor_mimicking_portfolio(
    df,
    factor_col="value",
    return_col="next_return",
    time_col="date",
    top_pct=0.3,
    bottom_pct=0.3
)
# 返回: time, long_return, short_return, ls_return
```

#### 第三步：稳健性检验

验证因子在不同样本和时间段的稳定性：

```python
# 1. 子期检验（分时段检验）
subsample_result = factor.subsample_test(
    df,
    factor_col="value",
    return_col="next_return",
    time_col="date",
    n_splits=3
)
# 返回: period, start_date, end_date, mean_ic, t_stat, p_value

# 2. 分组检验（如行业、市值分组）
subgroup_result = factor.subgroup_test(
    df,
    factor_col="value",
    return_col="next_return",
    group_col="industry"
)
# 返回: group, mean_ic, t_stat, p_value

# 3. 滚动IC检验
rolling_result = factor.rolling_ic(
    df,
    factor_col="value",
    return_col="next_return",
    time_col="date",
    window=60
)
# 返回: time, rolling_ic, rolling_ir
```

#### 完整检验流程示例

```python
import polars as pl
from polars_quant import Factor, returns

# 准备数据
df = pl.read_parquet("stock_data.parquet")
df = returns(df, price_col="close", period=1, return_col="next_return")

factor = Factor()

# 预处理：多因子正交化
df = factor.clean(
    df,
    factor_cols=["size", "value", "momentum"],
    method="orthogonalize"
)

# 第一步：单因子检验
ic_result = factor.ic_test(df, "value", "next_return", "date")
sorts_result = factor.portfolio_sorts(df, "value", "next_return", "date", 5)
fr_result = factor.factor_return(df, "value", "next_return", "date")
decay_result = factor.ic_decay(df, "value", "next_return", "date", "stock_code", 10)

# 第二步：多因子回归分析
fm_result = factor.fama_macbeth(
    df, ["size", "value", "momentum"], "next_return", "date"
)
ts_result = factor.time_series_regression(
    df, ["size", "value", "momentum"], "next_return", "stock_code"
)

# 第三步：稳健性检验
subsample_result = factor.subsample_test(df, "value", "next_return", "date", 3)
subgroup_result = factor.subgroup_test(df, "value", "next_return", "industry")
rolling_result = factor.rolling_ic(df, "value", "next_return", "date", 60)

# 分析结果
print("IC均值:", ic_result["ic"].mean())
print("分层收益:", sorts_result)
print("Fama-MacBeth系数:", fm_result)
print("子期稳定性:", subsample_result)
```

### 因子评估标准

| 指标 | 优秀标准 | 较好标准 | 一般标准 |
|------|---------|---------|---------|
| IC | \|IC\| > 0.10 | \|IC\| > 0.05 | \|IC\| > 0.03 |
| IR | IR > 2.0 | IR > 1.0 | IR > 0.5 |
| Rank IC | \|Rank IC\| > 0.10 | \|Rank IC\| > 0.05 | \|Rank IC\| > 0.03 |
| IC胜率 | > 0.7 | > 0.6 | > 0.5 |
| 换手率 | < 0.3 | < 0.5 | < 0.7 |

---

## 🔧 开发

```bash
# 克隆仓库
git clone https://github.com/Firstastor/polars-quant.git
cd polars-quant

# 安装开发依赖
pip install maturin

# 开发模式编译
maturin develop

# 发布模式编译（更快）
maturin develop --release
```

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系

如有问题或建议，请提交 [Issue](https://github.com/Firstastor/polars-quant/issues)。

