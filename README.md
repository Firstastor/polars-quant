# polars-quant 🧮📊

> 基于 Rust + Polars 的高性能量化分析与回测工具集，提供丰富的技术指标计算和独立资金池回测引擎。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.90+-orange.svg)](https://www.rust-lang.org/)

## ✨ 特性

- 🚀 **高性能**：基于 Rust 实现，底层使用 Polars 数据处理，速度快、内存占用低
- 📊 **丰富指标**：提供 50+ 常用技术指标（移动平均、动量、震荡、成交量等）
- 💰 **独立资金池**：每只股票使用独立资金池回测，智能并行处理
- 🎯 **真实模拟**：支持佣金（含最低佣金）、滑点、整百股交易等实盘规则
- 📈 **详细统计**：提供夏普比率、索提诺比率、卡尔马比率等 12 类详细指标
- 🔍 **灵活分析**：支持全局汇总和单股票深度分析

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

### 一、回测类 (Backtest)

#### 1. 构造函数

##### `Backtest(prices, buy_signals, sell_signals, initial_capital, commission_rate, min_commission, slippage)`

创建回测实例。

**参数**：
- `prices` (DataFrame): 价格数据，第一列为日期，其余列为各股票价格
- `buy_signals` (DataFrame): 买入信号，第一列为日期，其余列为布尔值（True 表示买入，False 表示不买入）
- `sell_signals` (DataFrame): 卖出信号，第一列为日期，其余列为布尔值（True 表示卖出，False 表示不卖出）
- `initial_capital` (float): 初始资金，默认 100000.0
- `commission_rate` (float): 佣金费率，默认 0.0003（万三）
- `min_commission` (float): 最低佣金，默认 5.0 元
- `slippage` (float): 滑点，默认 0.0（0.001 表示 0.1%）

**示例**：
```python
from polars_quant import Backtest

bt = Backtest(
    prices=prices_df,
    buy_signals=buy_df,
    sell_signals=sell_df,
    initial_capital=100000.0,
    commission_rate=0.0003,  # 万三
    min_commission=5.0,       # 最低5元
    slippage=0.001            # 0.1%滑点
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

##### `summary()`

打印所有股票的综合统计摘要。

**返回**：None（直接打印输出）

**包含统计项**：
1. **基本信息**：总资产、总收益、收益率
2. **交易统计**：交易次数、胜率、盈亏比
3. **收益分析**：最大盈利/亏损、平均盈利/亏损
4. **风险指标**：最大回撤、最大回撤百分比
5. **风险调整收益**：夏普比率、索提诺比率、卡尔马比率
6. **时间分析**：平均持仓天数、最长/最短持仓
7. **连续统计**：最大连续盈利/亏损次数
8. **月度/年度收益**
9. **波动率指标**
10. **分布统计**：收益率分布
11. **资金曲线**：最高/最低点
12. **交易频率**

**示例**：
```python
bt.summary()  # 直接打印，无需 print()
```

---

#### 4. 结果查询 - 单只股票

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

### 二、技术指标函数

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
| `ma(series, period)` | 通用移动平均（默认SMA） | series: 价格序列<br>period: 周期 | Series |
| `t3(series, period, vfactor)` | T3 移动平均 | series: 价格序列<br>period: 周期<br>vfactor: 体积因子 | Series |

**示例**：
```python
import polars as pl
from polars_quant import sma, ema, wma

df = pl.DataFrame({"close": [100, 102, 101, 105, 107, 110]})
df = df.with_columns([
    sma(pl.col("close"), 3).alias("sma_3"),
    ema(pl.col("close"), 3).alias("ema_3"),
    wma(pl.col("close"), 3).alias("wma_3"),
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

### 示例 1：基础回测

```python
import polars as pl
from polars_quant import Backtest

# 准备数据
dates = pl.date_range(
    start=pl.date(2023, 1, 1),
    end=pl.date(2023, 12, 31),
    interval="1d",
    eager=True
).cast(str)

n = len(dates)

# 价格数据
prices_df = pl.DataFrame({
    "date": dates,
    "AAPL": [100 + i * 0.3 for i in range(n)],
    "MSFT": [200 + i * 0.5 for i in range(n)],
})

# 买卖信号（布尔值：True 表示买入/卖出，False 表示不操作）
buy_signals_df = pl.DataFrame({
    "date": dates,
    "AAPL": [i in [10, 100, 200] for i in range(n)],  # 第10、100、200天买入
    "MSFT": [i in [20, 120, 220] for i in range(n)],  # 第20、120、220天买入
})

sell_signals_df = pl.DataFrame({
    "date": dates,
    "AAPL": [i in [60, 150, 280] for i in range(n)],  # 第60、150、280天卖出
    "MSFT": [i in [70, 170, 300] for i in range(n)],  # 第70、170、300天卖出
})

# 创建并运行回测
bt = Backtest(
    prices=prices_df,
    buy_signals=buy_signals_df,
    sell_signals=sell_signals_df,
    initial_capital=100000.0,
    commission_rate=0.0003,
    min_commission=5.0,
    slippage=0.001
)

bt.run()

# 查看结果
bt.summary()  # 综合统计
daily = bt.get_daily_records()  # 每日记录
positions = bt.get_position_records()  # 交易记录
```

---

### 示例 2：技术指标计算

```python
import polars as pl
from polars_quant import sma, ema, rsi, macd, bband

# 读取数据
df = pl.read_csv("stock_data.csv")

# 计算多个指标
df = df.with_columns([
    # 移动平均
    sma(pl.col("close"), 5).alias("sma_5"),
    sma(pl.col("close"), 20).alias("sma_20"),
    ema(pl.col("close"), 12).alias("ema_12"),
    
    # RSI
    rsi(pl.col("close"), 14).alias("rsi"),
])

# MACD
macd_line, signal_line, hist = macd(pl.col("close"), 12, 26, 9)
df = df.with_columns([
    macd_line.alias("macd"),
    signal_line.alias("signal"),
    hist.alias("hist"),
])

# 布林带
upper, middle, lower = bband(pl.col("close"), 20, 2.0)
df = df.with_columns([
    upper.alias("bb_upper"),
    middle.alias("bb_middle"),
    lower.alias("bb_lower"),
])

print(df)
```

---

### 示例 3：单股票深度分析

```python
# 运行回测后
bt.run()

# 查看单只股票
stock_daily = bt.get_stock_daily("AAPL")
stock_positions = bt.get_stock_positions("AAPL")
print(bt.get_stock_summary("AAPL"))

# 时间段筛选
q1_data = bt.get_stock_daily("AAPL").filter(
    (pl.col("date") >= "2023-01-01") & (pl.col("date") <= "2023-03-31")
)

# 找出最佳/最差交易
all_positions = bt.get_position_records()
best_trade = all_positions.filter(pl.col("symbol") == "AAPL").sort("pnl", descending=True).head(1)
worst_trade = all_positions.filter(pl.col("symbol") == "AAPL").sort("pnl").head(1)
```

---

## 🎯 回测特性说明

### 独立资金池
每只股票使用独立的初始资金池，互不影响。适合测试多策略或对比不同股票表现。

### 智能并行
- **< 4 只股票**：串行执行（避免线程开销）
- **≥ 4 只股票**：并行执行，线程数 = min(股票数, CPU核心数)

### 交易规则
- **整百股交易**：自动计算可买入的 100 股倍数
- **佣金计算**：`max(交易金额 × 费率, 最低佣金)`
- **滑点模拟**：买入价上浮，卖出价下调

### 强制平仓
回测结束时自动平仓所有持仓，按最后一日收盘价计算。

---

## 📊 统计指标说明

`summary()` 提供的详细统计包括：

- **夏普比率** (Sharpe Ratio)：风险调整后收益
- **索提诺比率** (Sortino Ratio)：只考虑下行风险的收益率
- **卡尔马比率** (Calmar Ratio)：年化收益率 / 最大回撤
- **最大回撤** (Max Drawdown)：资金曲线最大跌幅
- **胜率** (Win Rate)：盈利交易占比
- **盈亏比** (Profit Factor)：总盈利 / 总亏损
- **持仓统计**：平均/最长/最短持仓天数
- **连续统计**：最大连续盈利/亏损次数

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

