# 使用示例

本页展示最常用的三类能力：技术指标、回测、数据获取。

## 技术指标（TA）

大多数指标接收一个含数值列的 DataFrame，返回 `list[pl.Series]` 或在原 DataFrame 上追加列。

```python
import polars as pl
import polars_quant as plqt

# 移动平均
df = pl.DataFrame({'close': [100, 101, 102, 101, 103]})
ma_list = plqt.ma(df, timeperiod=3)  # 返回若干 Series，可直接 with_columns 挂回
res = df.with_columns(ma_list)
print(res)

# MACD（返回 [dif, dea, macd] 三列）
macd_cols = plqt.macd(df, fast=12, slow=26, signal=9)
res2 = df.with_columns(macd_cols)
print(res2)

# ADX（要求列名：high、low、close）
df_ohlc = pl.DataFrame({
  'high':[10.0,10.5,11.0], 'low':[9.5,9.8,10.2], 'close':[10.0,10.4,10.8]
})
res3 = plqt.adx(df_ohlc, timeperiod=14)  # 直接返回带指标列的 DataFrame
print(res3)
```

## 回测（Backtrade / Portfolio）

- Backtrade.run：每个标的独立使用一份初始资金，互不影响。
- Backtrade.portfolio 或 Portfolio.run：多标的共享资金池，更贴近组合实盘。

```python
import polars as pl
from polars_quant import Backtrade

# 简单单标的示例
data = pl.DataFrame({
  'Date': ['2023-01-01','2023-01-02','2023-01-03','2023-01-04'],
  'AAPL': [100.0, 102.0, 101.0, 105.0]
})
entries = pl.DataFrame({'Date': data['Date'], 'AAPL': [True, False, False, True]})
exits   = pl.DataFrame({'Date': data['Date'], 'AAPL': [False, True, True, False]})

bt = Backtrade.run(data, entries, exits, init_cash=100_000.0, fee=0.001)
bt.summary()         # 控制台打印绩效摘要
print(bt.results())  # 资金曲线 / 现金
print(bt.trades())   # 交易明细
```

组合级回测：

```python
from polars_quant import Portfolio

btp = Portfolio.run(data, entries, exits, init_cash=200_000.0, size=0.5)
btp.summary()
```

注意事项：
- 三个 DataFrame 的时间列需一致且为第一列；标的列名需一致且布尔信号对齐。
- fee/slip/size 为全局参数，具体以策略需要调整。

## A 股数据函数

```python
import polars as pl
import polars_quant as plqt

# 历史数据（来自新浪）
df = plqt.history('sz000001', scale=240, datalen=100)
print(df)

# 保存到 Parquet
plqt.history_save('sz000001', datalen=500)

# 基础信息（全市场）
info = plqt.info()
print(info.head())

# 保存基础信息
plqt.info_save('stocks.parquet')
```
