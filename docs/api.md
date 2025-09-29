# API 参考

以 `python/polars_quant/polars_quant.pyi` 中的签名为准。以下为常用接口速览。

## 回测

### Backtrade

- 类方法
  - `Backtrade.run(data, entries, exits, init_cash=100_000.0, fee=0.0, slip=0.0, size=1.0) -> Backtrade`
  - `Backtrade.portfolio(data, entries, exits, init_cash=100_000.0, fee=0.0, slip=0.0, size=1.0) -> Backtrade`
- 实例方法
  - `results() -> pl.DataFrame | None`
  - `trades() -> pl.DataFrame | None`
  - `summary() -> None`

说明：`run` 为各标的独立资金回测；`portfolio` 为共享资金的组合级回测。

示例：

```python
bt = Backtrade.run(data, entries, exits, init_cash=100000, fee=0.001)
print(bt.results())
print(bt.trades())
bt.summary()
```

### Portfolio

- 类方法
  - `Portfolio.run(data, entries, exits, init_cash=100_000.0, fee=0.0, slip=0.0, size=1.0) -> Portfolio`
- 实例方法同 Backtrade。

## 数据函数（qstock）

- `history(stock_code: str, scale: int = 240, datalen: int = 365*10, timeout: int = 10) -> pl.DataFrame | None`
- `history_save(stock_code: str, scale: int = 240, datalen: int = 365*10, timeout: int = 10) -> None`
- `info() -> pl.DataFrame`
- `info_save(path: str) -> None`

返回列约定：历史数据包含 `date, open, close, high, low, volume` 等。

## 技术指标（qtalib）

下列函数通常接受 `pl.DataFrame`，返回 `list[pl.Series]` 或 `pl.DataFrame`：

- 均线类：`ma, sma, ema, dema, kama, mama, mavp, t3, tema, trima, wma`
- 趋势/动量：`macd, rsi, roc, mom, apo, ppo`
- 通道/摆动：`bband, stoch, stochf, stochrsi`
- 波动/强弱：`adx, adxr, dx, cci, cmo, mfi, ultosc, willr`
- 成交/能量：`obv, ad, adosc, bop`

典型用法：

```python
# 对数值列计算指标后挂回 DataFrame
cols = plqt.rsi(df, timeperiod=14)
df = df.with_columns(cols)
```

命名与列要求请参照各函数文档或源码注释（例如 ADX 需要 `high/low/close`）。
