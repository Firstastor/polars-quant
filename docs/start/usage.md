# Usage

Polars-Quant provides an easy-to-use API for backtesting trading strategies using Polars DataFrames. Below is a basic guide on how to get started.

## Quick Example

Here's a simple example of how to run a backtest using the `Backtrade` class.

```python
import polars as pl
import polars as plqt
df = plqt.history("sz000001")["date","close"].rename({"close":"sz000001"})
entries = plqt.MA.run(df,[5,10]).cross(5,10)
exits = plqt.MA.run(df,[5,10]).cross(10,5)
print(plqt.Backtrade.portfolio(df,entries,exits).summary())
print(plqt.Backtrade.run(df,entries,exits).summary())
```
