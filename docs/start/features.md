# Features of Polars-Quant

Polars-Quant is designed for high-performance backtesting using **Polars DataFrames**, with the following key features:

## Backtesting with Polars
- **Vectorized Backtesting**: Leverage Polars' fast DataFrame operations to run backtests efficiently.
- **Flexible Backtesting**: Supports both per-symbol independent backtests and portfolio-level backtests, allowing you to tailor the strategy to your needs.

## Per-symbol Independent Backtests
- The `Backtrade.run` method allows you to run backtests independently for each symbol, which can be useful for testing strategies on single assets.
- Example usage:
  ```python
  import polars_quant as plqt
  bt = plqt.Backtrade.run(data, entries, exits, init_cash=50_000, fee=0.001)
  results = bt.results()
  trades = bt.trades()
  print(bt.summary())
  ```