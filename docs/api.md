# API Documentation

The **Polars-Quant** API provides methods for backtesting trading strategies using Polars DataFrames. Below is an overview of the classes and methods available.

## `Backtrade` Class

The `Backtrade` class allows you to run backtests using historical price data and entry/exit signals.

### Attributes:
- **`results`** (`pl.DataFrame | None`): A DataFrame containing the equity curve and cash over time.
- **`trades`** (`pl.DataFrame | None`): A DataFrame containing the executed trades, including entry and exit details.
- **`_summary`** (`dict | None`): An optional cached dictionary of performance statistics.

### Methods:

#### __init__()
- Initializes a `Backtrade` object with optional results and trades.

#### run()
- Runs a per-symbol independent backtest.
- **Parameters**:
  - `data`: DataFrame with historical price data.
  - `entries`: DataFrame indicating entry signals for each symbol.
  - `exits`: DataFrame indicating exit signals for each symbol.
  - `init_cash`: Initial cash for the backtest (default: 100,000).
  - `fee`: Transaction fee (default: 0.0).
  - `slip`: Slippage (default: 0.0).
  - `size`: Trade size (default: 1.0).
- **Returns**: A `Backtrade` object with the backtest results.

#### portfolio()
- Runs a portfolio-level backtest with shared capital across all symbols.
- **Parameters**: Same as `run`.
- **Returns**: A `Backtrade` object with the portfolio-level backtest results.

#### results()
- Returns the backtest equity/cash DataFrame or `None` if not available.

#### trades()
- Returns the trade log DataFrame or `None` if not available.

#### summary()
- Returns a text summary of final equity and performance statistics.


