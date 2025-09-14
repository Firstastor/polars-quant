# polars-quant docs 📚

This documentation is short and code-driven. Use the examples in `docs/start/` to get started.

- Quick start: `docs/start/installation.md`
- Usage examples: `docs/start/usage.md`
- Features summary: `docs/start/features.md`
- API reference: `docs/api.md`

For exact Python function signatures consult `python/polars_quant/polars_quant.pyi`.

If you need more examples or a function parity table, open an issue or ask here.

# Welcome to Polars-Quant

Welcome to the **Polars-Quant** project! This high-performance library provides comprehensive tools for quantitative trading and backtesting using the **Polars** DataFrame library. Built with Rust for speed and Python bindings for ease of use, it offers both per-symbol independent backtests and portfolio-level backtests.

## 🚀 Key Features

### Backtesting Engine
- **Vectorized Backtesting**: High-performance backtesting using Polars' efficient DataFrame operations
- **Per-symbol Independent Backtests**: Run tests on individual symbols with separate capital allocation
- **Portfolio-level Backtests**: Perform backtests with shared capital across multiple symbols
- **Real-time Performance**: Optimized Rust implementation for fast execution

### Data Integration
- **Stock Data Fetching**: Built-in functions to fetch historical stock data from Sina Finance
- **Data Persistence**: Save and load data in efficient Parquet format
- **Multi-source Support**: Extensible architecture for various data sources

### Technical Analysis
- **TA-Lib Integration**: Comprehensive technical analysis indicators
- **Custom Indicators**: Easy-to-extend framework for custom technical indicators
- **Signal Generation**: Built-in crossover and signal generation utilities

### Performance Analytics
- **Detailed Statistics**: Comprehensive performance metrics including Sharpe ratio, max drawdown
- **Trade Analysis**: Per-trade and portfolio-level trade statistics
- **Risk Metrics**: Advanced risk analysis and portfolio optimization tools

## 📊 Quick Start

```python
import polars as pl
import polars_quant as plqt

# Fetch historical data
df = plqt.history("sz000001")[["date", "close"]].rename({"close": "sz000001"})

# Calculate moving averages
df = df.with_columns([
    plqt.sma(df, 5).alias("MA5"),
- Issue tracking and feature requests
- Code of conduct
    # polars-quant — Documentation index

    Short index with links to core documentation pages.

    - Installation: `start/installation.md`
    - Quick usage and examples: `start/usage.md`
    - API summary: `api.md`
    - Feature notes: `start/features.md`

    Notes

    - The Python type hints file `python/polars_quant/polars_quant.pyi` contains exact function signatures and return types. Use it as the authoritative source for calling conventions.
    - This project is distributed under the MIT License (see `LICENSE`).

