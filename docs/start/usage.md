# Usage Guide

This guide provides comprehensive examples and tutorials for using Polars-Quant effectively.

## 🚀 Quick Start

### Basic Usage

```python
import polars as pl
import polars_quant as plqt

# Fetch stock data
df = plqt.history("sz000001", datalen=100)
print(df.head())

# Calculate technical indicators
df = df.with_columns([
    plqt.ma(df, 5).alias("MA5"),
    plqt.ma(df, 20).alias("MA20"),
    plqt.rsi(df, 14).alias("RSI")
])

print(df.tail())
```

## 📊 Data Fetching

### Stock Data

```python
# Fetch single stock
df = plqt.history("sz000001", datalen=200)  # Ping An Bank

# Fetch multiple stocks
stocks = ["sz000001", "sh600036", "sz000002"]
dfs = [plqt.history(stock, datalen=100) for stock in stocks]

# Get stock information
info = plqt.info("sz000001")
print(info)
```

### Data Structure

The fetched data includes:
- `date`: Trading date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `amount`: Trading amount

## 📈 Technical Analysis

### Moving Averages

```python
# Simple Moving Average
df = df.with_columns([
    plqt.ma(df, 5).alias("MA5"),
    plqt.ma(df, 10).alias("MA10"),
    plqt.ma(df, 20).alias("MA20"),
    plqt.ma(df, 30).alias("MA30")
])

# Exponential Moving Average
df = df.with_columns([
    plqt.ema(df, 12).alias("EMA12"),
    plqt.ema(df, 26).alias("EMA26")
])
Usage

Short examples showing typical use.

Simple moving average

```python
import polars as pl
import polars_quant as plqt

df = pl.DataFrame({'close': [100, 101, 102, 103, 104, 105]})
res = plqt.ma(df, 3)
print(res)
```

Backtesting (very small example)

```python
from polars_quant import Backtrade

bt = Backtrade()
# bt.load_data(...)  # supply your DataFrame or CSV
# bt.run()
# print(bt.report())
```

Notes

- This project provides tools for analysis and simple backtests; it is not investment advice.
- For full API details, see `docs/api.md` or `python/polars_quant/polars_quant.pyi` for exact signatures.
])

# Average True Range
df = df.with_columns([
    plqt.atr(df, 14).alias("ATR")
])
```

### Volume Indicators

```python
# On Balance Volume
df = df.with_columns([
    plqt.obv(df).alias("OBV")
])

# Volume Weighted Average Price (Note: VWAP may not be implemented)
# df = df.with_columns([
#     plqt.vwap(df).alias("VWAP")
# ])
```

## 📊 Backtesting

### Basic Backtesting

```python
from polars_quant import Backtrade

# Define trading strategy
def simple_strategy(data):
    # Buy signal: MA5 crosses above MA20
    ma5 = plqt.ma(data, 5)
    ma20 = plqt.ma(data, 20)
    buy_signal = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))

    # Sell signal: MA5 crosses below MA20
    sell_signal = (ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1))

    return buy_signal, sell_signal

# Run backtest
results = Backtrade.run(df, buy_signal, sell_signal, init_cash=100000)

# Print results
print("Backtest completed")
print(results.summary())
```

### Advanced Strategy

```python
def advanced_strategy(data):
    # Multiple indicators
    rsi = plqt.rsi(data, 14)
    macd_result = plqt.macd(data, 12, 26, 9)
    bb_result = plqt.bband(data, 20, 2.0, 2.0)

    # Buy conditions
    buy_rsi = rsi < 30
    buy_macd = macd_result.struct.field("macd") > macd_result.struct.field("signal")
    buy_bb = data["close"] < bb_result.struct.field("lower")
    buy_signal = buy_rsi & buy_macd & buy_bb

    # Sell conditions
    sell_rsi = rsi > 70
    sell_macd = macd_result.struct.field("macd") < macd_result.struct.field("signal")
    sell_bb = data["close"] > bb_result.struct.field("upper")
    sell_signal = sell_rsi | sell_macd | sell_bb

    return buy_signal, sell_signal

# Run advanced backtest
results = Backtrade.run(df, buy_signal, sell_signal, init_cash=100000)
print("Advanced backtest completed")
```

### Portfolio Management

```python
# Create portfolio
portfolio = Portfolio(initial_capital=100000)

# Add positions
portfolio.buy("sz000001", 100, 10.0)  # Buy 100 shares at $10
portfolio.sell("sz000001", 50, 12.0)  # Sell 50 shares at $12

# Get portfolio status
print(f"Current Value: ${portfolio.value()}")
print(f"Total Return: {portfolio.total_return():.2%}")
print(f"Positions: {portfolio.positions()}")
```

## 📈 Performance Analysis

### Risk Metrics

```python
# Calculate Sharpe ratio
returns = df["close"].pct_change()
sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # Annualized

# Calculate maximum drawdown
cumulative = (1 + returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
```

### Benchmark Comparison

```python
# Compare strategy vs benchmark
benchmark_returns = plqt.history("sh000001", datalen=len(df))["close"].pct_change()

# Calculate alpha and beta
cov_matrix = pl.cov(returns, benchmark_returns)
beta = cov_matrix[0, 1] / benchmark_returns.var()
alpha = returns.mean() - beta * benchmark_returns.mean()

print(f"Alpha: {alpha:.4f}")
print(f"Beta: {beta:.4f}")
```

## 🔧 Advanced Features

### Parallel Processing

```python
import concurrent.futures

# Process multiple stocks in parallel
def analyze_stock(stock_code):
    df = plqt.history(stock_code, datalen=200)
    df = df.with_columns([
        plqt.ma(df, 20).alias("MA20"),
        plqt.rsi(df, 14).alias("RSI")
    ])
    return df

stocks = ["sz000001", "sh600036", "sz000002", "sh600000"]

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(analyze_stock, stocks))

print(f"Analyzed {len(results)} stocks")
```

### Custom Indicators

```python
def custom_indicator(high, low, close, period=14):
    """Custom indicator example"""
    # Calculate typical price
    typical_price = (high + low + close) / 3

    # Calculate custom moving average
    custom_ma = typical_price.rolling_mean(window_size=period)

    return custom_ma

# Use custom indicator
df = df.with_columns([
    custom_indicator(df["high"], df["low"], df["close"], 14).alias("Custom_MA")
])
```

## 📊 Visualization

### Basic Plotting

```python
import matplotlib.pyplot as plt

# Plot price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["close"], label="Close Price", alpha=0.7)
plt.plot(df["date"], df["MA5"], label="MA5", alpha=0.7)
plt.plot(df["date"], df["MA20"], label="MA20", alpha=0.7)

plt.title("Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
```

### Technical Analysis Plot

```python
# Create subplot for price and RSI
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price chart
ax1.plot(df["date"], df["close"], label="Close")
ax1.plot(df["date"], df["MA20"], label="MA20")
ax1.set_title("Stock Price")
ax1.legend()

# RSI chart
ax2.plot(df["date"], df["RSI"], label="RSI", color="orange")
ax2.axhline(y=70, color="red", linestyle="--", alpha=0.5)
ax2.axhline(y=30, color="green", linestyle="--", alpha=0.5)
ax2.set_title("RSI Indicator")
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.show()
```

## 🛠️ Best Practices

### Performance Optimization

1. **Use vectorized operations** instead of loops
2. **Pre-calculate indicators** when possible
3. **Use appropriate data types** for memory efficiency
4. **Leverage parallel processing** for multiple stocks

### Memory Management

```python
# Use lazy evaluation for large datasets
df_lazy = df.lazy()

# Process in chunks for memory efficiency
chunk_size = 1000
for i in range(0, len(df), chunk_size):
    chunk = df[i:i+chunk_size]
    # Process chunk
    processed_chunk = process_data(chunk)
```

### Error Handling

```python
def safe_data_fetch(stock_code, max_retries=3):
    """Safely fetch data with error handling"""
    for attempt in range(max_retries):
        try:
            df = plqt.history(stock_code, datalen=100)
            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Wait before retry

# Usage
try:
    df = safe_data_fetch("sz000001")
    print("Data fetched successfully")
except Exception as e:
    print(f"Failed to fetch data: {e}")
```

## 📚 Further Reading

- [API Reference](api.md) - Complete API documentation
- [Features](features.md) - Detailed feature descriptions
- [GitHub Repository](https://github.com/Firstastor/polars-quant) - Source code and examples
