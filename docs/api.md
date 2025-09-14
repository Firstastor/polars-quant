# API (short reference) 🧾

This page lists the most-used functions and their short signatures. For exact types and overloads see `python/polars_quant/polars_quant.pyi`.

Top-level items

- Backtest helpers
  - `Backtrade.run(data, entries, exits, init_cash=100000.0, fee=0.0, slip=0.0, size=1.0)` → Backtrade
  - `Backtrade.portfolio(...)` → Backtrade
  - `Portfolio.run(...)` → Portfolio

- Data utilities (qstock)
  - `history(stock_code, scale=240, datalen=3650, timeout=10)` → pl.DataFrame | None
  - `history_save(stock_code, ...)` → None
  - `info()` → pl.DataFrame
  - `info_save(path)` → None

- Technical indicators (qtalib)
  - `ma(data, timeperiod=30)` → list[pl.Series]
  - `ema(data, timeperiod=30)` → list[pl.Series]
  - `sma(data, timeperiod=20)` → list[pl.Series]
  - `wma(data, timeperiod=20)` → list[pl.Series]
  - `bband(data, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)` → list[pl.Series]
  - `macd(data, fast=12, slow=26, signal=9)` → list[pl.Series]
  - `rsi(data, timeperiod=14)` → list[pl.Series]
  - `adx(data, timeperiod=14)` → pl.DataFrame
  - `adxr(data, timeperiod=14)` → pl.DataFrame
  - `obv(data)` → pl.DataFrame
  - `mfi(data, period=14)` → pl.DataFrame
  - `ultosc(data, short_period=7, medium_period=14, long_period=28)` → pl.DataFrame

Notes

- Most indicator functions accept a Polars DataFrame and either return a DataFrame (with new columns) or a list of Series (one per numeric input column). Check `python/polars_quant/polars_quant.pyi` for exact return types.
- Many functions expect lowercase OHLC column names: `open`, `high`, `low`, `close`, `volume`.

Want a machine-readable mapping (function → output columns)? Ask and I'll generate it from the code.
## API Reference

本页列出 `qtalib`（Rust 实现并通过 PyO3 暴露）的主要函数签名与返回类型说明。文档以事实为主，省略性能夸张性描述。

注意：这些函数通常接受一个 Polars `DataFrame`（列为浮点数）并返回新增列或新的 `DataFrame`。请参考 `python/polars_quant/polars_quant.pyi` 以获取精确的类型提示。

常见输入列：`open`, `high`, `low`, `close`, `volume`（若有）

示例函数签名（局部摘要）：

- bband(data: DataFrame, timeperiod: int = 5, nbdevup: float = 2.0, nbdevdn: float = 2.0) -> list[Series]
- dema(data: DataFrame, timeperiod: int = 30) -> list[Series]
- ema(data: DataFrame, timeperiod: int = 30) -> list[Series]
- kama(data: DataFrame, timeperiod: int = 30, fast_limit: float = 2.0, slow_limit: float = 30.0) -> list[Series]
- ma(data: DataFrame, timeperiod: int = 30) -> list[Series]
- mama(data: DataFrame, c: float = 10.0) -> list[Series]
- mavp(data: DataFrame, timeperiod: int = 30) -> list[Series]
- sma(data: DataFrame, timeperiod: int = 20) -> list[Series]
- t3(data: DataFrame, timeperiod: int = 20, b: float = 0.7) -> list[Series]
- tema(data: DataFrame, timeperiod: int = 20) -> list[Series]
- trima(data: DataFrame, timeperiod: int = 20) -> list[Series]
- wma(data: DataFrame, timeperiod: int = 20) -> list[Series]

- adx(data: DataFrame, timeperiod: int = 14) -> DataFrame  ✅ 返回含 `adx` 列的 DataFrame
- adxr(data: DataFrame, timeperiod: int = 14) -> DataFrame
- apo(data: DataFrame, fastperiod: int = 12, slowperiod: int = 26) -> list[Series]

- aroon(data: DataFrame, timeperiod: int = 14) -> DataFrame (包含 `aroon_up{n}` / `aroon_down{n}`)
- aroonosc(data: DataFrame, timeperiod: int = 14) -> DataFrame
- macd(data: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> list[Series]

- mfi(data: DataFrame, period: int = 14) -> DataFrame
- mom(data: DataFrame, period: int = 14) -> DataFrame
- ppo(data: DataFrame, fastperiod: int = 12, slowperiod: int = 26) -> list[Series]
- roc(data: DataFrame, timeperiod: int = 10) -> DataFrame
- rsi(data: DataFrame, timeperiod: int = 14) -> list[Series]

- stoch(...), stochf(...), stochrsi(...) -> DataFrame (详见类型提示文件)
- trix(data: DataFrame, timeperiod: int = 30) -> DataFrame
- ultosc(data: DataFrame, short_period: int = 7, medium_period: int = 14, long_period: int = 28) -> DataFrame
- willr(data: DataFrame, timeperiod: int = 14) -> DataFrame
- ad(data: DataFrame) -> DataFrame
- adosc(data: DataFrame, fastperiod: int = 3, slowperiod: int = 10) -> DataFrame
- obv(data: DataFrame) -> DataFrame

更多细节（列名、返回列格式）请直接参考 `python/polars_quant/polars_quant.pyi` 或在 Python 中查看运行时返回值示例。若需我把每个函数的输出列名一一列出，我可以继续生成完整表格。
Calculates Relative Strength Index.

```python
qtalib.rsi(data, timeperiod=14)
```

**Parameters:**
- `data` (DataFrame): Price data
- `timeperiod` (int): Period for calculation (default: 14)

**Output Columns:**
- `rsi`: RSI values

### stoch - Stochastic Oscillator
Calculates Stochastic Oscillator.

```python
qtalib.stoch(data, fastk_period=5, slowk_period=3, slowd_period=3)
```

**Parameters:**
- `data` (DataFrame): OHLC data
- `fastk_period` (int): Fast %K period (default: 5)
- `slowk_period` (int): Slow %K period (default: 3)
- `slowd_period` (int): Slow %D period (default: 3)

**Output Columns:**
- `stoch_k`: Slow %K values
- `stoch_d`: Slow %D values

### stochf - Fast Stochastic Oscillator
Calculates Fast Stochastic Oscillator.

```python
qtalib.stochf(data, fastk_period=5, fastd_period=3)
```

**Parameters:**
- `data` (DataFrame): OHLC data
- `fastk_period` (int): Fast %K period (default: 5)
- `fastd_period` (int): Fast %D period (default: 3)

**Output Columns:**
- `stoch_k`: Fast %K values
- `stoch_d`: Fast %D values

### stochrsi - Stochastic RSI
Calculates Stochastic RSI.

```python
qtalib.stochrsi(data, timeperiod=14, fastk_period=5, fastd_period=3)
```

**Parameters:**
- `data` (DataFrame): Price data
- `timeperiod` (int): RSI period (default: 14)
- `fastk_period` (int): Fast %K period (default: 5)
- `fastd_period` (int): Fast %D period (default: 3)

**Output Columns:**
- `stochrsi_k`: Stochastic RSI %K
- `stochrsi_d`: Stochastic RSI %D

### ultosc - Ultimate Oscillator
Calculates Ultimate Oscillator.

```python
qtalib.ultosc(data, short_period=7, medium_period=14, long_period=28)
```

**Parameters:**
- `data` (DataFrame): OHLC data
- `short_period` (int): Short period (default: 7)
- `medium_period` (int): Medium period (default: 14)
- `long_period` (int): Long period (default: 28)

**Output Columns:**
- `ultosc`: Ultimate Oscillator values

### willr - Williams' %R
Calculates Williams' %R.

```python
qtalib.willr(data, timeperiod=14)
```

**Parameters:**
- `data` (DataFrame): OHLC data
- `timeperiod` (int): Period for calculation (default: 14)

**Output Columns:**
- `willr{timeperiod}`: Williams' %R values

## 📊 Volume Indicators

### ad - Accumulation/Distribution Line
Calculates Accumulation/Distribution Line.

```python
qtalib.ad(data)
```

**Parameters:**
- `data` (DataFrame): OHLCV data

**Output Columns:**
- `clv`: Chaikin Line values
- `ad`: Accumulation/Distribution values

### adosc - Chaikin A/D Oscillator
Calculates Chaikin A/D Oscillator.

```python
qtalib.adosc(data, fastperiod=3, slowperiod=10)
```

**Parameters:**
- `data` (DataFrame): OHLCV data
- `fastperiod` (int): Fast period (default: 3)
- `slowperiod` (int): Slow period (default: 10)

**Output Columns:**
- Chaikin A/D Oscillator values

### mfi - Money Flow Index
Calculates Money Flow Index.

```python
qtalib.mfi(data, period=14)
```

**Parameters:**
- `data` (DataFrame): OHLCV data
- `period` (int): Period for calculation (default: 14)

**Output Columns:**
- `mfi`: Money Flow Index values

### obv - On Balance Volume
Calculates On Balance Volume.

```python
qtalib.obv(data)
```

**Parameters:**
- `data` (DataFrame): Price and volume data

**Output Columns:**
- `obv`: On Balance Volume values

## 🔧 Usage Examples

### Basic Technical Analysis

```python
import polars as pl
import polars_quant as plqt

# Fetch data
df = plqt.history("sz000001", datalen=200)

# Calculate multiple indicators
df = df.with_columns([
    plqt.qtalib.sma(df, 20).alias("SMA20"),
    plqt.qtalib.rsi(df, 14).alias("RSI"),
    plqt.qtalib.macd(df, 12, 26, 9).struct.field("macd").alias("MACD")
])

print(df.tail())
```

### Advanced Multi-Indicator Analysis

```python
# Calculate trend indicators
trend_indicators = df.with_columns([
    plqt.qtalib.adx(df, 14).alias("ADX"),
    plqt.qtalib.aroon(df, 14).struct.field("aroon_up").alias("Aroon_Up"),
    plqt.qtalib.aroon(df, 14).struct.field("aroon_down").alias("Aroon_Down")
])

# Calculate momentum indicators
momentum_indicators = df.with_columns([
    plqt.qtalib.stoch(df, 14, 3, 3).struct.field("stoch_k").alias("Stoch_K"),
    plqt.qtalib.stoch(df, 14, 3, 3).struct.field("stoch_d").alias("Stoch_D"),
    plqt.qtalib.cci(df, 20).alias("CCI")
])

# Calculate volume indicators
volume_indicators = df.with_columns([
    plqt.qtalib.mfi(df, 14).alias("MFI"),
    plqt.qtalib.obv(df).alias("OBV")
])
```

## ⚠️ Important Notes

1. **Data Types**: All functions expect Float64 columns for calculations
2. **Column Names**: Output column names follow the pattern `{original_column}_{indicator}{parameters}`
3. **NaN Values**: Initial values may be NaN due to calculation requirements
4. **Performance**: Functions are optimized for performance with vectorized operations
5. **Memory**: Large datasets are handled efficiently with streaming operations

## 📚 Related Documentation

- [Installation Guide](start/installation.md) - How to install Polars-Quant
- [Usage Guide](start/usage.md) - Comprehensive usage examples
- [Features](start/features.md) - Overview of all features</content>
<parameter name="filePath">c:\Users\28767\Documents\Project\polars-quant\docs\api.md
