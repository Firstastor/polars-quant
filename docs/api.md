# **API Documentation**

The **Polars-Quant** API provides methods for backtesting trading strategies using Polars DataFrames. Below is an overview of the classes and methods available, including the integration of TA-Lib indicators.

## `Backtrade` Class

The `Backtrade` class allows you to run backtests using historical price data and entry/exit signals.

### Attributes:
- **`results`** (`pl.DataFrame | None`): A DataFrame containing the equity curve and cash over time.
- **`trades`** (`pl.DataFrame | None`): A DataFrame containing the executed trades, including entry and exit details.
- **`_summary`** (`dict | None`): An optional cached dictionary of performance statistics.

### Methods:

#### __init__()
- Initializes a `Backtrade` object with optional results and trades.

#### `run()`
- Runs a per-symbol independent backtest.
- **Parameters**:
  - `data`: DataFrame with historical price data.
  - `entries`: DataFrame indicating entry signals for each symbol.
  - `exits`: DataFrame indicating exit signals for each symbol.
  - `init_cash`: Initial cash for the backtest (default: 100,000).
  - `fee`: Transaction fee (default: 0.0).
  - `slip`: Slippage (default: 0.0).
  - `size`: Trade size (default: 1.0).
  - `indicators`: Optional list of TA-Lib indicators to apply to the data (default: None).
- **Returns**: A `Backtrade` object with the backtest results.

#### `portfolio()`
- Runs a portfolio-level backtest with shared capital across all symbols.
- **Parameters**: Same as `run`.
- **Returns**: A `Backtrade` object with the portfolio-level backtest results.

#### `results()`
- Returns the backtest equity/cash DataFrame or `None` if not available.

#### `trades()`
- Returns the trade log DataFrame or `None` if not available.

#### `summary()`
- Returns a text summary of final equity and performance statistics.

---

## **TA-Lib**

The following classes provide easy integration with **TA-Lib** indicators, allowing you to apply popular technical analysis functions to your trading data.

### `ATR` Class
This class applies the **Average True Range (ATR)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **ATR** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the ATR calculation (default: 14).
- **Returns**: A `ATR` object containing the results with the ATR values added to the DataFrame.

### `BBANDS` Class
This class applies the **Bollinger Bands (BBANDS)** indicator from TA-Lib.

#### Methods:

#### `run()`
- Applies the **BBANDS** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the calculation (default: 20).
  - `nbdevup`: The number of standard deviations for the upper band (default: 2).
  - `nbdevdn`: The number of standard deviations for the lower band (default: 2).
- **Returns**: A `BBANDS` object containing the results with the Bollinger Bands added to the DataFrame.

### `CCI` Class
This class applies the **Commodity Channel Index (CCI)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **CCI** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the CCI calculation (default: 20).
- **Returns**: A `CCI` object containing the results with the CCI values added to the DataFrame.

### `EMA` Class
This class applies the **Exponential Moving Average (EMA)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **EMA** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the EMA calculation.
  - `adjust`: Whether to adjust for the initial values (default: False).
- **Returns**: A `EMA` object containing the results with the EMA values added to the DataFrame.

#### `cross()`
- Checks for crossovers between two EMAs.
- **Parameters**:
  - `first_ma`: The first moving average period.
  - `second_ma`: The second moving average period.
- **Returns**: A dictionary with cross-over points.

### `KAMA` Class
This class applies the **Kaufman Adaptive Moving Average (KAMA)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **KAMA** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the KAMA calculation (default: 14).
  - `fast_period`: The fast period for KAMA calculation (default: 2).
  - `slow_period`: The slow period for KAMA calculation (default: 30).
- **Returns**: A `KAMA` object containing the results with the KAMA values added to the DataFrame.

### `MACD` Class
This class applies the **Moving Average Convergence Divergence (MACD)** indicator from TA-Lib.

#### Methods:

#### `run()`
- Applies the **MACD** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `fastperiod`: The fast period for MACD calculation (default: 12).
  - `slowperiod`: The slow period for MACD calculation (default: 26).
  - `signalperiod`: The period for the MACD signal line (default: 9).
- **Returns**: A `MACD` object containing the results with the MACD values added to the DataFrame.

### `NATR` Class
This class applies the **Normalized Average True Range (NATR)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **NATR** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the NATR calculation (default: 14).
- **Returns**: A `NATR` object containing the results with the NATR values added to the DataFrame.

### `OBV` Class
This class applies the **On-Balance Volume (OBV)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **OBV** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `volume_col`: The name of the volume column (default: "Volume").
  - `price_col`: The name of the price column (default: "Close").
- **Returns**: A `OBV` object containing the results with the OBV values added to the DataFrame.

### `RSI` Class
This class applies the **Relative Strength Index (RSI)** indicator from TA-Lib to your price data.

#### Methods:

#### `run()`
- Applies the **RSI** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the RSI calculation (default: 14).
- **Returns**: A `RSI` object containing the results with the RSI values added to the DataFrame.

### `SMA` Class
This class applies the **Simple Moving Average (SMA)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **SMA** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the SMA calculation.
- **Returns**: A `SMA` object containing the results with the SMA values added to the DataFrame.

#### `cross()`
- Checks for crossovers between two SMAs.
- **Parameters**:
  - `first_ma`: The first moving average period.
  - `second_ma`: The second moving average period.
- **Returns**: A dictionary with cross-over points.

### `TEMA` Class
This class applies the **Triple Exponential Moving Average (TEMA)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **TEMA** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the TEMA calculation (default: 14).
- **Returns**: A `TEMA` object containing the results with the TEMA values added to the DataFrame.

### `TRANGE` Class
This class calculates the **True Range (TRANGE)** indicator.

#### Methods:

#### `run()`
- Calculates the **TRANGE** indicator.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
- **Returns**: A `TRANGE` object containing the results with the True Range values added to the DataFrame.

### `TRIMA` Class
This class applies the **Triangular Moving Average (TRIMA)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **TRIMA** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the TRIMA calculation (default: 14).
- **Returns**: A `TRIMA` object containing the results with the TRIMA values added to the DataFrame.

### `VOL` Class
This class calculates the **Volatility (VOL)** indicator.

#### Methods:

#### `run()`
- Calculates the **Volatility** indicator.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the Volatility calculation (default: 20).
  - `method`: The method to calculate volatility ("std" or "range"; default: "std").
- **Returns**: A `VOL` object containing the results with the Volatility values added to the DataFrame.

### `WMA` Class
This class applies the **Weighted Moving Average (WMA)** indicator to your price data.

#### Methods:

#### `run()`
- Applies the **WMA** indicator to the provided data.
- **Parameters**:
  - `data`: A Polars DataFrame with historical price data (must include numeric columns).
  - `timeperiod`: The number of periods to use for the WMA calculation.
- **Returns**: A `WMA` object containing the results with the WMA values added to the DataFrame.

#### `cross()`
- Checks for crossovers between two WMAs.
- **Parameters**:
  - `first_ma`: The first moving average period.
  - `second_ma`: The second moving average period.
- **Returns**: A dictionary with cross-over points.
