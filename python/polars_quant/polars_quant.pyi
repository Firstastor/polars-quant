import polars as pl

class Backtrade:
    """
    Backtrade class for vectorized backtesting using Polars DataFrames.
    Provides both per-symbol independent backtests (`run`) and
    portfolio-level backtests with shared capital (`portfolio`).

    This class enables high-performance backtesting of trading strategies
    using Polars' efficient DataFrame operations and Rust's parallelism.
    It supports multiple symbols, customizable fees, slippage, and position sizing.

    Attributes
    ----------
    results : pl.DataFrame | None
        DataFrame containing equity curve and cash over time.
    trades : pl.DataFrame | None
        DataFrame of executed trades with entry and exit details.
    _summary : dict | None
        Cached summary statistics for performance analysis.

    Examples
    --------
    Basic usage with per-symbol backtesting:

    >>> import polars as pl
    >>> from polars_quant import Backtrade
    >>>
    >>> # Sample data
    >>> data = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [100, 105, 110],
    ...     "TSLA": [200, 195, 210],
    ... })
    >>>
    >>> # Entry signals
    >>> entries = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [True, False, False],
    ...     "TSLA": [False, True, False],
    ... })
    >>>
    >>> # Exit signals
    >>> exits = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [False, False, True],
    ...     "TSLA": [False, False, True],
    ... })
    >>>
    >>> # Run backtest
    >>> bt = Backtrade.run(data, entries, exits, init_cash=100000, fee=0.001)
    >>> bt.summary()  # Prints summary to console
    >>> results_df = bt.results()
    >>> trades_df = bt.trades()

    Portfolio-level backtesting:

    >>> bt_port = Backtrade.portfolio(data, entries, exits, init_cash=200000, size=0.5)
    >>> bt_port.summary()  # Prints portfolio summary
    """

    results: pl.DataFrame | None
    """DataFrame of equity curve and cash over time."""

    trades: pl.DataFrame | None
    """DataFrame of executed trades, including entry and exit details."""

    _summary: dict | None
    """Optional cached summary statistics for performance analysis."""

    def __init__(
        self,
        results: pl.DataFrame | None = None,
        trades: pl.DataFrame | None = None
    ) -> None: 
        """Initialize a Backtrade object with optional results and trades."""

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
        init_cash: float = 100_000.0,
        fee: float = 0.0,
        slip: float = 0.0,
        size: float = 1.0,
    ) -> "Backtrade": 
        """
        Run per-symbol independent backtests.

        Each symbol is backtested separately with its own capital allocation.
        This is useful for analyzing individual symbol performance.

        Parameters
        ----------
        data : pl.DataFrame
            Price data with dates in first column and symbols in subsequent columns.
        entries : pl.DataFrame
            Boolean signals for trade entries.
        exits : pl.DataFrame
            Boolean signals for trade exits.
        init_cash : float, default 100000.0
            Initial cash per symbol.
        fee : float, default 0.0
            Trading fee as a fraction (e.g., 0.001 for 0.1%).
        slip : float, default 0.0
            Slippage as a fraction of price.
        size : float, default 1.0
            Position size multiplier.

        Returns
        -------
        Backtrade
            Backtest results object.

        Examples
        --------
        >>> bt = Backtrade.run(data, entries, exits, init_cash=50000, fee=0.001)
        >>> results = bt.results()
        >>> trades = bt.trades()
        >>> bt.summary()  # Prints performance summary
        """

    @classmethod
    def portfolio(
        cls,
        data: pl.DataFrame,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
        init_cash: float = 100_000.0,
        fee: float = 0.0,
        slip: float = 0.0,
        size: float = 1.0,
    ) -> "Backtrade": 
        """
        Run portfolio-level backtest with shared cash across all symbols.

        All symbols share the same capital pool, allowing for more realistic
        portfolio-level risk management and position sizing.

        Parameters
        ----------
        data : pl.DataFrame
            Price data with dates in first column and symbols in subsequent columns.
        entries : pl.DataFrame
            Boolean signals for trade entries.
        exits : pl.DataFrame
            Boolean signals for trade exits.
        init_cash : float, default 100000.0
            Total initial cash for the portfolio.
        fee : float, default 0.0
            Trading fee as a fraction.
        slip : float, default 0.0
            Slippage as a fraction of price.
        size : float, default 1.0
            Position size multiplier.

        Returns
        -------
        Backtrade
            Portfolio backtest results object.

        Examples
        --------
        >>> bt = Backtrade.portfolio(data, entries, exits, init_cash=100000, size=0.5)
        >>> bt.summary()  # Prints portfolio performance summary
        """

    def results(self) -> pl.DataFrame | None: 
        """Return the backtest equity/cash DataFrame, or None if not available."""

    def trades(self) -> pl.DataFrame | None: 
        """Return the trade log DataFrame, or None if not available."""

    def summary(self) -> None: 
        """
        Print a comprehensive summary of backtest performance to the console.

        Includes overall statistics like total return, Sharpe ratio, max drawdown,
        and per-symbol breakdowns with win rates and profit factors.
        """

class Portfolio:
    """
    Portfolio class for backtesting with shared capital across multiple symbols.

    This class provides portfolio-level backtesting where all positions share
    the same capital pool, enabling realistic risk management and position sizing
    across correlated assets.

    Attributes
    ----------
    results : pl.DataFrame | None
        DataFrame containing portfolio equity and cash over time.
    trades : pl.DataFrame | None
        DataFrame of executed trades.
    _summary : dict | None
        Cached summary statistics.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_quant import Portfolio
    >>>
    >>> # Sample multi-symbol data
    >>> data = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [100, 105, 110],
    ...     "TSLA": [200, 195, 210],
    ...     "GOOGL": [150, 152, 148],
    ... })
    >>>
    >>> entries = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [True, False, False],
    ...     "TSLA": [False, True, False],
    ...     "GOOGL": [False, False, True],
    ... })
    >>>
    >>> exits = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [False, False, True],
    ...     "TSLA": [False, False, True],
    ...     "GOOGL": [False, True, False],
    ... })
    >>>
    >>> port = Portfolio.run(data, entries, exits, init_cash=200000, fee=0.002)
    >>> port.summary()  # Prints portfolio summary
    >>> equity = port.results()
    """

    results: pl.DataFrame | None
    """DataFrame of portfolio equity and cash over time."""

    trades: pl.DataFrame | None
    """DataFrame of executed trades."""

    _summary: dict | None
    """Cached summary statistics."""

    def __init__(
        self,
        results: pl.DataFrame | None = None,
        trades: pl.DataFrame | None = None
    ) -> None:
        """Initialize a Portfolio object with optional results and trades."""

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
        init_cash: float = 100_000.0,
        fee: float = 0.0,
        slip: float = 0.0,
        size: float = 1.0,
    ) -> "Portfolio":
        """
        Run portfolio-level backtest with shared capital.

        Parameters
        ----------
        data : pl.DataFrame
            Price data with dates and multiple symbols.
        entries : pl.DataFrame
            Entry signals for each symbol.
        exits : pl.DataFrame
            Exit signals for each symbol.
        init_cash : float, default 100000.0
            Total portfolio capital.
        fee : float, default 0.0
            Trading fee fraction.
        slip : float, default 0.0
            Slippage fraction.
        size : float, default 1.0
            Position size multiplier.

        Returns
        -------
        Portfolio
            Portfolio backtest results.

        Examples
        --------
        >>> port = Portfolio.run(data, entries, exits, init_cash=150000, size=0.8)
        >>> port.summary()
        """

    def results(self) -> pl.DataFrame | None:
        """Return portfolio equity/cash DataFrame."""

    def trades(self) -> pl.DataFrame | None:
        """Return trade log DataFrame."""

    def summary(self) -> None:
        """
        Print portfolio performance summary to the console.

        Includes total return, Sharpe ratio, drawdown, trade statistics,
        win rate, and profit factor for the entire portfolio.
        """


# Stock data functions from qstock.rs

def history(
    stock_code: str,
    scale: int = 240,
    datalen: int = 365 * 10,
    timeout: int = 10,
) -> pl.DataFrame | None:
    """
    Fetch historical stock data from Sina Finance API.

    Retrieves OHLCV (Open, High, Low, Close, Volume) data for a given stock code.

    Parameters
    ----------
    stock_code : str
        Stock symbol/code (e.g., '000001' for Shanghai Composite).
    scale : int, default 240
        Time scale in minutes (e.g., 240 for daily data).
    datalen : int, default 3650
        Number of data points to retrieve (approximately 10 years of daily data).
    timeout : int, default 10
        Request timeout in seconds.

    Returns
    -------
    pl.DataFrame | None
        DataFrame with columns: date, open, close, high, low, volume.
        Returns None if data fetch fails.

    Examples
    --------
    >>> df = history('000001', scale=240, datalen=100)
    >>> print(df.head())
    """


def history_save(
    stock_code: str,
    scale: int = 240,
    datalen: int = 365 * 10,
    timeout: int = 10,
) -> None:
    """
    Fetch and save historical stock data to a Parquet file.

    Retrieves stock data and saves it as a compressed Parquet file.

    Parameters
    ----------
    stock_code : str
        Stock symbol/code.
    scale : int, default 240
        Time scale in minutes.
    datalen : int, default 3650
        Number of data points to retrieve.
    timeout : int, default 10
        Request timeout in seconds.

    Examples
    --------
    >>> history_save('000001', datalen=500)
    # Saves data to '000001.parquet'
    """


def info() -> pl.DataFrame:
    """
    Fetch information for all A-share stocks from Sina Finance.

    Retrieves basic information including stock codes and names.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: symbol, name.

    Examples
    --------
    >>> stock_info = info()
    >>> print(stock_info.head())
    """


def info_save(path: str) -> None:
    """
    Fetch and save stock information to a Parquet file.

    Parameters
    ----------
    path : str
        File path to save the Parquet file.

    Examples
    --------
    >>> info_save('stocks.parquet')
    """


# Technical analysis functions from qtalib.rs

def bband(
    data: pl.DataFrame,
    timeperiod: int = 5,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> list[pl.Series]:
    """
    Bollinger Bands.

    Calculates upper and lower Bollinger Bands based on moving average and standard deviation.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price columns.
    timeperiod : int, default 5
        Period for moving average.
    nbdevup : float, default 2.0
        Standard deviations for upper band.
    nbdevdn : float, default 2.0
        Standard deviations for lower band.

    Returns
    -------
    list[pl.Series]
        New Series: middle, upper, lower bands.

    Examples
    --------
    >>> result = bband(data, timeperiod=20)
    """


def dema(data: pl.DataFrame, timeperiod: int = 30) -> list[pl.Series]:
    """
    Double Exponential Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 30
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        DEMA series list.
    """


def ema(data: pl.DataFrame, timeperiod: int = 30) -> list[pl.Series]:
    """
    Exponential Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 30
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        EMA series list.
    """


def kama(
    data: pl.DataFrame,
    timeperiod: int = 30,
    fast_limit: float = 2.0,
    slow_limit: float = 30.0,
) -> list[pl.Series]:
    """
    Kaufman Adaptive Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 30
        Period for calculation.
    fast_limit : float, default 2.0
        Fast limit parameter.
    slow_limit : float, default 30.0
        Slow limit parameter.

    Returns
    -------
    list[pl.Series]
        KAMA series list.
    """


def ma(data: pl.DataFrame, timeperiod: int = 30) -> list[pl.Series]:
    """
    Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 30
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        MA series list.
    """


def mama(data: pl.DataFrame, c: float = 10.0) -> list[pl.Series]:
    """
    MESA Adaptive Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    c : float, default 10.0
        Cycle limit parameter.

    Returns
    -------
    list[pl.Series]
        MESA adaptive series list.
    """


def mavp(data: pl.DataFrame, timeperiod: int = 30) -> list[pl.Series]:
    """
    Moving Average with Variable Period.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 30
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        MAVP series list.
    """


def sma(data: pl.DataFrame, timeperiod: int = 20) -> list[pl.Series]:
    """
    Simple Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 20
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        SMA series list.
    """


def t3(data: pl.DataFrame, timeperiod: int = 20, b: float = 0.7) -> list[pl.Series]:
    """
    Triple Exponential Moving Average (T3).

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 20
        Period for calculation.
    b : float, default 0.7
        Volume factor.

    Returns
    -------
    list[pl.Series]
        T3 series list.
    """


def tema(data: pl.DataFrame, timeperiod: int = 20) -> list[pl.Series]:
    """
    Triple Exponential Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 20
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        TEMA series list.
    """


def trima(data: pl.DataFrame, timeperiod: int = 20) -> list[pl.Series]:
    """
    Triangular Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 20
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        TRIMA series list.
    """


def wma(data: pl.DataFrame, timeperiod: int = 20) -> list[pl.Series]:
    """
    Weighted Moving Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 20
        Period for calculation.

    Returns
    -------
    list[pl.Series]
        WMA series list.
    """


def adx(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Average Directional Movement Index.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with ADX columns added.
    """


def adxr(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Average Directional Movement Index Rating.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with ADXR columns added.
    """


def apo(
    data: pl.DataFrame,
    fastperiod: int = 12,
    slowperiod: int = 26,
) -> list[pl.Series]:
    """
    Absolute Price Oscillator.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    fastperiod : int, default 12
        Fast period.
    slowperiod : int, default 26
        Slow period.

    Returns
    -------
    pl.DataFrame
        DataFrame with APO columns added.
    """


def aroon(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Aroon Indicator.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with Aroon columns added.
    """


def aroonosc(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Aroon Oscillator.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with Aroon Oscillator columns added.
    """


def bop(data: pl.DataFrame) -> pl.DataFrame:
    """
    Balance of Power.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.

    Returns
    -------
    pl.DataFrame
        DataFrame with BOP columns added.
    """


def cci(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Commodity Channel Index.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with CCI columns added.
    """


def cmo(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Chande Momentum Oscillator.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with CMO columns added.
    """


def dx(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Directional Movement Index.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with DX columns added.
    """


def macd(
    data: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> list[pl.Series]:
    """
    Moving Average Convergence Divergence.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    fast : int, default 12
        Fast period.
    slow : int, default 26
        Slow period.
    signal : int, default 9
        Signal period.

    Returns
    -------
    pl.DataFrame
        DataFrame with MACD columns added.
    """


def mfi(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Money Flow Index.

    Parameters
    ----------
    data : pl.DataFrame
        OHLCV data.
    period : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with MFI columns added.
    """


def mom(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Momentum.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    period : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with MOM columns added.
    """


def ppo(
    data: pl.DataFrame,
    fastperiod: int = 12,
    slowperiod: int = 26,
) -> list[pl.Series]:
    """
    Percentage Price Oscillator.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    fastperiod : int, default 12
        Fast period.
    slowperiod : int, default 26
        Slow period.

    Returns
    -------
    pl.DataFrame
        DataFrame with PPO columns added.
    """


def roc(data: pl.DataFrame, timeperiod: int = 10) -> pl.DataFrame:
    """
    Rate of Change.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 10
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with ROC columns added.
    """


def rsi(data: pl.DataFrame, timeperiod: int = 14) -> list[pl.Series]:
    """
    Relative Strength Index.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with RSI columns added.
    """


def stoch(
    data: pl.DataFrame,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pl.DataFrame:
    """
    Stochastic Oscillator.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    fastk_period : int, default 5
        Fast K period.
    slowk_period : int, default 3
        Slow K period.
    slowd_period : int, default 3
        Slow D period.

    Returns
    -------
    pl.DataFrame
        DataFrame with Stochastic columns added.
    """


def stochf(
    data: pl.DataFrame,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.DataFrame:
    """
    Stochastic Fast.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    fastk_period : int, default 5
        Fast K period.
    fastd_period : int, default 3
        Fast D period.

    Returns
    -------
    pl.DataFrame
        DataFrame with Stochastic Fast columns added.
    """


def stochrsi(
    data: pl.DataFrame,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.DataFrame:
    """
    Stochastic RSI.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 14
        RSI period.
    fastk_period : int, default 5
        Fast K period.
    fastd_period : int, default 3
        Fast D period.

    Returns
    -------
    pl.DataFrame
        DataFrame with Stochastic RSI columns added.
    """


def trix(data: pl.DataFrame, timeperiod: int = 30) -> pl.DataFrame:
    """
    Triple Exponential Average.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    timeperiod : int, default 30
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with TRIX columns added.
    """


def ultosc(
    data: pl.DataFrame,
    short_period: int = 7,
    medium_period: int = 14,
    long_period: int = 28,
) -> pl.DataFrame:
    """
    Ultimate Oscillator.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    short_period : int, default 7
        Short period.
    medium_period : int, default 14
        Medium period.
    long_period : int, default 28
        Long period.

    Returns
    -------
    pl.DataFrame
        DataFrame with Ultimate Oscillator columns added.
    """


def willr(data: pl.DataFrame, timeperiod: int = 14) -> pl.DataFrame:
    """
    Williams' %R.

    Parameters
    ----------
    data : pl.DataFrame
        OHLC data.
    timeperiod : int, default 14
        Period for calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with Williams' %R columns added.
    """


def ad(data: pl.DataFrame) -> pl.DataFrame:
    """
    Accumulation/Distribution Line.

    Parameters
    ----------
    data : pl.DataFrame
        OHLCV data.

    Returns
    -------
    pl.DataFrame
        DataFrame with AD columns added.
    """


def adosc(data: pl.DataFrame, fastperiod: int = 3, slowperiod: int = 10) -> pl.DataFrame:
    """
    Chaikin A/D Oscillator.

    Parameters
    ----------
    data : pl.DataFrame
        OHLCV data.
    fastperiod : int, default 3
        Fast period.
    slowperiod : int, default 10
        Slow period.

    Returns
    -------
    pl.DataFrame
        DataFrame with AD Oscillator columns added.
    """


def obv(data: pl.DataFrame) -> pl.DataFrame:
    """
    On Balance Volume.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with close and volume.

    Returns
    -------
    pl.DataFrame
        DataFrame with OBV columns added.
    """
