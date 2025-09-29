"""
polars-quant 类型提示文件
提供所有技术分析函数的类型注释和使用示例
"""

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
    ...

def mfi(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series, period: int) -> pl.Series:
    """
    资金流量指标 (Money Flow Index)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        period: 计算周期 (通常为14)
    
    Returns:
        MFI序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.mfi(high, low, close, volume, 14)
    """
    ...

def mom(series: pl.Series, period: int) -> pl.Series:
    """
    动量指标 (Momentum)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        动量序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.mom(prices, 10)
    """
    ...

def trix(series: pl.Series, period: int) -> pl.Series:
    """
    TRIX指标 (1-day Rate-Of-Change (ROC) of a Triple Smooth EMA)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为14)
    
    Returns:
        TRIX序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.trix(prices, 14)
    """
    ...

def roc(series: pl.Series, period: int) -> pl.Series:
    """
    变化率 (Rate of change)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROC序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.roc(prices, 10)
    """
    ...

def rocp(series: pl.Series, period: int) -> pl.Series:
    """
    变化率百分比 (Rate of change Percentage)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROCP序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.rocp(prices, 10)
    """
    ...

def rocr(series: pl.Series, period: int) -> pl.Series:
    """
    变化率比率 (Rate of change ratio)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROCR序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.rocr(prices, 10)
    """
    ...

def rocr100(series: pl.Series, period: int) -> pl.Series:
    """
    变化率比率*100 (Rate of change ratio 100 scale)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROCR100序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.rocr100(prices, 10)
    """
    ...

# ====================================================================
# 成交量指标 (Volume Indicators) - 成交量相关指标
# ====================================================================

def ad(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series) -> pl.Series:
    """
    累积/派发线 (Accumulation/Distribution Line)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
    
    Returns:
        A/D线序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.ad(high, low, close, volume)
    """
    ...

def adosc(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series, fast_period: int, slow_period: int) -> pl.Series:
    """
    累积/派发摆动指标 (Accumulation/Distribution Oscillator)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        fast_period: 快速周期 (通常为3)
        slow_period: 慢速周期 (通常为10)
    
    Returns:
        A/D摆动指标序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.adosc(high, low, close, volume, 3, 10)
    """
    ...

def obv(close: pl.Series, volume: pl.Series) -> pl.Series:
    """
    能量潮指标 (On Balance Volume)
    
    Args:
        close: 收盘价序列
        volume: 成交量序列
    
    Returns:
        OBV序列
    
    Example:
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.obv(close, volume)
    """
    ...

# ====================================================================
# 波动率指标 (Volatility Indicators) - 价格波动测量
# ====================================================================

def atr(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    平均真实波幅 (Average True Range)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期 (通常为14)
    
    Returns:
        ATR序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.atr(high, low, close, 14)
    """
    ...

def natr(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    标准化平均真实波幅 (Normalized Average True Range)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期 (通常为14)
    
    Returns:
        NATR序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.natr(high, low, close, 14)
    """
    ...

def trange(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    真实波幅 (True Range)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        真实波幅序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.trange(high, low, close)
    """
    ...

# ====================================================================
# 价格变换函数 (Price Transform) - 价格数据变换
# ====================================================================

def avgprice(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    平均价格 (Average Price)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        平均价格序列 (OHLC/4)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.avgprice(open, high, low, close)
    """
    ...

def medprice(high: pl.Series, low: pl.Series) -> pl.Series:
    """
    中位价格 (Median Price)
    
    Args:
        high: 最高价序列
        low: 最低价序列
    
    Returns:
        中位价格序列 (HL/2)
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.medprice(high, low)
    """
    ...

def typprice(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    典型价格 (Typical Price)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        典型价格序列 (HLC/3)
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.typprice(high, low, close)
    """
    ...

def wclprice(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    加权收盘价格 (Weighted Close Price)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        加权收盘价格序列 (HLC2/4)
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.wclprice(high, low, close)
    """
    ...

# ====================================================================
# 周期指标 (Cycle Indicators) - 希尔伯特变换系列
# ====================================================================

def ht_trendline(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-趋势线 (Hilbert Transform - Instantaneous Trendline)
    
    Args:
        series: 价格序列
    
    Returns:
        瞬时趋势线序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_trendline(prices)
    """
    ...

def ht_dcperiod(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-主导周期 (Hilbert Transform - Dominant Cycle Period)
    
    Args:
        series: 价格序列
    
    Returns:
        主导周期序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_dcperiod(prices)
    """
    ...

def ht_dcphase(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-主导周期相位 (Hilbert Transform - Dominant Cycle Phase)
    
    Args:
        series: 价格序列
    
    Returns:
        主导周期相位序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_dcphase(prices)
    """
    ...

def ht_phasor(series: pl.Series) -> Tuple[pl.Series, pl.Series]:
    """
    希尔伯特变换-相量分量 (Hilbert Transform - Phasor Components)
    
    Args:
        series: 价格序列
    
    Returns:
        Tuple[同相分量, 正交分量]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> inphase, quadrature = pq.ht_phasor(prices)
    """
    ...

def ht_sine(series: pl.Series) -> Tuple[pl.Series, pl.Series]:
    """
    希尔伯特变换-正弦波 (Hilbert Transform - SineWave)
    
    Args:
        series: 价格序列
    
    Returns:
        Tuple[正弦波, 余弦波]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> sine, leadsine = pq.ht_sine(prices)
    """
    ...

def ht_trendmode(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-趋势模式 (Hilbert Transform - Trend vs Cycle Mode)
    
    Args:
        series: 价格序列
    
    Returns:
        趋势模式序列 (0=周期模式, 1=趋势模式)
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_trendmode(prices)
    """
    ...

# ====================================================================
# 蜡烛图模式识别 (Candlestick Pattern Recognition) - K线形态
# ====================================================================

def cdldoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    十字星 (Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldoji(open, high, low, close)
    """
    ...

def cdldojistar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    十字星 (Doji Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldojistar(open, high, low, close)
    """
    ...

def cdldragonflydoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    蜻蜓十字 (Dragonfly Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldragonflydoji(open, high, low, close)
    """
    ...

def cdlgravestonedoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    墓石十字 (Gravestone Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlgravestonedoji(open, high, low, close)
    """
    ...

def cdllongleggeddoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    长腿十字 (Long Legged Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdllongleggeddoji(open, high, low, close)
    """
    ...

def cdlhammer(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    锤头线 (Hammer)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhammer(open, high, low, close)
    """
    ...

def cdlhangingman(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    上吊线 (Hanging Man)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhangingman(open, high, low, close)
    """
    ...

def cdlinvertedhammer(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    倒锤头线 (Inverted Hammer)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlinvertedhammer(open, high, low, close)
    """
    ...

def cdlshootingstar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    射击之星 (Shooting Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlshootingstar(open, high, low, close)
    """
    ...

def cdlengulfing(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    吞没形态 (Engulfing Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlengulfing(open, high, low, close)
    """
    ...

def cdlharami(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    孕线形态 (Harami Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlharami(open, high, low, close)
    """
    ...

def cdlharamicross(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    孕线十字 (Harami Cross Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlharamicross(open, high, low, close)
    """
    ...

def cdlmorningstar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    早晨之星 (Morning Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmorningstar(open, high, low, close, 0.3)
    """
    ...

def cdleveningstar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    黄昏之星 (Evening Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdleveningstar(open, high, low, close, 0.3)
    """
    ...

def cdlmorningdojistar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    早晨十字星 (Morning Doji Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmorningdojistar(open, high, low, close, 0.3)
    """
    ...

def cdleveningdojistar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    黄昏十字星 (Evening Doji Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdleveningdojistar(open, high, low, close, 0.3)
    """
    ...

def cdl3blackcrows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三只乌鸦 (Three Black Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3blackcrows(open, high, low, close)
    """
    ...

def cdl3whitesoldiers(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三个白武士 (Three Advancing White Soldiers)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3whitesoldiers(open, high, low, close)
    """
    ...

def cdlpiercing(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    穿透形态 (Piercing Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlpiercing(open, high, low, close)
    """
    ...

def cdldarkcloudcover(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    乌云盖顶 (Dark Cloud Cover)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透百分比 (通常为0.5)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldarkcloudcover(open, high, low, close, 0.5)
    """
    ...

def cdlabandonedbaby(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    弃婴形态 (Abandoned Baby)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透百分比 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlabandonedbaby(open, high, low, close, 0.3)
    """
    ...

def cdltristar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三星形态 (Tri-Star Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdltristar(open, high, low, close)
    """
    ...

def cdl3inside(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三内部上升/下降 (Three Inside Up/Down)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3inside(open, high, low, close)
    """
    ...

def cdl3outside(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三外部上升/下降 (Three Outside Up/Down)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3outside(open, high, low, close)
    """
    ...

def cdl3linestrike(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三线攻击 (Three-Line Strike)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3linestrike(open, high, low, close)
    """
    ...

def cdl3starsinsouth(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    南方三星 (Three Stars In The South)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3starsinsouth(open, high, low, close)
    """
    ...

def cdlidentical3crows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    相同三乌鸦 (Identical Three Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlidentical3crows(open, high, low, close)
    """
    ...

def cdlgapsidesidewhite(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    向上跳空并列阳线 (Up-side Gap Two Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlgapsidesidewhite(open, high, low, close)
    """
    ...

def cdlupsidegap2crows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    向上跳空两只乌鸦 (Upside Gap Two Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlupsidegap2crows(open, high, low, close)
    """
    ...

def cdltasukigap(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    跳空并列阴阳线 (Tasuki Gap)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdltasukigap(open, high, low, close)
    """
    ...

def cdlxsidegap3methods(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    上升/下降跳空三法 (Upside/Downside Gap Three Methods)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlxsidegap3methods(open, high, low, close)
    """
    ...

def cdl2crows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    两只乌鸦 (Two Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl2crows(open, high, low, close)
    """
    ...

def cdladvanceblock(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    前进阻挡 (Advance Block)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdladvanceblock(open, high, low, close)
    """
    ...

def cdlbelthold(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    捉腰带线 (Belt-hold)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlbelthold(open, high, low, close)
    """
    ...

def cdlbreakaway(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    脱离形态 (Breakaway)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlbreakaway(open, high, low, close)
    """
    ...

def cdlclosingmarubozu(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    收盘秃头 (Closing Marubozu)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlclosingmarubozu(open, high, low, close)
    """
    ...

def cdlconcealbabyswall(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    藏婴吞没 (Concealing Baby Swallow)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlconcealbabyswall(open, high, low, close)
    """
    ...

def cdlcounterattack(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    反击线 (Counterattack)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlcounterattack(open, high, low, close)
    """
    ...

def cdlhighwave(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    长影线 (High-Wave Candle)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhighwave(open, high, low, close)
    """
    ...

def cdlhikkake(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    陷阱形态 (Hikkake Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhikkake(open, high, low, close)
    """
    ...

def cdlhikkakemod(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    修正陷阱形态 (Modified Hikkake Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhikkakemod(open, high, low, close)
    """
    ...

def cdlhomingpigeon(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    家鸽形态 (Homing Pigeon)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhomingpigeon(open, high, low, close)
    """
    ...

def cdlinneck(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    颈内线 (In-Neck Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlinneck(open, high, low, close)
    """
    ...

def cdlkicking(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    反冲形态 (Kicking)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlkicking(open, high, low, close)
    """
    ...

def cdlkickingbylength(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    由长度决定的反冲形态 (Kicking - bull/bear determined by the longer marubozu)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlkickingbylength(open, high, low, close)
    """
    ...

def cdlladderbottom(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    梯底形态 (Ladder Bottom)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlladderbottom(open, high, low, close)
    """
    ...

def cdllongline(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    长线形态 (Long Line Candle)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdllongline(open, high, low, close)
    """
    ...

def cdlmarubozu(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    秃头形态 (Marubozu)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmarubozu(open, high, low, close)
    """
    ...

def cdlmatchinglow(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    相同低价 (Matching Low)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmatchinglow(open, high, low, close)
    """
    ...

def cdlmathold(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    铺垫形态 (Mat Hold)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透百分比 (通常为0.5)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmathold(open, high, low, close, 0.5)
    """
    ...

def cdlonneck(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    颈上线 (On-Neck Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlonneck(open, high, low, close)
    """
    ...

def cdlrickshawman(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    黄包车夫 (Rickshaw Man)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlrickshawman(open, high, low, close)
    """
    ...

def cdlrisefall3methods(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    上升/下降三法 (Rising/Falling Three Methods)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlrisefall3methods(open, high, low, close)
    """
    ...

def cdlseparatinglines(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    分离线 (Separating Lines)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlseparatinglines(open, high, low, close)
    """
    ...

def cdlshortline(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    短线形态 (Short Line Candle)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlshortline(open, high, low, close)
    """
    ...

def cdlspinningtop(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    纺锤形态 (Spinning Top)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlspinningtop(open, high, low, close)
    """
    ...

def cdlstalledpattern(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    停顿形态 (Stalled Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlstalledpattern(open, high, low, close)
    """
    ...

def cdlsticksandwich(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    条形三明治 (Stick Sandwich)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlsticksandwich(open, high, low, close)
    """
    ...

def cdltakuri(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    探水竿 (Takuri - Dragonfly Doji with very long lower shadow)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdltakuri(open, high, low, close)
    """
    ...

def cdlthrusting(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    插入形态 (Thrusting Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlthrusting(open, high, low, close)
    """
    ...

def cdlunique3river(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    奇特三河床 (Unique 3 River)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlunique3river(open, high, low, close)
    """
    ...