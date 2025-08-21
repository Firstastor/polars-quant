import polars as pl

class Backtrade:
    """
    Backtrade class for vectorized backtesting using Polars DataFrames.
    Provides both per-symbol independent backtests (`run`) and
    portfolio-level backtests with shared capital (`portfolio`).

    Example
    -------
    >>> import polars as pl
    >>> from backtrade import Backtrade
    >>>
    >>> data = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [100, 105, 110],
    ...     "TSLA": [200, 195, 210],
    ... })
    >>>
    >>> entries = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [True, False, False],
    ...     "TSLA": [False, True, False],
    ... })
    >>>
    >>> exits = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     "AAPL": [False, False, True],
    ...     "TSLA": [False, False, True],
    ... })
    >>>
    >>> bt = Backtrade.run(data, entries, exits)
    >>> print(bt.summary())
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

        Example
        -------
        >>> bt = Backtrade.run(data, entries, exits, init_cash=50_000, fee=0.001)
        >>> results = bt.results()
        >>> trades = bt.trades()
        >>> print(bt.summary())
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

        Example
        -------
        >>> bt = Backtrade.portfolio(data, entries, exits, init_cash=100_000, size=0.5)
        >>> print(bt.summary())
        """

    def results(self) -> pl.DataFrame | None: 
        """Return the backtest equity/cash DataFrame, or None if not available."""

    def trades(self) -> pl.DataFrame | None: 
        """Return the trade log DataFrame, or None if not available."""

    def summary(self) -> str: 
        """Return a text summary of final equity and performance statistics."""
