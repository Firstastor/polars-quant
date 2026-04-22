from typing import Callable, Dict, Optional, Tuple

import polars as pl

class OrderContext:
    """Order context for safely delegating trades from Python strategies."""

    def buy(
        self, target_asset: str, target_quantity: float, execution_price: float
    ) -> None:
        """Commits a pending buy order."""
        ...

    def sell(
        self, target_asset: str, target_quantity: float, execution_price: float
    ) -> None:
        """Commits a pending sell order."""
        ...

class VectorizedBacktester:
    """Vectorized backtester for a single asset."""

    def __init__(
        self,
        price: pl.Series,
        buy_signal: pl.Series,
        sell_signal: pl.Series,
        benchmark: Optional[pl.DataFrame] = None,
        initial_capital: float = 100000.0,
        buy_slippage: float = 0.0,
        sell_slippage: float = 0.0,
        buy_commission_rate: float = 0.0003,
        sell_commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        position_size: float = 1.0,
    ) -> None:
        """Creates a new VectorizedBacktester instance."""
        ...

    def run(self) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
        """Executes the vectorized backtesting process.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
                - Positions DataFrame
                - Capital/Equity DataFrame
                - Summary metrics dictionary
        """
        ...

class SequentialBacktester:
    """Sequential backtesting environment designed for multi-asset portfolio evaluation."""

    def __init__(
        self,
        historical_data: pl.DataFrame,
        benchmark: Optional[pl.DataFrame] = None,
        initial_capital: float = 100000.0,
        buy_slippage: float = 0.0,
        sell_slippage: float = 0.0,
        buy_commission_rate: float = 0.0003,
        sell_commission_rate: float = 0.0003,
        minimum_commission_fee: float = 5.0,
    ) -> None:
        """Creates and configures a new SequentialBacktester instance."""
        ...

    def run(
        self, strategy_callback: Callable[[OrderContext, int], None]
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
        """Executes the event-driven sequential backtesting simulation.

        Args:
            strategy_callback (Callable[[OrderContext], None]): Function called per step.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
                - Positions DataFrame
                - Capital/Equity DataFrame
                - Summary metrics dictionary
        """
        ...

class talib:
    """TA-Lib bindings implemented in Rust."""

    @staticmethod
    def ht_dcperiod(real: pl.Series | pl.Expr) -> pl.Series | pl.Expr: ...
    @staticmethod
    def ht_dcphase(real: pl.Series | pl.Expr) -> pl.Series | pl.Expr: ...
    @staticmethod
    def ht_phasor(
        real: pl.Series | pl.Expr,
    ) -> Tuple[pl.Series | pl.Expr, pl.Series | pl.Expr]: ...
    @staticmethod
    def ht_sine(
        real: pl.Series | pl.Expr,
    ) -> Tuple[pl.Series | pl.Expr, pl.Series | pl.Expr]: ...
    @staticmethod
    def ht_trendline(real: pl.Series | pl.Expr) -> pl.Series | pl.Expr: ...
    @staticmethod
    def ht_trendmode(real: pl.Series | pl.Expr) -> pl.Series | pl.Expr: ...
    @staticmethod
    def mama(
        real: pl.Series | pl.Expr, fastlimit: float = 0.5, slowlimit: float = 0.05
    ) -> Tuple[pl.Series | pl.Expr, pl.Series | pl.Expr]: ...
    @staticmethod
    def sar(
        high: pl.Series | pl.Expr,
        low: pl.Series | pl.Expr,
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> pl.Series | pl.Expr: ...
    @staticmethod
    def sarext(
        high: pl.Series | pl.Expr,
        low: pl.Series | pl.Expr,
        startvalue: float = 0.0,
        offsetonreverse: float = 0.0,
        accelerationinitlong: float = 0.02,
        accelerationlong: float = 0.02,
        accelerationmaxlong: float = 0.2,
        accelerationinitshort: float = 0.02,
        accelerationshort: float = 0.02,
        accelerationmaxshort: float = 0.2,
    ) -> pl.Series | pl.Expr: ...
    @staticmethod
    def mavp(
        real: pl.Series | pl.Expr,
        periods: pl.Series | pl.Expr,
        minperiod: int = 2,
        maxperiod: int = 30,
        matype: int = 0,
    ) -> pl.Series | pl.Expr: ...
