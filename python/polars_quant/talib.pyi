import polars as pl
from typing import Union, List, Dict

class ATR:
    def __init__(self) -> None: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: int = 14) -> "ATR":
        """
        Calculate the Average True Range (ATR).

        Args:
            data: Input DataFrame with High, Low, Close columns.
            timeperiod: Period for ATR calculation (default=14).

        Returns:
            ATR instance with results stored in `.frame`.

        Example:
            >>> atr = ATR.run(df, timeperiod=14)
            >>> atr.frame["ATR"].head()
        """
        ...

class BBANDS:
    def __init__(self) -> None: ...
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 20,
        nbdevup: float = 2,
        nbdevdn: float = 2,
    ) -> "BBANDS":
        """
        Calculate Bollinger Bands (BBANDS).

        Args:
            data: Input DataFrame with numeric columns.
            timeperiod: Period for moving average (default=20).
            nbdevup: Number of standard deviations above.
            nbdevdn: Number of standard deviations below.

        Returns:
            BBANDS instance with results stored in `.frame`.

        Example:
            >>> bb = BBANDS.run(df, timeperiod=20)
            >>> bb.frame["Close"].head()
        """
        ...

class CCI:
    def __init__(self) -> None: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: int = 20) -> "CCI":
        """
        Calculate Commodity Channel Index (CCI).

        Args:
            data: DataFrame with High, Low, Close.
            timeperiod: Period for CCI (default=20).

        Returns:
            CCI instance with `.frame` containing CCI values.

        Example:
            >>> cci = CCI.run(df, timeperiod=20)
            >>> cci.frame["CCI"].tail()
        """
        ...

class EMA:
    def __init__(self) -> None: ...
    def cross(self, first_ma: int, second_ma: int) -> Dict[str, pl.DataFrame]:
        """
        Detect EMA crossover signals.

        Args:
            first_ma: First EMA period.
            second_ma: Second EMA period.

        Returns:
            Dictionary of DataFrames with boolean crossover signals.

        Example:
            >>> ema = EMA.run(df, [12, 26])
            >>> cross = ema.cross(12, 26)
        """
        ...
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: Union[int, List[int]],
        adjust: bool = False,
    ) -> "EMA":
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            data: Input DataFrame.
            timeperiod: EMA period(s).
            adjust: Use adjusted EMA (default=False).

        Returns:
            EMA instance.

        Example:
            >>> ema = EMA.run(df, [12, 26])
            >>> ema.frame["Close"].head()
        """
        ...

class KAMA:
    def __init__(self) -> None: ...
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 14,
        fast_period: int = 2,
        slow_period: int = 30,
    ) -> "KAMA":
        """
        Calculate Kaufman's Adaptive Moving Average (KAMA).

        Args:
            data: Input DataFrame.
            timeperiod: Base period (default=14).
            fast_period: Fast smoothing constant (default=2).
            slow_period: Slow smoothing constant (default=30).

        Returns:
            KAMA instance.

        Example:
            >>> kama = KAMA.run(df, timeperiod=10)
            >>> kama.frame["Close"].tail()
        """
        ...

class MA:
    def __init__(self) -> None: ...
    def cross(self, first_ma: int, second_ma: int) -> Dict[str, pl.DataFrame]:
        """
        Detect SMA/MA crossover signals.

        Args:
            first_ma: First MA period.
            second_ma: Second MA period.

        Returns:
            Dictionary of crossover signals.
        """
        ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: Union[int, List[int]]) -> "MA":
        """
        Calculate Simple Moving Average (SMA).

        Args:
            data: Input DataFrame.
            timeperiod: Window(s) for MA.

        Returns:
            MA instance.

        Example:
            >>> ma = MA.run(df, [10, 50])
            >>> ma.frame["Close"].head()
        """
        ...

class MACD:
    def __init__(self) -> None: ...
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> "MACD":
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Args:
            data: Input DataFrame.
            fastperiod: Fast EMA period.
            slowperiod: Slow EMA period.
            signalperiod: Signal line EMA.

        Returns:
            MACD instance.

        Example:
            >>> macd = MACD.run(df)
            >>> macd.frame["Close"].head()
        """
        ...

class NATR:
    def __init__(self) -> None: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: int = 14) -> "NATR":
        """
        Calculate Normalized ATR (NATR).

        Args:
            data: Input DataFrame with High, Low, Close.
            timeperiod: ATR period.

        Returns:
            NATR instance.
        """
        ...

class OBV:
    def __init__(self) -> None: ...
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        volume_col: str = "Volume",
        price_col: str = "Close",
    ) -> "OBV":
        """
        Calculate On-Balance Volume (OBV).

        Args:
            data: Input DataFrame.
            volume_col: Volume column name (default="Volume").
            price_col: Price column name (default="Close").

        Returns:
            OBV instance.
        """
        ...

class RSI:
    def __init__(self) -> None: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: int = 14) -> "RSI":
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: Input DataFrame.
            timeperiod: Period for RSI (default=14).

        Returns:
            RSI instance.

        Example:
            >>> rsi = RSI.run(df, 14)
            >>> rsi.frame["Close"].tail()
        """
        ...

class SMA:
    def __init__(self) -> None: ...
    def cross(self, first_ma: int, second_ma: int) -> Dict[str, pl.DataFrame]: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: Union[int, List[int]]) -> "SMA":
        """
        Calculate Simple Moving Average (SMA).

        Args:
            data: Input DataFrame.
            timeperiod: Window(s).

        Returns:
            SMA instance.
        """
        ...

class TEMA:
    def __init__(self) -> None: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: int = 14) -> "TEMA":
        """
        Calculate Triple Exponential Moving Average (TEMA).

        Args:
            data: Input DataFrame.
            timeperiod: Period for EMA.

        Returns:
            TEMA instance.
        """
        ...

class TRANGE:
    def __init__(self) -> None: ...
    @classmethod
    def run(cls, data: pl.DataFrame) -> "TRANGE":
        """
        Calculate True Range (TRANGE).

        Args:
            data: Input DataFrame with High, Low, Close.

        Returns:
            TRANGE instance.
        """
        ...

class TRIMA:
    def __init__(self) -> None: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: int = 14) -> "TRIMA":
        """
        Calculate Triangular Moving Average (TRIMA).

        Args:
            data: Input DataFrame.
            timeperiod: Period.

        Returns:
            TRIMA instance.
        """
        ...

class VOL:
    def __init__(self) -> None: ...
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 20,
        method: str = "std",
    ) -> "VOL":
        """
        Calculate volatility indicator.

        Args:
            data: Input DataFrame.
            timeperiod: Window size (default=20).
            method: "std" or "range".

        Returns:
            VOL instance.
        """
        ...

class WMA:
    def __init__(self) -> None: ...
    def cross(self, first_ma: int, second_ma: int) -> Dict[str, pl.DataFrame]: ...
    @classmethod
    def run(cls, data: pl.DataFrame, timeperiod: Union[int, List[int]]) -> "WMA":
        """
        Calculate Weighted Moving Average (WMA).

        Args:
            data: Input DataFrame.
            timeperiod: Window(s).

        Returns:
            WMA instance.

        Example:
            >>> wma = WMA.run(df, [10, 20])
            >>> wma.frame["Close"].head()
        """
        ...
