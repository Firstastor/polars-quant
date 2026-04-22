# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
import inspect
from pathlib import Path

_LIB = Path(__file__).parent.parent / "polars_quant.abi3.so"

import polars as pl

def ADX(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int=14) -> pl.Series:
    """ADX - Average Directional Movement Index"""
    expr = register_plugin_function(args=[high, low, close, timeperiod], plugin_path=_LIB, function_name='adx', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def ADXR(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int=14) -> pl.Series:
    """ADXR - Average Directional Movement Index Rating"""
    expr = register_plugin_function(args=[high, low, close, timeperiod], plugin_path=_LIB, function_name='adxr', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def APO(real: pl.Series, fastperiod: int=12, slowperiod: int=26, matype: int=0) -> pl.Series:
    """APO - Absolute Price Oscillator"""
    expr = register_plugin_function(args=[real, fastperiod, slowperiod, matype], plugin_path=_LIB, function_name='apo', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def AROON(high: pl.Series, low: pl.Series, timeperiod: int=14) -> tuple[pl.Series, pl.Series]:
    """AROON - Aroon"""
    expr = register_plugin_function(args=[high, low, timeperiod], plugin_path=_LIB, function_name='aroon', is_elementwise=False)
    if isinstance(high, pl.Series):
        df = high.to_frame().select(expr).unnest(expr.meta.output_name())
        return tuple((df[col] for col in df.columns))
    return (expr.struct.field('aroon_up'), expr.struct.field('aroon_down'))

def AROONOSC(high: pl.Series, low: pl.Series, timeperiod: int=14) -> pl.Series:
    """AROONOSC - Aroon Oscillator"""
    expr = register_plugin_function(args=[high, low, timeperiod], plugin_path=_LIB, function_name='aroonosc', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def BOP(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """BOP - Balance Of Power"""
    expr = register_plugin_function(args=[open, high, low, close], plugin_path=_LIB, function_name='bop', is_elementwise=False)
    if isinstance(open, pl.Series):
        return open.to_frame().select(expr).to_series()
    return expr

def CCI(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int=14) -> pl.Series:
    """CCI - Commodity Channel Index"""
    expr = register_plugin_function(args=[high, low, close, timeperiod], plugin_path=_LIB, function_name='cci', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def CMO(real: pl.Series, timeperiod: int=14) -> pl.Series:
    """CMO - Chande Momentum Oscillator"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='cmo', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def DX(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int=14) -> pl.Series:
    """DX - Directional Movement Index"""
    expr = register_plugin_function(args=[high, low, close, timeperiod], plugin_path=_LIB, function_name='dx', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def MACD(real: pl.Series, fastperiod: int=12, slowperiod: int=26, signalperiod: int=9) -> tuple[pl.Series, pl.Series, pl.Series]:
    """MACD - Moving Average Convergence/Divergence (MACD, Signal, Hist)"""
    expr = register_plugin_function(args=[real, fastperiod, slowperiod, signalperiod], plugin_path=_LIB, function_name='macd', is_elementwise=False)
    if isinstance(real, pl.Series):
        df = real.to_frame().select(expr).unnest(expr.meta.output_name())
        return tuple((df[col] for col in df.columns))
    return (expr.struct.field('macd'), expr.struct.field('macd_signal'), expr.struct.field('macd_hist'))

def MACDEXT(real: pl.Series, fastperiod: int=12, fastmatype: int=0, slowperiod: int=26, slowmatype: int=0, signalperiod: int=9, signalmatype: int=0) -> tuple[pl.Series, pl.Series, pl.Series]:
    """MACDEXT - MACD with controllable MA type"""
    from .overlap import MA
    macd_line = MA(real, fastperiod, fastmatype) - MA(real, slowperiod, slowmatype)
    signal_line = MA(macd_line, signalperiod, signalmatype)
    return (macd_line.alias('macd_dif'), signal_line.alias('macd_dea'), (macd_line - signal_line).alias('macd_hist'))

def MACDFIX(real: pl.Series, signalperiod: int=9) -> tuple[pl.Series, pl.Series, pl.Series]:
    """MACDFIX - Moving Average Convergence/Divergence Fixed 12/26/9"""
    return MACD(real, 12, 26, signalperiod)

def MFI(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series, timeperiod: int=14) -> pl.Series:
    """MFI - Money Flow Index"""
    expr = register_plugin_function(args=[high, low, close, volume, timeperiod], plugin_path=_LIB, function_name='mfi', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def MINUS_DI(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int=14) -> pl.Series:
    """MINUS_DI - Minus Directional Indicator"""
    expr = register_plugin_function(args=[high, low, close, timeperiod], plugin_path=_LIB, function_name='minus_di', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def MINUS_DM(high: pl.Series, low: pl.Series, timeperiod: int=14) -> pl.Series:
    """MINUS_DM - Minus Directional Movement"""
    expr = register_plugin_function(args=[high, low, timeperiod], plugin_path=_LIB, function_name='minus_dm', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def MOM(real: pl.Series, timeperiod: int=10) -> pl.Series:
    """MOM - Momentum"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='mom', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def PLUS_DI(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int=14) -> pl.Series:
    """PLUS_DI - Plus Directional Indicator"""
    expr = register_plugin_function(args=[high, low, close, timeperiod], plugin_path=_LIB, function_name='plus_di', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def PLUS_DM(high: pl.Series, low: pl.Series, timeperiod: int=14) -> pl.Series:
    """PLUS_DM - Plus Directional Movement"""
    expr = register_plugin_function(args=[high, low, timeperiod], plugin_path=_LIB, function_name='plus_dm', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def PPO(real: pl.Series, fastperiod: int=12, slowperiod: int=26, matype: int=0) -> pl.Series:
    """PPO - Percentage Price Oscillator"""
    expr = register_plugin_function(args=[real, fastperiod, slowperiod, matype], plugin_path=_LIB, function_name='ppo', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def ROC(real: pl.Series, timeperiod: int=10) -> pl.Series:
    """ROC - Rate of change"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='roc', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def ROCP(real: pl.Series, timeperiod: int=10) -> pl.Series:
    """ROCP - Rate of change Percentage"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='rocp', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def ROCR(real: pl.Series, timeperiod: int=10) -> pl.Series:
    """ROCR - Rate of change ratio: (price/prevPrice)"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='rocr', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def ROCR100(real: pl.Series, timeperiod: int=10) -> pl.Series:
    """ROCR100 - Rate of change ratio 100: (price/prevPrice)*100"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='rocr100', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def RSI(real: pl.Series, timeperiod: int=14) -> pl.Series:
    """RSI - Relative Strength Index"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='rsi', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def STOCH(high: pl.Series, low: pl.Series, close: pl.Series, fastk_period: int=5, slowk_period: int=3, slowk_matype: int=0, slowd_period: int=3, slowd_matype: int=0) -> tuple[pl.Series, pl.Series]:
    """STOCH - Stochastic (SlowK, SlowD)"""
    from .overlap import MA
    ln = low.rolling_min(fastk_period)
    hn = high.rolling_max(fastk_period)
    fastk = (close - ln) * 100.0 / (hn - ln)
    slowk = MA(fastk, slowk_period, slowk_matype).alias('slowk')
    slowd = MA(slowk, slowd_period, slowd_matype).alias('slowd')
    return (slowk, slowd)

def STOCHF(high: pl.Series, low: pl.Series, close: pl.Series, fastk_period: int=5, fastd_period: int=3, fastd_matype: int=0) -> tuple[pl.Series, pl.Series]:
    """STOCHF - Stochastic Fast (FastK, FastD)"""
    from .overlap import MA
    ln = low.rolling_min(fastk_period)
    hn = high.rolling_max(fastk_period)
    fastk = ((close - ln) * 100.0 / (hn - ln)).alias('fastk')
    fastd = MA(fastk, fastd_period, fastd_matype).alias('fastd')
    return (fastk, fastd)

def STOCHRSI(real: pl.Series, timeperiod: int=14, fastk_period: int=5, fastd_period: int=3, fastd_matype: int=0) -> tuple[pl.Series, pl.Series]:
    """STOCHRSI - Stochastic Relative Strength Index (FastK, FastD)"""
    from .overlap import MA
    rsi = RSI(real, timeperiod)
    ln = rsi.rolling_min(fastk_period)
    hn = rsi.rolling_max(fastk_period)
    fastk = ((rsi - ln) * 100.0 / (hn - ln)).alias('fastk_rsi')
    fastd = MA(fastk, fastd_period, fastd_matype).alias('fastd_rsi')
    return (fastk, fastd)

def TRIX(real: pl.Series, timeperiod: int=30) -> pl.Series:
    """TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
    expr = register_plugin_function(args=[real, timeperiod], plugin_path=_LIB, function_name='trix', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def ULTOSC(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod1: int=7, timeperiod2: int=14, timeperiod3: int=28) -> pl.Series:
    """ULTOSC - Ultimate Oscillator"""
    expr = register_plugin_function(args=[high, low, close, timeperiod1, timeperiod2, timeperiod3], plugin_path=_LIB, function_name='ultosc', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def WILLR(high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int=14) -> pl.Series:
    """WILLR - Williams' %R"""
    expr = register_plugin_function(args=[high, low, close, timeperiod], plugin_path=_LIB, function_name='willr', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr