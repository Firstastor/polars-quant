import polars as pl


def ADX(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """ADX - Average Directional Movement Index"""
    return (
        DX(high, low, close, timeperiod).ewm_mean(alpha=1.0 / timeperiod, adjust=False)
    ).rename("adx")


def ADXR(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """ADXR - Average Directional Movement Index Rating"""
    adx = ADX(high, low, close, timeperiod)
    return (adx + adx.shift(timeperiod)).rename("adxr") * 0.5


def APO(
    real: pl.Series, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0
) -> pl.Series:
    """APO - Absolute Price Oscillator"""
    from .overlap import MA

    return (MA(real, fastperiod, matype) - MA(real, slowperiod, matype)).rename("apo")


def AROON(
    high: pl.Series, low: pl.Series, timeperiod: int = 14
) -> tuple[pl.Series, pl.Series]:
    """AROON - Aroon"""
    return (
        (
            100.0 - high.rolling_rank(timeperiod, method="max") / timeperiod * 100.0
        ).rename("aroon_up"),
        (
            100.0 - low.rolling_rank(timeperiod, method="min") / timeperiod * 100.0
        ).rename("aroon_down"),
    )


def AROONOSC(high: pl.Series, low: pl.Series, timeperiod: int = 14) -> pl.Series:
    """AROONOSC - Aroon Oscillator"""
    up, down = AROON(high, low, timeperiod)
    return (up - down).rename("aroon_osc")


def BOP(
    open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series
) -> pl.Series:
    """BOP - Balance Of Power"""
    return ((close - open) / (high - low)).rename("bop")


def CCI(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """CCI - Commodity Channel Index"""
    tp = (high + low + close) / 3.0
    ma_tp = tp.rolling_mean(timeperiod)
    return (
        (tp - ma_tp) / ((tp - ma_tp).abs().rolling_mean(timeperiod) * 0.015)
    ).rename("cci")


def CMO(real: pl.Series, timeperiod: int = 14) -> pl.Series:
    """CMO - Chande Momentum Oscillator"""
    diff = real.diff()
    zero = pl.zeros(len(real), dtype=pl.Float64, eager=True)
    up = diff.zip_with(diff > 0, zero).rolling_sum(timeperiod)
    down = (-diff).zip_with(diff < 0, zero).rolling_sum(timeperiod)
    return ((up - down) * 100.0 / (up + down)).rename("cmo")


def DX(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """DX - Directional Movement Index"""
    pdi = PLUS_DI(high, low, close, timeperiod)
    mdi = MINUS_DI(high, low, close, timeperiod)
    return ((pdi - mdi).abs() * 100.0 / (pdi + mdi)).rename("dx")


def MACD(
    real: pl.Series,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """MACD - Moving Average Convergence/Divergence (MACD, Signal, Hist)"""
    macd_line = real.ewm_mean(span=fastperiod, adjust=False) - real.ewm_mean(
        span=slowperiod, adjust=False
    )
    signal_line = macd_line.ewm_mean(span=signalperiod, adjust=False)
    return (
        macd_line.rename("macd_dif"),
        signal_line.rename("macd_dea"),
        (macd_line - signal_line).rename("macd_hist"),
    )


def MACDEXT(
    real: pl.Series,
    fastperiod: int = 12,
    fastmatype: int = 0,
    slowperiod: int = 26,
    slowmatype: int = 0,
    signalperiod: int = 9,
    signalmatype: int = 0,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """MACDEXT - MACD with controllable MA type"""
    from .overlap import MA

    macd_line = MA(real, fastperiod, fastmatype) - MA(real, slowperiod, slowmatype)
    signal_line = MA(macd_line, signalperiod, signalmatype)
    return (
        macd_line.rename("macd_dif"),
        signal_line.rename("macd_dea"),
        (macd_line - signal_line).rename("macd_hist"),
    )


def MACDFIX(
    real: pl.Series, signalperiod: int = 9
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """MACDFIX - Moving Average Convergence/Divergence Fixed 12/26/9"""
    return MACD(real, 12, 26, signalperiod)


def MFI(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    volume: pl.Series,
    timeperiod: int = 14,
) -> pl.Series:
    """MFI - Money Flow Index"""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    diff = tp.diff()
    zero = pl.zeros(len(high), dtype=pl.Float64, eager=True)
    mr = mf.zip_with(diff > 0, zero).rolling_sum(timeperiod) / mf.zip_with(
        diff < 0, zero
    ).rolling_sum(timeperiod)
    return (100.0 - (100.0 / (1.0 + mr))).rename("mfi")


def MINUS_DI(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """MINUS_DI - Minus Directional Indicator"""
    from .volatility import ATR

    return (
        MINUS_DM(high, low, timeperiod) / ATR(high, low, close, timeperiod) * 100.0
    ).rename("minus_di")


def MINUS_DM(high: pl.Series, low: pl.Series, timeperiod: int = 14) -> pl.Series:
    """MINUS_DM - Minus Directional Movement"""
    up = high - high.shift(1)
    down = low.shift(1) - low
    zero = pl.zeros(len(high), dtype=pl.Float64, eager=True)
    return (
        down.zip_with((down > up) & (down > 0), zero)
        .ewm_mean(alpha=1.0 / timeperiod, adjust=False)
        .rename("minus_dm")
    )


def MOM(real: pl.Series, timeperiod: int = 10) -> pl.Series:
    """MOM - Momentum"""
    return real.diff(timeperiod).rename("mom")


def PLUS_DI(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """PLUS_DI - Plus Directional Indicator"""
    from .volatility import ATR

    return (
        PLUS_DM(high, low, timeperiod) / ATR(high, low, close, timeperiod) * 100.0
    ).rename("plus_di")


def PLUS_DM(high: pl.Series, low: pl.Series, timeperiod: int = 14) -> pl.Series:
    """PLUS_DM - Plus Directional Movement"""
    up = high - high.shift(1)
    down = low.shift(1) - low
    zero = pl.zeros(len(high), dtype=pl.Float64, eager=True)
    return (
        up.zip_with((up > down) & (up > 0), zero)
        .ewm_mean(alpha=1.0 / timeperiod, adjust=False)
        .rename("plus_dm")
    )


def PPO(
    real: pl.Series, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0
) -> pl.Series:
    """PPO - Percentage Price Oscillator"""
    from .overlap import MA

    return (
        (MA(real, fastperiod, matype) / MA(real, slowperiod, matype)) * 100.0 - 100.0
    ).rename("ppo")


def ROC(real: pl.Series, timeperiod: int = 10) -> pl.Series:
    """ROC - Rate of change"""
    return (real / real.shift(timeperiod) * 100.0 - 100.0).rename("roc")


def ROCP(real: pl.Series, timeperiod: int = 10) -> pl.Series:
    """ROCP - Rate of change Percentage"""
    return (real / real.shift(timeperiod) - 1.0).rename("rocp")


def ROCR(real: pl.Series, timeperiod: int = 10) -> pl.Series:
    """ROCR - Rate of change ratio: (price/prevPrice)"""
    return (real / real.shift(timeperiod)).rename("rocr")


def ROCR100(real: pl.Series, timeperiod: int = 10) -> pl.Series:
    """ROCR100 - Rate of change ratio 100: (price/prevPrice)*100"""
    return (real / real.shift(timeperiod) * 100.0).rename("rocr100")


def RSI(real: pl.Series, timeperiod: int = 14) -> pl.Series:
    """RSI - Relative Strength Index"""
    diff = real.diff()
    zero = pl.zeros(len(real), dtype=pl.Float64, eager=True)
    rs = diff.zip_with(diff > 0, zero).ewm_mean(alpha=1 / timeperiod, adjust=False) / (
        (-diff).zip_with(diff < 0, zero).ewm_mean(alpha=1 / timeperiod, adjust=False)
    )
    return (100.0 - (100.0 / (1.0 + rs))).rename("rsi")


def STOCH(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0,
) -> tuple[pl.Series, pl.Series]:
    """STOCH - Stochastic (SlowK, SlowD)"""
    from .overlap import MA

    ln = low.rolling_min(fastk_period)
    hn = high.rolling_max(fastk_period)
    fastk = (close - ln) * 100.0 / (hn - ln)
    slowk = MA(fastk, slowk_period, slowk_matype).rename("slowk")
    slowd = MA(slowk, slowd_period, slowd_matype).rename("slowd")
    return slowk, slowd


def STOCHF(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[pl.Series, pl.Series]:
    """STOCHF - Stochastic Fast (FastK, FastD)"""
    from .overlap import MA

    ln = low.rolling_min(fastk_period)
    hn = high.rolling_max(fastk_period)
    fastk = ((close - ln) * 100.0 / (hn - ln)).rename("fastk")
    fastd = MA(fastk, fastd_period, fastd_matype).rename("fastd")
    return fastk, fastd


def STOCHRSI(
    real: pl.Series,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[pl.Series, pl.Series]:
    """STOCHRSI - Stochastic Relative Strength Index (FastK, FastD)"""
    from .overlap import MA

    rsi = RSI(real, timeperiod)
    ln = rsi.rolling_min(fastk_period)
    hn = rsi.rolling_max(fastk_period)
    fastk = ((rsi - ln) * 100.0 / (hn - ln)).rename("fastk_rsi")
    fastd = MA(fastk, fastd_period, fastd_matype).rename("fastd_rsi")
    return fastk, fastd


def TRIX(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
    ema3 = (
        real.ewm_mean(span=timeperiod, adjust=False)
        .ewm_mean(span=timeperiod, adjust=False)
        .ewm_mean(span=timeperiod, adjust=False)
    )
    return (ema3.diff() / ema3.shift(1) * 100.0).rename("trix")


def ULTOSC(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> pl.Series:
    """ULTOSC - Ultimate Oscillator"""
    pc = close.shift(1)
    bp = close - pc.zip_with(pc < low, low)
    tr = pc.zip_with(pc > high, high) - pc.zip_with(pc < low, low)
    avg1 = bp.rolling_sum(timeperiod1) / tr.rolling_sum(timeperiod1)
    avg2 = bp.rolling_sum(timeperiod2) / tr.rolling_sum(timeperiod2)
    avg3 = bp.rolling_sum(timeperiod3) / tr.rolling_sum(timeperiod3)
    return ((4.0 * avg1 + 2.0 * avg2 + avg3) * 100.0).rename("ultosc")


def WILLR(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """WILLR - Williams' %R"""
    hn = high.rolling_max(timeperiod)
    ln = low.rolling_min(timeperiod)
    return ((hn - close) * -100.0 / (hn - ln)).rename("willr")
