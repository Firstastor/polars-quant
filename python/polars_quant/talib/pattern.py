# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

_LIB = str(Path(__file__).parent.parent)


def CDL2CROWS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDL2CROWS - Two Crows"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdl2crows",
        is_elementwise=False,
    )
    return expr


def CDL3BLACKCROWS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDL3BLACKCROWS - Three Black Crows"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdl3blackcrows",
        is_elementwise=False,
    )
    return expr


def CDL3INSIDE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDL3INSIDE - Three Inside Up/Down"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdl3inside",
        is_elementwise=False,
    )
    return expr


def CDL3LINESTRIKE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDL3LINESTRIKE - Three-Line Strike"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdl3linestrike",
        is_elementwise=False,
    )
    return expr


def CDL3OUTSIDE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDL3OUTSIDE - Three Outside Up/Down"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdl3outside",
        is_elementwise=False,
    )
    return expr


def CDL3STARSINSOUTH(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDL3STARSINSOUTH - Three Stars In The South"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdl3starsinsouth",
        is_elementwise=False,
    )
    return expr


def CDL3WHITESOLDIERS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDL3WHITESOLDIERS - Three Advancing White Soldiers"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdl3whitesoldiers",
        is_elementwise=False,
    )
    return expr


def CDLABANDONEDBABY(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.3,
) -> pl.Expr:
    """CDLABANDONEDBABY - Abandoned Baby"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdlabandonedbaby",
        is_elementwise=False,
    )
    return expr


def CDLADVANCEBLOCK(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLADVANCEBLOCK - Advance Block"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdladvanceblock",
        is_elementwise=False,
    )
    return expr


def CDLBELTHOLD(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLBELTHOLD - Belt-hold"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlbelthold",
        is_elementwise=False,
    )
    return expr


def CDLBREAKAWAY(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLBREAKAWAY - Breakaway"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlbreakaway",
        is_elementwise=False,
    )
    return expr


def CDLCLOSINGMARUBOZU(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLCLOSINGMARUBOZU - Closing Marubozu"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlclosingmarubozu",
        is_elementwise=False,
    )
    return expr


def CDLCONCEALBABYSWALL(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLCONCEALBABYSWALL - Concealing Baby Swallow"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlconcealbabyswall",
        is_elementwise=False,
    )
    return expr


def CDLCOUNTERATTACK(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLCOUNTERATTACK - Counterattack"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlcounterattack",
        is_elementwise=False,
    )
    return expr


def CDLDARKCLOUDCOVER(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.5,
) -> pl.Expr:
    """CDLDARKCLOUDCOVER - Dark Cloud Cover"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdldarkcloudcover",
        is_elementwise=False,
    )
    return expr


def CDLDOJI(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLDOJI - Doji"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdldoji",
        is_elementwise=False,
    )
    return expr


def CDLDOJISTAR(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLDOJISTAR - Doji Star"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdldojistar",
        is_elementwise=False,
    )
    return expr


def CDLDRAGONFLYDOJI(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLDRAGONFLYDOJI - Dragonfly Doji"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdldragonflydoji",
        is_elementwise=False,
    )
    return expr


def CDLENGULFING(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLENGULFING - Engulfing Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlengulfing",
        is_elementwise=False,
    )
    return expr


def CDLEVENINGDOJISTAR(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.3,
) -> pl.Expr:
    """CDLEVENINGDOJISTAR - Evening Doji Star"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdleveningdojistar",
        is_elementwise=False,
    )
    return expr


def CDLEVENINGSTAR(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.3,
) -> pl.Expr:
    """CDLEVENINGSTAR - Evening Star"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdleveningstar",
        is_elementwise=False,
    )
    return expr


def CDLGAPSIDESIDEWHITE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlgapsidesidewhite",
        is_elementwise=False,
    )
    return expr


def CDLGRAVESTONEDOJI(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLGRAVESTONEDOJI - Gravestone Doji"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlgravestonedoji",
        is_elementwise=False,
    )
    return expr


def CDLHAMMER(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHAMMER - Hammer"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlhammer",
        is_elementwise=False,
    )
    return expr


def CDLHANGINGMAN(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHANGINGMAN - Hanging Man"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlhangingman",
        is_elementwise=False,
    )
    return expr


def CDLHARAMI(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHARAMI - Harami Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlharami",
        is_elementwise=False,
    )
    return expr


def CDLHARAMICROSS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHARAMICROSS - Harami Cross Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlharamicross",
        is_elementwise=False,
    )
    return expr


def CDLHIGHWAVE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHIGHWAVE - High-Wave Candle"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlhighwave",
        is_elementwise=False,
    )
    return expr


def CDLHIKKAKE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHIKKAKE - Hikkake Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlhikkake",
        is_elementwise=False,
    )
    return expr


def CDLHIKKAKEMOD(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHIKKAKEMOD - Modified Hikkake Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlhikkakemod",
        is_elementwise=False,
    )
    return expr


def CDLHOMINGPIGEON(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLHOMINGPIGEON - Homing Pigeon"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlhomingpigeon",
        is_elementwise=False,
    )
    return expr


def CDLIDENTICAL3CROWS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLIDENTICAL3CROWS - Identical Three Crows"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlidentical3crows",
        is_elementwise=False,
    )
    return expr


def CDLINNECK(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLINNECK - In-Neck Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlinneck",
        is_elementwise=False,
    )
    return expr


def CDLINVERTEDHAMMER(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLINVERTEDHAMMER - Inverted Hammer"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlinvertedhammer",
        is_elementwise=False,
    )
    return expr


def CDLKICKING(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLKICKING - Kicking"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlkicking",
        is_elementwise=False,
    )
    return expr


def CDLKICKINGBYLENGTH(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLKICKINGBYLENGTH - Kicking bull/bear determined by longer marubozu"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlkickingbylength",
        is_elementwise=False,
    )
    return expr


def CDLLADDERBOTTOM(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLLADDERBOTTOM - Ladder Bottom"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlladderbottom",
        is_elementwise=False,
    )
    return expr


def CDLLONGLEGGEDDOJI(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLLONGLEGGEDDOJI - Long Legged Doji"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdllongleggeddoji",
        is_elementwise=False,
    )
    return expr


def CDLLONGLINE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLLONGLINE - Long Line Candle"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdllongline",
        is_elementwise=False,
    )
    return expr


def CDLMARUBOZU(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLMARUBOZU - Marubozu"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlmarubozu",
        is_elementwise=False,
    )
    return expr


def CDLMATCHINGLOW(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLMATCHINGLOW - Matching Low"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlmatchinglow",
        is_elementwise=False,
    )
    return expr


def CDLMATHOLD(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.5,
) -> pl.Expr:
    """CDLMATHOLD - Mat Hold"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdlmathold",
        is_elementwise=False,
    )
    return expr


def CDLMORNINGDOJISTAR(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.3,
) -> pl.Expr:
    """CDLMORNINGDOJISTAR - Morning Doji Star"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdlmorningdojistar",
        is_elementwise=False,
    )
    return expr


def CDLMORNINGSTAR(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.3,
) -> pl.Expr:
    """CDLMORNINGSTAR - Morning Star"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdlmorningstar",
        is_elementwise=False,
    )
    return expr


def CDLONNECK(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLONNECK - On-Neck Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlonneck",
        is_elementwise=False,
    )
    return expr


def CDLPIERCING(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
    penetration: float = 0.5,
) -> pl.Expr:
    """CDLPIERCING - Piercing Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c, penetration],
        plugin_path=_LIB,
        function_name="cdlpiercing",
        is_elementwise=False,
    )
    return expr


def CDLRICKSHAWMAN(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLRICKSHAWMAN - Rickshaw Man"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlrickshawman",
        is_elementwise=False,
    )
    return expr


def CDLRISEFALL3METHODS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLRISEFALL3METHODS - Rising/Falling Three Methods"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlrisefall3methods",
        is_elementwise=False,
    )
    return expr


def CDLSEPARATINGLINES(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLSEPARATINGLINES - Separating Lines"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlseparatinglines",
        is_elementwise=False,
    )
    return expr


def CDLSHOOTINGSTAR(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLSHOOTINGSTAR - Shooting Star"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlshootingstar",
        is_elementwise=False,
    )
    return expr


def CDLSHORTLINE(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLSHORTLINE - Short Line Candle"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlshortline",
        is_elementwise=False,
    )
    return expr


def CDLSPINNINGTOP(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLSPINNINGTOP - Spinning Top"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlspinningtop",
        is_elementwise=False,
    )
    return expr


def CDLSTALLEDPATTERN(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLSTALLEDPATTERN - Stalled Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlstalledpattern",
        is_elementwise=False,
    )
    return expr


def CDLSTICKSANDWICH(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLSTICKSANDWICH - Stick Sandwich"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlsticksandwich",
        is_elementwise=False,
    )
    return expr


def CDLTAKURI(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLTAKURI - Takuri (Long legged dragonfly doji)"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdltakuri",
        is_elementwise=False,
    )
    return expr


def CDLTASUKIGAP(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLTASUKIGAP - Tasuki Gap"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdltasukigap",
        is_elementwise=False,
    )
    return expr


def CDLTHRUSTING(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLTHRUSTING - Thrusting Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlthrusting",
        is_elementwise=False,
    )
    return expr


def CDLTRISTAR(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLTRISTAR - Tristar Pattern"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdltristar",
        is_elementwise=False,
    )
    return expr


def CDLUNIQUE3RIVER(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLUNIQUE3RIVER - Unique 3 River"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlunique3river",
        is_elementwise=False,
    )
    return expr


def CDLUPSIDEGAP2CROWS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLUPSIDEGAP2CROWS - Upside Gap Two Crows"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlupsidegap2crows",
        is_elementwise=False,
    )
    return expr


def CDLXSIDEGAP3METHODS(
    o: pl.Expr,
    h: pl.Expr,
    l: pl.Expr,
    c: pl.Expr,
) -> pl.Expr:
    """CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods"""
    expr = register_plugin_function(
        args=[o, h, l, c],
        plugin_path=_LIB,
        function_name="cdlxsidegap3methods",
        is_elementwise=False,
    )
    return expr
