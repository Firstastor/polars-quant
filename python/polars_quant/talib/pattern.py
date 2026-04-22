# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
import inspect
from pathlib import Path

_LIB = Path(__file__).parent.parent / "polars_quant.abi3.so"

import polars as pl

def CDL2CROWS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL2CROWS - Two Crows"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdl2crows', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDL3BLACKCROWS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3BLACKCROWS - Three Black Crows"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdl3blackcrows', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDL3INSIDE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3INSIDE - Three Inside Up/Down"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdl3inside', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDL3LINESTRIKE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3LINESTRIKE - Three-Line Strike"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdl3linestrike', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDL3OUTSIDE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3OUTSIDE - Three Outside Up/Down"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdl3outside', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDL3STARSINSOUTH(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3STARSINSOUTH - Three Stars In The South"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdl3starsinsouth', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDL3WHITESOLDIERS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3WHITESOLDIERS - Three Advancing White Soldiers"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdl3whitesoldiers', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLABANDONEDBABY(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.3) -> pl.Series:
    """CDLABANDONEDBABY - Abandoned Baby"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdlabandonedbaby', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLADVANCEBLOCK(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLADVANCEBLOCK - Advance Block"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdladvanceblock', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLBELTHOLD(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLBELTHOLD - Belt-hold"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlbelthold', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLBREAKAWAY(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLBREAKAWAY - Breakaway"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlbreakaway', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLCLOSINGMARUBOZU(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLCLOSINGMARUBOZU - Closing Marubozu"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlclosingmarubozu', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLCONCEALBABYSWALL(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLCONCEALBABYSWALL - Concealing Baby Swallow"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlconcealbabyswall', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLCOUNTERATTACK(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLCOUNTERATTACK - Counterattack"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlcounterattack', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLDARKCLOUDCOVER(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.5) -> pl.Series:
    """CDLDARKCLOUDCOVER - Dark Cloud Cover"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdldarkcloudcover', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLDOJI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLDOJI - Doji"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdldoji', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLDOJISTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLDOJISTAR - Doji Star"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdldojistar', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLDRAGONFLYDOJI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLDRAGONFLYDOJI - Dragonfly Doji"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdldragonflydoji', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLENGULFING(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLENGULFING - Engulfing Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlengulfing', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLEVENINGDOJISTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.3) -> pl.Series:
    """CDLEVENINGDOJISTAR - Evening Doji Star"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdleveningdojistar', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLEVENINGSTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.3) -> pl.Series:
    """CDLEVENINGSTAR - Evening Star"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdleveningstar', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLGAPSIDESIDEWHITE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlgapsidesidewhite', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLGRAVESTONEDOJI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLGRAVESTONEDOJI - Gravestone Doji"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlgravestonedoji', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHAMMER(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHAMMER - Hammer"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlhammer', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHANGINGMAN(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHANGINGMAN - Hanging Man"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlhangingman', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHARAMI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHARAMI - Harami Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlharami', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHARAMICROSS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHARAMICROSS - Harami Cross Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlharamicross', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHIGHWAVE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHIGHWAVE - High-Wave Candle"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlhighwave', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHIKKAKE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHIKKAKE - Hikkake Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlhikkake', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHIKKAKEMOD(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHIKKAKEMOD - Modified Hikkake Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlhikkakemod', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLHOMINGPIGEON(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHOMINGPIGEON - Homing Pigeon"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlhomingpigeon', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLIDENTICAL3CROWS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLIDENTICAL3CROWS - Identical Three Crows"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlidentical3crows', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLINNECK(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLINNECK - In-Neck Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlinneck', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLINVERTEDHAMMER(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLINVERTEDHAMMER - Inverted Hammer"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlinvertedhammer', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLKICKING(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLKICKING - Kicking"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlkicking', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLKICKINGBYLENGTH(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLKICKINGBYLENGTH - Kicking bull/bear determined by longer marubozu"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlkickingbylength', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLLADDERBOTTOM(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLLADDERBOTTOM - Ladder Bottom"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlladderbottom', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLLONGLEGGEDDOJI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLLONGLEGGEDDOJI - Long Legged Doji"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdllongleggeddoji', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLLONGLINE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLLONGLINE - Long Line Candle"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdllongline', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLMARUBOZU(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLMARUBOZU - Marubozu"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlmarubozu', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLMATCHINGLOW(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLMATCHINGLOW - Matching Low"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlmatchinglow', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLMATHOLD(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.5) -> pl.Series:
    """CDLMATHOLD - Mat Hold"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdlmathold', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLMORNINGDOJISTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.3) -> pl.Series:
    """CDLMORNINGDOJISTAR - Morning Doji Star"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdlmorningdojistar', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLMORNINGSTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.3) -> pl.Series:
    """CDLMORNINGSTAR - Morning Star"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdlmorningstar', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLONNECK(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLONNECK - On-Neck Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlonneck', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLPIERCING(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.5) -> pl.Series:
    """CDLPIERCING - Piercing Pattern"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdlpiercing', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLRICKSHAWMAN(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLRICKSHAWMAN - Rickshaw Man"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlrickshawman', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLRISEFALL3METHODS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLRISEFALL3METHODS - Rising/Falling Three Methods"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlrisefall3methods', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLSEPARATINGLINES(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSEPARATINGLINES - Separating Lines"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlseparatinglines', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLSHOOTINGSTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSHOOTINGSTAR - Shooting Star"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlshootingstar', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLSHORTLINE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSHORTLINE - Short Line Candle"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlshortline', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLSPINNINGTOP(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSPINNINGTOP - Spinning Top"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlspinningtop', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLSTALLEDPATTERN(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSTALLEDPATTERN - Stalled Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlstalledpattern', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLSTICKSANDWICH(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSTICKSANDWICH - Stick Sandwich"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlsticksandwich', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLTAKURI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLTAKURI - Takuri (Long legged dragonfly doji)"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdltakuri', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLTASUKIGAP(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLTASUKIGAP - Tasuki Gap"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdltasukigap', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLTHRUSTING(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float=0.3) -> pl.Series:
    """CDLTHRUSTING - Thrusting Pattern"""
    expr = register_plugin_function(args=[o, h, l, c, penetration], plugin_path=_LIB, function_name='cdlthrusting', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLTRISTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLTRISTAR - Tristar Pattern"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdltristar', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLUNIQUE3RIVER(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLUNIQUE3RIVER - Unique 3 River"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlunique3river', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLUPSIDEGAP2CROWS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLUPSIDEGAP2CROWS - Upside Gap Two Crows"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlupsidegap2crows', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr

def CDLXSIDEGAP3METHODS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods"""
    expr = register_plugin_function(args=[o, h, l, c], plugin_path=_LIB, function_name='cdlxsidegap3methods', is_elementwise=False)
    if isinstance(o, pl.Series):
        return o.to_frame().select(expr).to_series()
    return expr