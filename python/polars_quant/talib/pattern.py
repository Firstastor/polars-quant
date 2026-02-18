import polars as pl

# ====================================================================
# Pattern Recognition - 蜡烛图模式识别
# ====================================================================


def CDL2CROWS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL2CROWS - Two Crows"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h1 = _shift(h, 2)
    c1, c2 = _shift(c, 2), _shift(c, 1)
    l2 = _shift(l, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    bear2 = is_bearish(o2, c2)
    gap_up2 = o2 > c1
    bear3 = is_bearish(o, c)
    open_in2 = (o > o2) & (o < c2)
    close_in1 = (c > o1) & (c < c1)

    mask = bull1 & bear2 & gap_up2 & bear3 & open_in2 & close_in1
    return pattern_result(mask, -100)


def CDL3BLACKCROWS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3BLACKCROWS - Three Black Crows"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    bear2 = is_bearish(o2, c2) & is_body_long(o2, c2)
    bear3 = is_bearish(o, c) & is_body_long(o, c)
    opens_within1 = (o2 < o1) & (o2 > c1)
    opens_within2 = (o < o2) & (o > c2)
    lower_closes = (c2 < c1) & (c < c2)

    mask = bear1 & bear2 & bear3 & opens_within1 & opens_within2 & lower_closes
    return pattern_result(mask, -100)


def CDL3INSIDE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3INSIDE - Three Inside Up/Down"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h2 = _shift(h, 1)
    l2 = _shift(l, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull_pattern = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o2, c2)
        & (c2 < o1)
        & (o2 > c1)
        & is_bullish(o, c)
        & (c > o1)
    )
    bear_pattern = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_bearish(o2, c2)
        & (o2 < c1)
        & (c2 > o1)
        & is_bearish(o, c)
        & (c < o1)
    )
    return combine_bullish_bearish(bull_pattern, bear_pattern)


def CDL3LINESTRIKE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3LINESTRIKE - Three-Line Strike"""
    o1, o2, o3 = _shift(o, 3), _shift(o, 2), _shift(o, 1)
    c1, c2, c3 = _shift(c, 3), _shift(c, 2), _shift(c, 1)

    bull_three = (
        is_bearish(o1, c1)
        & is_bearish(o2, c2)
        & is_bearish(o3, c3)
        & (c2 < c1)
        & (c3 < c2)
        & (o2 > c1)
        & (o2 < o1)
        & (o3 > c2)
        & (o3 < o2)
    )
    bull_strike = is_bullish(o, c) & (o < c3) & (c > o1)

    bear_three = (
        is_bullish(o1, c1)
        & is_bullish(o2, c2)
        & is_bullish(o3, c3)
        & (c2 > c1)
        & (c3 > c2)
        & (o2 < c1)
        & (o2 > o1)
        & (o3 < c2)
        & (o3 > o2)
    )
    bear_strike = is_bearish(o, c) & (o > c3) & (c < o1)

    return combine_bullish_bearish(bull_three & bull_strike, bear_three & bear_strike)


def CDL3OUTSIDE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDL3OUTSIDE - Three Outside Up/Down"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull_pattern = (
        is_bearish(o1, c1)
        & is_bullish(o2, c2)
        & (o2 <= c1)
        & (c2 >= o1)
        & is_bullish(o, c)
        & (c > c2)
    )
    bear_pattern = (
        is_bullish(o1, c1)
        & is_bearish(o2, c2)
        & (o2 >= c1)
        & (c2 <= o1)
        & is_bearish(o, c)
        & (c < c2)
    )
    return combine_bullish_bearish(bull_pattern, bear_pattern)


def CDL3STARSINSOUTH(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDL3STARSINSOUTH - Three Stars In The South"""
    o1, o2, o3 = _shift(o, 2), _shift(o, 1), o
    h1, h2, h3 = _shift(h, 2), _shift(h, 1), h
    l1, l2, l3 = _shift(l, 2), _shift(l, 1), l
    c1, c2, c3 = _shift(c, 2), _shift(c, 1), c

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    has_ls1 = is_shadow_long_lower(o1, l1, c1)
    bear2 = is_bearish(o2, c2)
    lower_low2 = l2 > l1
    higher_close2 = c2 > c1
    bear3 = is_bearish(o3, c3) & is_body_short(o3, c3)
    inside3 = (h3 < h2) & (l3 > l2)

    mask = bear1 & has_ls1 & bear2 & lower_low2 & higher_close2 & bear3 & inside3
    return pattern_result(mask, 100)


def CDL3WHITESOLDIERS(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDL3WHITESOLDIERS - Three Advancing White Soldiers"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    bull2 = is_bullish(o2, c2) & is_body_long(o2, c2)
    bull3 = is_bullish(o, c) & is_body_long(o, c)
    opens_within1 = (o2 > o1) & (o2 <= c1)
    opens_within2 = (o > o2) & (o <= c2)
    higher_closes = (c2 > c1) & (c > c2)

    mask = bull1 & bull2 & bull3 & opens_within1 & opens_within2 & higher_closes
    return pattern_result(mask, 100)


def CDLABANDONEDBABY(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.3
) -> pl.Series:
    """CDLABANDONEDBABY - Abandoned Baby"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h1, h2 = _shift(h, 2), _shift(h, 1)
    l1, l2 = _shift(l, 2), _shift(l, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    doji2 = is_doji(o2, h2, l2, c2)

    bull_pattern = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & doji2
        & (h2 < l1)
        & is_bullish(o, c)
        & (l > h2)
    )
    bear_pattern = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & doji2
        & (l2 > h1)
        & is_bearish(o, c)
        & (h < l2)
    )
    return combine_bullish_bearish(bull_pattern, bear_pattern)


def CDLADVANCEBLOCK(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLADVANCEBLOCK - Advance Block"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h1, h2 = _shift(h, 2), _shift(h, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    bull2 = is_bullish(o2, c2)
    bull3 = is_bullish(o, c)
    opens_within1 = (o2 > o1) & (o2 <= c1)
    opens_within2 = (o > o2) & (o <= c2)
    higher_closes = (c2 > c1) & (c > c2)
    shrinking = body_abs(o, c) < body_abs(o2, c2)

    mask = (
        bull1
        & bull2
        & bull3
        & opens_within1
        & opens_within2
        & higher_closes
        & shrinking
    )
    return pattern_result(mask, -100)


def CDLBELTHOLD(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLBELTHOLD - Belt-hold"""
    bull = (
        is_bullish(o, c) & is_body_long(o, c) & is_shadow_very_short_lower(o, h, l, c)
    )
    bear = (
        is_bearish(o, c) & is_body_long(o, c) & is_shadow_very_short_upper(o, h, l, c)
    )
    return combine_bullish_bearish(bull, bear)


def CDLBREAKAWAY(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLBREAKAWAY - Breakaway"""
    o1, o2, o3, o4 = _shift(o, 4), _shift(o, 3), _shift(o, 2), _shift(o, 1)
    c1, c2, c3, c4 = _shift(c, 4), _shift(c, 3), _shift(c, 2), _shift(c, 1)
    h1 = _shift(h, 4)
    l1 = _shift(l, 4)

    bull_pattern = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bearish(o2, c2)
        & (o2 < c1)
        & (c3 < c2)
        & is_bullish(o, c)
        & (c > o2)
        & (c < c1)
    )
    bear_pattern = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o2, c2)
        & (o2 > c1)
        & (c3 > c2)
        & is_bearish(o, c)
        & (c < o2)
        & (c > c1)
    )
    return combine_bullish_bearish(bull_pattern, bear_pattern)


def CDLCLOSINGMARUBOZU(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLCLOSINGMARUBOZU - Closing Marubozu"""
    bull = (
        is_bullish(o, c) & is_body_long(o, c) & is_shadow_very_short_upper(o, h, l, c)
    )
    bear = (
        is_bearish(o, c) & is_body_long(o, c) & is_shadow_very_short_lower(o, h, l, c)
    )
    return combine_bullish_bearish(bull, bear)


def CDLCONCEALBABYSWALL(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLCONCEALBABYSWALL - Concealing Baby Swallow"""
    o1, o2, o3 = _shift(o, 3), _shift(o, 2), _shift(o, 1)
    h1, h2, h3 = _shift(h, 3), _shift(h, 2), _shift(h, 1)
    l1, l2, l3 = _shift(l, 3), _shift(l, 2), _shift(l, 1)
    c1, c2, c3 = _shift(c, 3), _shift(c, 2), _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    no_shadow1 = is_shadow_very_short_upper(
        o1, h1, l1, c1
    ) & is_shadow_very_short_lower(o1, h1, l1, c1)
    bear2 = is_bearish(o2, c2) & is_body_long(o2, c2)
    no_shadow2 = is_shadow_very_short_upper(
        o2, h2, l2, c2
    ) & is_shadow_very_short_lower(o2, h2, l2, c2)
    bear3 = is_bearish(o3, c3)
    high_gap3 = h3 > c2
    bear4 = is_bearish(o, c) & is_body_long(o, c)
    engulf = (o > h3) & (c < l2)

    mask = (
        bear1
        & no_shadow1
        & bear2
        & no_shadow2
        & (c2 < c1)
        & bear3
        & high_gap3
        & bear4
        & engulf
    )
    return pattern_result(mask, 100)


def CDLCOUNTERATTACK(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLCOUNTERATTACK - Counterattack"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    bull = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o, c)
        & is_body_long(o, c)
        & is_near(c, c1, h, l)
    )
    bear = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_bearish(o, c)
        & is_body_long(o, c)
        & is_near(c, c1, h, l)
    )
    return combine_bullish_bearish(bull, bear)


def CDLDARKCLOUDCOVER(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.5
) -> pl.Series:
    """CDLDARKCLOUDCOVER - Dark Cloud Cover"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    bear_cur = is_bearish(o, c)
    open_above = o > c1
    close_into = c < (c1 - body_abs(o1, c1) * penetration)
    close_above = c > o1

    mask = bull1 & bear_cur & open_above & close_into & close_above
    return pattern_result(mask, -100)


def CDLDOJI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLDOJI - Doji"""
    mask = is_doji(o, h, l, c)
    return pattern_result(mask, 100)


def CDLDOJISTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLDOJISTAR - Doji Star"""
    o1 = _shift(o, 1)
    h1 = _shift(h, 1)
    l1 = _shift(l, 1)
    c1 = _shift(c, 1)

    doji_cur = is_doji(o, h, l, c)
    cur_mid = (o + c) / 2

    bull = is_bearish(o1, c1) & is_body_long(o1, c1) & doji_cur & (cur_mid < c1)
    bear = is_bullish(o1, c1) & is_body_long(o1, c1) & doji_cur & (cur_mid > c1)
    return combine_bullish_bearish(bull, bear)


def CDLDRAGONFLYDOJI(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLDRAGONFLYDOJI - Dragonfly Doji"""
    mask = (
        is_doji(o, h, l, c)
        & is_shadow_long_lower(o, l, c)
        & is_shadow_very_short_upper(o, h, l, c)
    )
    return pattern_result(mask, 100)


def CDLENGULFING(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLENGULFING - Engulfing Pattern"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    bull = (
        is_bearish(o1, c1)
        & is_bullish(o, c)
        & (o <= c1)
        & (c >= o1)
        & ((o < c1) | (c > o1))
    )
    bear = (
        is_bullish(o1, c1)
        & is_bearish(o, c)
        & (o >= c1)
        & (c <= o1)
        & ((o > c1) | (c < o1))
    )
    return combine_bullish_bearish(bull, bear)


def CDLEVENINGDOJISTAR(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.3
) -> pl.Series:
    """CDLEVENINGDOJISTAR - Evening Doji Star"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h2 = _shift(h, 1)
    l2 = _shift(l, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    doji2 = is_doji(o2, h2, l2, c2)
    gap_up = oc_min(o2, c2) > c1
    bear3 = is_bearish(o, c)
    close_into = c < (c1 - body_abs(o1, c1) * penetration)

    mask = bull1 & doji2 & gap_up & bear3 & close_into
    return pattern_result(mask, -100)


def CDLEVENINGSTAR(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.3
) -> pl.Series:
    """CDLEVENINGSTAR - Evening Star"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    short2 = is_body_short(o2, c2)
    gap_up = oc_min(o2, c2) > c1
    bear3 = is_bearish(o, c)
    close_into = c < (c1 - body_abs(o1, c1) * penetration)

    mask = bull1 & short2 & gap_up & bear3 & close_into
    return pattern_result(mask, -100)


def CDLGAPSIDESIDEWHITE(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)
    h1, h2 = _shift(h, 2), _shift(h, 1)
    l1, l2 = _shift(l, 2), _shift(l, 1)

    bull2 = is_bullish(o2, c2)
    bull3 = is_bullish(o, c)
    similar_size = is_near(body_abs(o, c), body_abs(o2, c2), h, l)
    similar_open = is_near(o, o2, h, l)

    up_gap = (
        is_bullish(o1, c1) & (o2 > c1) & bull2 & bull3 & similar_size & similar_open
    )
    down_gap = (
        is_bearish(o1, c1) & (c2 < c1) & bull2 & bull3 & similar_size & similar_open
    )

    return combine_bullish_bearish(up_gap, down_gap)


def CDLGRAVESTONEDOJI(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLGRAVESTONEDOJI - Gravestone Doji"""
    mask = (
        is_doji(o, h, l, c)
        & is_shadow_long_upper(o, h, c)
        & is_shadow_very_short_lower(o, h, l, c)
    )
    return pattern_result(mask, -100)


def CDLHAMMER(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHAMMER - Hammer"""
    ba = body_abs(o, c)
    ls = lower_shadow(o, l, c)
    us = upper_shadow(o, h, c)

    mask = is_body_short(o, c) & (ls > 2 * ba) & is_shadow_very_short_upper(o, h, l, c)

    o1 = _shift(o, 1)
    c1 = _shift(c, 1)
    downtrend = is_bearish(o1, c1)

    return pattern_result(mask & downtrend, 100)


def CDLHANGINGMAN(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHANGINGMAN - Hanging Man"""
    ba = body_abs(o, c)
    ls = lower_shadow(o, l, c)

    mask = is_body_short(o, c) & (ls > 2 * ba) & is_shadow_very_short_upper(o, h, l, c)

    o1 = _shift(o, 1)
    c1 = _shift(c, 1)
    uptrend = is_bullish(o1, c1)

    return pattern_result(mask & uptrend, -100)


def CDLHARAMI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHARAMI - Harami Pattern"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    bull = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o, c)
        & is_body_short(o, c)
        & (o > c1)
        & (c < o1)
    )
    bear = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_bearish(o, c)
        & is_body_short(o, c)
        & (o < c1)
        & (c > o1)
    )
    return combine_bullish_bearish(bull, bear)


def CDLHARAMICROSS(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHARAMICROSS - Harami Cross Pattern"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    cur_doji = is_doji(o, h, l, c)

    bull = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & cur_doji
        & (oc_max(o, c) < o1)
        & (oc_min(o, c) > c1)
    )
    bear = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & cur_doji
        & (oc_max(o, c) < c1)
        & (oc_min(o, c) > o1)
    )
    return combine_bullish_bearish(bull, bear)


def CDLHIGHWAVE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHIGHWAVE - High-Wave Candle"""
    mask = (
        is_body_short(o, c)
        & is_shadow_long_upper(o, h, c)
        & is_shadow_long_lower(o, l, c)
    )
    return combine_bullish_bearish(mask & is_bullish(o, c), mask & is_bearish(o, c))


def CDLHIKKAKE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHIKKAKE - Hikkake Pattern"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h1, h2 = _shift(h, 2), _shift(h, 1)
    l1, l2 = _shift(l, 2), _shift(l, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    inside_bar = (h2 < h1) & (l2 > l1)

    bull = inside_bar & (c > h1) & is_bullish(o, c)
    bear = inside_bar & (c < l1) & is_bearish(o, c)
    return combine_bullish_bearish(bull, bear)


def CDLHIKKAKEMOD(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLHIKKAKEMOD - Modified Hikkake Pattern"""
    o1, o2, o3 = _shift(o, 3), _shift(o, 2), _shift(o, 1)
    h1, h2, h3 = _shift(h, 3), _shift(h, 2), _shift(h, 1)
    l1, l2, l3 = _shift(l, 3), _shift(l, 2), _shift(l, 1)
    c1, c2, c3 = _shift(c, 3), _shift(c, 2), _shift(c, 1)

    inside_bar = (h2 < h1) & (l2 > l1)
    second_inside = (h3 < h2) & (l3 > l2)

    bull = inside_bar & second_inside & (c > h1) & is_bullish(o, c)
    bear = inside_bar & second_inside & (c < l1) & is_bearish(o, c)
    return combine_bullish_bearish(bull, bear)


def CDLHOMINGPIGEON(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLHOMINGPIGEON - Homing Pigeon"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    mask = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bearish(o, c)
        & is_body_short(o, c)
        & (o < o1)
        & (c > c1)
    )
    return pattern_result(mask, 100)


def CDLIDENTICAL3CROWS(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLIDENTICAL3CROWS - Identical Three Crows"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    bear2 = is_bearish(o2, c2) & is_body_long(o2, c2)
    bear3 = is_bearish(o, c) & is_body_long(o, c)
    eq_open1 = is_equal(o2, c1, h, l)
    eq_open2 = is_equal(o, c2, h, l)
    lower_closes = (c2 < c1) & (c < c2)

    mask = bear1 & bear2 & bear3 & eq_open1 & eq_open2 & lower_closes
    return pattern_result(mask, -100)


def CDLINNECK(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLINNECK - In-Neck Pattern"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    mask = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o, c)
        & (o < c1)
        & is_near(c, c1, h, l)
    )
    return pattern_result(mask, -100)


def CDLINVERTEDHAMMER(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLINVERTEDHAMMER - Inverted Hammer"""
    ba = body_abs(o, c)
    us = upper_shadow(o, h, c)

    mask = is_body_short(o, c) & (us > 2 * ba) & is_shadow_very_short_lower(o, h, l, c)

    o1 = _shift(o, 1)
    c1 = _shift(c, 1)
    downtrend = is_bearish(o1, c1)

    return pattern_result(mask & downtrend, 100)


def CDLKICKING(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLKICKING - Kicking"""
    o1 = _shift(o, 1)
    h1 = _shift(h, 1)
    l1 = _shift(l, 1)
    c1 = _shift(c, 1)

    marubozu1_bear = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_shadow_very_short_upper(o1, h1, l1, c1)
        & is_shadow_very_short_lower(o1, h1, l1, c1)
    )
    marubozu1_bull = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_shadow_very_short_upper(o1, h1, l1, c1)
        & is_shadow_very_short_lower(o1, h1, l1, c1)
    )
    marubozu_cur_bull = (
        is_bullish(o, c)
        & is_body_long(o, c)
        & is_shadow_very_short_upper(o, h, l, c)
        & is_shadow_very_short_lower(o, h, l, c)
    )
    marubozu_cur_bear = (
        is_bearish(o, c)
        & is_body_long(o, c)
        & is_shadow_very_short_upper(o, h, l, c)
        & is_shadow_very_short_lower(o, h, l, c)
    )

    bull = marubozu1_bear & marubozu_cur_bull & (o > o1)
    bear = marubozu1_bull & marubozu_cur_bear & (o < o1)
    return combine_bullish_bearish(bull, bear)


def CDLKICKINGBYLENGTH(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLKICKINGBYLENGTH - Kicking bull/bear determined by longer marubozu"""
    o1 = _shift(o, 1)
    h1 = _shift(h, 1)
    l1 = _shift(l, 1)
    c1 = _shift(c, 1)

    marubozu1_bear = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_shadow_very_short_upper(o1, h1, l1, c1)
        & is_shadow_very_short_lower(o1, h1, l1, c1)
    )
    marubozu1_bull = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_shadow_very_short_upper(o1, h1, l1, c1)
        & is_shadow_very_short_lower(o1, h1, l1, c1)
    )
    marubozu_cur_bull = (
        is_bullish(o, c)
        & is_body_long(o, c)
        & is_shadow_very_short_upper(o, h, l, c)
        & is_shadow_very_short_lower(o, h, l, c)
    )
    marubozu_cur_bear = (
        is_bearish(o, c)
        & is_body_long(o, c)
        & is_shadow_very_short_upper(o, h, l, c)
        & is_shadow_very_short_lower(o, h, l, c)
    )

    ba1 = body_abs(o1, c1)
    ba0 = body_abs(o, c)

    bull_kick = marubozu1_bear & marubozu_cur_bull & (o > o1)
    bear_kick = marubozu1_bull & marubozu_cur_bear & (o < o1)

    bull_longer = bull_kick & (ba0 >= ba1)
    bear_longer = bear_kick & (ba0 >= ba1)

    return combine_bullish_bearish(
        bull_longer | (bull_kick & ~bear_longer),
        bear_longer | (bear_kick & ~bull_longer),
    )


def CDLLADDERBOTTOM(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLLADDERBOTTOM - Ladder Bottom"""
    o1, o2, o3, o4 = _shift(o, 4), _shift(o, 3), _shift(o, 2), _shift(o, 1)
    h4 = _shift(h, 1)
    c1, c2, c3, c4 = _shift(c, 4), _shift(c, 3), _shift(c, 2), _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    bear2 = is_bearish(o2, c2) & (c2 < c1)
    bear3 = is_bearish(o3, c3) & (c3 < c2)
    bear4 = is_bearish(o4, c4)
    has_upper4 = is_shadow_long_upper(o4, h4, c4)
    bull5 = is_bullish(o, c) & (o > o4)

    mask = bear1 & bear2 & bear3 & bear4 & has_upper4 & bull5
    return pattern_result(mask, 100)


def CDLLONGLEGGEDDOJI(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLLONGLEGGEDDOJI - Long Legged Doji"""
    mask = (
        is_doji(o, h, l, c)
        & is_shadow_long_upper(o, h, c)
        & is_shadow_long_lower(o, l, c)
    )
    return pattern_result(mask, 100)


def CDLLONGLINE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLLONGLINE - Long Line Candle"""
    long_body = is_body_long(o, c)
    short_shadows = is_shadow_short_upper(o, h, l, c) & is_shadow_short_lower(
        o, h, l, c
    )
    mask = long_body & short_shadows
    return combine_bullish_bearish(mask & is_bullish(o, c), mask & is_bearish(o, c))


def CDLMARUBOZU(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLMARUBOZU - Marubozu"""
    mask = (
        is_body_long(o, c)
        & is_shadow_very_short_upper(o, h, l, c)
        & is_shadow_very_short_lower(o, h, l, c)
    )
    return combine_bullish_bearish(mask & is_bullish(o, c), mask & is_bearish(o, c))


def CDLMATCHINGLOW(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLMATCHINGLOW - Matching Low"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    mask = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bearish(o, c)
        & is_equal(c, c1, h, l)
    )
    return pattern_result(mask, 100)


def CDLMATHOLD(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.5
) -> pl.Series:
    """CDLMATHOLD - Mat Hold"""
    o1, o2, o3, o4 = _shift(o, 4), _shift(o, 3), _shift(o, 2), _shift(o, 1)
    h2, h3, h4 = _shift(h, 3), _shift(h, 2), _shift(h, 1)
    l2, l3, l4 = _shift(l, 3), _shift(l, 2), _shift(l, 1)
    c1, c2, c3, c4 = _shift(c, 4), _shift(c, 3), _shift(c, 2), _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    small2 = is_body_short(o2, c2) & (o2 > c1)
    small3 = is_body_short(o3, c3)
    small4 = is_body_short(o4, c4)
    hold_above = (l2 > o1) & (l3 > o1) & (l4 > o1)
    bull5 = is_bullish(o, c) & (c > c1)

    mask = bull1 & small2 & small3 & small4 & hold_above & bull5
    return pattern_result(mask, 100)


def CDLMORNINGDOJISTAR(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.3
) -> pl.Series:
    """CDLMORNINGDOJISTAR - Morning Doji Star"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h2 = _shift(h, 1)
    l2 = _shift(l, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    doji2 = is_doji(o2, h2, l2, c2)
    gap_down = oc_max(o2, c2) < c1
    bull3 = is_bullish(o, c)
    close_into = c > (c1 + body_abs(o1, c1) * penetration)

    mask = bear1 & doji2 & gap_down & bull3 & close_into
    return pattern_result(mask, 100)


def CDLMORNINGSTAR(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.3
) -> pl.Series:
    """CDLMORNINGSTAR - Morning Star"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    short2 = is_body_short(o2, c2)
    gap_down = oc_max(o2, c2) < c1
    bull3 = is_bullish(o, c)
    close_into = c > (c1 + body_abs(o1, c1) * penetration)

    mask = bear1 & short2 & gap_down & bull3 & close_into
    return pattern_result(mask, 100)


def CDLONNECK(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLONNECK - On-Neck Pattern"""
    o1 = _shift(o, 1)
    l1 = _shift(l, 1)
    c1 = _shift(c, 1)

    mask = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o, c)
        & (o < c1)
        & is_near(c, l1, h, l)
    )
    return pattern_result(mask, -100)


def CDLPIERCING(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.5
) -> pl.Series:
    """CDLPIERCING - Piercing Pattern"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    bull_cur = is_bullish(o, c)
    open_below = o < c1
    close_into = c > (c1 + body_abs(o1, c1) * penetration)
    close_below = c < o1

    mask = bear1 & bull_cur & open_below & close_into & close_below
    return pattern_result(mask, 100)


def CDLRICKSHAWMAN(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLRICKSHAWMAN - Rickshaw Man"""
    ba = body_abs(o, c)
    us = upper_shadow(o, h, c)
    ls = lower_shadow(o, l, c)
    cr = candle_range(h, l)

    mask = (
        is_doji(o, h, l, c)
        & is_shadow_long_upper(o, h, c)
        & is_shadow_long_lower(o, l, c)
        & is_near(us, ls, h, l)
    )
    return pattern_result(mask, 100)


def CDLRISEFALL3METHODS(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLRISEFALL3METHODS - Rising/Falling Three Methods"""
    o1, o2, o3, o4 = _shift(o, 4), _shift(o, 3), _shift(o, 2), _shift(o, 1)
    h1, h2, h3, h4 = _shift(h, 4), _shift(h, 3), _shift(h, 2), _shift(h, 1)
    l1, l2, l3, l4 = _shift(l, 4), _shift(l, 3), _shift(l, 2), _shift(l, 1)
    c1, c2, c3, c4 = _shift(c, 4), _shift(c, 3), _shift(c, 2), _shift(c, 1)

    rising = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_body_short(o2, c2)
        & is_body_short(o3, c3)
        & is_body_short(o4, c4)
        & (h2 < h1)
        & (h3 < h1)
        & (h4 < h1)
        & (l2 > l1)
        & (l3 > l1)
        & (l4 > l1)
        & is_bullish(o, c)
        & is_body_long(o, c)
        & (c > c1)
    )
    falling = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_body_short(o2, c2)
        & is_body_short(o3, c3)
        & is_body_short(o4, c4)
        & (l2 > l1)
        & (l3 > l1)
        & (l4 > l1)
        & (h2 < h1)
        & (h3 < h1)
        & (h4 < h1)
        & is_bearish(o, c)
        & is_body_long(o, c)
        & (c < c1)
    )
    return combine_bullish_bearish(rising, falling)


def CDLSEPARATINGLINES(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLSEPARATINGLINES - Separating Lines"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    bull = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o, c)
        & is_body_long(o, c)
        & is_equal(o, o1, h, l)
    )
    bear = (
        is_bullish(o1, c1)
        & is_body_long(o1, c1)
        & is_bearish(o, c)
        & is_body_long(o, c)
        & is_equal(o, o1, h, l)
    )
    return combine_bullish_bearish(bull, bear)


def CDLSHOOTINGSTAR(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLSHOOTINGSTAR - Shooting Star"""
    ba = body_abs(o, c)
    us = upper_shadow(o, h, c)

    mask = is_body_short(o, c) & (us > 2 * ba) & is_shadow_very_short_lower(o, h, l, c)

    o1 = _shift(o, 1)
    c1 = _shift(c, 1)
    uptrend = is_bullish(o1, c1)
    gap_up = oc_min(o, c) > oc_max(o1, c1)

    return pattern_result(mask & uptrend, -100)


def CDLSHORTLINE(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSHORTLINE - Short Line Candle"""
    short_body = is_body_short(o, c)
    short_shadows = is_shadow_short_upper(o, h, l, c) & is_shadow_short_lower(
        o, h, l, c
    )
    mask = short_body & short_shadows
    return combine_bullish_bearish(mask & is_bullish(o, c), mask & is_bearish(o, c))


def CDLSPINNINGTOP(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLSPINNINGTOP - Spinning Top"""
    mask = (
        is_body_short(o, c)
        & (upper_shadow(o, h, c) > body_abs(o, c))
        & (lower_shadow(o, l, c) > body_abs(o, c))
    )
    return combine_bullish_bearish(mask & is_bullish(o, c), mask & is_bearish(o, c))


def CDLSTALLEDPATTERN(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLSTALLEDPATTERN - Stalled Pattern"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    bull2 = is_bullish(o2, c2) & is_body_long(o2, c2) & (c2 > c1)
    bull3 = is_bullish(o, c) & is_body_short(o, c) & (c > c2)
    opens_near = (o > o2) & (o <= c2)

    mask = bull1 & bull2 & bull3 & opens_near
    return pattern_result(mask, -100)


def CDLSTICKSANDWICH(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLSTICKSANDWICH - Stick Sandwich"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    mask = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o2, c2)
        & is_body_long(o2, c2)
        & (o2 > c1)
        & is_bearish(o, c)
        & is_body_long(o, c)
        & is_equal(c, c1, h, l)
    )
    return pattern_result(mask, 100)


def CDLTAKURI(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLTAKURI - Takuri (Long legged dragonfly doji)"""
    mask = (
        is_doji(o, h, l, c)
        & is_shadow_very_long_lower(o, l, c)
        & is_shadow_very_short_upper(o, h, l, c)
    )
    return pattern_result(mask, 100)


def CDLTASUKIGAP(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLTASUKIGAP - Tasuki Gap"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull = (
        is_bullish(o1, c1)
        & is_bullish(o2, c2)
        & (o2 > c1)
        & is_bearish(o, c)
        & (o > o2)
        & (o < c2)
        & (c > o1)
        & (c < c1)
    )
    bear = (
        is_bearish(o1, c1)
        & is_bearish(o2, c2)
        & (o2 < c1)
        & is_bullish(o, c)
        & (o < o2)
        & (o > c2)
        & (c < o1)
        & (c > c1)
    )
    return combine_bullish_bearish(bull, bear)


def CDLTHRUSTING(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series, penetration: float = 0.3
) -> pl.Series:
    """CDLTHRUSTING - Thrusting Pattern"""
    o1 = _shift(o, 1)
    c1 = _shift(c, 1)

    midpoint = c1 + body_abs(o1, c1) * 0.5

    mask = (
        is_bearish(o1, c1)
        & is_body_long(o1, c1)
        & is_bullish(o, c)
        & (o < c1)
        & (c > c1)
        & (c < midpoint)
    )
    return pattern_result(mask, -100)


def CDLTRISTAR(o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series) -> pl.Series:
    """CDLTRISTAR - Tristar Pattern"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h1, h2 = _shift(h, 2), _shift(h, 1)
    l1, l2 = _shift(l, 2), _shift(l, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    doji1 = is_doji(o1, h1, l1, c1)
    doji2 = is_doji(o2, h2, l2, c2)
    doji3 = is_doji(o, h, l, c)

    mid1 = (o1 + c1) / 2
    mid2 = (o2 + c2) / 2
    mid3 = (o + c) / 2

    bull = doji1 & doji2 & doji3 & (mid2 < mid1) & (mid3 > mid2)
    bear = doji1 & doji2 & doji3 & (mid2 > mid1) & (mid3 < mid2)
    return combine_bullish_bearish(bull, bear)


def CDLUNIQUE3RIVER(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLUNIQUE3RIVER - Unique 3 River"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    h2 = _shift(h, 1)
    l1, l2 = _shift(l, 2), _shift(l, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bear1 = is_bearish(o1, c1) & is_body_long(o1, c1)
    bear2 = is_bearish(o2, c2) & (l2 < l1) & (c2 > l2)
    harami = (o2 < o1) & (o2 > c1)
    bull3 = is_bullish(o, c) & is_body_short(o, c) & (c < c2)

    mask = bear1 & bear2 & harami & bull3
    return pattern_result(mask, 100)


def CDLUPSIDEGAP2CROWS(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLUPSIDEGAP2CROWS - Upside Gap Two Crows"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull1 = is_bullish(o1, c1) & is_body_long(o1, c1)
    bear2 = is_bearish(o2, c2) & (o2 > c1) & (c2 > c1)
    bear3 = is_bearish(o, c) & (o > o2) & (c > c1) & (c < c2)

    mask = bull1 & bear2 & bear3
    return pattern_result(mask, -100)


def CDLXSIDEGAP3METHODS(
    o: pl.Series, h: pl.Series, l: pl.Series, c: pl.Series
) -> pl.Series:
    """CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods"""
    o1, o2 = _shift(o, 2), _shift(o, 1)
    c1, c2 = _shift(c, 2), _shift(c, 1)

    bull = (
        is_bullish(o1, c1)
        & is_bullish(o2, c2)
        & (o2 > c1)
        & is_bearish(o, c)
        & (o < c2)
        & (o > o2)
        & (c > o1)
        & (c < c1)
    )
    bear = (
        is_bearish(o1, c1)
        & is_bearish(o2, c2)
        & (o2 < c1)
        & is_bullish(o, c)
        & (o > c2)
        & (o < o2)
        & (c < o1)
        & (c > c1)
    )
    return combine_bullish_bearish(bull, bear)
