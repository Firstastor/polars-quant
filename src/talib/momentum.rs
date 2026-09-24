use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use serde::Deserialize;
use std::collections::VecDeque;

#[derive(Deserialize)]
struct ApoKwargs {
    fastperiod: Option<usize>,
    slowperiod: Option<usize>,
    matype: Option<usize>,
}

#[derive(Deserialize)]
struct MacdKwargs {
    fastperiod: Option<usize>,
    slowperiod: Option<usize>,
    signalperiod: Option<usize>,
}

#[derive(Deserialize)]
struct TimeperiodKwargs {
    timeperiod: Option<usize>,
}

#[derive(Deserialize)]
struct UltoscKwargs {
    timeperiod1: Option<usize>,
    timeperiod2: Option<usize>,
    timeperiod3: Option<usize>,
}

#[derive(Deserialize)]
struct StochKwargs {
    fastk_period: Option<usize>,
    slowk_period: Option<usize>,
    slowk_matype: Option<usize>,
    slowd_period: Option<usize>,
    slowd_matype: Option<usize>,
}

#[derive(Deserialize)]
struct StochfKwargs {
    fastk_period: Option<usize>,
    fastd_period: Option<usize>,
    fastd_matype: Option<usize>,
}

#[derive(Deserialize)]
struct StochrsiKwargs {
    timeperiod: Option<usize>,
    fastk_period: Option<usize>,
    fastd_period: Option<usize>,
    fastd_matype: Option<usize>,
}

fn aroon_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("aroon_down".into(), DataType::Float64);
    let f2 = Field::new("aroon_up".into(), DataType::Float64);
    Ok(Field::new("aroon".into(), DataType::Struct(vec![f1, f2])))
}

fn macd_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("macd".into(), DataType::Float64);
    let f2 = Field::new("macd_signal".into(), DataType::Float64);
    let f3 = Field::new("macd_hist".into(), DataType::Float64);
    Ok(Field::new(
        "macd".into(),
        DataType::Struct(vec![f1, f2, f3]),
    ))
}

fn stoch_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("slowk".into(), DataType::Float64);
    let f2 = Field::new("slowd".into(), DataType::Float64);
    Ok(Field::new("stoch".into(), DataType::Struct(vec![f1, f2])))
}

fn stochf_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("fastk".into(), DataType::Float64);
    let f2 = Field::new("fastd".into(), DataType::Float64);
    Ok(Field::new("stochf".into(), DataType::Struct(vec![f1, f2])))
}

fn stochrsi_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("fastk".into(), DataType::Float64);
    let f2 = Field::new("fastd".into(), DataType::Float64);
    Ok(Field::new(
        "stochrsi".into(),
        DataType::Struct(vec![f1, f2]),
    ))
}

#[polars_expr(output_type=Float64)]
pub fn adx(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_adx(high, low, close, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn adxr(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_adxr(high, low, close, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn apo(inputs: &[Series], kwargs: ApoKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let n = real.len();
    let fastperiod = kwargs.fastperiod.unwrap_or(12);
    let slowperiod = kwargs.slowperiod.unwrap_or(26);
    let matype = kwargs.matype.unwrap_or(0);

    if fastperiod == 0 || slowperiod == 0 || matype != 0 {
        use crate::talib::overlap::calc_ma;
        let fast = calc_ma(real, fastperiod, matype);
        let slow = calc_ma(real, slowperiod, matype);
        return Ok((&fast - &slow).into_series());
    }

    if real.null_count() != 0 {
        return Ok(Float64Chunked::full_null("apo".into(), n).into_series());
    }

    let real = real.rechunk();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("apo".into(), n);

    let mut fast_buf: VecDeque<f64> = VecDeque::with_capacity(fastperiod);
    let mut slow_buf: VecDeque<f64> = VecDeque::with_capacity(slowperiod);
    let mut fast_sum: f64 = 0.0;
    let mut slow_sum: f64 = 0.0;

    for val in real.no_null_iter() {
        fast_sum += val;
        fast_buf.push_back(val);
        if fast_buf.len() > fastperiod {
            let old = fast_buf.pop_front().unwrap();
            fast_sum -= old;
        }
        slow_sum += val;
        slow_buf.push_back(val);
        if slow_buf.len() > slowperiod {
            let old = slow_buf.pop_front().unwrap();
            slow_sum -= old;
        }
        if fast_buf.len() >= fastperiod && slow_buf.len() >= slowperiod {
            let fast_sma = fast_sum / fastperiod as f64;
            let slow_sma = slow_sum / slowperiod as f64;
            builder.append_value(fast_sma - slow_sma);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish().into_series())
}

#[polars_expr(output_type_func=aroon_output)]
pub fn aroon(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    let n = high.len();

    let (down, up) = calc_aroon(high, low, timeperiod);
    let s1 = down.into_series();
    let s2 = up.into_series();
    Ok(StructChunked::from_series("aroon".into(), n, [s1, s2].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn aroonosc(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let high = high.f64()?;
    let low = low.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_aroonosc(high, low, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn bop(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?;
    let high = inputs[1].cast(&DataType::Float64)?;
    let low = inputs[2].cast(&DataType::Float64)?;
    let close = inputs[3].cast(&DataType::Float64)?;

    let open = open.f64()?;
    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    Ok(calc_bop(open, high, low, close).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn cci(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_cci(high, low, close, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn cmo(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_cmo(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn dx(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_dx(high, low, close, timeperiod).into_series())
}

#[derive(Deserialize)]
struct MacdextKwargs {
    fastperiod: Option<usize>,
    fastmatype: Option<usize>,
    slowperiod: Option<usize>,
    slowmatype: Option<usize>,
    signalperiod: Option<usize>,
    signalmatype: Option<usize>,
}

fn macd_core(
    real: &[f64],
    mut fast_period: usize,
    mut slow_period: usize,
    signal_period: usize,
    fixed_k: Option<(f64, f64)>,
) -> (Float64Chunked, Float64Chunked, Float64Chunked) {
    let n = real.len();
    if slow_period < fast_period {
        std::mem::swap(&mut fast_period, &mut slow_period);
    }
    let (fast_k, slow_k) = match fixed_k {
        Some((fk, sk)) => (fk, sk),
        None => (
            2.0 / (fast_period as f64 + 1.0),
            2.0 / (slow_period as f64 + 1.0),
        ),
    };
    let signal_k = 2.0 / (signal_period as f64 + 1.0);
    let lookback = slow_period + signal_period - 2;

    let mut macd_b = PrimitiveChunkedBuilder::<Float64Type>::new("macd".into(), n);
    let mut signal_b = PrimitiveChunkedBuilder::<Float64Type>::new("macd_signal".into(), n);
    let mut hist_b = PrimitiveChunkedBuilder::<Float64Type>::new("macd_hist".into(), n);
    for _ in 0..lookback {
        macd_b.append_null();
        signal_b.append_null();
        hist_b.append_null();
    }

    let mut slow_sum = 0.0f64;
    for &x in &real[..slow_period] {
        slow_sum += x;
    }
    let mut fast_sum = 0.0f64;
    for &x in &real[(slow_period - fast_period)..slow_period] {
        fast_sum += x;
    }
    let mut prev_fast = fast_sum / fast_period as f64;
    let mut prev_slow = slow_sum / slow_period as f64;

    let mut cur_macd = prev_fast - prev_slow;
    let mut sig_sum = cur_macd;
    let mut bar = slow_period;
    while bar <= lookback {
        let x = real[bar];
        bar += 1;
        prev_fast = (x - prev_fast).mul_add(fast_k, prev_fast);
        prev_slow = (x - prev_slow).mul_add(slow_k, prev_slow);
        cur_macd = prev_fast - prev_slow;
        sig_sum += cur_macd;
    }
    let mut prev_signal = sig_sum / signal_period as f64;
    macd_b.append_value(cur_macd);
    signal_b.append_value(prev_signal);
    hist_b.append_value(cur_macd - prev_signal);

    for b in (lookback + 1)..n {
        let x = real[b];
        prev_fast = (x - prev_fast).mul_add(fast_k, prev_fast);
        prev_slow = (x - prev_slow).mul_add(slow_k, prev_slow);
        cur_macd = prev_fast - prev_slow;
        if signal_period == 1 {
            prev_signal = cur_macd;
        } else {
            prev_signal = (cur_macd - prev_signal).mul_add(signal_k, prev_signal);
        }
        macd_b.append_value(cur_macd);
        signal_b.append_value(prev_signal);
        hist_b.append_value(cur_macd - prev_signal);
    }
    (macd_b.finish(), signal_b.finish(), hist_b.finish())
}

fn null_macd_struct(n: usize) -> PolarsResult<Series> {
    let nulls = Float64Chunked::full_null("macd".into(), n);
    let s1 = nulls.clone().into_series();
    let s2 = nulls.clone().into_series();
    let s3 = nulls.into_series();
    StructChunked::from_series("macd".into(), n, [s1, s2, s3].iter()).map(|sc| sc.into_series())
}

#[polars_expr(output_type_func=macd_output)]
pub fn macd(inputs: &[Series], kwargs: MacdKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let n = real.len();

    let fast = kwargs.fastperiod.unwrap_or(12);
    let slow = kwargs.slowperiod.unwrap_or(26);
    let signal = kwargs.signalperiod.unwrap_or(9);

    if fast == 0 || slow == 0 || signal == 0 || real.null_count() != 0 {
        return null_macd_struct(n);
    }
    if n < slow.max(fast) + signal - 1 {
        return null_macd_struct(n);
    }

    let real: Vec<f64> = real.no_null_iter().collect();
    let (m, sg, h) = macd_core(&real, fast, slow, signal, None);
    let s1 = m.into_series();
    let s2 = sg.into_series();
    let s3 = h.into_series();
    Ok(StructChunked::from_series("macd".into(), n, [s1, s2, s3].iter())?.into_series())
}

#[derive(Deserialize)]
struct MacdfixKwargs {
    signalperiod: Option<usize>,
}

#[polars_expr(output_type_func=macd_output)]
pub fn macdfix(inputs: &[Series], kwargs: MacdfixKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let n = real.len();
    let signal = kwargs.signalperiod.unwrap_or(9);

    if signal == 0 || real.null_count() != 0 {
        return null_macd_struct(n);
    }
    if n < 26 + signal - 1 {
        return null_macd_struct(n);
    }
    let real: Vec<f64> = real.no_null_iter().collect();
    let (m, sg, h) = macd_core(&real, 12, 26, signal, Some((0.15, 0.075)));
    let s1 = m.into_series();
    let s2 = sg.into_series();
    let s3 = h.into_series();
    Ok(StructChunked::from_series("macd".into(), n, [s1, s2, s3].iter())?.into_series())
}

#[polars_expr(output_type_func=macd_output)]
pub fn macdext(inputs: &[Series], kwargs: MacdextKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let n = real.len();

    let fastperiod = kwargs.fastperiod.unwrap_or(12);
    let fastmatype = kwargs.fastmatype.unwrap_or(0);
    let slowperiod = kwargs.slowperiod.unwrap_or(26);
    let slowmatype = kwargs.slowmatype.unwrap_or(0);
    let signalperiod = kwargs.signalperiod.unwrap_or(9);
    let signalmatype = kwargs.signalmatype.unwrap_or(0);

    if fastperiod == 0 || slowperiod == 0 || signalperiod == 0 || real.null_count() != 0 {
        return null_macd_struct(n);
    }

    use crate::talib::overlap::calc_ma;
    let vals: Vec<f64> = real.no_null_iter().collect();
    let full = Float64Chunked::from_vec("real".into(), vals.clone());

    let l_fast = calc_ma(&full, fastperiod, fastmatype)
        .first_non_null()
        .unwrap_or(n);
    let l_slow = calc_ma(&full, slowperiod, slowmatype)
        .first_non_null()
        .unwrap_or(n);
    let anchor = l_fast.max(l_slow);
    if anchor >= n {
        return null_macd_struct(n);
    }

    let fast_slice = Float64Chunked::from_vec("fast".into(), vals[anchor - l_fast..].to_vec());
    let slow_slice = Float64Chunked::from_vec("slow".into(), vals[anchor - l_slow..].to_vec());
    let fast_anch = calc_ma(&fast_slice, fastperiod, fastmatype);
    let slow_anch = calc_ma(&slow_slice, slowperiod, slowmatype);

    let mut macd_vals = Vec::with_capacity(n - anchor);
    for b in anchor..n {
        let f = fast_anch.get(b - (anchor - l_fast)).unwrap();
        let g = slow_anch.get(b - (anchor - l_slow)).unwrap();
        macd_vals.push(f - g);
    }
    let macd_chunk = Float64Chunked::from_vec("macd_line".into(), macd_vals.clone());
    let sig_ma = calc_ma(&macd_chunk, signalperiod, signalmatype);
    let sig_rel0 = sig_ma.first_non_null().unwrap_or(sig_ma.len());
    let start = anchor + sig_rel0;
    if start >= n {
        return null_macd_struct(n);
    }

    let mut m = vec![f64::NAN; n];
    let mut sg = vec![f64::NAN; n];
    let mut h = vec![f64::NAN; n];
    for b in start..n {
        let mv = macd_vals[b - anchor];
        let sv = sig_ma.get(b - anchor).unwrap();
        m[b] = mv;
        sg[b] = sv;
        h[b] = mv - sv;
    }
    let s1 = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("macd".into(), n);
        for v in m {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish().into_series()
    };
    let s2 = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("macd_signal".into(), n);
        for v in sg {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish().into_series()
    };
    let s3 = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("macd_hist".into(), n);
        for v in h {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish().into_series()
    };
    Ok(StructChunked::from_series("macd".into(), n, [s1, s2, s3].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn mfi(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;
    let volume = inputs[3].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;
    let volume = volume.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_mfi(high, low, close, volume, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn minus_di(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let (_, minus_di) = calc_di(high, low, close, timeperiod);
    Ok(minus_di.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn minus_dm(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let high = high.f64()?;
    let low = low.f64()?;

    let (_, minus_dm) = calc_dm(high, low, timeperiod);
    Ok(minus_dm.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn plus_di(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let (plus_di, _) = calc_di(high, low, close, timeperiod);
    Ok(plus_di.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn plus_dm(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let (plus_dm, _) = calc_dm(high, low, timeperiod);
    Ok(plus_dm.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn ppo(inputs: &[Series], kwargs: ApoKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let n = real.len();
    let fastperiod = kwargs.fastperiod.unwrap_or(12);
    let slowperiod = kwargs.slowperiod.unwrap_or(26);
    let matype = kwargs.matype.unwrap_or(0);

    if fastperiod == 0 || slowperiod == 0 || matype != 0 {
        use crate::talib::overlap::calc_ma;
        let fast = calc_ma(real, fastperiod, matype);
        let slow = calc_ma(real, slowperiod, matype);
        let diff = &fast - &slow;
        return Ok((&diff / &slow * 100.0).into_series());
    }

    if real.null_count() != 0 {
        return Ok(Float64Chunked::full_null("ppo".into(), n).into_series());
    }

    let real = real.rechunk();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ppo".into(), n);

    let mut fast_buf: VecDeque<f64> = VecDeque::with_capacity(fastperiod);
    let mut slow_buf: VecDeque<f64> = VecDeque::with_capacity(slowperiod);
    let mut fast_sum: f64 = 0.0;
    let mut slow_sum: f64 = 0.0;

    for val in real.no_null_iter() {
        fast_sum += val;
        fast_buf.push_back(val);
        if fast_buf.len() > fastperiod {
            let old = fast_buf.pop_front().unwrap();
            fast_sum -= old;
        }
        slow_sum += val;
        slow_buf.push_back(val);
        if slow_buf.len() > slowperiod {
            let old = slow_buf.pop_front().unwrap();
            slow_sum -= old;
        }
        if fast_buf.len() >= fastperiod && slow_buf.len() >= slowperiod {
            let fast_sma = fast_sum / fastperiod as f64;
            let slow_sma = slow_sum / slowperiod as f64;
            if slow_sma == 0.0 {
                builder.append_value(0.0);
            } else {
                builder.append_value((fast_sma - slow_sma) / slow_sma * 100.0);
            }
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn mom(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(10);
    Ok((real - &real.shift(timeperiod as i64)).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn rsi(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_rsi(real, timeperiod).into_series())
}

#[polars_expr(output_type_func=stoch_output)]
pub fn stoch(inputs: &[Series], kwargs: StochKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let fastk_period = kwargs.fastk_period.unwrap_or(5);
    let slowk_period = kwargs.slowk_period.unwrap_or(3);
    let slowk_matype = kwargs.slowk_matype.unwrap_or(0);
    let slowd_period = kwargs.slowd_period.unwrap_or(3);
    let slowd_matype = kwargs.slowd_matype.unwrap_or(0);

    let n = high.len();
    let (slowk, slowd) = calc_stoch(
        high,
        low,
        close,
        fastk_period,
        slowk_period,
        slowk_matype,
        slowd_period,
        slowd_matype,
    );
    let s1 = slowk.into_series();
    let s2 = slowd.into_series();
    Ok(StructChunked::from_series("stoch".into(), n, [s1, s2].iter())?.into_series())
}

#[polars_expr(output_type_func=stochf_output)]
pub fn stochf(inputs: &[Series], kwargs: StochfKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let fastk_period = kwargs.fastk_period.unwrap_or(5);
    let fastd_period = kwargs.fastd_period.unwrap_or(3);
    let fastd_matype = kwargs.fastd_matype.unwrap_or(0);

    let n = high.len();
    let (fastk, fastd) = calc_stochf(high, low, close, fastk_period, fastd_period, fastd_matype);
    let s1 = fastk.into_series();
    let s2 = fastd.into_series();
    Ok(StructChunked::from_series("stochf".into(), n, [s1, s2].iter())?.into_series())
}

#[polars_expr(output_type_func=stochrsi_output)]
pub fn stochrsi(inputs: &[Series], kwargs: StochrsiKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    let fastk_period = kwargs.fastk_period.unwrap_or(5);
    let fastd_period = kwargs.fastd_period.unwrap_or(3);
    let fastd_matype = kwargs.fastd_matype.unwrap_or(0);

    let n = real.len();
    let (fastk, fastd) = calc_stochrsi(real, timeperiod, fastk_period, fastd_period, fastd_matype);
    let s1 = fastk.into_series();
    let s2 = fastd.into_series();
    Ok(StructChunked::from_series("stochrsi".into(), n, [s1, s2].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn trix(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);
    Ok(calc_trix(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn ultosc(inputs: &[Series], kwargs: UltoscKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let p1 = kwargs.timeperiod1.unwrap_or(7);
    let p2 = kwargs.timeperiod2.unwrap_or(14);
    let p3 = kwargs.timeperiod3.unwrap_or(28);
    Ok(calc_ultosc(high, low, close, p1, p2, p3).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn willr(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_willr(high, low, close, timeperiod).into_series())
}

fn calc_adx(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0
        || n < 2 * timeperiod
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("adx".into(), n);
    }

    let h: Vec<f64> = high.no_null_iter().collect();
    let l: Vec<f64> = low.no_null_iter().collect();
    let c: Vec<f64> = close.no_null_iter().collect();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("adx".into(), n);
    let inv = timeperiod as f64;

    let mut minus_dm = 0.0f64;
    let mut plus_dm = 0.0f64;
    let mut tr = 0.0f64;
    let mut sumdx = 0.0f64;
    let mut adx = 0.0f64;

    let mut h_pre = h[0];
    let mut l_pre = l[0];
    let mut c_pre = c[0];
    builder.append_null();

    for i in 1..n {
        let _up = h[i] - h_pre;
        let _down = l_pre - l[i];
        let p_dm = if _up > 0.0 && _up > _down { _up } else { 0.0 };
        let m_dm = if _down > 0.0 && _down > _up {
            _down
        } else {
            0.0
        };
        let tr_i = (h[i] - l[i])
            .max((h[i] - c_pre).abs())
            .max((l[i] - c_pre).abs());
        h_pre = h[i];
        l_pre = l[i];
        c_pre = c[i];

        if i < timeperiod {
            plus_dm += p_dm;
            minus_dm += m_dm;
            tr += tr_i;
            builder.append_null();
        } else if i < 2 * timeperiod {
            plus_dm = plus_dm - plus_dm / inv + p_dm;
            minus_dm = minus_dm - minus_dm / inv + m_dm;
            tr = tr - tr / inv + tr_i;
            if tr > 0.0 {
                let minus_di = 100.0 * (minus_dm / tr);
                let plus_di = 100.0 * (plus_dm / tr);
                let sum_di = minus_di + plus_di;
                if sum_di.abs() >= 1e-14 {
                    sumdx += 100.0 * ((minus_di - plus_di).abs() / sum_di);
                }
            }
            if i == 2 * timeperiod - 1 {
                adx = sumdx / inv;
                builder.append_value(adx);
            } else {
                builder.append_null();
            }
        } else {
            plus_dm = plus_dm - plus_dm / inv + p_dm;
            minus_dm = minus_dm - minus_dm / inv + m_dm;
            tr = tr - tr / inv + tr_i;
            if tr > 0.0 {
                let minus_di = 100.0 * (minus_dm / tr);
                let plus_di = 100.0 * (plus_dm / tr);
                let sum_di = minus_di + plus_di;
                if sum_di.abs() >= 1e-14 {
                    let dx = 100.0 * ((minus_di - plus_di).abs() / sum_di);
                    adx = (adx * (timeperiod as f64 - 1.0) + dx) / inv;
                }
            }
            builder.append_value(adx);
        }
    }

    builder.finish()
}

fn calc_adxr(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0
        || n < 3 * timeperiod - 1
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("adxr".into(), n);
    }
    let adx = calc_adx(high, low, close, timeperiod);
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("adxr".into(), n);
    let lag = timeperiod - 1;
    for i in 0..n {
        if i < 3 * timeperiod - 2 {
            builder.append_null();
        } else {
            let a = adx.get(i).unwrap();
            let b = adx.get(i - lag).unwrap();
            builder.append_value((a + b) / 2.0);
        }
    }
    builder.finish()
}

fn calc_aroon(
    high: &Float64Chunked,
    low: &Float64Chunked,
    timeperiod: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = high.len();

    if timeperiod == 0 || n < timeperiod || high.null_count() + low.null_count() != 0 {
        let nulls = Float64Chunked::full_null("aroon".into(), n);
        return (nulls.clone(), nulls);
    }
    let high = high.rechunk();
    let low = low.rechunk();
    let h = high.cont_slice().unwrap();
    let l = low.cont_slice().unwrap();

    let inv_period = 100.0 / timeperiod as f64;
    let mut hi_q: VecDeque<(usize, f64)> = VecDeque::new();
    let mut lo_q: VecDeque<(usize, f64)> = VecDeque::new();
    let mut up = Vec::with_capacity(n - timeperiod);
    let mut down = Vec::with_capacity(n - timeperiod);

    for i in 0..n {
        while hi_q.back().map_or(false, |&(_, v)| v <= h[i]) {
            hi_q.pop_back();
        }
        hi_q.push_back((i, h[i]));
        while lo_q.back().map_or(false, |&(_, v)| v >= l[i]) {
            lo_q.pop_back();
        }
        lo_q.push_back((i, l[i]));

        if i >= timeperiod {
            let window_start = i - timeperiod;
            while hi_q.front().map_or(false, |&(i0, _)| i0 < window_start) {
                hi_q.pop_front();
            }
            while lo_q.front().map_or(false, |&(i0, _)| i0 < window_start) {
                lo_q.pop_front();
            }
            let up_v = if let Some(&(i0, _)) = hi_q.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };
            let down_v = if let Some(&(i0, _)) = lo_q.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };
            up.push(up_v);
            down.push(down_v);
        }
    }
    let out_down = {
        let mut ch = Float64Chunked::full_null("aroon_down".into(), timeperiod);
        ch.append(&Float64Chunked::from_vec("aroon_down".into(), down))
            .expect("append");
        ch
    };
    let out_up = {
        let mut ch = Float64Chunked::full_null("aroon_up".into(), timeperiod);
        ch.append(&Float64Chunked::from_vec("aroon_up".into(), up))
            .expect("append");
        ch
    };
    (out_down, out_up)
}

fn calc_aroonosc(high: &Float64Chunked, low: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = high.len();

    if timeperiod == 0 || n < timeperiod || high.null_count() + low.null_count() != 0 {
        return Float64Chunked::full_null("aroonosc".into(), n);
    }
    let high = high.rechunk();
    let low = low.rechunk();
    let h = high.cont_slice().unwrap();
    let l = low.cont_slice().unwrap();

    let inv_period = 100.0 / timeperiod as f64;
    let mut hi_q: VecDeque<(usize, f64)> = VecDeque::new();
    let mut lo_q: VecDeque<(usize, f64)> = VecDeque::new();
    let mut out = Vec::with_capacity(n - timeperiod);

    for i in 0..n {
        while hi_q.back().map_or(false, |&(_, v)| v <= h[i]) {
            hi_q.pop_back();
        }
        hi_q.push_back((i, h[i]));
        while lo_q.back().map_or(false, |&(_, v)| v >= l[i]) {
            lo_q.pop_back();
        }
        lo_q.push_back((i, l[i]));

        if i >= timeperiod {
            let window_start = i - timeperiod;
            while hi_q.front().map_or(false, |&(i0, _)| i0 < window_start) {
                hi_q.pop_front();
            }
            while lo_q.front().map_or(false, |&(i0, _)| i0 < window_start) {
                lo_q.pop_front();
            }
            let up = if let Some(&(i0, _)) = hi_q.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };
            let down = if let Some(&(i0, _)) = lo_q.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };
            out.push(up - down);
        }
    }
    let mut ch = Float64Chunked::full_null("aroonosc".into(), timeperiod);
    ch.append(&Float64Chunked::from_vec("aroonosc".into(), out))
        .expect("append");
    ch
}

fn calc_bop(
    open: &Float64Chunked,
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
) -> Float64Chunked {
    let n = open.len();

    if open.null_count() + high.null_count() + low.null_count() + close.null_count() != 0 {
        return Float64Chunked::full_null("bop".into(), n);
    }

    let open = open.rechunk();
    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("bop".into(), n);

    for (o, h, l, c) in izip!(
        open.no_null_iter(),
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    ) {
        if h == l {
            builder.append_value(0.0);
        } else {
            builder.append_value((c - o) / (h - l));
        }
    }

    builder.finish()
}

fn calc_cci(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();

    if timeperiod == 0
        || n < timeperiod
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("cci".into(), n);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("cci".into(), n);

    let mut tp_window: VecDeque<f64> = VecDeque::with_capacity(timeperiod);
    let mut sma_sum: f64 = 0.0;
    let mut i: usize = 0;

    for (h, l, c) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    ) {
        let tp = (h + l + c) / 3.0;

        sma_sum += tp;
        tp_window.push_back(tp);
        if i >= timeperiod {
            let old = tp_window.pop_front().unwrap();
            sma_sum -= old;
        }

        if i >= timeperiod - 1 {
            let sma_val = sma_sum / timeperiod as f64;
            let mut dev_sum = 0.0f64;
            for &v in tp_window.iter() {
                dev_sum += (v - sma_val).abs();
            }
            let mean_dev = dev_sum / timeperiod as f64;
            if mean_dev == 0.0 {
                builder.append_value(0.0);
            } else {
                builder.append_value((tp - sma_val) / (0.015 * mean_dev));
            }
        } else {
            builder.append_null();
        }

        i += 1;
    }

    builder.finish()
}

fn calc_cmo(real: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = real.len();
    if timeperiod == 0 || n < timeperiod + 1 || real.null_count() != 0 {
        return Float64Chunked::full_null("cmo".into(), n);
    }
    let real = real.rechunk();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("cmo".into(), n);
    let mut gain: f64 = 0.0;
    let mut loss: f64 = 0.0;
    let mut prev: f64 = f64::NAN;
    let inv: f64 = 1.0 / timeperiod as f64;
    for (i, cur) in real.no_null_iter().enumerate() {
        if i == 0 {
            builder.append_null();
            prev = cur;
            continue;
        }
        let diff = cur - prev;
        let g = if diff > 0.0 { diff } else { 0.0 };
        let l = if diff < 0.0 { -diff } else { 0.0 };
        if i < timeperiod {
            gain += g;
            loss += l;
            builder.append_null();
        } else if i == timeperiod {
            gain = (gain + g) / timeperiod as f64;
            loss = (loss + l) / timeperiod as f64;
            let denom = gain + loss;
            builder.append_value(if denom == 0.0 {
                0.0
            } else {
                100.0 * (gain - loss) / denom
            });
        } else {
            gain = gain - gain * inv + g * inv;
            loss = loss - loss * inv + l * inv;
            let denom = gain + loss;
            builder.append_value(if denom == 0.0 {
                0.0
            } else {
                100.0 * (gain - loss) / denom
            });
        }
        prev = cur;
    }
    builder.finish()
}

fn raw_stoch_k(h: &[f64], l: &[f64], c: &[f64], fastk: usize) -> Vec<f64> {
    let m = h.len();
    let mut out = Vec::with_capacity(m - (fastk - 1));
    let mut hi_deque: VecDeque<(usize, f64)> = VecDeque::new();
    let mut lo_deque: VecDeque<(usize, f64)> = VecDeque::new();
    for i in 0..m {
        while let Some(&(_, v)) = hi_deque.back() {
            if v <= h[i] {
                hi_deque.pop_back();
            } else {
                break;
            }
        }
        hi_deque.push_back((i, h[i]));
        while let Some(&(_, v)) = lo_deque.back() {
            if v >= l[i] {
                lo_deque.pop_back();
            } else {
                break;
            }
        }
        lo_deque.push_back((i, l[i]));

        if i + 1 >= fastk {
            let start = i + 1 - fastk;
            while let Some(&(i0, _)) = hi_deque.front() {
                if i0 < start {
                    hi_deque.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(i0, _)) = lo_deque.front() {
                if i0 < start {
                    lo_deque.pop_front();
                } else {
                    break;
                }
            }
            let highest = hi_deque.front().unwrap().1;
            let lowest = lo_deque.front().unwrap().1;
            let diff = (highest - lowest) / 100.0;
            let k = if (highest - lowest).abs() <= 1e-14 * (highest.abs() + lowest.abs()) {
                0.0
            } else {
                (c[i] - lowest) / diff
            };
            out.push(k);
        }
    }
    out
}

fn stochf_vectors(
    h: &[f64],
    l: &[f64],
    c: &[f64],
    fastk: usize,
    fastd: usize,
    fastd_matype: usize,
) -> (Vec<f64>, Vec<f64>) {
    let m = h.len();
    let mut fastk_out = vec![f64::NAN; m];
    let mut fastd_out = vec![f64::NAN; m];

    let raw = raw_stoch_k(h, l, c, fastk);
    let raw_chunk = Float64Chunked::from_vec("fastk".into(), raw.to_vec());
    use crate::talib::overlap::calc_ma;
    let d_ma = calc_ma(&raw_chunk, fastd, fastd_matype);
    let d_rel0 = d_ma.first_non_null().unwrap_or(d_ma.len());
    let start = (fastk - 1) + d_rel0;
    if start < m {
        for i in start..m {
            fastk_out[i] = raw[i - (fastk - 1)];
            fastd_out[i] = d_ma.get(i - (fastk - 1)).unwrap();
        }
    }
    (fastk_out, fastd_out)
}

fn calc_stoch(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    fastk_period: usize,
    slowk_period: usize,
    slowk_matype: usize,
    slowd_period: usize,
    slowd_matype: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = high.len();
    let nulls = || {
        (
            Float64Chunked::full_null("slowk".into(), n),
            Float64Chunked::full_null("slowd".into(), n),
        )
    };
    if fastk_period == 0
        || slowk_period == 0
        || slowd_period == 0
        || n < fastk_period
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return nulls();
    }
    let h: Vec<f64> = high.no_null_iter().collect();
    let l: Vec<f64> = low.no_null_iter().collect();
    let c: Vec<f64> = close.no_null_iter().collect();
    let m = n;

    let raw = raw_stoch_k(&h, &l, &c, fastk_period);
    let raw_chunk = Float64Chunked::from_vec("fastk".into(), raw.to_vec());
    use crate::talib::overlap::calc_ma;
    let k_ma = calc_ma(&raw_chunk, slowk_period, slowk_matype);
    let k_rel0 = k_ma.first_non_null().unwrap_or(k_ma.len());
    let mut k_vals = Vec::with_capacity(k_ma.len() - k_rel0);
    for p in k_rel0..k_ma.len() {
        k_vals.push(k_ma.get(p).unwrap());
    }
    let k_chunk = Float64Chunked::from_vec("slowk".into(), k_vals.to_vec());
    let d_ma = calc_ma(&k_chunk, slowd_period, slowd_matype);
    let d_rel0 = d_ma.first_non_null().unwrap_or(d_ma.len());
    let start = (fastk_period - 1) + k_rel0 + d_rel0;

    let mut slowk = vec![f64::NAN; m];
    let mut slowd = vec![f64::NAN; m];
    if start < m {
        for i in start..m {
            slowk[i] = k_ma.get(i - (fastk_period - 1)).unwrap();
            slowd[i] = d_ma.get(i - (fastk_period - 1) - k_rel0).unwrap();
        }
    }
    let sk_chunk = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("slowk".into(), n);
        for v in slowk {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish()
    };
    let sd_chunk = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("slowd".into(), n);
        for v in slowd {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish()
    };
    (sk_chunk, sd_chunk)
}

fn calc_stochf(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = high.len();
    let nulls = || {
        (
            Float64Chunked::full_null("fastk".into(), n),
            Float64Chunked::full_null("fastd".into(), n),
        )
    };
    if fastk_period == 0
        || fastd_period == 0
        || n < fastk_period
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return nulls();
    }
    let h: Vec<f64> = high.no_null_iter().collect();
    let l: Vec<f64> = low.no_null_iter().collect();
    let c: Vec<f64> = close.no_null_iter().collect();
    let (fk, fd) = stochf_vectors(&h, &l, &c, fastk_period, fastd_period, fastd_matype);
    if fk.iter().all(|v| v.is_nan()) {
        return nulls();
    }
    let a_chunk = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("fastk".into(), n);
        for v in fk {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish()
    };
    let b_chunk = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("fastd".into(), n);
        for v in fd {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish()
    };
    (a_chunk, b_chunk)
}

fn calc_stochrsi(
    real: &Float64Chunked,
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = real.len();
    let nulls = || {
        (
            Float64Chunked::full_null("fastk".into(), n),
            Float64Chunked::full_null("fastd".into(), n),
        )
    };
    if timeperiod == 0
        || fastk_period == 0
        || fastd_period == 0
        || n < timeperiod + fastk_period
        || real.null_count() != 0
    {
        return nulls();
    }
    let rsi = calc_rsi(real, timeperiod);
    let r0 = rsi.first_non_null().unwrap_or(n);
    if r0 >= n {
        return nulls();
    }
    let mut rsi_vals = Vec::with_capacity(n - r0);
    for i in r0..n {
        rsi_vals.push(rsi.get(i).unwrap());
    }
    let (fk, fd) = stochf_vectors(
        &rsi_vals,
        &rsi_vals,
        &rsi_vals,
        fastk_period,
        fastd_period,
        fastd_matype,
    );
    if fk.iter().all(|v| v.is_nan()) {
        return nulls();
    }
    let mut fastk = vec![f64::NAN; n];
    let mut fastd = vec![f64::NAN; n];
    for i in 0..n {
        if i >= r0 {
            fastk[i] = fk[i - r0];
            fastd[i] = fd[i - r0];
        }
    }
    let a_chunk = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("fastk".into(), n);
        for v in fastk {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish()
    };
    let b_chunk = {
        let mut b = PrimitiveChunkedBuilder::<Float64Type>::new("fastd".into(), n);
        for v in fastd {
            if v.is_nan() {
                b.append_null();
            } else {
                b.append_value(v);
            }
        }
        b.finish()
    };
    (a_chunk, b_chunk)
}

fn calc_trix(real: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = real.len();

    if timeperiod == 0 || n < 3 * timeperiod - 2 || real.null_count() != 0 {
        return Float64Chunked::full_null("trix".into(), n);
    }

    let real = real.rechunk();

    let alpha: f64 = 2.0 / (timeperiod + 1) as f64;
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("trix".into(), n);

    let mut ema1_sum: f64 = 0.0;
    let mut ema1_val: f64 = 0.0;
    let mut ema1_warm: bool = false;

    let mut ema2_sum: f64 = 0.0;
    let mut ema2_val: f64 = 0.0;
    let mut ema2_count: usize = 0;
    let mut ema2_warm: bool = false;

    let mut ema3_sum: f64 = 0.0;
    let mut ema3_val: f64 = 0.0;
    let mut ema3_count: usize = 0;
    let mut ema3_warm: bool = false;

    let mut trix_prev: f64 = f64::NAN;
    let mut count: usize = 0;

    for val in real.no_null_iter() {
        count += 1;

        if count < timeperiod {
            ema1_sum += val;
        } else if count == timeperiod {
            ema1_sum += val;
            ema1_val = ema1_sum / timeperiod as f64;
            ema1_warm = true;
        } else {
            ema1_val += alpha * (val - ema1_val);
        }

        let ema2_input: Option<f64> = if ema1_warm { Some(ema1_val) } else { None };
        if let Some(v) = ema2_input {
            ema2_count += 1;
            if ema2_count < timeperiod {
                ema2_sum += v;
            } else if ema2_count == timeperiod {
                ema2_sum += v;
                ema2_val = ema2_sum / timeperiod as f64;
                ema2_warm = true;
            } else {
                ema2_val += alpha * (v - ema2_val);
            }
        }

        let ema3_input: Option<f64> = if ema2_warm { Some(ema2_val) } else { None };
        if let Some(v) = ema3_input {
            ema3_count += 1;
            if ema3_count < timeperiod {
                ema3_sum += v;
            } else if ema3_count == timeperiod {
                ema3_sum += v;
                ema3_val = ema3_sum / timeperiod as f64;
                ema3_warm = true;
            } else {
                ema3_val += alpha * (v - ema3_val);
            }
        }

        if ema3_warm {
            if trix_prev.is_nan() {
                builder.append_null();
            } else if trix_prev == 0.0 {
                builder.append_value(0.0);
            } else {
                builder.append_value((ema3_val - trix_prev) / trix_prev * 100.0);
            }
            trix_prev = ema3_val;
        } else {
            builder.append_null();
        }
    }

    builder.finish()
}

fn calc_ultosc(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> Float64Chunked {
    let n = high.len();
    let max_period = timeperiod1.max(timeperiod2).max(timeperiod3);

    if timeperiod1 == 0
        || timeperiod2 == 0
        || timeperiod3 == 0
        || n <= max_period
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("ultosc".into(), n);
    }
    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();
    let h = high.cont_slice().unwrap();
    let l = low.cont_slice().unwrap();
    let c = close.cont_slice().unwrap();

    let mut bp1 = vec![0.0f64; timeperiod1];
    let mut tr1 = vec![0.0f64; timeperiod1];
    let mut bp2 = vec![0.0f64; timeperiod2];
    let mut tr2 = vec![0.0f64; timeperiod2];
    let mut bp3 = vec![0.0f64; timeperiod3];
    let mut tr3 = vec![0.0f64; timeperiod3];
    let (mut head1, mut head2, mut head3) = (0usize, 0usize, 0usize);
    let mut s_bp1 = 0.0f64;
    let mut s_tr1 = 0.0f64;
    let mut s_bp2 = 0.0f64;
    let mut s_tr2 = 0.0f64;
    let mut s_bp3 = 0.0f64;
    let mut s_tr3 = 0.0f64;

    let mut out = Vec::with_capacity(n - max_period);
    for i in 0..n {
        let (bp, tr) = if i == 0 {
            (c[0] - l[0], h[0] - l[0])
        } else {
            let min_lp = l[i].min(c[i - 1]);
            let max_hp = h[i].max(c[i - 1]);
            (c[i] - min_lp, max_hp - min_lp)
        };

        s_bp1 += bp - bp1[head1];
        bp1[head1] = bp;
        s_tr1 += tr - tr1[head1];
        tr1[head1] = tr;
        head1 = (head1 + 1) % timeperiod1;

        s_bp2 += bp - bp2[head2];
        bp2[head2] = bp;
        s_tr2 += tr - tr2[head2];
        tr2[head2] = tr;
        head2 = (head2 + 1) % timeperiod2;

        s_bp3 += bp - bp3[head3];
        bp3[head3] = bp;
        s_tr3 += tr - tr3[head3];
        tr3[head3] = tr;
        head3 = (head3 + 1) % timeperiod3;

        if i >= max_period {
            let avg1 = if s_tr1 == 0.0 { 0.0 } else { s_bp1 / s_tr1 };
            let avg2 = if s_tr2 == 0.0 { 0.0 } else { s_bp2 / s_tr2 };
            let avg3 = if s_tr3 == 0.0 { 0.0 } else { s_bp3 / s_tr3 };
            out.push(100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0);
        }
    }
    let mut ch = Float64Chunked::full_null("ultosc".into(), max_period);
    ch.append(&Float64Chunked::from_vec("ultosc".into(), out))
        .expect("append");
    ch
}

fn calc_di(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = high.len();
    let nulls = || {
        (
            Float64Chunked::full_null("minus_di".into(), n),
            Float64Chunked::full_null("plus_di".into(), n),
        )
    };
    if timeperiod == 0
        || n < timeperiod + 1
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return nulls();
    }

    let h: Vec<f64> = high.no_null_iter().collect();
    let l: Vec<f64> = low.no_null_iter().collect();
    let c: Vec<f64> = close.no_null_iter().collect();

    let mut minus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("minus_di".into(), n);
    let mut plus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("plus_di".into(), n);
    let inv = timeperiod as f64;

    let mut minus_dm = 0.0f64;
    let mut plus_dm = 0.0f64;
    let mut tr = 0.0f64;

    let mut h_pre = h[0];
    let mut l_pre = l[0];
    let mut c_pre = c[0];
    minus_builder.append_null();
    plus_builder.append_null();

    for i in 1..n {
        let _up = h[i] - h_pre;
        let _down = l_pre - l[i];
        let p_dm = if _up > 0.0 && _up > _down { _up } else { 0.0 };
        let m_dm = if _down > 0.0 && _down > _up {
            _down
        } else {
            0.0
        };
        let tr_i = (h[i] - l[i])
            .max((h[i] - c_pre).abs())
            .max((l[i] - c_pre).abs());
        h_pre = h[i];
        l_pre = l[i];
        c_pre = c[i];

        if i < timeperiod {
            minus_dm += m_dm;
            plus_dm += p_dm;
            tr += tr_i;
            minus_builder.append_null();
            plus_builder.append_null();
        } else {
            minus_dm = minus_dm - minus_dm / inv + m_dm;
            plus_dm = plus_dm - plus_dm / inv + p_dm;
            tr = tr - tr / inv + tr_i;
            if tr > 0.0 {
                minus_builder.append_value(100.0 * (minus_dm / tr));
                plus_builder.append_value(100.0 * (plus_dm / tr));
            } else {
                minus_builder.append_value(0.0);
                plus_builder.append_value(0.0);
            }
        }
    }

    (plus_builder.finish(), minus_builder.finish())
}

fn calc_dm(
    high: &Float64Chunked,
    low: &Float64Chunked,
    timeperiod: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = high.len();
    let nulls = || {
        (
            Float64Chunked::full_null("minus_dm".into(), n),
            Float64Chunked::full_null("plus_dm".into(), n),
        )
    };
    if timeperiod == 0 || n < timeperiod || high.null_count() + low.null_count() != 0 {
        return nulls();
    }

    let h: Vec<f64> = high.no_null_iter().collect();
    let l: Vec<f64> = low.no_null_iter().collect();

    let mut minus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("minus_dm".into(), n);
    let mut plus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("plus_dm".into(), n);
    let inv = timeperiod as f64;

    let mut minus_dm = 0.0f64;
    let mut plus_dm = 0.0f64;

    let mut h_pre = h[0];
    let mut l_pre = l[0];
    minus_builder.append_null();
    plus_builder.append_null();

    for i in 1..n {
        let _up = h[i] - h_pre;
        let _down = l_pre - l[i];
        let p_dm = if _up > 0.0 && _up > _down { _up } else { 0.0 };
        let m_dm = if _down > 0.0 && _down > _up {
            _down
        } else {
            0.0
        };
        h_pre = h[i];
        l_pre = l[i];

        if i < timeperiod - 1 {
            plus_dm += p_dm;
            minus_dm += m_dm;
            minus_builder.append_null();
            plus_builder.append_null();
        } else if i == timeperiod - 1 {
            plus_dm += p_dm;
            minus_dm += m_dm;
            minus_builder.append_value(minus_dm);
            plus_builder.append_value(plus_dm);
        } else {
            plus_dm = plus_dm - plus_dm / inv + p_dm;
            minus_dm = minus_dm - minus_dm / inv + m_dm;
            minus_builder.append_value(minus_dm);
            plus_builder.append_value(plus_dm);
        }
    }

    (plus_builder.finish(), minus_builder.finish())
}

fn calc_dx(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0
        || n < timeperiod + 1
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("dx".into(), n);
    }

    let h: Vec<f64> = high.no_null_iter().collect();
    let l: Vec<f64> = low.no_null_iter().collect();
    let c: Vec<f64> = close.no_null_iter().collect();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("dx".into(), n);
    let inv = timeperiod as f64;

    let mut minus_dm = 0.0f64;
    let mut plus_dm = 0.0f64;
    let mut tr = 0.0f64;

    let mut h_pre = h[0];
    let mut l_pre = l[0];
    let mut c_pre = c[0];
    builder.append_null();

    for i in 1..n {
        let _up = h[i] - h_pre;
        let _down = l_pre - l[i];
        let p_dm = if _up > 0.0 && _up > _down { _up } else { 0.0 };
        let m_dm = if _down > 0.0 && _down > _up {
            _down
        } else {
            0.0
        };
        let tr_i = (h[i] - l[i])
            .max((h[i] - c_pre).abs())
            .max((l[i] - c_pre).abs());
        h_pre = h[i];
        l_pre = l[i];
        c_pre = c[i];

        if i < timeperiod {
            minus_dm += m_dm;
            plus_dm += p_dm;
            tr += tr_i;
            builder.append_null();
        } else {
            minus_dm = minus_dm - minus_dm / inv + m_dm;
            plus_dm = plus_dm - plus_dm / inv + p_dm;
            tr = tr - tr / inv + tr_i;
            let dx = if tr > 0.0 {
                let minus_di = 100.0 * (minus_dm / tr);
                let plus_di = 100.0 * (plus_dm / tr);
                let sum_di = minus_di + plus_di;
                if sum_di.abs() >= 1e-14 {
                    100.0 * ((minus_di - plus_di).abs() / sum_di)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            builder.append_value(dx);
        }
    }

    builder.finish()
}

fn calc_mfi(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    volume: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0
        || n < timeperiod + 1
        || high.null_count() + low.null_count() + close.null_count() + volume.null_count() != 0
    {
        return Float64Chunked::full_null("mfi".into(), n);
    }

    let h: Vec<f64> = high.no_null_iter().collect();
    let l: Vec<f64> = low.no_null_iter().collect();
    let c: Vec<f64> = close.no_null_iter().collect();
    let v: Vec<f64> = volume.no_null_iter().collect();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("mfi".into(), n);

    let mut pos_buf = vec![0.0f64; timeperiod];
    let mut neg_buf = vec![0.0f64; timeperiod];
    let mut buf_idx = 0usize;
    let max_idx = timeperiod - 1;

    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;
    let mut null_run = 0usize;

    let mut prev_tp = (h[0] + l[0] + c[0]) / 3.0;
    builder.append_null();

    for bar in 1..=timeperiod {
        let tp = (h[bar] + l[bar] + c[bar]) / 3.0;
        let dtp = tp - prev_tp;
        let scale = tp.abs() + prev_tp.abs();
        prev_tp = tp;
        let mf = tp * v[bar];

        let money_flow = if dtp.abs() <= 1e-14 * scale { 0.0 } else { mf };
        let pos_flow = if dtp < 0.0 { 0.0 } else { money_flow };
        let neg_flow = if dtp < 0.0 { money_flow } else { 0.0 };

        pos_buf[buf_idx] = pos_flow;
        neg_buf[buf_idx] = neg_flow;
        pos_sum += pos_flow;
        neg_sum += neg_flow;

        null_run = if money_flow == 0.0 { null_run + 1 } else { 0 };
        if null_run >= timeperiod {
            null_run = timeperiod;
            pos_sum = 0.0;
            neg_sum = 0.0;
        }
        buf_idx += 1;
        if buf_idx > max_idx {
            buf_idx = 0;
        }
        if bar < timeperiod {
            builder.append_null();
        }
    }

    let total = pos_sum + neg_sum;
    let clamped = pos_sum.clamp(0.0, total);
    if total <= 0.0 {
        builder.append_value(0.0);
    } else {
        builder.append_value(100.0 * (clamped / total));
    }

    for bar in (timeperiod + 1)..n {
        pos_sum -= pos_buf[buf_idx];
        neg_sum -= neg_buf[buf_idx];

        let tp = (h[bar] + l[bar] + c[bar]) / 3.0;
        let dtp = tp - prev_tp;
        let scale = tp.abs() + prev_tp.abs();
        prev_tp = tp;
        let mf = tp * v[bar];

        let money_flow = if dtp.abs() <= 1e-14 * scale { 0.0 } else { mf };
        let pos_flow = if dtp < 0.0 { 0.0 } else { money_flow };
        let neg_flow = if dtp < 0.0 { money_flow } else { 0.0 };

        pos_buf[buf_idx] = pos_flow;
        neg_buf[buf_idx] = neg_flow;
        pos_sum += pos_flow;
        neg_sum += neg_flow;

        null_run = if money_flow == 0.0 { null_run + 1 } else { 0 };
        if null_run >= timeperiod {
            null_run = timeperiod;
            pos_sum = 0.0;
            neg_sum = 0.0;
        }
        buf_idx += 1;
        if buf_idx > max_idx {
            buf_idx = 0;
        }

        let total = pos_sum + neg_sum;
        let clamped = pos_sum.clamp(0.0, total);
        if total <= 0.0 {
            builder.append_value(0.0);
        } else {
            builder.append_value(100.0 * (clamped / total));
        }
    }

    builder.finish()
}

fn calc_rsi(real: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = real.len();
    if timeperiod == 0 || n < timeperiod + 1 || real.null_count() != 0 {
        return Float64Chunked::full_null("rsi".into(), n);
    }

    let real = real.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("rsi".into(), n);

    let alpha = 1. / timeperiod as f64;
    let mut gain: f64 = 0.0;
    let mut loss: f64 = 0.0;
    let mut val_pre: f64 = f64::NAN;

    for (i, val) in real.no_null_iter().enumerate() {
        if i == 0 {
            builder.append_null();
            val_pre = val;
            continue;
        }
        let diff = val - val_pre;
        let g = if diff > 0.0 { diff } else { 0.0 };
        let l = if diff < 0.0 { -diff } else { 0.0 };
        if i < timeperiod {
            gain += g;
            loss += l;
            builder.append_null();
        } else if i == timeperiod {
            gain = (gain + g) * alpha;
            loss = (loss + l) * alpha;
            let denom = gain + loss;
            builder.append_value(if denom != 0.0 {
                100.0 * gain / denom
            } else {
                0.0
            });
        } else {
            gain += alpha * (g - gain);
            loss += alpha * (l - loss);
            let denom = gain + loss;
            builder.append_value(if denom != 0.0 {
                100.0 * gain / denom
            } else {
                0.0
            });
        }
        val_pre = val;
    }

    builder.finish()
}

fn calc_willr(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0
        || n < timeperiod
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("willr".into(), n);
    }
    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();
    let h = high.cont_slice().unwrap();
    let l = low.cont_slice().unwrap();
    let c = close.cont_slice().unwrap();

    let mut hi_q: VecDeque<(usize, f64)> = VecDeque::new();
    let mut lo_q: VecDeque<(usize, f64)> = VecDeque::new();

    let head = timeperiod - 1;
    let mut out = Vec::with_capacity(n - head);
    for i in 0..n {
        while hi_q.back().map_or(false, |&(_, v)| v <= h[i]) {
            hi_q.pop_back();
        }
        hi_q.push_back((i, h[i]));
        while lo_q.back().map_or(false, |&(_, v)| v >= l[i]) {
            lo_q.pop_back();
        }
        lo_q.push_back((i, l[i]));

        if i >= timeperiod {
            let start = i + 1 - timeperiod;
            while hi_q.front().map_or(false, |&(i0, _)| i0 < start) {
                hi_q.pop_front();
            }
            while lo_q.front().map_or(false, |&(i0, _)| i0 < start) {
                lo_q.pop_front();
            }
        }
        if i >= head {
            let hi = hi_q.front().unwrap().1;
            let lo = lo_q.front().unwrap().1;
            out.push(-100.0 * (hi - c[i]) / (hi - lo));
        }
    }
    let mut ch = Float64Chunked::full_null("willr".into(), head);
    ch.append(&Float64Chunked::from_vec("willr".into(), out))
        .expect("append");
    ch
}
