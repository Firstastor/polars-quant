use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use ringbuffer::{AllocRingBuffer, RingBuffer};
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
    let _close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_adx(high, low, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn adxr(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let _close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_adxr(high, low, timeperiod).into_series())
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
    let _close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    Ok(calc_dx(high, low, timeperiod).into_series())
}

#[polars_expr(output_type_func=macd_output)]
pub fn macd(inputs: &[Series], kwargs: MacdKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let n = real.len();

    let fast = kwargs.fastperiod.unwrap_or(12);
    let slow = kwargs.slowperiod.unwrap_or(26);
    let signal = kwargs.signalperiod.unwrap_or(9);

    if fast == 0 || slow == 0 || signal == 0 || n < slow || real.null_count() != 0 {
        let nulls = Float64Chunked::full_null("macd".into(), n);
        let s1 = nulls.clone().into_series();
        let s2 = nulls.clone().into_series();
        let s3 = nulls.into_series();
        return Ok(
            StructChunked::from_series("macd".into(), n, [s1, s2, s3].iter())?.into_series(),
        );
    }

    let real = real.rechunk();

    let alpha_fast: f64 = 2.0 / (fast + 1) as f64;
    let alpha_slow: f64 = 2.0 / (slow + 1) as f64;
    let mut macd_builder = PrimitiveChunkedBuilder::<Float64Type>::new("macd".into(), n);

    let mut fast_sum: f64 = 0.0;
    let mut slow_sum: f64 = 0.0;
    let mut fast_ema: f64 = 0.0;
    let mut slow_ema: f64 = 0.0;
    let mut fast_warm = false;
    let mut slow_warm = false;

    let mut count: usize = 0;
    for val in real.no_null_iter() {
        count += 1;

        if count < fast {
            fast_sum += val;
        } else if count == fast {
            fast_sum += val;
            fast_ema = fast_sum / fast as f64;
            fast_warm = true;
        } else {
            fast_ema += alpha_fast * (val - fast_ema);
        }

        if count < slow {
            slow_sum += val;
        } else if count == slow {
            slow_sum += val;
            slow_ema = slow_sum / slow as f64;
            slow_warm = true;
        } else {
            slow_ema += alpha_slow * (val - slow_ema);
        }

        if fast_warm && slow_warm {
            macd_builder.append_value(fast_ema - slow_ema);
        } else {
            macd_builder.append_null();
        }
    }

    let macd_line = macd_builder.finish();

    let alpha_signal: f64 = 2.0 / (signal + 1) as f64;
    let mut signal_builder = PrimitiveChunkedBuilder::<Float64Type>::new("macd_signal".into(), n);
    let mut sig_valid_count: usize = 0;
    let mut sig_sum: f64 = 0.0;
    let mut sig_ema: f64 = 0.0;
    let mut sig_warm = false;

    for opt in macd_line.iter() {
        if let Some(val) = opt {
            sig_valid_count += 1;
            if !sig_warm {
                sig_sum += val;
                if sig_valid_count == signal {
                    sig_ema = sig_sum / signal as f64;
                    sig_warm = true;
                    signal_builder.append_value(sig_ema);
                } else {
                    signal_builder.append_null();
                }
            } else {
                sig_ema += alpha_signal * (val - sig_ema);
                signal_builder.append_value(sig_ema);
            }
        } else {
            signal_builder.append_null();
        }
    }
    let signal_line = signal_builder.finish();

    let hist = &macd_line - &signal_line;

    let s1 = macd_line.with_name("macd".into()).into_series();
    let s2 = signal_line.with_name("macd_signal".into()).into_series();
    let s3 = hist.with_name("macd_hist".into()).into_series();
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
pub fn mom(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(10);
    Ok((real - &real.shift(timeperiod as i64)).into_series())
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
pub fn roc(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(10);
    let shifted = real.shift(timeperiod as i64);
    Ok((&(real - &shifted) / &shifted * 100.0).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn rocp(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(10);
    let shifted = real.shift(timeperiod as i64);
    Ok((&(real - &shifted) / &shifted).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn rocr(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(10);
    Ok((real / &real.shift(timeperiod as i64)).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn rocr100(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(10);
    Ok((real / &real.shift(timeperiod as i64) * 100.0).into_series())
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

fn calc_adx(high: &Float64Chunked, low: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0 || n < 2 * timeperiod {
        return Float64Chunked::full_null("adx".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("adx".into(), n);

    let alpha: f64 = 1.0 / timeperiod as f64;
    let mut adx: f64 = f64::NAN;
    let mut minus_dm: f64 = 0.0;
    let mut plus_dm: f64 = 0.0;
    let mut h_pre: f64 = f64::NAN;
    let mut l_pre: f64 = f64::NAN;

    for (i, (h, l)) in izip!(high.no_null_iter(), low.no_null_iter()).enumerate() {
        if i == 0 {
            builder.append_null();
            h_pre = h;
            l_pre = l;
            continue;
        }

        let down = l_pre - l;
        let up = h - h_pre;

        let minus_dm_val: f64;
        let plus_dm_val: f64;
        if up > down {
            minus_dm_val = 0.0;
            plus_dm_val = up.max(0.);
        } else if down > up {
            minus_dm_val = down.max(0.);
            plus_dm_val = 0.0;
        } else {
            minus_dm_val = 0.;
            plus_dm_val = 0.;
        }

        h_pre = h;
        l_pre = l;

        if i < timeperiod {
            minus_dm += minus_dm_val;
            plus_dm += plus_dm_val;
            builder.append_null();
        } else if i == timeperiod {
            minus_dm = alpha * (minus_dm + minus_dm_val);
            plus_dm = alpha * (plus_dm + plus_dm_val);
            adx += 100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm);
            builder.append_null();
        } else if i < 2 * timeperiod - 1 {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            adx += 100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm);
            builder.append_null();
        } else if i == 2 * timeperiod - 1 {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            adx = alpha * (100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm) + adx);
            builder.append_value(adx);
        } else {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            adx += alpha * (100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm) - adx);
            builder.append_value(adx);
        }
    }

    builder.finish()
}

fn calc_adxr(high: &Float64Chunked, low: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0 || n < 3 * timeperiod - 1 {
        return Float64Chunked::full_null("adx".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("adx".into(), n);

    let alpha: f64 = 1.0 / timeperiod as f64;
    let mut adx: f64 = f64::NAN;
    let mut adx_pre = AllocRingBuffer::new(timeperiod - 1);
    let mut minus_dm: f64 = 0.0;
    let mut plus_dm: f64 = 0.0;
    let mut h_pre: f64 = f64::NAN;
    let mut l_pre: f64 = f64::NAN;

    for (i, (h, l)) in izip!(high.no_null_iter(), low.no_null_iter()).enumerate() {
        if i == 0 {
            builder.append_null();
            h_pre = h;
            l_pre = l;
            continue;
        }

        let down = l_pre - l;
        let up = h - h_pre;

        let minus_dm_val: f64;
        let plus_dm_val: f64;
        if up > down {
            minus_dm_val = 0.0;
            plus_dm_val = up.max(0.);
        } else if down > up {
            minus_dm_val = down.max(0.);
            plus_dm_val = 0.0;
        } else {
            minus_dm_val = 0.;
            plus_dm_val = 0.;
        }

        h_pre = h;
        l_pre = l;

        if i < timeperiod {
            minus_dm += minus_dm_val;
            plus_dm += plus_dm_val;
            builder.append_null();
        } else if i == timeperiod {
            minus_dm = alpha * (minus_dm + minus_dm_val);
            plus_dm = alpha * (plus_dm + plus_dm_val);
            adx += 100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm);
            builder.append_null();
        } else if i < 2 * timeperiod - 1 {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            adx += 100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm);
            builder.append_null();
        } else if i == 2 * timeperiod - 1 {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            adx = alpha * (100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm) + adx);
            adx_pre.enqueue(adx);
            builder.append_null();
        } else if i < 3 * timeperiod - 1 {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            adx += alpha * (100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm) - adx);
            adx_pre.enqueue(adx);
            builder.append_null();
        } else {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            adx += alpha * (100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm) - adx);
            builder.append_value((adx - adx_pre.enqueue(adx).unwrap()) / 2.0);
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

    let mut down_builder = PrimitiveChunkedBuilder::<Float64Type>::new("aroon_down".into(), n);
    let mut up_builder = PrimitiveChunkedBuilder::<Float64Type>::new("aroon_up".into(), n);
    let inv_period = 100.0 / timeperiod as f64;
    let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod + 1);
    let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod + 1);

    for (i, (h, l)) in izip!(high.no_null_iter(), low.no_null_iter()).enumerate() {
        while let Some(&(_, v)) = high_max.back() {
            if v <= h {
                high_max.pop_back();
            } else {
                break;
            }
        }
        high_max.push_back((i, h));

        while let Some(&(_, v)) = low_min.back() {
            if v >= l {
                low_min.pop_back();
            } else {
                break;
            }
        }
        low_min.push_back((i, l));

        if i < timeperiod {
            up_builder.append_null();
            down_builder.append_null();
        } else {
            let window_start = i - timeperiod;
            while let Some(&(i0, _)) = high_max.front() {
                if i0 < window_start {
                    high_max.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(i0, _)) = low_min.front() {
                if i0 < window_start {
                    low_min.pop_front();
                } else {
                    break;
                }
            }

            let up = if let Some(&(i0, _)) = high_max.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };
            let down = if let Some(&(i0, _)) = low_min.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };

            up_builder.append_value(up);
            down_builder.append_value(down);
        }
    }

    (down_builder.finish(), up_builder.finish())
}

fn calc_aroonosc(high: &Float64Chunked, low: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = high.len();

    if timeperiod == 0 || n < timeperiod || high.null_count() + low.null_count() != 0 {
        return Float64Chunked::full_null("aroonosc".into(), n);
    }

    let high = high.rechunk();
    let low = low.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("aroonosc".into(), n);
    let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod + 1);
    let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod + 1);
    let inv_period = 100.0 / timeperiod as f64;

    for (i, (h, l)) in izip!(high.no_null_iter(), low.no_null_iter()).enumerate() {
        while let Some(&(_, v)) = high_max.back() {
            if v <= h {
                high_max.pop_back();
            } else {
                break;
            }
        }
        high_max.push_back((i, h));

        while let Some(&(_, v)) = low_min.back() {
            if v >= l {
                low_min.pop_back();
            } else {
                break;
            }
        }
        low_min.push_back((i, l));

        if i < timeperiod {
            builder.append_null();
        } else {
            let window_start = i - timeperiod;
            while let Some(&(i0, _)) = high_max.front() {
                if i0 < window_start {
                    high_max.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(i0, _)) = low_min.front() {
                if i0 < window_start {
                    low_min.pop_front();
                } else {
                    break;
                }
            }

            let up = if let Some(&(i0, _)) = high_max.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };
            let down = if let Some(&(i0, _)) = low_min.front() {
                (timeperiod as f64 - (i - i0) as f64) * inv_period
            } else {
                0.0
            };

            builder.append_value(up - down);
        }
    }

    builder.finish()
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

    if fastk_period == 0 || slowk_period == 0 || slowd_period == 0 || n < fastk_period {
        let nulls = Float64Chunked::full_null("stoch".into(), n);
        return (nulls.clone(), nulls);
    }

    if slowk_matype != 0 || slowd_matype != 0 {
        use crate::talib::overlap::calc_ma;

        if high.null_count() + low.null_count() + close.null_count() != 0 {
            let nulls = Float64Chunked::full_null("stoch".into(), n);
            return (nulls.clone(), nulls);
        }

        let high = high.rechunk();
        let low = low.rechunk();
        let close = close.rechunk();

        let mut fastk_builder = PrimitiveChunkedBuilder::<Float64Type>::new("fastk".into(), n);
        let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
        let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);

        for (i, (h, l, c)) in izip!(
            high.no_null_iter(),
            low.no_null_iter(),
            close.no_null_iter()
        )
        .enumerate()
        {
            while let Some(&(_, v)) = high_max.back() {
                if v <= h {
                    high_max.pop_back();
                } else {
                    break;
                }
            }
            high_max.push_back((i, h));

            while let Some(&(_, v)) = low_min.back() {
                if v >= l {
                    low_min.pop_back();
                } else {
                    break;
                }
            }
            low_min.push_back((i, l));

            if i < fastk_period - 1 {
                fastk_builder.append_null();
            } else {
                let expired = i + 1 - fastk_period;
                while let Some(&(idx, _)) = high_max.front() {
                    if idx < expired {
                        high_max.pop_front();
                    } else {
                        break;
                    }
                }
                while let Some(&(idx, _)) = low_min.front() {
                    if idx < expired {
                        low_min.pop_front();
                    } else {
                        break;
                    }
                }

                if let (Some(&(_, max_h)), Some(&(_, min_l))) = (high_max.front(), low_min.front())
                {
                    let denom = max_h - min_l;
                    if denom == 0.0 {
                        fastk_builder.append_value(0.0);
                    } else {
                        fastk_builder.append_value(100.0 * (c - min_l) / denom);
                    }
                } else {
                    fastk_builder.append_value(0.0);
                }
            }
        }

        let fastk = fastk_builder.finish();
        let slowk = calc_ma(&fastk, slowk_period, slowk_matype);
        let slowd = calc_ma(&slowk, slowd_period, slowd_matype);
        return (slowk, slowd);
    }

    if high.null_count() + low.null_count() + close.null_count() != 0 {
        let nulls = Float64Chunked::full_null("stoch".into(), n);
        return (nulls.clone(), nulls);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut slowk_builder = PrimitiveChunkedBuilder::<Float64Type>::new("slowk".into(), n);
    let mut slowd_builder = PrimitiveChunkedBuilder::<Float64Type>::new("slowd".into(), n);

    let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
    let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);

    let mut slowk_buf: VecDeque<f64> = VecDeque::with_capacity(slowk_period);
    let mut slowk_sum: f64 = 0.0;

    let mut slowd_buf: VecDeque<f64> = VecDeque::with_capacity(slowd_period);
    let mut slowd_sum: f64 = 0.0;

    for (i, (h, l, c)) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    )
    .enumerate()
    {
        while let Some(&(_, v)) = high_max.back() {
            if v <= h {
                high_max.pop_back();
            } else {
                break;
            }
        }
        high_max.push_back((i, h));

        while let Some(&(_, v)) = low_min.back() {
            if v >= l {
                low_min.pop_back();
            } else {
                break;
            }
        }
        low_min.push_back((i, l));

        let fastk_val: Option<f64> = if i < fastk_period - 1 {
            None
        } else {
            let expired = i + 1 - fastk_period;
            while let Some(&(idx, _)) = high_max.front() {
                if idx < expired {
                    high_max.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(idx, _)) = low_min.front() {
                if idx < expired {
                    low_min.pop_front();
                } else {
                    break;
                }
            }

            if let (Some(&(_, max_h)), Some(&(_, min_l))) = (high_max.front(), low_min.front()) {
                let denom = max_h - min_l;
                if denom == 0.0 {
                    Some(0.0)
                } else {
                    Some(100.0 * (c - min_l) / denom)
                }
            } else {
                Some(0.0)
            }
        };

        let slowk_val: Option<f64>;
        if let Some(fk) = fastk_val {
            slowk_sum += fk;
            slowk_buf.push_back(fk);
            if slowk_buf.len() > slowk_period {
                let old = slowk_buf.pop_front().unwrap();
                slowk_sum -= old;
            }
            if slowk_buf.len() == slowk_period {
                slowk_val = Some(slowk_sum / slowk_period as f64);
            } else {
                slowk_val = None;
            }
        } else {
            slowk_val = None;
        }

        let slowd_val: Option<f64>;
        if let Some(sk) = slowk_val {
            slowd_sum += sk;
            slowd_buf.push_back(sk);
            if slowd_buf.len() > slowd_period {
                let old = slowd_buf.pop_front().unwrap();
                slowd_sum -= old;
            }
            if slowd_buf.len() == slowd_period {
                slowd_val = Some(slowd_sum / slowd_period as f64);
            } else {
                slowd_val = None;
            }
        } else {
            slowd_val = None;
        }

        slowk_builder.append_option(slowk_val);
        slowd_builder.append_option(slowd_val);
    }

    (slowk_builder.finish(), slowd_builder.finish())
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

    if fastk_period == 0 || fastd_period == 0 || n < fastk_period {
        let nulls = Float64Chunked::full_null("stochf".into(), n);
        return (nulls.clone(), nulls);
    }

    if fastd_matype != 0 {
        use crate::talib::overlap::calc_ma;

        if high.null_count() + low.null_count() + close.null_count() != 0 {
            let nulls = Float64Chunked::full_null("stochf".into(), n);
            return (nulls.clone(), nulls);
        }

        let high = high.rechunk();
        let low = low.rechunk();
        let close = close.rechunk();

        let mut fastk_builder = PrimitiveChunkedBuilder::<Float64Type>::new("fastk".into(), n);
        let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
        let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);

        for (i, (h, l, c)) in izip!(
            high.no_null_iter(),
            low.no_null_iter(),
            close.no_null_iter()
        )
        .enumerate()
        {
            while let Some(&(_, v)) = high_max.back() {
                if v <= h {
                    high_max.pop_back();
                } else {
                    break;
                }
            }
            high_max.push_back((i, h));
            while let Some(&(_, v)) = low_min.back() {
                if v >= l {
                    low_min.pop_back();
                } else {
                    break;
                }
            }
            low_min.push_back((i, l));

            if i < fastk_period - 1 {
                fastk_builder.append_null();
            } else {
                let expired = i + 1 - fastk_period;
                while let Some(&(idx, _)) = high_max.front() {
                    if idx < expired {
                        high_max.pop_front();
                    } else {
                        break;
                    }
                }
                while let Some(&(idx, _)) = low_min.front() {
                    if idx < expired {
                        low_min.pop_front();
                    } else {
                        break;
                    }
                }
                if let (Some(&(_, max_h)), Some(&(_, min_l))) = (high_max.front(), low_min.front())
                {
                    let denom = max_h - min_l;
                    if denom == 0.0 {
                        fastk_builder.append_value(0.0);
                    } else {
                        fastk_builder.append_value(100.0 * (c - min_l) / denom);
                    }
                } else {
                    fastk_builder.append_value(0.0);
                }
            }
        }

        let fastk = fastk_builder.finish();
        let fastd = calc_ma(&fastk, fastd_period, fastd_matype);
        return (fastk, fastd);
    }

    if high.null_count() + low.null_count() + close.null_count() != 0 {
        let nulls = Float64Chunked::full_null("stochf".into(), n);
        return (nulls.clone(), nulls);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut fastk_builder = PrimitiveChunkedBuilder::<Float64Type>::new("fastk".into(), n);
    let mut fastd_builder = PrimitiveChunkedBuilder::<Float64Type>::new("fastd".into(), n);

    let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
    let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);

    let mut fastd_buf: VecDeque<f64> = VecDeque::with_capacity(fastd_period);
    let mut fastd_sum: f64 = 0.0;

    for (i, (h, l, c)) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    )
    .enumerate()
    {
        while let Some(&(_, v)) = high_max.back() {
            if v <= h {
                high_max.pop_back();
            } else {
                break;
            }
        }
        high_max.push_back((i, h));

        while let Some(&(_, v)) = low_min.back() {
            if v >= l {
                low_min.pop_back();
            } else {
                break;
            }
        }
        low_min.push_back((i, l));

        let fastk_val: Option<f64> = if i < fastk_period - 1 {
            None
        } else {
            let expired = i + 1 - fastk_period;
            while let Some(&(idx, _)) = high_max.front() {
                if idx < expired {
                    high_max.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(idx, _)) = low_min.front() {
                if idx < expired {
                    low_min.pop_front();
                } else {
                    break;
                }
            }

            if let (Some(&(_, max_h)), Some(&(_, min_l))) = (high_max.front(), low_min.front()) {
                let denom = max_h - min_l;
                if denom == 0.0 {
                    Some(0.0)
                } else {
                    Some(100.0 * (c - min_l) / denom)
                }
            } else {
                Some(0.0)
            }
        };

        fastk_builder.append_option(fastk_val);

        let fastd_val: Option<f64>;
        if let Some(fk) = fastk_val {
            fastd_sum += fk;
            fastd_buf.push_back(fk);
            if fastd_buf.len() > fastd_period {
                let old = fastd_buf.pop_front().unwrap();
                fastd_sum -= old;
            }
            if fastd_buf.len() == fastd_period {
                fastd_val = Some(fastd_sum / fastd_period as f64);
            } else {
                fastd_val = None;
            }
        } else {
            fastd_val = None;
        }

        fastd_builder.append_option(fastd_val);
    }

    (fastk_builder.finish(), fastd_builder.finish())
}

fn calc_stochrsi(
    real: &Float64Chunked,
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = real.len();

    if timeperiod == 0 || fastk_period == 0 || fastd_period == 0 || n < timeperiod + fastk_period {
        let nulls = Float64Chunked::full_null("stochrsi".into(), n);
        return (nulls.clone(), nulls);
    }

    if fastd_matype != 0 {
        use crate::talib::overlap::calc_ma;
        let rsi_vals = calc_rsi(real, timeperiod);

        if fastd_period == 0 || n < fastk_period {
            let nulls = Float64Chunked::full_null("stochrsi".into(), n);
            return (nulls.clone(), nulls);
        }

        let rsi_vals = rsi_vals.rechunk();

        let mut fastk_builder = PrimitiveChunkedBuilder::<Float64Type>::new("fastk".into(), n);
        let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
        let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);

        for (i, opt) in rsi_vals.iter().enumerate() {
            if let Some(v) = opt {
                while let Some(&(_, vv)) = high_max.back() {
                    if vv <= v {
                        high_max.pop_back();
                    } else {
                        break;
                    }
                }
                high_max.push_back((i, v));
                while let Some(&(_, vv)) = low_min.back() {
                    if vv >= v {
                        low_min.pop_back();
                    } else {
                        break;
                    }
                }
                low_min.push_back((i, v));
            }

            if i < fastk_period - 1 || opt.is_none() {
                fastk_builder.append_null();
            } else {
                let expired = i + 1 - fastk_period;
                while let Some(&(idx, _)) = high_max.front() {
                    if idx < expired {
                        high_max.pop_front();
                    } else {
                        break;
                    }
                }
                while let Some(&(idx, _)) = low_min.front() {
                    if idx < expired {
                        low_min.pop_front();
                    } else {
                        break;
                    }
                }
                if let (Some(&(_, max_v)), Some(&(_, min_v))) = (high_max.front(), low_min.front())
                {
                    let denom = max_v - min_v;
                    if denom == 0.0 {
                        fastk_builder.append_value(0.0);
                    } else {
                        fastk_builder.append_value(100.0 * (opt.unwrap() - min_v) / denom);
                    }
                } else {
                    fastk_builder.append_null();
                }
            }
        }

        let fastk = fastk_builder.finish();
        let fastd = calc_ma(&fastk, fastd_period, fastd_matype);
        return (fastk, fastd);
    }

    if real.null_count() != 0 {
        let nulls = Float64Chunked::full_null("stochrsi".into(), n);
        return (nulls.clone(), nulls);
    }

    let real = real.rechunk();

    let alpha: f64 = 1.0 / timeperiod as f64;
    let mut gain_sum: f64 = 0.0;
    let mut loss_sum: f64 = 0.0;
    let mut gain_rma: f64 = 0.0;
    let mut loss_rma: f64 = 0.0;
    let mut rma_count: usize = 0;

    let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);
    let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(fastk_period);

    let mut fastd_buf: VecDeque<f64> = VecDeque::with_capacity(fastd_period);
    let mut fastd_sum: f64 = 0.0;

    let mut fastk_builder = PrimitiveChunkedBuilder::<Float64Type>::new("fastk".into(), n);
    let mut fastd_builder = PrimitiveChunkedBuilder::<Float64Type>::new("fastd".into(), n);

    let mut prev: f64 = f64::NAN;

    for (i, cur) in real.no_null_iter().enumerate() {
        let rsi_val: Option<f64>;
        if i == 0 {
            rsi_val = None;
            prev = cur;
        } else {
            let diff = cur - prev;
            let g = if diff > 0.0 { diff } else { 0.0 };
            let l = if diff < 0.0 { -diff } else { 0.0 };
            rma_count += 1;

            if rma_count <= timeperiod {
                gain_sum += g;
                loss_sum += l;
                if rma_count == timeperiod {
                    gain_rma = gain_sum / timeperiod as f64;
                    loss_rma = loss_sum / timeperiod as f64;
                    rsi_val = if loss_rma == 0.0 {
                        Some(100.0)
                    } else {
                        Some(100.0 - 100.0 / (1.0 + gain_rma / loss_rma))
                    };
                } else {
                    rsi_val = None;
                }
            } else {
                gain_rma += alpha * (g - gain_rma);
                loss_rma += alpha * (l - loss_rma);
                rsi_val = if loss_rma == 0.0 {
                    Some(100.0)
                } else {
                    Some(100.0 - 100.0 / (1.0 + gain_rma / loss_rma))
                };
            }
            prev = cur;
        }

        if let Some(v) = rsi_val {
            while let Some(&(_, vv)) = high_max.back() {
                if vv <= v {
                    high_max.pop_back();
                } else {
                    break;
                }
            }
            high_max.push_back((i, v));
            while let Some(&(_, vv)) = low_min.back() {
                if vv >= v {
                    low_min.pop_back();
                } else {
                    break;
                }
            }
            low_min.push_back((i, v));
        }

        let fastk_val: Option<f64> = if i < timeperiod || i < fastk_period - 1 || rsi_val.is_none()
        {
            None
        } else {
            let expired = i + 1 - fastk_period;
            while let Some(&(idx, _)) = high_max.front() {
                if idx < expired {
                    high_max.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(idx, _)) = low_min.front() {
                if idx < expired {
                    low_min.pop_front();
                } else {
                    break;
                }
            }
            if let (Some(&(_, max_v)), Some(&(_, min_v))) = (high_max.front(), low_min.front()) {
                let denom = max_v - min_v;
                if denom == 0.0 {
                    Some(0.0)
                } else {
                    Some(100.0 * (rsi_val.unwrap() - min_v) / denom)
                }
            } else {
                None
            }
        };

        fastk_builder.append_option(fastk_val);

        let fastd_val: Option<f64>;
        if let Some(fk) = fastk_val {
            fastd_sum += fk;
            fastd_buf.push_back(fk);
            if fastd_buf.len() > fastd_period {
                let old = fastd_buf.pop_front().unwrap();
                fastd_sum -= old;
            }
            if fastd_buf.len() == fastd_period {
                fastd_val = Some(fastd_sum / fastd_period as f64);
            } else {
                fastd_val = None;
            }
        } else {
            fastd_val = None;
        }

        fastd_builder.append_option(fastd_val);
    }

    (fastk_builder.finish(), fastd_builder.finish())
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

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ultosc".into(), n);

    let mut bp1_buf: VecDeque<f64> = VecDeque::with_capacity(timeperiod1);
    let mut tr1_buf: VecDeque<f64> = VecDeque::with_capacity(timeperiod1);
    let mut bp2_buf: VecDeque<f64> = VecDeque::with_capacity(timeperiod2);
    let mut tr2_buf: VecDeque<f64> = VecDeque::with_capacity(timeperiod2);
    let mut bp3_buf: VecDeque<f64> = VecDeque::with_capacity(timeperiod3);
    let mut tr3_buf: VecDeque<f64> = VecDeque::with_capacity(timeperiod3);

    let mut bp1_sum: f64 = 0.0;
    let mut tr1_sum: f64 = 0.0;
    let mut bp2_sum: f64 = 0.0;
    let mut tr2_sum: f64 = 0.0;
    let mut bp3_sum: f64 = 0.0;
    let mut tr3_sum: f64 = 0.0;

    let mut c_pre: f64 = f64::NAN;

    for (i, (h, l, c)) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    )
    .enumerate()
    {
        let (bp, tr): (f64, f64);
        if i == 0 {
            bp = c - l;
            tr = h - l;
        } else {
            let min_lp = l.min(c_pre);
            let max_hp = h.max(c_pre);
            bp = c - min_lp;
            tr = max_hp - min_lp;
        }
        c_pre = c;

        bp1_sum += bp;
        bp1_buf.push_back(bp);
        let bp1_was_full = bp1_buf.len() > timeperiod1;
        if bp1_was_full {
            let old = bp1_buf.pop_front().unwrap();
            bp1_sum -= old;
        }
        tr1_sum += tr;
        tr1_buf.push_back(tr);
        let tr1_was_full = tr1_buf.len() > timeperiod1;
        if tr1_was_full {
            let old = tr1_buf.pop_front().unwrap();
            tr1_sum -= old;
        }
        bp2_sum += bp;
        bp2_buf.push_back(bp);
        let bp2_was_full = bp2_buf.len() > timeperiod2;
        if bp2_was_full {
            let old = bp2_buf.pop_front().unwrap();
            bp2_sum -= old;
        }
        tr2_sum += tr;
        tr2_buf.push_back(tr);
        let tr2_was_full = tr2_buf.len() > timeperiod2;
        if tr2_was_full {
            let old = tr2_buf.pop_front().unwrap();
            tr2_sum -= old;
        }
        bp3_sum += bp;
        bp3_buf.push_back(bp);
        let bp3_was_full = bp3_buf.len() > timeperiod3;
        if bp3_was_full {
            let old = bp3_buf.pop_front().unwrap();
            bp3_sum -= old;
        }
        tr3_sum += tr;
        tr3_buf.push_back(tr);
        let tr3_was_full = tr3_buf.len() > timeperiod3;
        if tr3_was_full {
            let old = tr3_buf.pop_front().unwrap();
            tr3_sum -= old;
        }

        if bp1_was_full
            && tr1_was_full
            && bp2_was_full
            && tr2_was_full
            && bp3_was_full
            && tr3_was_full
        {
            let sma_bp1 = bp1_sum / timeperiod1 as f64;
            let sma_tr1 = tr1_sum / timeperiod1 as f64;
            let sma_bp2 = bp2_sum / timeperiod2 as f64;
            let sma_tr2 = tr2_sum / timeperiod2 as f64;
            let sma_bp3 = bp3_sum / timeperiod3 as f64;
            let sma_tr3 = tr3_sum / timeperiod3 as f64;

            let avg1 = if sma_tr1 == 0.0 {
                0.0
            } else {
                sma_bp1 / sma_tr1
            };
            let avg2 = if sma_tr2 == 0.0 {
                0.0
            } else {
                sma_bp2 / sma_tr2
            };
            let avg3 = if sma_tr3 == 0.0 {
                0.0
            } else {
                sma_bp3 / sma_tr3
            };
            let osc = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;
            builder.append_value(osc);
        } else {
            builder.append_null();
        }
    }

    builder.finish()
}

fn calc_di(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> (Float64Chunked, Float64Chunked) {
    let n = high.len();

    if timeperiod == 0
        || n < timeperiod + 1
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        (
            Float64Chunked::full_null("minus_di".into(), n),
            Float64Chunked::full_null("plus_di".into(), n),
        );
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut minus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("minus_di".into(), n);
    let mut plus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("plus_di".into(), n);

    let alpha: f64 = 1.0 / timeperiod as f64;
    let mut minus_dm: f64 = 0.0;
    let mut plus_dm: f64 = 0.0;
    let mut c_pre: f64 = f64::NAN;
    let mut h_pre: f64 = f64::NAN;
    let mut l_pre: f64 = f64::NAN;
    let mut tr: f64 = 0.0;

    for (i, (c, h, l)) in izip!(
        close.no_null_iter(),
        high.no_null_iter(),
        low.no_null_iter()
    )
    .enumerate()
    {
        if i == 0 {
            minus_builder.append_null();
            plus_builder.append_null();
            c_pre = c;
            h_pre = h;
            l_pre = l;
            continue;
        }

        let down = l_pre - l;
        let up = h - h_pre;

        let minus_dm_val: f64;
        let plus_dm_val: f64;
        if up > down {
            minus_dm_val = 0.0;
            plus_dm_val = up.max(0.);
        } else if down > up {
            minus_dm_val = down.max(0.);
            plus_dm_val = 0.0;
        } else {
            minus_dm_val = 0.;
            plus_dm_val = 0.;
        }

        let tr_val = if c_pre > h {
            c_pre - l
        } else if c_pre < l {
            h - c_pre
        } else {
            h - l
        };

        c_pre = c;
        h_pre = h;
        l_pre = l;

        if i < timeperiod {
            minus_dm += minus_dm_val;
            plus_dm += plus_dm_val;
            tr += tr_val;
            minus_builder.append_null();
            plus_builder.append_null();
        } else if i == timeperiod {
            minus_dm = alpha * (minus_dm + minus_dm_val);
            plus_dm = alpha * (plus_dm + plus_dm_val);
            tr = alpha * (tr + tr_val);

            if tr == 0.0 {
                minus_builder.append_value(0.0);
                plus_builder.append_value(0.0);
            } else {
                minus_builder.append_value(100.0 * minus_dm / tr);
                plus_builder.append_value(100.0 * plus_dm / tr);
            }
        } else {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            tr += alpha * (tr_val - tr);

            if tr == 0.0 {
                minus_builder.append_value(0.0);
                plus_builder.append_value(0.0);
            } else {
                minus_builder.append_value(100.0 * minus_dm / tr);
                plus_builder.append_value(100.0 * plus_dm / tr);
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

    if timeperiod == 0 || n < timeperiod + 1 || high.null_count() + low.null_count() != 0 {
        (
            Float64Chunked::full_null("minus_dm".into(), n),
            Float64Chunked::full_null("plus_dm".into(), n),
        );
    }

    let high = high.rechunk();
    let low = low.rechunk();

    let mut minus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("minus_di".into(), n);
    let mut plus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("plus_di".into(), n);

    let alpha: f64 = 1.0 / timeperiod as f64;
    let mut minus_dm: f64 = 0.;
    let mut plus_dm: f64 = 0.;
    let mut h_pre: f64 = f64::NAN;
    let mut l_pre: f64 = f64::NAN;

    for (i, (h, l)) in izip!(high.no_null_iter(), low.no_null_iter()).enumerate() {
        if i == 0 {
            minus_builder.append_null();
            plus_builder.append_null();
            h_pre = h;
            l_pre = l;
            continue;
        }

        let down = l_pre - l;
        let up = h - h_pre;

        let minus_dm_val: f64;
        let plus_dm_val: f64;
        if up > down {
            minus_dm_val = 0.0;
            plus_dm_val = up.max(0.);
        } else if down > up {
            minus_dm_val = down.max(0.);
            plus_dm_val = 0.0;
        } else {
            minus_dm_val = 0.;
            plus_dm_val = 0.;
        }
        h_pre = h;
        l_pre = l;

        if i < timeperiod {
            minus_dm += minus_dm_val;
            plus_dm += plus_dm_val;
            minus_builder.append_null();
            plus_builder.append_null();
        } else if i == timeperiod {
            minus_dm += minus_dm_val;
            plus_dm += plus_dm_val;
            minus_dm *= alpha;
            plus_dm *= alpha;
        } else {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
        }
    }

    (plus_builder.finish(), minus_builder.finish())
}

fn calc_dx(high: &Float64Chunked, low: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = high.len();
    if timeperiod == 0 || n < timeperiod + 1 {
        return Float64Chunked::full_null("dx".into(), n);
    }
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("dx".into(), n);

    let alpha: f64 = 1.0 / timeperiod as f64;
    let mut minus_dm: f64 = 0.0;
    let mut plus_dm: f64 = 0.0;
    let mut h_pre: f64 = f64::NAN;
    let mut l_pre: f64 = f64::NAN;

    for (i, (h, l)) in izip!(high.no_null_iter(), low.no_null_iter()).enumerate() {
        if i == 0 {
            builder.append_null();
            h_pre = h;
            l_pre = l;
            continue;
        }

        let down = l_pre - l;
        let up = h - h_pre;

        let minus_dm_val: f64;
        let plus_dm_val: f64;
        if up > down {
            minus_dm_val = 0.0;
            plus_dm_val = up.max(0.);
        } else if down > up {
            minus_dm_val = down.max(0.);
            plus_dm_val = 0.0;
        } else {
            minus_dm_val = 0.;
            plus_dm_val = 0.;
        }

        h_pre = h;
        l_pre = l;

        if i < timeperiod {
            minus_dm += minus_dm_val;
            plus_dm += plus_dm_val;
            builder.append_null();
        } else if i == timeperiod {
            minus_dm = alpha * (minus_dm + minus_dm_val);
            plus_dm = alpha * (plus_dm + plus_dm_val);
            builder.append_value(100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm));
        } else {
            minus_dm += alpha * (minus_dm_val - minus_dm);
            plus_dm += alpha * (plus_dm_val - plus_dm);
            builder.append_value(100. * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm));
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

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();
    let volume = volume.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("mfi".into(), n);

    let mut pre_tp: f64 = f64::NAN;
    let mut neg_buf = AllocRingBuffer::new(timeperiod);
    let mut pos_buf = AllocRingBuffer::new(timeperiod);
    let mut neg_sum = 0.0;
    let mut pos_sum: f64 = 0.0;

    for (i, (h, l, c, v)) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter(),
        volume.no_null_iter()
    )
    .enumerate()
    {
        let tp = (h + l + c) / 3.0;
        let mf = tp * v;

        if i == 0 {
            pre_tp = tp;
            builder.append_null();
            continue;
        }

        if i < timeperiod {
            if tp > pre_tp {
                pos_sum += mf;
                neg_buf.enqueue(0.);
                pos_buf.enqueue(mf);
            } else if tp < pre_tp {
                neg_sum += mf;
                neg_buf.enqueue(mf);
                pos_buf.enqueue(0.);
            } else {
                neg_buf.enqueue(0.);
                pos_buf.enqueue(0.);
            }
            builder.append_null();
        } else if i == timeperiod {
            if tp > pre_tp {
                pos_sum += mf;
                neg_buf.enqueue(0.);
                pos_buf.enqueue(mf);
            } else if tp < pre_tp {
                neg_sum += mf;
                neg_buf.enqueue(mf);
                pos_buf.enqueue(0.);
            } else {
                neg_buf.enqueue(0.);
                pos_buf.enqueue(0.);
            }

            if neg_sum != 0. {
                builder.append_value(100. * pos_sum / (pos_sum + neg_sum));
            } else {
                builder.append_value(50.);
            }
        } else {
            if tp > pre_tp {
                pos_sum += mf - pos_buf.enqueue(mf).unwrap();
                neg_buf.enqueue(0.);
            } else if tp < pre_tp {
                neg_sum += mf - neg_buf.enqueue(mf).unwrap();
                pos_buf.enqueue(0.);
            } else {
                neg_buf.enqueue(0.);
                pos_buf.enqueue(0.);
            }

            if neg_sum != 0. {
                builder.append_value(100. * pos_sum / (pos_sum + neg_sum));
            } else {
                builder.append_value(50.);
            }
        }

        pre_tp = tp;
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
            builder.append_value(if loss == 0.0 {
                100.0
            } else {
                100.0 * gain / (gain + loss)
            });
        } else {
            gain += alpha * (g - gain);
            loss += alpha * (l - loss);
            builder.append_value(if loss == 0.0 {
                100.0
            } else {
                100.0 * gain / (gain + loss)
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

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("willr".into(), n);
    let mut window_high: AllocRingBuffer<f64> = AllocRingBuffer::new(timeperiod);
    let mut window_low: AllocRingBuffer<f64> = AllocRingBuffer::new(timeperiod);
    let mut window_max: VecDeque<f64> = VecDeque::with_capacity(timeperiod);
    let mut window_min: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    for (i, (h, l, c)) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    )
    .enumerate()
    {
        window_high.enqueue(h);
        window_low.enqueue(l);

        while let Some(last) = window_max.back() {
            if &h >= last {
                window_max.pop_back();
            } else {
                break;
            }
        }
        window_max.push_back(h);

        while let Some(last) = window_min.back() {
            if &l <= last {
                window_min.pop_back();
            } else {
                break;
            }
        }
        window_min.push_back(l);

        if i < timeperiod - 1 {
            builder.append_null();
        } else {
            let max = window_max.front().unwrap();
            let min = window_min.front().unwrap();
            builder.append_value(-100. * (max - c) / (max - min));

            if i >= timeperiod {
                if let Some(pop) = window_high.dequeue() {
                    window_max.pop_front_if(|&mut front| pop == front);
                };
                if let Some(pop) = window_low.dequeue() {
                    window_min.pop_front_if(|&mut front| pop == front);
                };
            }
        }
    }

    builder.finish()
}
