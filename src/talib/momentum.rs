use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_arrow::array::Array;
use serde::Deserialize;
use std::collections::VecDeque;

use crate::talib::overlap::{calc_ema, calc_ma, calc_rma, calc_sma};
use crate::talib::volatility::calc_trange;

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
    let fastperiod = kwargs.fastperiod.unwrap_or(12);
    let slowperiod = kwargs.slowperiod.unwrap_or(26);
    let matype = kwargs.matype.unwrap_or(0);
    let fast = calc_ma(real, fastperiod, matype);
    let slow = calc_ma(real, slowperiod, matype);
    Ok((&fast - &slow).into_series())
}

#[polars_expr(output_type_func=aroon_output)]
pub fn aroon(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let n = high.len();

    if timeperiod == 0 || n < timeperiod {
        let up = Float64Chunked::full_null("aroon_up".into(), n);
        let down = Float64Chunked::full_null("aroon_down".into(), n);
        let s1 = up.into_series();
        let s2 = down.into_series();
        return Ok(StructChunked::from_series("aroon".into(), n, [s1, s2].iter())?.into_series());
    }

    let mut up_builder = PrimitiveChunkedBuilder::<Float64Type>::new("aroon_up".into(), n);
    let mut down_builder = PrimitiveChunkedBuilder::<Float64Type>::new("aroon_down".into(), n);
    let inv_period = 100.0 / timeperiod as f64;
    let mut idx: usize = 0;
    let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod + 1);
    let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod + 1);

    for (h_arr, l_arr) in high.downcast_iter().zip(low.downcast_iter()) {
        let fast = h_arr.null_count() == 0 && l_arr.null_count() == 0;
        if fast {
            for (&h, &l) in h_arr.values_iter().zip(l_arr.values_iter()) {
                while let Some(&(_, v)) = high_max.back() {
                    if v <= h {
                        high_max.pop_back();
                    } else {
                        break;
                    }
                }
                high_max.push_back((idx, h));

                while let Some(&(_, v)) = low_min.back() {
                    if v >= l {
                        low_min.pop_back();
                    } else {
                        break;
                    }
                }
                low_min.push_back((idx, l));

                if idx < timeperiod {
                    up_builder.append_null();
                    down_builder.append_null();
                } else {
                    let window_start = idx - timeperiod;
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
                        let days_since = idx - i0;
                        (timeperiod as f64 - days_since as f64) * inv_period
                    } else {
                        0.0
                    };
                    let down = if let Some(&(i0, _)) = low_min.front() {
                        let days_since = idx - i0;
                        (timeperiod as f64 - days_since as f64) * inv_period
                    } else {
                        0.0
                    };

                    up_builder.append_value(up);
                    down_builder.append_value(down);
                }
                idx += 1;
            }
        } else {
            for (h_opt, l_opt) in h_arr.iter().zip(l_arr.iter()) {
                if let Some(&h) = h_opt {
                    while let Some(&(_, v)) = high_max.back() {
                        if v <= h {
                            high_max.pop_back();
                        } else {
                            break;
                        }
                    }
                    high_max.push_back((idx, h));
                }

                if let Some(&l) = l_opt {
                    while let Some(&(_, v)) = low_min.back() {
                        if v >= l {
                            low_min.pop_back();
                        } else {
                            break;
                        }
                    }
                    low_min.push_back((idx, l));
                }

                if idx < timeperiod {
                    up_builder.append_null();
                    down_builder.append_null();
                } else {
                    let window_start = idx - timeperiod;
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
                        let days_since = idx - i0;
                        (timeperiod as f64 - days_since as f64) * inv_period
                    } else {
                        0.0
                    };
                    let down = if let Some(&(i0, _)) = low_min.front() {
                        let days_since = idx - i0;
                        (timeperiod as f64 - days_since as f64) * inv_period
                    } else {
                        0.0
                    };

                    up_builder.append_value(up);
                    down_builder.append_value(down);
                }
                idx += 1;
            }
        }
    }

    let s1 = up_builder.finish().into_series();
    let s2 = down_builder.finish().into_series();
    Ok(StructChunked::from_series("aroon".into(), n, [s2, s1].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn aroonosc(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let high = high.f64()?;
    let low = low.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let n = high.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("aroonosc".into(), n);

    if timeperiod == 0 || n < timeperiod {
        for _ in 0..n {
            builder.append_null();
        }
        return Ok(builder.finish().into_series());
    }

    let window = timeperiod + 1;
    let mut high_buf: VecDeque<Option<f64>> = VecDeque::with_capacity(window);
    let mut low_buf: VecDeque<Option<f64>> = VecDeque::with_capacity(window);
    let mut idx: usize = 0;

    for (h_arr, l_arr) in high.downcast_iter().zip(low.downcast_iter()) {
        let fast = h_arr.null_count() == 0 && l_arr.null_count() == 0;
        if fast {
            for (&h, &l) in h_arr.values_iter().zip(l_arr.values_iter()) {
                high_buf.push_back(Some(h));
                low_buf.push_back(Some(l));
                if high_buf.len() > window {
                    high_buf.pop_front();
                    low_buf.pop_front();
                }
                if idx < timeperiod {
                    builder.append_null();
                } else {
                    let mut max_idx = 0usize;
                    let mut max_val = f64::NEG_INFINITY;
                    for (j, opt) in high_buf.iter().enumerate() {
                        if let Some(v) = opt {
                            if *v >= max_val {
                                max_val = *v;
                                max_idx = j;
                            }
                        }
                    }
                    let mut min_idx = 0usize;
                    let mut min_val = f64::INFINITY;
                    for (j, opt) in low_buf.iter().enumerate() {
                        if let Some(v) = opt {
                            if *v <= min_val {
                                min_val = *v;
                                min_idx = j;
                            }
                        }
                    }
                    let up = max_idx as f64 / timeperiod as f64 * 100.0;
                    let down = min_idx as f64 / timeperiod as f64 * 100.0;
                    builder.append_value(up - down);
                }
                idx += 1;
            }
        } else {
            for (h_opt, l_opt) in h_arr.iter().zip(l_arr.iter()) {
                high_buf.push_back(h_opt.copied());
                low_buf.push_back(l_opt.copied());
                if high_buf.len() > window {
                    high_buf.pop_front();
                    low_buf.pop_front();
                }
                if idx < timeperiod {
                    builder.append_null();
                } else {
                    let mut max_idx = 0usize;
                    let mut max_val = f64::NEG_INFINITY;
                    for (j, opt) in high_buf.iter().enumerate() {
                        if let Some(v) = opt {
                            if *v >= max_val {
                                max_val = *v;
                                max_idx = j;
                            }
                        }
                    }
                    let mut min_idx = 0usize;
                    let mut min_val = f64::INFINITY;
                    for (j, opt) in low_buf.iter().enumerate() {
                        if let Some(v) = opt {
                            if *v <= min_val {
                                min_val = *v;
                                min_idx = j;
                            }
                        }
                    }
                    let up = max_idx as f64 / timeperiod as f64 * 100.0;
                    let down = min_idx as f64 / timeperiod as f64 * 100.0;
                    builder.append_value(up - down);
                }
                idx += 1;
            }
        }
    }

    Ok(builder.finish().into_series())
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

    let n = open.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("bop".into(), n);

    for (((o_arr, h_arr), l_arr), c_arr) in open
        .downcast_iter()
        .zip(high.downcast_iter())
        .zip(low.downcast_iter())
        .zip(close.downcast_iter())
    {
        let fast = o_arr.null_count() == 0
            && h_arr.null_count() == 0
            && l_arr.null_count() == 0
            && c_arr.null_count() == 0;
        if fast {
            for (((&o, &h), &l), &c) in o_arr
                .values_iter()
                .zip(h_arr.values_iter())
                .zip(l_arr.values_iter())
                .zip(c_arr.values_iter())
            {
                if h == l {
                    builder.append_value(0.0);
                } else {
                    builder.append_value((c - o) / (h - l));
                }
            }
        } else {
            for (((o_opt, h_opt), l_opt), c_opt) in o_arr
                .iter()
                .zip(h_arr.iter())
                .zip(l_arr.iter())
                .zip(c_arr.iter())
            {
                match (o_opt, h_opt, l_opt, c_opt) {
                    (Some(&o), Some(&h), Some(&l), Some(&c)) => {
                        if h == l {
                            builder.append_value(0.0);
                        } else {
                            builder.append_value((c - o) / (h - l));
                        }
                    }
                    _ => builder.append_null(),
                }
            }
        }
    }

    Ok(builder.finish().into_series())
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
    let n = high.len();

    if timeperiod == 0 || n < timeperiod {
        return Ok(Float64Chunked::full_null("cci".into(), n).into_series());
    }

    let mut tp_builder = PrimitiveChunkedBuilder::<Float64Type>::new("tp".into(), n);
    for ((h_arr, l_arr), c_arr) in high
        .downcast_iter()
        .zip(low.downcast_iter())
        .zip(close.downcast_iter())
    {
        let fast = h_arr.null_count() == 0 && l_arr.null_count() == 0 && c_arr.null_count() == 0;
        if fast {
            for ((&h, &l), &c) in h_arr
                .values_iter()
                .zip(l_arr.values_iter())
                .zip(c_arr.values_iter())
            {
                tp_builder.append_value((h + l + c) / 3.0);
            }
        } else {
            for ((h_opt, l_opt), c_opt) in h_arr.iter().zip(l_arr.iter()).zip(c_arr.iter()) {
                match (h_opt, l_opt, c_opt) {
                    (Some(&h), Some(&l), Some(&c)) => tp_builder.append_value((h + l + c) / 3.0),
                    _ => tp_builder.append_null(),
                }
            }
        }
    }
    let tp = tp_builder.finish();

    let sma_tp = calc_sma(&tp, timeperiod);
    let mut cci_builder = PrimitiveChunkedBuilder::<Float64Type>::new("cci".into(), n);
    let mut tp_window: VecDeque<f64> = VecDeque::with_capacity(timeperiod);
    let mut i: usize = 0;

    for (tp_arr, sma_arr) in tp.downcast_iter().zip(sma_tp.downcast_iter()) {
        let fast = tp_arr.null_count() == 0 && sma_arr.null_count() == 0;
        if fast {
            for (&tp_val, &s_val) in tp_arr.values_iter().zip(sma_arr.values_iter()) {
                tp_window.push_back(tp_val);
                if i >= timeperiod {
                    tp_window.pop_front();
                }
                if i >= timeperiod - 1 {
                    let mut dev_sum = 0.0f64;
                    for &v in tp_window.iter() {
                        dev_sum += (v - s_val).abs();
                    }
                    let mean_dev = dev_sum / timeperiod as f64;
                    if mean_dev == 0.0 {
                        cci_builder.append_value(0.0);
                    } else {
                        cci_builder.append_value((tp_val - s_val) / (0.015 * mean_dev));
                    }
                } else {
                    cci_builder.append_null();
                }
                i += 1;
            }
        } else {
            for (tp_opt, s_opt) in tp_arr.iter().zip(sma_arr.iter()) {
                let tp_val = tp_opt.copied().unwrap_or(0.0);
                tp_window.push_back(tp_val);
                if i >= timeperiod {
                    tp_window.pop_front();
                }
                if i >= timeperiod - 1 {
                    if let Some(s_val) = s_opt {
                        let mut dev_sum = 0.0f64;
                        for &v in tp_window.iter() {
                            dev_sum += (v - s_val).abs();
                        }
                        let mean_dev = dev_sum / timeperiod as f64;
                        if mean_dev == 0.0 {
                            cci_builder.append_value(0.0);
                        } else {
                            cci_builder.append_value((tp_val - s_val) / (0.015 * mean_dev));
                        }
                    } else {
                        cci_builder.append_null();
                    }
                } else {
                    cci_builder.append_null();
                }
                i += 1;
            }
        }
    }

    Ok(cci_builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn cmo(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let real = real.rechunk();
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let n = real.len();

    if timeperiod == 0 || n < timeperiod {
        return Ok(Float64Chunked::full_null("cmo".into(), n).into_series());
    }

    let prev = real.shift(1i64);
    let prev = prev.rechunk();
    let mut gain_b = PrimitiveChunkedBuilder::<Float64Type>::new("gain".into(), n);
    let mut loss_b = PrimitiveChunkedBuilder::<Float64Type>::new("loss".into(), n);

    for (cur_arr, prev_arr) in real.downcast_iter().zip(prev.downcast_iter()) {
        let fast = cur_arr.null_count() == 0 && prev_arr.null_count() == 0;
        if fast {
            for (&cur, &prev) in cur_arr.values_iter().zip(prev_arr.values_iter()) {
                let diff = cur - prev;
                if diff > 0.0 {
                    gain_b.append_value(diff);
                    loss_b.append_value(0.0);
                } else {
                    gain_b.append_value(0.0);
                    loss_b.append_value(-diff);
                }
            }
        } else {
            for (cur_opt, prev_opt) in cur_arr.iter().zip(prev_arr.iter()) {
                match (cur_opt, prev_opt) {
                    (Some(&cur), Some(&prev)) => {
                        let diff = cur - prev;
                        if diff > 0.0 {
                            gain_b.append_value(diff);
                            loss_b.append_value(0.0);
                        } else {
                            gain_b.append_value(0.0);
                            loss_b.append_value(-diff);
                        }
                    }
                    _ => {
                        gain_b.append_null();
                        loss_b.append_null();
                    }
                }
            }
        }
    }

    let smooth_gain = calc_rma(&gain_b.finish(), timeperiod);
    let smooth_loss = calc_rma(&loss_b.finish(), timeperiod);

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("cmo".into(), n);
    for (g_arr, l_arr) in smooth_gain.downcast_iter().zip(smooth_loss.downcast_iter()) {
        let fast = g_arr.null_count() == 0 && l_arr.null_count() == 0;
        if fast {
            for (&g, &l) in g_arr.values_iter().zip(l_arr.values_iter()) {
                let denom = g + l;
                if denom == 0.0 {
                    builder.append_value(0.0);
                } else {
                    builder.append_value(100.0 * (g - l) / denom);
                }
            }
        } else {
            for (g_opt, l_opt) in g_arr.iter().zip(l_arr.iter()) {
                match (g_opt, l_opt) {
                    (Some(&g), Some(&l)) => {
                        let denom = g + l;
                        if denom == 0.0 {
                            builder.append_value(0.0);
                        } else {
                            builder.append_value(100.0 * (g - l) / denom);
                        }
                    }
                    _ => builder.append_null(),
                }
            }
        }
    }

    Ok(builder.finish().into_series())
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
    let n = high.len();

    if timeperiod == 0 || n < timeperiod {
        return Ok(Float64Chunked::full_null("dx".into(), n).into_series());
    }

    let (plus_di, minus_di) = calc_di(high, low, close, timeperiod);

    let mut dx_builder = PrimitiveChunkedBuilder::<Float64Type>::new("dx".into(), n);
    for (pdi_arr, mdi_arr) in plus_di.downcast_iter().zip(minus_di.downcast_iter()) {
        let fast = pdi_arr.null_count() == 0 && mdi_arr.null_count() == 0;
        if fast {
            for (&pdi, &mdi) in pdi_arr.values_iter().zip(mdi_arr.values_iter()) {
                let denom = pdi + mdi;
                if denom == 0.0 {
                    dx_builder.append_value(0.0);
                } else {
                    dx_builder.append_value(100.0 * (pdi - mdi).abs() / denom);
                }
            }
        } else {
            for (pdi_opt, mdi_opt) in pdi_arr.iter().zip(mdi_arr.iter()) {
                match (pdi_opt, mdi_opt) {
                    (Some(&pdi), Some(&mdi)) => {
                        let denom = pdi + mdi;
                        if denom == 0.0 {
                            dx_builder.append_value(0.0);
                        } else {
                            dx_builder.append_value(100.0 * (pdi - mdi).abs() / denom);
                        }
                    }
                    _ => dx_builder.append_null(),
                }
            }
        }
    }

    Ok(dx_builder.finish().into_series())
}

#[polars_expr(output_type_func=macd_output)]
pub fn macd(inputs: &[Series], kwargs: MacdKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;

    let fast = kwargs.fastperiod.unwrap_or(12);
    let slow = kwargs.slowperiod.unwrap_or(26);
    let signal = kwargs.signalperiod.unwrap_or(9);

    let n = real.len();

    let ema_fast = calc_ema(real, fast);
    let ema_slow = calc_ema(real, slow);
    let macd_line = &ema_fast - &ema_slow;
    let signal_line = calc_ema(&macd_line, signal);
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
    let n = high.len();

    if timeperiod == 0 || n < timeperiod {
        return Ok(Float64Chunked::full_null("mfi".into(), n).into_series());
    }

    let mut tp_builder = PrimitiveChunkedBuilder::<Float64Type>::new("tp".into(), n);
    for ((h_arr, l_arr), c_arr) in high
        .downcast_iter()
        .zip(low.downcast_iter())
        .zip(close.downcast_iter())
    {
        let fast = h_arr.null_count() == 0 && l_arr.null_count() == 0 && c_arr.null_count() == 0;
        if fast {
            for ((&h, &l), &c) in h_arr
                .values_iter()
                .zip(l_arr.values_iter())
                .zip(c_arr.values_iter())
            {
                tp_builder.append_value((h + l + c) / 3.0);
            }
        } else {
            for ((h_opt, l_opt), c_opt) in h_arr.iter().zip(l_arr.iter()).zip(c_arr.iter()) {
                match (h_opt, l_opt, c_opt) {
                    (Some(&h), Some(&l), Some(&c)) => tp_builder.append_value((h + l + c) / 3.0),
                    _ => tp_builder.append_null(),
                }
            }
        }
    }
    let tp = tp_builder.finish();
    let tp = tp.rechunk();
    let prev_tp = tp.shift(1i64);
    let prev_tp = prev_tp.rechunk();

    let mut pos_mf_builder = PrimitiveChunkedBuilder::<Float64Type>::new("pos_mf".into(), n);
    let mut neg_mf_builder = PrimitiveChunkedBuilder::<Float64Type>::new("neg_mf".into(), n);

    for ((tp_arr, prev_arr), vol_arr) in tp
        .downcast_iter()
        .zip(prev_tp.downcast_iter())
        .zip(volume.downcast_iter())
    {
        let fast =
            tp_arr.null_count() == 0 && prev_arr.null_count() == 0 && vol_arr.null_count() == 0;
        if fast {
            for ((&tp_val, &prev), &vol) in tp_arr
                .values_iter()
                .zip(prev_arr.values_iter())
                .zip(vol_arr.values_iter())
            {
                let mf = tp_val * vol;
                if tp_val > prev {
                    pos_mf_builder.append_value(mf);
                    neg_mf_builder.append_value(0.0);
                } else if tp_val < prev {
                    pos_mf_builder.append_value(0.0);
                    neg_mf_builder.append_value(mf);
                } else {
                    pos_mf_builder.append_value(0.0);
                    neg_mf_builder.append_value(0.0);
                }
            }
        } else {
            for ((tp_opt, prev_opt), vol_opt) in
                tp_arr.iter().zip(prev_arr.iter()).zip(vol_arr.iter())
            {
                match (tp_opt, prev_opt, vol_opt) {
                    (Some(&tp_val), Some(&prev), Some(&vol)) => {
                        let mf = tp_val * vol;
                        if tp_val > prev {
                            pos_mf_builder.append_value(mf);
                            neg_mf_builder.append_value(0.0);
                        } else if tp_val < prev {
                            pos_mf_builder.append_value(0.0);
                            neg_mf_builder.append_value(mf);
                        } else {
                            pos_mf_builder.append_value(0.0);
                            neg_mf_builder.append_value(0.0);
                        }
                    }
                    _ => {
                        pos_mf_builder.append_null();
                        neg_mf_builder.append_null();
                    }
                }
            }
        }
    }

    let pos_mf = pos_mf_builder.finish();
    let neg_mf = neg_mf_builder.finish();

    let sma_pos = calc_sma(&pos_mf, timeperiod);
    let sma_neg = calc_sma(&neg_mf, timeperiod);

    let mut mfi_builder = PrimitiveChunkedBuilder::<Float64Type>::new("mfi".into(), n);
    for (p_arr, n_arr) in sma_pos.downcast_iter().zip(sma_neg.downcast_iter()) {
        let fast = p_arr.null_count() == 0 && n_arr.null_count() == 0;
        if fast {
            for (&p, &nval) in p_arr.values_iter().zip(n_arr.values_iter()) {
                if nval == 0.0 {
                    if p == 0.0 {
                        mfi_builder.append_value(50.0);
                    } else {
                        mfi_builder.append_value(100.0);
                    }
                } else {
                    let mr = p / nval;
                    mfi_builder.append_value(100.0 - 100.0 / (1.0 + mr));
                }
            }
        } else {
            for (p_opt, n_opt) in p_arr.iter().zip(n_arr.iter()) {
                match (p_opt, n_opt) {
                    (Some(&p), Some(&nval)) => {
                        if nval == 0.0 {
                            if p == 0.0 {
                                mfi_builder.append_value(50.0);
                            } else {
                                mfi_builder.append_value(100.0);
                            }
                        } else {
                            let mr = p / nval;
                            mfi_builder.append_value(100.0 - 100.0 / (1.0 + mr));
                        }
                    }
                    _ => mfi_builder.append_null(),
                }
            }
        }
    }

    Ok(mfi_builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn minus_di(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    let (_, minus_di) = calc_di(high, low, close, timeperiod);
    Ok(minus_di.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn minus_dm(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);
    let (_, minus_dm) = calc_dm(high, low);
    Ok((calc_rma(&minus_dm, timeperiod) * timeperiod as f64).into_series())
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
    let (plus_dm, _) = calc_dm(high, low);
    Ok((calc_rma(&plus_dm, timeperiod) * timeperiod as f64).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn ppo(inputs: &[Series], kwargs: ApoKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let fastperiod = kwargs.fastperiod.unwrap_or(12);
    let slowperiod = kwargs.slowperiod.unwrap_or(26);
    let matype = kwargs.matype.unwrap_or(0);
    let fast = calc_ma(real, fastperiod, matype);
    let slow = calc_ma(real, slowperiod, matype);
    let diff = &fast - &slow;
    Ok((&diff / &slow * 100.0).into_series())
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
    let real = real.rechunk();
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let n = real.len();

    if timeperiod == 0 || n < timeperiod {
        return Ok(Float64Chunked::full_null("rsi".into(), n).into_series());
    }

    let prev = real.shift(1i64);
    let prev = prev.rechunk();

    let mut gain_builder = PrimitiveChunkedBuilder::<Float64Type>::new("gain".into(), n);
    let mut loss_builder = PrimitiveChunkedBuilder::<Float64Type>::new("loss".into(), n);

    for (cur_arr, prev_arr) in real.downcast_iter().zip(prev.downcast_iter()) {
        let fast = cur_arr.null_count() == 0 && prev_arr.null_count() == 0;
        if fast {
            for (&cur, &prev) in cur_arr.values_iter().zip(prev_arr.values_iter()) {
                let diff = cur - prev;
                if diff > 0.0 {
                    gain_builder.append_value(diff);
                    loss_builder.append_value(0.0);
                } else {
                    gain_builder.append_value(0.0);
                    loss_builder.append_value(-diff);
                }
            }
        } else {
            for (cur_opt, prev_opt) in cur_arr.iter().zip(prev_arr.iter()) {
                match (cur_opt, prev_opt) {
                    (Some(&cur), Some(&prev)) => {
                        let diff = cur - prev;
                        if diff > 0.0 {
                            gain_builder.append_value(diff);
                            loss_builder.append_value(0.0);
                        } else {
                            gain_builder.append_value(0.0);
                            loss_builder.append_value(-diff);
                        }
                    }
                    _ => {
                        gain_builder.append_null();
                        loss_builder.append_null();
                    }
                }
            }
        }
    }

    let smooth_gain = calc_rma(&gain_builder.finish(), timeperiod);
    let smooth_loss = calc_rma(&loss_builder.finish(), timeperiod);

    let mut rsi_builder = PrimitiveChunkedBuilder::<Float64Type>::new("rsi".into(), n);
    for (g_arr, l_arr) in smooth_gain.downcast_iter().zip(smooth_loss.downcast_iter()) {
        let fast = g_arr.null_count() == 0 && l_arr.null_count() == 0;
        if fast {
            for (&g, &l) in g_arr.values_iter().zip(l_arr.values_iter()) {
                if l == 0.0 {
                    rsi_builder.append_value(100.0);
                } else {
                    rsi_builder.append_value(100.0 - 100.0 / (1.0 + g / l));
                }
            }
        } else {
            for (g_opt, l_opt) in g_arr.iter().zip(l_arr.iter()) {
                match (g_opt, l_opt) {
                    (Some(&g), Some(&l)) => {
                        if l == 0.0 {
                            rsi_builder.append_value(100.0);
                        } else {
                            rsi_builder.append_value(100.0 - 100.0 / (1.0 + g / l));
                        }
                    }
                    _ => rsi_builder.append_null(),
                }
            }
        }
    }

    Ok(rsi_builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn trix(inputs: &[Series], kwargs: TimeperiodKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    let n = real.len();

    let ema1 = calc_ema(real, timeperiod);
    let ema2 = calc_ema(&ema1, timeperiod);
    let ema3 = calc_ema(&ema2, timeperiod);
    let ema3 = ema3.rechunk();
    let ema3_prev = ema3.shift(1i64);
    let ema3_prev = ema3_prev.rechunk();

    let mut trix_builder = PrimitiveChunkedBuilder::<Float64Type>::new("trix".into(), n);
    for (cur_arr, prev_arr) in ema3.downcast_iter().zip(ema3_prev.downcast_iter()) {
        let fast = cur_arr.null_count() == 0 && prev_arr.null_count() == 0;
        if fast {
            for (&cur, &prev) in cur_arr.values_iter().zip(prev_arr.values_iter()) {
                if prev == 0.0 {
                    trix_builder.append_value(0.0);
                } else {
                    trix_builder.append_value((cur - prev) / prev * 100.0);
                }
            }
        } else {
            for (cur_opt, prev_opt) in cur_arr.iter().zip(prev_arr.iter()) {
                match (cur_opt, prev_opt) {
                    (Some(&cur), Some(&prev)) => {
                        if prev == 0.0 {
                            trix_builder.append_value(0.0);
                        } else {
                            trix_builder.append_value((cur - prev) / prev * 100.0);
                        }
                    }
                    _ => trix_builder.append_null(),
                }
            }
        }
    }

    Ok(trix_builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn ultosc(inputs: &[Series], kwargs: UltoscKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let p1 = kwargs.timeperiod1.unwrap_or(7);
    let p2 = kwargs.timeperiod2.unwrap_or(14);
    let p3 = kwargs.timeperiod3.unwrap_or(28);
    let max_period = p1.max(p2).max(p3);

    let n = high.len();

    if p1 == 0 || p2 == 0 || p3 == 0 || n < max_period {
        return Ok(Float64Chunked::full_null("ultosc".into(), n).into_series());
    }

    let prev_close = close.shift(1i64);
    let prev_close = prev_close.rechunk();

    let mut bp_builder = PrimitiveChunkedBuilder::<Float64Type>::new("bp".into(), n);
    let mut tr_builder = PrimitiveChunkedBuilder::<Float64Type>::new("tr".into(), n);

    for (((h_arr, l_arr), c_arr), pc_arr) in high
        .downcast_iter()
        .zip(low.downcast_iter())
        .zip(close.downcast_iter())
        .zip(prev_close.downcast_iter())
    {
        let fast = h_arr.null_count() == 0
            && l_arr.null_count() == 0
            && c_arr.null_count() == 0
            && pc_arr.null_count() == 0;
        if fast {
            for (((&h, &l), &c), &pc) in h_arr
                .values_iter()
                .zip(l_arr.values_iter())
                .zip(c_arr.values_iter())
                .zip(pc_arr.values_iter())
            {
                let min_lp = l.min(pc);
                let max_hp = h.max(pc);
                bp_builder.append_value(c - min_lp);
                tr_builder.append_value(max_hp - min_lp);
            }
        } else {
            for (((h_opt, l_opt), c_opt), pc_opt) in h_arr
                .iter()
                .zip(l_arr.iter())
                .zip(c_arr.iter())
                .zip(pc_arr.iter())
            {
                match (h_opt, l_opt, c_opt, pc_opt) {
                    (Some(&h), Some(&l), Some(&c), Some(&pc)) => {
                        let min_lp = l.min(pc);
                        let max_hp = h.max(pc);
                        bp_builder.append_value(c - min_lp);
                        tr_builder.append_value(max_hp - min_lp);
                    }
                    (Some(&h), Some(&l), Some(&c), None) => {
                        bp_builder.append_value(c - l);
                        tr_builder.append_value(h - l);
                    }
                    _ => {
                        bp_builder.append_null();
                        tr_builder.append_null();
                    }
                }
            }
        }
    }

    let bp = bp_builder.finish();
    let tr = tr_builder.finish();

    let sma_bp1 = calc_sma(&bp, p1);
    let sma_tr1 = calc_sma(&tr, p1);
    let sma_bp2 = calc_sma(&bp, p2);
    let sma_tr2 = calc_sma(&tr, p2);
    let sma_bp3 = calc_sma(&bp, p3);
    let sma_tr3 = calc_sma(&tr, p3);

    let mut ultosc_builder = PrimitiveChunkedBuilder::<Float64Type>::new("ultosc".into(), n);

    for (((bp1_arr, tr1_arr), (bp2_arr, tr2_arr)), (bp3_arr, tr3_arr)) in sma_bp1
        .downcast_iter()
        .zip(sma_tr1.downcast_iter())
        .zip(sma_bp2.downcast_iter().zip(sma_tr2.downcast_iter()))
        .zip(sma_bp3.downcast_iter().zip(sma_tr3.downcast_iter()))
    {
        let fast = bp1_arr.null_count() == 0
            && tr1_arr.null_count() == 0
            && bp2_arr.null_count() == 0
            && tr2_arr.null_count() == 0
            && bp3_arr.null_count() == 0
            && tr3_arr.null_count() == 0;
        if fast {
            for (((&bp1, &tr1), (&bp2, &tr2)), (&bp3, &tr3)) in bp1_arr
                .values_iter()
                .zip(tr1_arr.values_iter())
                .zip(bp2_arr.values_iter().zip(tr2_arr.values_iter()))
                .zip(bp3_arr.values_iter().zip(tr3_arr.values_iter()))
            {
                let avg1 = if tr1 == 0.0 { 0.0 } else { bp1 / tr1 };
                let avg2 = if tr2 == 0.0 { 0.0 } else { bp2 / tr2 };
                let avg3 = if tr3 == 0.0 { 0.0 } else { bp3 / tr3 };
                let osc = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;
                ultosc_builder.append_value(osc);
            }
        } else {
            for (((bp1_opt, tr1_opt), (bp2_opt, tr2_opt)), (bp3_opt, tr3_opt)) in bp1_arr
                .iter()
                .zip(tr1_arr.iter())
                .zip(bp2_arr.iter().zip(tr2_arr.iter()))
                .zip(bp3_arr.iter().zip(tr3_arr.iter()))
            {
                match (bp1_opt, tr1_opt, bp2_opt, tr2_opt, bp3_opt, tr3_opt) {
                    (Some(&bp1), Some(&tr1), Some(&bp2), Some(&tr2), Some(&bp3), Some(&tr3)) => {
                        let avg1 = if tr1 == 0.0 { 0.0 } else { bp1 / tr1 };
                        let avg2 = if tr2 == 0.0 { 0.0 } else { bp2 / tr2 };
                        let avg3 = if tr3 == 0.0 { 0.0 } else { bp3 / tr3 };
                        let osc = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;
                        ultosc_builder.append_value(osc);
                    }
                    _ => ultosc_builder.append_null(),
                }
            }
        }
    }

    Ok(ultosc_builder.finish().into_series())
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
    let n = high.len();

    if timeperiod == 0 || n < timeperiod {
        return Ok(Float64Chunked::full_null("willr".into(), n).into_series());
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("willr".into(), n);
    let mut high_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod);
    let mut low_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod);
    let mut count: usize = 0;

    for ((h_arr, l_arr), c_arr) in high
        .downcast_iter()
        .zip(low.downcast_iter())
        .zip(close.downcast_iter())
    {
        let fast = h_arr.null_count() == 0 && l_arr.null_count() == 0 && c_arr.null_count() == 0;
        if fast {
            for ((&h, &l), &c) in h_arr
                .values_iter()
                .zip(l_arr.values_iter())
                .zip(c_arr.values_iter())
            {
                while let Some(&(_, v)) = high_max.back() {
                    if v <= h {
                        high_max.pop_back();
                    } else {
                        break;
                    }
                }
                high_max.push_back((count, h));

                while let Some(&(_, v)) = low_min.back() {
                    if v >= l {
                        low_min.pop_back();
                    } else {
                        break;
                    }
                }
                low_min.push_back((count, l));

                count += 1;

                if count < timeperiod {
                    builder.append_null();
                } else {
                    let expired = count.saturating_sub(timeperiod);
                    if let Some(&(idx, _)) = high_max.front() {
                        if idx < expired {
                            high_max.pop_front();
                        }
                    }
                    if let Some(&(idx, _)) = low_min.front() {
                        if idx < expired {
                            low_min.pop_front();
                        }
                    }

                    let max_h = high_max.front().unwrap().1;
                    let min_l = low_min.front().unwrap().1;
                    let denom = max_h - min_l;

                    if denom == 0.0 {
                        builder.append_value(0.0);
                    } else {
                        builder.append_value(-100.0 * (max_h - c) / denom);
                    }
                }
            }
        } else {
            for ((h_opt, l_opt), c_opt) in h_arr.iter().zip(l_arr.iter()).zip(c_arr.iter()) {
                match (h_opt, l_opt, c_opt) {
                    (Some(&h), Some(&l), Some(&c)) => {
                        while let Some(&(_, v)) = high_max.back() {
                            if v <= h {
                                high_max.pop_back();
                            } else {
                                break;
                            }
                        }
                        high_max.push_back((count, h));

                        while let Some(&(_, v)) = low_min.back() {
                            if v >= l {
                                low_min.pop_back();
                            } else {
                                break;
                            }
                        }
                        low_min.push_back((count, l));

                        count += 1;

                        if count < timeperiod {
                            builder.append_null();
                        } else {
                            let expired = count.saturating_sub(timeperiod);
                            if let Some(&(idx, _)) = high_max.front() {
                                if idx < expired {
                                    high_max.pop_front();
                                }
                            }
                            if let Some(&(idx, _)) = low_min.front() {
                                if idx < expired {
                                    low_min.pop_front();
                                }
                            }

                            let max_h = high_max.front().unwrap().1;
                            let min_l = low_min.front().unwrap().1;
                            let denom = max_h - min_l;

                            if denom == 0.0 {
                                builder.append_value(0.0);
                            } else {
                                builder.append_value(-100.0 * (max_h - c) / denom);
                            }
                        }
                    }
                    _ => {
                        builder.append_null();
                        count += 1;
                    }
                }
            }
        }
    }

    Ok(builder.finish().into_series())
}

pub fn calc_dm(high: &Float64Chunked, low: &Float64Chunked) -> (Float64Chunked, Float64Chunked) {
    let high = high.rechunk();
    let low = low.rechunk();
    let n = high.len();
    let prev_high = high.shift(1i64);
    let prev_high = prev_high.rechunk();
    let prev_low = low.shift(1i64);
    let prev_low = prev_low.rechunk();

    let mut plus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("plus_dm".into(), n);
    let mut minus_builder = PrimitiveChunkedBuilder::<Float64Type>::new("minus_dm".into(), n);
    let mut idx: usize = 0;

    for ((h_arr, ph_arr), (l_arr, pl_arr)) in high
        .downcast_iter()
        .zip(prev_high.downcast_iter())
        .zip(low.downcast_iter().zip(prev_low.downcast_iter()))
    {
        let fast = h_arr.null_count() == 0
            && ph_arr.null_count() == 0
            && l_arr.null_count() == 0
            && pl_arr.null_count() == 0;
        if fast {
            for ((&h, &ph), (&l, &pl)) in h_arr
                .values_iter()
                .zip(ph_arr.values_iter())
                .zip(l_arr.values_iter().zip(pl_arr.values_iter()))
            {
                if idx == 0 {
                    plus_builder.append_value(0.0);
                    minus_builder.append_value(0.0);
                    idx += 1;
                    continue;
                }
                let up = h - ph;
                let down = pl - l;
                if up > down && up > 0.0 {
                    plus_builder.append_value(up);
                    minus_builder.append_value(0.0);
                } else if down > up && down > 0.0 {
                    plus_builder.append_value(0.0);
                    minus_builder.append_value(down);
                } else {
                    plus_builder.append_value(0.0);
                    minus_builder.append_value(0.0);
                }
                idx += 1;
            }
        } else {
            for ((h_opt, ph_opt), (l_opt, pl_opt)) in h_arr
                .iter()
                .zip(ph_arr.iter())
                .zip(l_arr.iter().zip(pl_arr.iter()))
            {
                if idx == 0 {
                    match (h_opt, l_opt) {
                        (Some(_), Some(_)) => {
                            plus_builder.append_value(0.0);
                            minus_builder.append_value(0.0);
                        }
                        _ => {
                            plus_builder.append_null();
                            minus_builder.append_null();
                        }
                    }
                    idx += 1;
                    continue;
                }
                match (h_opt, l_opt) {
                    (Some(&h), Some(&l)) => match (ph_opt, pl_opt) {
                        (Some(&ph), Some(&pl)) => {
                            let up = h - ph;
                            let down = pl - l;
                            if up > down && up > 0.0 {
                                plus_builder.append_value(up);
                                minus_builder.append_value(0.0);
                            } else if down > up && down > 0.0 {
                                plus_builder.append_value(0.0);
                                minus_builder.append_value(down);
                            } else {
                                plus_builder.append_value(0.0);
                                minus_builder.append_value(0.0);
                            }
                        }
                        _ => {
                            plus_builder.append_value(0.0);
                            minus_builder.append_value(0.0);
                        }
                    },
                    _ => {
                        plus_builder.append_null();
                        minus_builder.append_null();
                    }
                }
                idx += 1;
            }
        }
    }

    (plus_builder.finish(), minus_builder.finish())
}

pub fn calc_adx(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("adx".into(), n);
    }

    let (plus_di, minus_di) = calc_di(high, low, close, timeperiod);

    let mut dx_builder = PrimitiveChunkedBuilder::<Float64Type>::new("dx".into(), n);
    for (pdi_arr, mdi_arr) in plus_di.downcast_iter().zip(minus_di.downcast_iter()) {
        let fast = pdi_arr.null_count() == 0 && mdi_arr.null_count() == 0;
        if fast {
            for (&pdi, &mdi) in pdi_arr.values_iter().zip(mdi_arr.values_iter()) {
                let denom = pdi + mdi;
                if denom == 0.0 {
                    dx_builder.append_value(0.0);
                } else {
                    dx_builder.append_value(100.0 * (pdi - mdi).abs() / denom);
                }
            }
        } else {
            for (pdi_opt, mdi_opt) in pdi_arr.iter().zip(mdi_arr.iter()) {
                match (pdi_opt, mdi_opt) {
                    (Some(&pdi), Some(&mdi)) => {
                        let denom = pdi + mdi;
                        if denom == 0.0 {
                            dx_builder.append_value(0.0);
                        } else {
                            dx_builder.append_value(100.0 * (pdi - mdi).abs() / denom);
                        }
                    }
                    _ => dx_builder.append_null(),
                }
            }
        }
    }

    let dx = dx_builder.finish();
    calc_rma(&dx, timeperiod)
}

pub fn calc_adxr(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("adxr".into(), n);
    }

    let adx = calc_adx(high, low, close, timeperiod);
    let adx = adx.rechunk();
    let adx_lag = adx.shift((timeperiod - 1) as i64);
    let adx_lag = adx_lag.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("adxr".into(), n);
    for (cur_arr, lag_arr) in adx.downcast_iter().zip(adx_lag.downcast_iter()) {
        let fast = cur_arr.null_count() == 0 && lag_arr.null_count() == 0;
        if fast {
            for (&cur, &lag) in cur_arr.values_iter().zip(lag_arr.values_iter()) {
                builder.append_value((cur + lag) / 2.0);
            }
        } else {
            for (cur_opt, lag_opt) in cur_arr.iter().zip(lag_arr.iter()) {
                match (cur_opt, lag_opt) {
                    (Some(&cur), Some(&lag)) => builder.append_value((cur + lag) / 2.0),
                    _ => builder.append_null(),
                }
            }
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

    if timeperiod == 0 || n < timeperiod {
        let nulls = Float64Chunked::full_null("di".into(), n);
        return (nulls.clone(), nulls);
    }

    let (plus_dm, minus_dm) = calc_dm(high, low);
    let tr = calc_trange(high, low, close);

    let smooth_plus_dm = calc_rma(&plus_dm, timeperiod);
    let smooth_plus_dm = smooth_plus_dm.rechunk();
    let smooth_minus_dm = calc_rma(&minus_dm, timeperiod);
    let smooth_minus_dm = smooth_minus_dm.rechunk();
    let smooth_tr = calc_rma(&tr, timeperiod);
    let smooth_tr = smooth_tr.rechunk();

    let mut plus_di_builder = PrimitiveChunkedBuilder::<Float64Type>::new("plus_di".into(), n);
    let mut minus_di_builder = PrimitiveChunkedBuilder::<Float64Type>::new("minus_di".into(), n);

    for ((pdm_arr, mdm_arr), tr_arr) in smooth_plus_dm
        .downcast_iter()
        .zip(smooth_minus_dm.downcast_iter())
        .zip(smooth_tr.downcast_iter())
    {
        let fast =
            pdm_arr.null_count() == 0 && mdm_arr.null_count() == 0 && tr_arr.null_count() == 0;
        if fast {
            for ((&pdm, &mdm), &tr_val) in pdm_arr
                .values_iter()
                .zip(mdm_arr.values_iter())
                .zip(tr_arr.values_iter())
            {
                if tr_val == 0.0 {
                    plus_di_builder.append_value(0.0);
                    minus_di_builder.append_value(0.0);
                } else {
                    plus_di_builder.append_value(100.0 * pdm / tr_val);
                    minus_di_builder.append_value(100.0 * mdm / tr_val);
                }
            }
        } else {
            for ((pdm_opt, mdm_opt), tr_opt) in
                pdm_arr.iter().zip(mdm_arr.iter()).zip(tr_arr.iter())
            {
                match (pdm_opt, mdm_opt, tr_opt) {
                    (Some(&pdm), Some(&mdm), Some(&tr_val)) => {
                        if tr_val == 0.0 {
                            plus_di_builder.append_value(0.0);
                            minus_di_builder.append_value(0.0);
                        } else {
                            plus_di_builder.append_value(100.0 * pdm / tr_val);
                            minus_di_builder.append_value(100.0 * mdm / tr_val);
                        }
                    }
                    _ => {
                        plus_di_builder.append_null();
                        minus_di_builder.append_null();
                    }
                }
            }
        }
    }

    (plus_di_builder.finish(), minus_di_builder.finish())
}
